
import os
import glob
import torch
import datasets.utils
import cv2
import argparse
import time
from path import Path
from tensorboardX import SummaryWriter
from tqdm import tqdm

import numpy as np
from torchvision.transforms import Compose
import torch.nn.functional as F

from imutils.video import VideoStream
# from midas.model_loader import default_models, load_model
from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

from datasets.UCL import UCL_Dataset, SimCol3D_Dataset, C3VD_Dataset

def save_model(model, save_path, epoch, is_best=False):
    save_path.makedirs_p()
    state = model.state_dict()
    filename = "{}".format(epoch) if not is_best else "best_{}".format(epoch)
    torch.save(
        state,
        save_path/'{}.pth.tar'.format(filename)
    )

def compute_loss(disp_pred, disp_gt):
    """depth = norm(1/output)

    Args:
        disp_pred (_type_): _description_
        disp_gt (_type_): _description_

    Returns:
        _type_: _description_
    """
    B,H,W = disp_pred.shape
    disp_pred_ = 1/(disp_pred+22.2) # ! 加上一个数防止 div0
    # disp_pred_ = 1/(disp_pred+44.4) # ! 加上一个数防止 div0
    # disp_pred_ = 1/(disp_pred+88.8) # ! 加上一个数防止 div0
    # disp_pred_ = 1/(disp_pred+166.6) # ! 加上一个数防止 div0
    # disp_pred_ = 1/(disp_pred+333.3) # ! 加上一个数防止 div0
    max_value,min_value = torch.max(disp_pred_.view((B,-1)), axis=1,keepdim=True)[0], torch.min(disp_pred_.view((B,-1)),axis=1,keepdim=True)[0]
    disp_norm = (disp_pred_.view((B,-1))-min_value)/(max_value-min_value)

    disp_norm = torch.nn.functional.interpolate(disp_norm.unsqueeze(1).view(B,1,H,W),disp_gt.shape[1:]).squeeze(1) # 插值改变图片大小
    loss = torch.norm(disp_norm.view((B,-1))-disp_gt.view((B,-1)))
    return loss, disp_norm

def compute_loss2(disp_pred, depth_gt):
    """depth = 1-norm(output)

    Args:
        disp_pred (_type_): _description_
        depth_gt (_type_): _description_

    Returns:
        _type_: _description_
    """
    B,H,W = disp_pred.shape
    
    max_value,min_value = torch.max(disp_pred.view((B,-1)), axis=1,keepdim=True)[0], torch.min(disp_pred.view((B,-1)),axis=1,keepdim=True)[0]
    disp_norm = (disp_pred.view((B,-1))-min_value)/(max_value-min_value)
    
    disp_norm = torch.nn.functional.interpolate(disp_norm.unsqueeze(1).view(B,1,H,W),depth_gt.shape[1:]).squeeze(1) # 插值改变图片大小
    loss = torch.norm( (1-disp_norm.view((B,-1))) - depth_gt.view((B,-1)) ) # 将预测值转换为 depth 的方法
    
    return loss, 1-disp_norm

def compute_loss3(disp_pred, depth_gt):
    """depth = 1-sigmoid((output-5000)/10000)

    Args:
        disp_pred (_type_): _description_
        depth_gt (_type_): _description_

    Returns:
        _type_: _description_
    """
    B,H,W = disp_pred.shape
    
    disp_pred = (disp_pred-5000)/10000.0 # NOTE 经验的系数
    disp_norm = torch.sigmoid(disp_pred) # 输出归一化
    
    disp_norm = torch.nn.functional.interpolate(disp_norm.unsqueeze(1).view(B,1,H,W),depth_gt.shape[1:]).squeeze(1) # 插值改变图片大小
    loss = torch.norm( (1-disp_norm.view((B,-1))) - depth_gt.view((B,-1)) ) # 将预测值转换为 depth 的方法
    
    return loss, 1-disp_norm

def compute_loss4(disp_pred, disp_gt):
    """depth = log(1+norm(1/output))

    Args:
        disp_pred (_type_): _description_
        disp_gt (_type_): _description_

    Returns:
        _type_: _description_
    """
    B,H,W = disp_pred.shape
    disp_pred_ = 1/disp_pred
    max_value,min_value = torch.max(disp_pred_.view((B,-1)), axis=1,keepdim=True)[0], torch.min(disp_pred_.view((B,-1)),axis=1,keepdim=True)[0]
    disp_norm = torch.log2(1+(disp_pred_.view((B,-1))-min_value)/(max_value-min_value))

    disp_norm = torch.nn.functional.interpolate(disp_norm.unsqueeze(1).view(B,1,H,W),disp_gt.shape[1:]).squeeze(1) # 插值改变图片大小
    loss = torch.norm(disp_norm.view((B,-1))-disp_gt.view((B,-1)))
    return loss, disp_norm


best_error = -1
n_iter = 0
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.autograd.set_detect_anomaly(True)

my_writers = {}

def train(train_loader, model, optimizer, epoch, training_writer):
    loss_sum = 0
    n = 0
    print("=> training")
    for i,(input,disp_gt) in enumerate(tqdm(train_loader)):
        input, disp_gt = input.to('cuda'), disp_gt.to('cuda')
        _,h,w = disp_gt.shape
        
        disp_pred = model.forward(input)
        disp_pred = F.interpolate(disp_pred[None], (h, w), mode='bilinear', align_corners=False)[0]
        loss,_ = compute_loss(disp_pred,disp_gt) # depth = norm(1/pred)
        # loss,_ = compute_loss2(disp_pred,disp_gt) # depth = 1-pred 计算深度
        # loss,_ = compute_loss3(disp_pred,disp_gt) # depth = 1-sigmoid(pred) 计算深度

        training_writer.add_scalar(
            'L2_loss', loss.item(), n
        )
        
        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_sum += loss.item()
        n += 1

    return loss_sum / n
  
@torch.no_grad()
def validate(val_loader, model, optimizer, epoch, output_writers):
    loss_sum = 0
    n = 0
    print("=> validating")
    for i,(input,disp_gt, ori_input) in enumerate(tqdm(val_loader)):
        input, disp_gt = input.to('cuda'), disp_gt.to('cuda')
        _,h,w = disp_gt.shape
        
        disp_pred = model.forward(input)
        disp_pred = F.interpolate(disp_pred[None], (h, w), mode='bilinear', align_corners=False)[0]
        loss,disp_norm = compute_loss(disp_pred,disp_gt)
        # loss,disp_norm = compute_loss2(disp_pred,disp_gt) # depth = 1-pred 计算深度
        # loss,disp_norm = compute_loss3(disp_pred,disp_gt) # depth = 1-sigmoid(pred) 计算深度
        # loss,disp_norm = compute_loss4(disp_pred,disp_gt) # depth = log(1+norm(1/output)) 计算深度

        output_writers[-1].add_scalar(
            'val L2_loss', loss.item(), n
        )
        
        if i < len(output_writers)-1:
            if epoch == 0:
                output_writers[i].add_image('val Input', ori_input[0].detach().cpu().numpy(), 0)
                output_writers[i].add_image(
                    'val GT', disp_gt[0].unsqueeze(0).detach().cpu().numpy(), 0
                )
            output_writers[i].add_image(
                'val Pred', disp_norm[0].unsqueeze(0).detach().cpu().numpy(), epoch
            )
        
        loss_sum += loss.item()
        n += 1

    return loss_sum / n
    

def main(input_path, output_path, model_path, model_type="dpt_beit_large_512", optimize=False, side=False, height=None,
        square=False, grayscale=False):
    # select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: %s" % device)

    import datetime
    timestamp = datetime.datetime.now().strftime("%m-%d-%H:%M")
    args.save_path = Path(args.save_path)/timestamp
    print('=> will save everything to {}'.format(args.save_path))
    args.save_path.makedirs_p()

    training_writer = SummaryWriter(args.save_path)
    output_writers = []
    if True:
        for i in range(4):
            output_writers.append(SummaryWriter(args.save_path/'valid'/str(i)))
    
    # 加载模型
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    depth_anything = DepthAnything.from_pretrained('./checkpoints/depth_anything_vitl14', local_files_only=True).to(DEVICE)
    model = depth_anything

    model.train()
    
    transform = Compose([
            Resize(
                # width=518,
                # height=518,
                width=256,
                height=256,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])
    
    # model, transform, net_w, net_h = load_model(device, model_path, model_type, optimize, height, square)
    # ! UCL
    # train_set = UCL_Dataset(input_path, transform=transform, train=True)
    # val_set = UCL_Dataset(input_path, transform=transform, train=False)
    # ! SimCol3D
    # train_set = SimCol3D_Dataset(input_path, transform=transform, train=True)
    # val_set = SimCol3D_Dataset(input_path, transform=transform, train=False)
    # ! C3VD
    train_set = C3VD_Dataset(input_path, transform=transform, train=True)
    val_set = C3VD_Dataset(input_path, transform=transform, train=False)
    
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers)
    
    optim_params = [
        {'params': model.parameters(), 'lr': args.lr}
    ]
    optimizer = torch.optim.Adam(optim_params,
                                 betas=(0.9, 0.999),
                                 weight_decay=0)
    
    train_losses, val_losses = [], []
    best_loss = 1e5
    for epoch in range(args.epochs):
        print("------------------ {} epoch -------------------".format(epoch))
        train_loss = train(train_loader, model, optimizer, epoch, training_writer)

        val_loss = validate(val_loader, model, optimizer, epoch, output_writers)
        
        if val_loss < best_loss:
            best_loss = val_loss
            save_model(model, args.save_path/'checkpoints', epoch=epoch, is_best=True)
        if epoch % 1 == 0:
            save_model(model, args.save_path/'checkpoints', epoch=epoch, is_best=False)
            
        print("train loss: {}, val loss: {}".format(train_loss, val_loss))
        print("------------------ ******** -------------------")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input_path',
                        default=None,
                        help='Folder with input images (if no input path is specified, images are tried to be grabbed '
                             'from camera)'
                        )

    parser.add_argument('-o', '--output_path',
                        default=None,
                        help='Folder for output images'
                        )

    parser.add_argument('-m', '--model_weights',
                        default=None,
                        help='Path to the trained weights of model'
                        )

    parser.add_argument('-t', '--model_type',
                        default='dpt_beit_large_512',
                        help='Model type: '
                             'dpt_beit_large_512, dpt_beit_large_384, dpt_beit_base_384, dpt_swin2_large_384, '
                             'dpt_swin2_base_384, dpt_swin2_tiny_256, dpt_swin_large_384, dpt_next_vit_large_384, '
                             'dpt_levit_224, dpt_large_384, dpt_hybrid_384, midas_v21_384, midas_v21_small_256 or '
                             'openvino_midas_v21_small_256'
                        )

    parser.add_argument('-s', '--side',
                        action='store_true',
                        help='Output images contain RGB and depth images side by side'
                        )

    parser.add_argument('--optimize', dest='optimize', action='store_true', help='Use half-float optimization')
    parser.set_defaults(optimize=False)

    parser.add_argument('--height',
                        type=int, default=None,
                        help='Preferred height of images feed into the encoder during inference. Note that the '
                             'preferred height may differ from the actual height, because an alignment to multiples of '
                             '32 takes place. Many models support only the height chosen during training, which is '
                             'used automatically if this parameter is not set.'
                        )
    parser.add_argument('--square',
                        action='store_true',
                        help='Option to resize images to a square resolution by changing their widths when images are '
                             'fed into the encoder during inference. If this parameter is not set, the aspect ratio of '
                             'images is tried to be preserved if supported by the model.'
                        )
    parser.add_argument('--grayscale',
                        action='store_true',
                        help='Use a grayscale colormap instead of the inferno one. Although the inferno colormap, '
                             'which is used by default, is better for visibility, it does not allow storing 16-bit '
                             'depth values in PNGs but only 8-bit ones due to the precision limitation of this '
                             'colormap.'
                        )
    parser.add_argument("--batch_size", default=1)
    parser.add_argument("--workers", default=4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", default=1e-5)
    parser.add_argument("--save_path", default="./log")
    
    
    args = parser.parse_args()


    # if args.model_weights is None:
    #     args.model_weights = default_models[args.model_type]

    # set torch options
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # compute depth maps
    main(args.input_path, args.output_path, args.model_weights, args.model_type, args.optimize, args.side, args.height,
        args.square, args.grayscale)

