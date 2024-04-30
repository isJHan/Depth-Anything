
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
from datasets.utils import read_pfm

# from datasets.UCL import UCL_Dataset, SimCol3D_Dataset, C3VD_Dataset
import torch.utils.data as data



import random

def read_image(path):
    """Read image and output RGB image (0-1).

    Args:
        path (str): path to file

    Returns:
        array: RGB image (0-1)
    """
    img = cv2.imread(path)

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0

    return img

class C3VD_Dataset(data.Dataset):
    """A sequence data loader where the files are arranged in this way:
        root/scene_1/0000000.jpg
        root/scene_1/0000001.jpg
        ..
        root/scene_1/cam.txt
        root/scene_2/0000000.jpg
        .
        transform functions must take in a list a images and a numpy array (usually intrinsics matrix)
    """

    def __init__(self, input_path, transform, train, n_selected=4, selected_size=256, if_transform=True):
        np.random.seed(1)
        random.seed(1)
        self.is_train = train
        self.root = Path(input_path)
        self.transform = transform
        self.if_transform = if_transform
        self.n_selected = n_selected
        self.selected_size = selected_size
        self.h, self.w = 0, 0
        self.scale_factor = 0.5
        
        self.generateSample()
    
    
    def generate_random_points(self, w, h, N):
        points = []
        for _ in range(N):
            x = random.randint(0, w - 1)
            y = random.randint(0, h-1)
            points.append((x, y))
        return points

    def select_parts(self, N, size, input, h,w):
        start_points = self.generate_random_points(w-size ,h-size, N)
        croped_imgs = []
        for i in range(N):
            x_start,y_start = start_points[i][0], start_points[i][1]
            croped = input[y_start:y_start+size,x_start:x_start+size]
            # croped = transform({"image": croped})["image"] # jh in [-1, 1]
            croped_imgs.append(croped)
        return croped_imgs, start_points

    
    def generateSample(self):
        self.sample_input, self.sample_gt, self.sample_disp_base = [], [], []

        scenes = []
        filename = self.root/'train4Midas.txt' if self.is_train else self.root/'val4Midas.txt'
        with open(filename, 'r') as f:
            lines = f.readlines()
        scenes = [self.root/line[:-1] for line in lines]
        
        for scene in scenes:
            rgb_files, depth_files = sorted(scene.listdir("*.jpg")), sorted((scene/'depth_gt').listdir("*.npy"))
            self.sample_input += rgb_files
            self.sample_gt += depth_files
            self.sample_disp_base += sorted((scene/'output_monodepth').listdir("*.pfm"))
        
        

    def __getitem__(self, index):
        image_name = self.sample_input[index]
        depth_gt = np.load(self.sample_gt[index])
        disp_base, _ = read_pfm(self.sample_disp_base[index]) # read basic depth map
        
        # down sample
        depth_gt = cv2.resize(depth_gt, None, fx=self.scale_factor, fy=self.scale_factor)
        disp_base = cv2.resize(disp_base, None, fx=self.scale_factor, fy=self.scale_factor)
        original_image_rgb = read_image(image_name)  # in [0, 1] [H,W,3]
        original_image_rgb = cv2.resize(original_image_rgb,None, fx=self.scale_factor,fy=self.scale_factor) # ! 由于OOM 下采样
        h,w,_ = original_image_rgb.shape
        
        parts, start_points = self.select_parts(self.n_selected, self.selected_size, original_image_rgb, h, w)
        
        if self.if_transform: image = self.transform({"image": original_image_rgb})["image"] # jh in [-1, 1]
        image = torch.from_numpy(image)
        if self.if_transform:
            for i in range(len(parts)):
                parts[i] = self.transform({"image": parts[i]})["image"] # jh in [-1, 1]
                parts[i] = torch.from_numpy(parts[i])
        
        
        if self.is_train:
            return image, parts,start_points, original_image_rgb, disp_base
        else:
            return image, original_image_rgb, depth_gt

    def __len__(self):
        return len(self.sample_input)



def save_model(model, save_path, epoch, is_best=False):
    save_path.makedirs_p()
    state = model.state_dict()
    filename = "{}".format(epoch) if not is_best else "best_{}".format(epoch)
    torch.save(
        state,
        save_path/'{}.pth.tar'.format(filename)
    )

def _discard_compute_loss(disp_pred, disp_gt):
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

    pred, gt = disp_pred[0].detach().cpu().numpy(), disp_gt[0].detach().cpu().numpy()
    pred = pred * np.median(gt)/np.median(pred)
    loss = abs(pred-gt).mean()
    return loss, disp_norm


best_error = -1
n_iter = 0
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.autograd.set_detect_anomaly(True)

my_writers = {}

def min_max_norm(value):
    return (value-value.min())/(value.max()-value.min())

def l1_error(gt, pred):
    '''suppose gt and pred are normlized.'''
    return abs(gt-pred).mean()

def pearson_error(gt, pred):
    '''
    nagitive pearson loss
    Suppose gt and pred are NOT normalized.
    '''
    B = 1
    tmp_gt, tmp_pred = gt.reshape((B,-1)), pred.reshape((B,-1))
    mean_gt, mean_pred = torch.mean(tmp_gt, dim=-1), torch.mean(tmp_pred, dim=-1)
    tmp_gt_centered, tmp_pred_centered = tmp_gt-mean_gt, tmp_pred-mean_pred
    cov = torch.sum(tmp_gt_centered*tmp_pred_centered,dim=-1)
    var_gt = torch.sum(tmp_gt_centered**2,dim=-1)
    var_pred = torch.sum(tmp_pred_centered**2,dim=-1)
    
    return ((1 - cov/torch.sqrt(var_gt*var_pred)) /2).mean()

def RPNL_error(gt, pred):
    B = 1
    tmp_gt, tmp_pred = gt.reshape((B,-1)), pred.reshape((B,-1))
    median_gt, median_pred = torch.median(tmp_gt,dim=-1), torch.median(tmp_pred,dim=-1)
    scale_gt, scale_pred = abs(tmp_gt-median_gt).mean(), abs(tmp_pred-median_pred).mean()
    
    return abs( scale_gt*(tmp_gt-median_gt) - scale_pred*(tmp_pred-median_pred) ).mean()

# TODO - SSIM

# TODO - 傅立叶变换，中低频监督，高频过滤
def __generate_mask(h,w, radius):
    mask = np.ones((h,w))
    center_h = h // 2
    center_w = w // 2

    # 生成半径为radius内的掩码为0
    for i in range(h):
        for j in range(w):
            if (i - center_h)**2 + (j - center_w)**2 <= radius**2:
                mask[i, j] = 0
                
    return mask
def __display_fft_value(fft_value):
    magnitude_spectrum = torch.log(torch.abs(fft_value) + 1)  # 加1避免log(0)
    phase_spectrum = torch.angle(fft_value)
    return phase_spectrum, magnitude_spectrum

def compute_fourier_error(gt, pred):
    B,h,w = gt.shape
    filter_mask = None
    fft_gt, fft_pred = torch.fft.fft2(gt), torch.fft.fft2(pred)
    filter_mask = torch.from_numpy(__generate_mask(h,w, 20)[None,]).to(device)
    fft_gt, fft_pred = fft_gt * filter_mask, fft_pred * filter_mask
    error = torch.linalg.norm(fft_gt-fft_pred)/(h*w) # L2 error
    return error

def compute_self_loss(start_points, parts_pred, disp_pred, size):
    parts_gt = []
    loss = 0.0
    B,h,w = disp_pred.shape
    for i in range(len(start_points)):
        p = start_points[i]
        for j in range(B):
            x,y = p[0][j], p[1][j]
            loss += l1_error(min_max_norm(disp_pred[j][y:y+size,x:x+size]), min_max_norm(parts_pred[i][j])) # l1 error
            # loss += pearson_error(disp_pred[j][y:y+size,x:x+size], parts_pred[i][j]) # pearson error
    return loss

def compute_semi_loss(disp_pred,disp_base):
    # TODO
    # loss = pearson_error(disp_base, disp_pred)
    # loss = l1_error(disp_base, disp_pred)
    loss = compute_fourier_error(disp_base, disp_pred) # Fourier error
    
    return loss

def train(args, train_loader, model, optimizer, epoch, training_writer):
    N_selected, selected_size = args.self_number, args.self_size # 数量和大小
    
    loss_sum = 0
    n = 0
    print("=> training")
    for i,(input,parts,start_points, ori_img, disp_base) in enumerate(tqdm(train_loader)):
        input = input.to('cuda')
        for i in range(len(parts)): parts[i] = parts[i].to('cuda')
        disp_base = disp_base.to('cuda')
        _, h,w,_ = ori_img.shape
        
        # with torch.no_grad(): 
        disp_pred = model.forward(input)
        disp_pred = F.interpolate(disp_pred[None], (h, w), mode='bilinear', align_corners=False)[0]
        
        parts_pred = []
        with torch.no_grad():
            for p in parts:
                pred = model.forward(p)
                pred = F.interpolate(pred[None], (selected_size, selected_size), mode='bilinear', align_corners=False)[0]
                parts_pred.append(pred)
        
        
        loss_self = compute_self_loss(start_points, parts_pred, disp_pred, selected_size)
        loss_semi = compute_semi_loss(disp_pred, disp_base)
        loss = 5.0*loss_self + 0.1*loss_semi
        # loss,_ = compute_loss(disp_pred,disp_gt) # depth = norm(1/pred)
        # loss,_ = compute_loss2(disp_pred,disp_gt) # depth = 1-pred 计算深度
        # loss,_ = compute_loss3(disp_pred,disp_gt) # depth = 1-sigmoid(pred) 计算深度

        training_writer.add_scalar(
            'self-L1_loss', loss_self.item(), n
        )
        training_writer.add_scalar(
            'semi-L1_loss', loss_semi.item(), n
        )
        training_writer.add_scalar(
            'loss', loss.item(), n
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
    for i,(input, ori_img, depth_gt) in enumerate(tqdm(val_loader)):
        # input, disp_gt = input.to('cuda'), disp_gt.to('cuda')
        depth_gt = depth_gt.to('cuda')
        input = input.to('cuda')
        # print(ori_img.shape)
        _, h,w,_ = ori_img.shape
        
        disp_pred = model.forward(input)
        disp_pred = F.interpolate(disp_pred[None], (h, w), mode='bilinear', align_corners=False)[0]
        # loss = compute_self_loss(start_points, parts_pred, disp_pred, selected_size)
        loss,disp_norm = compute_loss(disp_pred,depth_gt)
        # loss,disp_norm = compute_loss2(disp_pred,disp_gt) # depth = 1-pred 计算深度
        # loss,disp_norm = compute_loss3(disp_pred,disp_gt) # depth = 1-sigmoid(pred) 计算深度
        # loss,disp_norm = compute_loss4(disp_pred,disp_gt) # depth = log(1+norm(1/output)) 计算深度

        output_writers[-1].add_scalar(
            'val L2_loss', loss.item(), n
        )
        
        if i < len(output_writers)-1:
            if epoch == 0:
                output_writers[i].add_image('val Input', ori_img[0].detach().cpu().numpy().transpose((2,0,1)), 0)
                # output_writers[i].add_image(
                    # 'val GT', disp_gt[0].unsqueeze(0).detach().cpu().numpy(), 0
                # )
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
    tmp = torch.load("log/UCL/04-11-00:00/checkpoints/4.pth.tar") # 22.2 UCL 4-epoch
    # tmp = torch.load("/home/jiahan/jiahan/codes/Depth-Anything/log/C3VD/03-20-19:43/checkpoints/0.pth.tar") # 22.2 C3VD
    depth_anything.load_state_dict(tmp)
    
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
    train_set = C3VD_Dataset(input_path, transform=transform, train=True, selected_size=args.self_size, n_selected=args.self_number)
    val_set = C3VD_Dataset(input_path, transform=transform, train=False, selected_size=args.self_size, n_selected=args.self_number)
    
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
        # val_loss = validate(val_loader, model, optimizer, epoch, output_writers)

        train_loss = train(args, train_loader, model, optimizer, epoch, training_writer)

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
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--save_path", default="./log")

    parser.add_argument("--self_number", default=2, help="Number of selected area on one image when training in self-surpervised menner.")
    parser.add_argument("--self_size", default=128, help="Size of cliped images when training in self-supervised manner.")
    
    
    args = parser.parse_args()


    # if args.model_weights is None:
    #     args.model_weights = default_models[args.model_type]

    # set torch options
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # compute depth maps
    main(args.input_path, args.output_path, args.model_weights, args.model_type, args.optimize, args.side, args.height,
        args.square, args.grayscale)

