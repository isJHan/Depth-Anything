import argparse
import cv2
import numpy as np
import os
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
from tqdm import tqdm
from path import Path
from path import Path
from tqdm.contrib import tzip

from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet


def RMSE_func(gt,pred):
    return np.sqrt(np.mean((gt-pred)**2))

def MAE_func(gt,pred):
    return abs(gt-pred).mean()


def Rel_func2(gt,pred):
    """rel in paper SimCol3D using median

    Args:
        gt (_type_): _description_
        pred (_type_): _description_

    Returns:
        _type_: _description_
    """
    mask = gt>0
    return np.median(abs(gt-pred)/(gt+1e-5))


# 加载模型
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print("=> device ", DEVICE)

depth_anything = DepthAnything.from_pretrained('./checkpoints/depth_anything_vitl14', local_files_only=True).to(DEVICE)

transform = Compose([
        Resize(
            width=518,
            height=518,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])



def infer(filename):
    raw_image = cv2.imread(filename)
    image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
    
    h, w = image.shape[:2]
    
    image = transform({'image': image})['image']
    image = torch.from_numpy(image).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        depth = depth_anything(image)
    
    depth = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
    depth = (depth - depth.min()) / (depth.max() - depth.min())
    depth = depth.cpu().numpy()
    
    return depth

def infer_inv(filename):
    raw_image = cv2.imread(filename)
    image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
    
    h, w = image.shape[:2]
    
    image = transform({'image': image})['image']
    image = torch.from_numpy(image).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        depth = depth_anything(image)
    
    
    depth = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
    depth = depth.cpu().numpy()
    
    return depth


# 加载图片

# root_path = Path("/root/autodl-tmp/datasets/SimCol")
root_path = Path("/home/jiahan/jiahan/datasets/C3VD/.dataset4SCDepth") # ! C3VD
scenes = []

with open(root_path/'all.txt') as f:
    lines = f.readlines()
for line in lines:
    line = line[:-1]
    scenes.append(root_path/line)

print(type(scenes[0]))


RMSEs_all, MAEs_all, AbsRels_all = [], [], []

for scene in scenes:
    print("=> processing ", scene)
    rgbs = sorted(scene.listdir("*.jpg")) # ! C3VD
    gts = sorted( (scene/'depth_gt').listdir("*.npy") )
    # rgbs = sorted(scene.listdir("F*.png"))
    # gts = sorted( (scene/'depth_gt').listdir("*.npy") )
    
    RMSEs, MAEs, AbsRels = [], [], []
    
    for rgb_name,gt_name in tzip(rgbs,gts):
        gt_depth = np.load(gt_name)
        # pred_depth = 1.0 - infer(rgb_name)
        pred_depth = 1.0/infer_inv(rgb_name)

        scale = np.median(gt_depth)/np.median(pred_depth)
        pred_depth *= scale

        RMSEs.append(RMSE_func(gt_depth,pred_depth))
        MAEs.append(MAE_func(gt_depth,pred_depth))
        AbsRels.append(Rel_func2(gt_depth,pred_depth))
    
    with open("./C3VD_all_inv2.txt", 'a') as f:
        f.write("\n\n\n\n")
        f.write(scene)
        f.write("\n")
        f.write("RMSE\n")
        f.write(str(RMSEs))
        f.write("\n")
        f.write("MAE\n")
        f.write(str(MAEs))
        f.write("\n")
        f.write("AbsRel\n")
        f.write(str(AbsRels))
    
    RMSEs_all.append(RMSEs)
    MAEs_all.append(MAEs)
    AbsRels_all.append(AbsRels)
    

rmses_sum = [j for i in RMSEs_all for j in i if j<100]
mae_sum = [j for i in MAEs_all for j in i if j < 100]
abs_sum = [j for i in AbsRels_all for j in i if j< 100]

print("=> RMSE is ", sum(rmses_sum)/len(rmses_sum))
print("=> MAE is ", sum(mae_sum)/len(mae_sum))
print("=> AbsRel is ", sum(abs_sum)/len(abs_sum))
