from datasets.utils import read_pfm, write_pfm


from path import Path
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

import matplotlib.pyplot as plt


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

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
# tmp = torch.load("/home/jiahan/jiahan/codes/Depth-Anything/log/C3VD/03-20-19:43/checkpoints/0.pth.tar")
# tmp = torch.load("log/C3VD/03-24-22:07/checkpoints/0.pth.tar")
# tmp = torch.load("/home/jiahan/jiahan/codes/Depth-Anything/log/C3VD/03-25-23:11/checkpoints/0.pth.tar") # 88.8
# tmp = torch.load("/home/jiahan/jiahan/codes/Depth-Anything/log/C3VD/03-27-00:54/checkpoints/0.pth.tar") # 166.6

tmp = torch.load("log/UCL/04-11-00:00/checkpoints/4.pth.tar") # UCL

depth_anything.load_state_dict(tmp)

@torch.no_grad()
def infer(filename):
    raw_image = cv2.imread(filename)
    image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
    
    h, w = image.shape[:2]
    
    image = transform({'image': image})['image']
    image = torch.from_numpy(image).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        depth = depth_anything(image)
    
    
    depth = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
    # depth = (depth - depth.min()) / (depth.max() - depth.min())
    depth = depth.cpu().numpy()
    
    return depth

def infer_c3vd():
    
    save_root_path = Path("/home/jiahan/jiahan/datasets/C3VD/DepthAnything_Depth_for_SCDepth/inv_22-6_5") # ! UCL 1/(output+22.2)转换, 第5轮的权重
    # save_root_path = Path("/home/jiahan/jiahan/datasets/C3VD/DepthAnything_Depth_for_SCDepth/inv_166-6_1") # 1/(output+88.8来转换)
    save_root_path.makedirs_p()
    scenes = [
        "scene_cecum_t1_a",
        "scene_cecum_t1_b",
        "scene_cecum_t2_a",
        "scene_cecum_t2_b",
        "scene_cecum_t2_c",
        "scene_cecum_t3_a",
        "scene_cecum_t4_a",
        "scene_cecum_t4_b",
        "scene_trans_t1_a",
        "scene_trans_t1_b",
        "scene_trans_t2_a",
        "scene_trans_t2_b",
        "scene_trans_t2_c",
        "scene_trans_t3_a",
        "scene_trans_t3_b",
        "scene_trans_t4_a",
        "scene_trans_t4_b",
        "scene_sigmoid_t1_a",
        "scene_sigmoid_t2_a",
        "scene_sigmoid_t3_a",
        "scene_sigmoid_t3_b",
        "scene_desc_t4_a"
    ]

    for scene in scenes:
            print("=> processing ", scene.split('/')[-1])
            save_path = save_root_path/scene
            save_path.makedirs_p()

            rgb = sorted(Path(f"/home/jiahan/jiahan/datasets/C3VD/.dataset4SCDepth/{scene}").listdir("*.jpg"))
            errors = []
            for rr in tqdm(rgb):
                name = rr.split('/')[-1][:-4]
                # pred = 1.0 - infer(rr)
                # pred = 1.0/infer(rr)
                output = infer(rr)
                write_pfm(save_path/name+'.pfm', output)
                cv2.imwrite(save_path/name+'.png', (65535*(output-output.min())/(output.max()-output.min())).astype(np.uint16))
                # np.save(save_path/name+'.npy', output)
            
def infer_simcol():
    scenes = [
        "SyntheticColon_I/Frames_S15",
        "SyntheticColon_I/Frames_S7",
        "SyntheticColon_I/Frames_S12",
        "SyntheticColon_I/Frames_S13",
        "SyntheticColon_I/Frames_S1",
        "SyntheticColon_I/Frames_S4",
        "SyntheticColon_I/Frames_S3",
        "SyntheticColon_I/Frames_S2",
        "SyntheticColon_I/Frames_S11",
        "SyntheticColon_I/Frames_S14",
        "SyntheticColon_I/Frames_S9",
        "SyntheticColon_I/Frames_S6",
        "SyntheticColon_I/Frames_S8",
        "SyntheticColon_I/Frames_S5",
        "SyntheticColon_I/Frames_S10",
        "SyntheticColon_II/Frames_B12",
        "SyntheticColon_II/Frames_B2",
        "SyntheticColon_II/Frames_B1",
        "SyntheticColon_II/Frames_B7",
        "SyntheticColon_II/Frames_B13",
        "SyntheticColon_II/Frames_B10",
        "SyntheticColon_II/Frames_B4",
        "SyntheticColon_II/Frames_B8",
        "SyntheticColon_II/Frames_B14",
        "SyntheticColon_II/Frames_B5",
        "SyntheticColon_II/Frames_B9",
        "SyntheticColon_II/Frames_B6",
        "SyntheticColon_II/Frames_B3",
        "SyntheticColon_II/Frames_B15",
        "SyntheticColon_II/Frames_B11",
        "SyntheticColon_III/Frames_O3",
        "SyntheticColon_III/Frames_O2",
        "SyntheticColon_III/Frames_O1",
    ]
    save_root_path = Path("/home/jiahan/jiahan/datasets/SimCol/Depth-Anything/inv_22-2_5") # 1/(y+22.2) train 5 epochs
    save_root_path.makedirs_p()
    img_root_path = Path("/home/jiahan/jiahan/datasets/SimCol/SimCol3D4SC_Depth")
    for scene in scenes:
        print("=> processing scene", scene)
        save_path = save_root_path/scene
        save_path.makedirs_p()
        print(save_path)
        print(img_root_path/scene)
        
        files = sorted((img_root_path/scene).listdir("F*.png"))
        for file in tqdm(files):
            name = file.split('/')[-1][:-4]
            output = infer(file)
            write_pfm(save_path/name+'_depth.pfm', output)
            cv2.imwrite(save_path/name+'_depth.png', (65535*(output-output.min())/(output.max()-output.min())).astype(np.uint16))
        

if __name__=="__main__":
    # infer_c3vd()     
    infer_simcol()
