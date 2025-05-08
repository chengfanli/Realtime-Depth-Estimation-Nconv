import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import glob
from PIL import Image
import numpy as np
import os
import random

class DataLoader_NYU(Dataset):
    def __init__(self, data_dir, mode, use_mask, add_noise, height=480, width=640, tp_min=50):

        self.depth_path = os.path.join(data_dir, mode, 'gt')
        self.lidar_path = os.path.join(data_dir, mode, 'depth')
        self.rgb_path = os.path.join(data_dir, mode, 'img')
        self.mask_path = os.path.join(data_dir, 'mask')
        

        self.depths = list(sorted(glob.iglob(self.depth_path + "/*.npy", recursive=True)))
        self.lidars = list(sorted(glob.iglob(self.lidar_path + "/*.npy", recursive=True)))
        self.rgbs = list(sorted(glob.iglob(self.rgb_path + "/*.png", recursive=True)))
        self.masks = list(sorted(glob.iglob(self.mask_path + "/*.npy", recursive=True)))

        self.height = height
        self.width = width
        self.tp_min = tp_min
        self.use_mask = use_mask
        self.add_noise = add_noise

        self.k = np.array([[582.62448, 0.0, 313.04476], [0.0, 582.69103, 238.44390], [0.0, 0.0, 1.0]])

    def __len__(self):
        return len(self.depths)

    def __getitem__(self, idx):
        return self.get_item(idx)
    
    def get_item(self, index):
        rgb = self.get_rgb(self.rgbs[index])
        depth = self.get_depth(self.lidars[index])
        gt = self.get_gt(self.depths[index])
        k = torch.FloatTensor(self.k)

        tp = rgb.shape[1] - self.height
        lp = (rgb.shape[2] - self.width) // 2
        rgb = rgb[:, tp:tp + self.height, lp:lp + self.width]
        depth = depth[:, tp:tp + self.height, lp:lp + self.width]
        gt = gt[:, tp:tp + self.height, lp:lp + self.width]
        k[0, 2] -= lp
        k[1, 2] -= tp

        # if (self.use_mask):
        #     depth = self.apply_random_mask(self.depths[index])
        
        # if (self.add_noise):
        #     depth = self.apply_random_noise(self.depths[index])

        depth = self.preprocess_depth(self.depths[index], self.use_mask, self.add_noise)

        sample = {'rgb': rgb, 'depth': depth, 'gt': gt, 'k': k}
        return sample

            # sample = {'rgb': rgb, 'depth': self.apply_random_mask(self.depths[index]), 'gt': gt, 'k': k}
            # return sample
        # else:
            # sample = {'rgb': rgb, 'depth': depth, 'gt': gt, 'k': k}
            # return sample
    
    def get_rgb(self, rgb_path):
        return torch.FloatTensor(cv2.imread(rgb_path)).permute(2, 0, 1)
    
    def get_depth(self, depth_path):
        d = np.load(depth_path).reshape(480, 640)
        exp_d = np.expand_dims(d, axis=0)
        depth = torch.FloatTensor(exp_d)
        return depth
    
    def get_gt(self, gt_path):
        gt_raw = np.load(gt_path).reshape(480, 640)
        gt_exp = np.expand_dims(gt_raw, axis=0)
        gt = torch.FloatTensor(gt_exp)
        return gt

    def preprocess_depth(self, depth_path, apply_mask, apply_noise):
        raw_depth = self.get_depth(depth_path)

        mask_path = random.choice(self.masks)
        mask_raw = np.load(mask_path)
        
        if mask_raw.shape != (480, 640):
            mask_image = Image.fromarray(mask_raw)
            mask_image = mask_image.resize((640, 480), Image.NEAREST)  # Using nearest neighbor to avoid interpolation of binary values
            mask = np.array(mask_image)
        else:
            mask = mask_raw

        if apply_noise:
            num_elements = raw_depth.numel()
            num_noisy_points = int(num_elements * 0.1)
            indices = torch.randperm(num_elements)[:num_noisy_points]

            noise = torch.FloatTensor(num_noisy_points).uniform_(-0.1, 0.1)

            flattened_depth = raw_depth.reshape(-1)
        
            flattened_depth[indices] += flattened_depth[indices] * noise
        
            depth = flattened_depth.view_as(raw_depth)
        else:
            depth = raw_depth

        if apply_mask:
            depth = depth * torch.FloatTensor(mask) 
        else:
            num_zeros_in_mask = np.count_nonzero(mask == 0)
            num_elements = depth.numel()

            num_points_to_zero = min(num_zeros_in_mask, num_elements)
            indices = torch.randperm(num_elements)[:num_points_to_zero]

            flattened_depth = depth.reshape(-1)
            flattened_depth[indices] = 0
            depth = flattened_depth.view_as(depth)

        return depth

class DataLoader_NYU_test(Dataset):
    def __init__(self, data_dir, mode, height=640, width=480, tp_min=50):

        self.lidar_path = os.path.join(data_dir, mode, 'depth')
        self.rgb_path = os.path.join(data_dir, mode, 'img')
        self.lidars = list(sorted(glob.iglob(self.lidar_path + "/*.npy", recursive=True)))
        self.rgbs = list(sorted(glob.iglob(self.rgb_path + "/*.png", recursive=True)))

        self.height = height
        self.width = width
        self.tp_min = tp_min

        self.k = np.array([[329.64, 0.0, 318.0], [0.0, 328.62, 236.0], [0.0, 0.0, 1.0]])

    def __len__(self):
        return len(self.lidars)

    def __getitem__(self, idx):
        return self.get_item(idx)
    
    def get_item(self, index):
        rgb = self.get_rgb(self.rgbs[index])
        depth = self.get_depth(self.lidars[index])
        k = torch.FloatTensor(self.k)

        # tp = rgb.shape[1] - self.height
        # lp = (rgb.shape[2] - self.width) // 2
        # rgb = rgb[:, tp:tp + self.height, lp:lp + self.width]
        # depth = depth[:, tp:tp + self.height, lp:lp + self.width]
        # k[0, 2] -= lp
        # k[1, 2] -= tp

        sample = {'rgb': rgb, 'depth': depth, 'k': k}
        # print(depth.shape)
        # print(rgb.shape)
        return sample
    
    def get_rgb(self, rgb_path):
        return torch.FloatTensor(cv2.imread(rgb_path)).permute(2, 0, 1)
    
    def get_depth(self, depth_path):
        d = np.load(depth_path).reshape(480, 640)
        exp_d = np.expand_dims(d, axis=0)
        depth = torch.FloatTensor(exp_d)
        return depth