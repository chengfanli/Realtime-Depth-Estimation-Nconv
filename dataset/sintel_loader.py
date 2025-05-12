import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import glob
from PIL import Image
import numpy as np
import os
import random
import matplotlib.pyplot as plt
from scipy import ndimage

TAG_FLOAT = 202021.25
TAG_CHAR = 'PIEH'
class DataLoader_Sintel(Dataset):
    def __init__(self, data_dir, mode, use_mask, add_noise, height=480, width=640, tp_min=50, overfit_samples=None):
        
        self.depth_path = os.path.join(data_dir, mode, 'depth')
        self.rgb_path = os.path.join(data_dir, mode, 'final')
        self.flow_path = os.path.join(data_dir, mode, 'flow')
        self.camdata_path = os.path.join(data_dir, mode, 'camdata')

        self.height = height
        self.width = width
        self.use_mask = use_mask

        self.samples = []
        self.k = np.array([[582.62448, 0.0, 313.04476], [0.0, 582.69103, 238.44390], [0.0, 0.0, 1.0]])

        ##IF OVERFITTING USE A SPECIFIC LIST
        
        sequence_names = sorted(os.listdir(self.depth_path))
        
        for seq_name in sequence_names:
            depth_seq_dir = os.path.join(self.depth_path, seq_name)
            if not os.path.isdir(depth_seq_dir):
                continue
            
            depth_files = sorted(glob.glob(os.path.join(depth_seq_dir, '*.dpt')))
            rgb_seq_dir = os.path.join(self.rgb_path, seq_name)
            cam_seq_dir = os.path.join(self.camdata_path, seq_name)

            for depth_file_path in depth_files:
                base_name = os.path.splitext(os.path.basename(depth_file_path))[0]
                
                rgb_file_path = os.path.join(rgb_seq_dir, base_name + '.png')
                cam_file_path = os.path.join(cam_seq_dir, base_name + '.cam')

                # Only use frames where all required files exist
                if os.path.exists(rgb_file_path) and os.path.exists(cam_file_path):
                    self.samples.append((rgb_file_path, depth_file_path, cam_file_path))
        
        if overfit_samples is not None:
            self.samples = [self.samples[i] for i in overfit_samples]

    
    def save_sample_images(self, sample, save_dir, idx):
        # Create directory for the sample
        sample_dir = os.path.join(save_dir, f"sample_{idx}")
        os.makedirs(sample_dir, exist_ok=True)

        # Save RGB image (assuming it's a tensor of shape [3, H, W])
        rgb = sample['rgb']
        if rgb.dim() == 4:
            rgb = rgb[0]  # Select the first image in the batch
        rgb = rgb.permute(1, 2, 0).numpy()  # Convert to HWC format for saving
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)  # Ensure values are in valid range
        rgb_path = os.path.join(sample_dir, "rgb.png")
        cv2.imwrite(rgb_path, rgb)

        # Save regular depth image (assuming it's a tensor of shape [H, W])
        depth = sample['gt']
        depth = depth.numpy()

        if depth.ndim != 2:
            depth = np.squeeze(depth, axis=0)
        depth_normalized = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))
        #depth_normalized = np.expand_dims(depth, axis=-1)
        

        # Save depth image as a grayscale image
        depth_path = os.path.join(sample_dir, "depth.png")
        plt.imsave(depth_path, depth_normalized, cmap='plasma')  # Use 'plasma' colormap for depth

        # # Save masked depth image (assuming it's a tensor of shape [H, W])
        masked_depth = sample['depth']
        masked_depth = masked_depth.numpy()

        if masked_depth.ndim != 2:
            masked_depth = np.squeeze(masked_depth, axis=0)

        # # Normalize masked depth to the range [0, 1] for saving
        masked_depth_normalized = (masked_depth - np.min(masked_depth)) / (np.max(masked_depth) - np.min(masked_depth))
        # masked_depth_normalized = np.expand_dims(masked_depth, axis=-1)
        # # Save masked depth image as a grayscale image
        masked_depth_path = os.path.join(sample_dir, "masked_depth.png")
        plt.imsave(masked_depth_path, masked_depth_normalized, cmap='plasma')  # Use 'plasma' colormap for masked depth

        print(f"Saved images for sample {idx} to {sample_dir}")



    
    def __len__(self):
        return len(self.samples)

    def test(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.get_item(idx)
    
    def sobel_filter(self, image, thickness=1.5):
        x_grad = ndimage.sobel(image, 0)
        y_grad = ndimage.sobel(image, 1)
        mag = np.sqrt(x_grad**2 + y_grad**2)
        cropped_edge = self.supress_cropped_edge(mag, 21, (479-21))
        # print(thickness)
        blurred_mag = ndimage.gaussian_filter(cropped_edge, sigma=thickness)
        blurred_mag = blurred_mag / np.max(blurred_mag)

        threshold = 0.1
        # print("This is the min and max value in our image")
        # print(np.min(blurred_mag), np.max(blurred_mag))
        # blurred_mag[blurred_mag < threshold] = 0
        binary_edge = (blurred_mag > threshold).astype(np.uint8)

        # print("This is the min and max value in our binary")
        # print(np.min(binary_edge), np.max(binary_edge))

        return binary_edge
    
    def supress_cropped_edge(self, mag, top, bottom):
        magc = mag
        magc[top - 3: top + 3, :] = 0
        magc[bottom - 3: bottom + 3, :] = 0
        return magc
    
    def get_item(self, idx):
        

        sample = self.samples[idx]
        rgb_path = sample[0]
        depth_path = sample[1]
        k_path = sample[2]

        rgb = self.get_rgb(rgb_path)  # Shape (C, H, W)
        gt, depth = self.get_depth(depth_path)  # Shape (H, W)
        depth = depth.unsqueeze(0).float()  # Shape [1, H, W]
        gt = gt.unsqueeze(0).float()

        k = torch.FloatTensor(self.k)

        #figure out rgb pre processing!



        #what is this cropping doing?
        # tp = rgb.shape[1] - self.height
        # lp = (rgb.shape[2] - self.width) // 2
        # rgb = rgb[:, tp:tp + self.height, lp:lp + self.width]
        # #is my depth okay?
        # print(f"Depth shape before slicing: {depth.shape}")
        # depth = depth[tp:tp + self.height, lp:lp + self.width]
        # gt = gt[:, tp:tp + self.height, lp:lp + self.width]
        # k[0, 2] -= lp
        # k[1, 2] -= tp

        # if (self.use_mask):
        #     depth = self.apply_random_mask(self.depths[index])
        
        # if (self.add_noise):
        #     depth = self.apply_random_noise(self.depths[index])

        # depth = self.preprocess_depth(self.depths[index], self.use_mask, self.add_noise)
        # print("Depth shape, Sentil")
        # print(depth.shape)
        # print("rgb_shape, Sentil")
        # print(rgb.shape)


        sample = {'rgb': rgb, 'depth': depth, 'gt': gt, 'k': k}
        return sample
    
    def get_rgb(self, rgb_path):
        rgb = cv2.imread(rgb_path)  # (H, W, C) = (436, 1024, 3)

        # Horizontal center crop (width: 1024 → 640)
        x_start = (1024 - 640) // 2
        rgb = rgb[:, x_start:x_start + 640, :]  # (436, 640, 3)

        # Vertical padding (height: 436 → 480)
        pad_top = (480 - 436) // 2
        pad_bottom = 480 - 436 - pad_top
        rgb = np.pad(rgb, ((pad_top, pad_bottom), (0, 0), (0, 0)), mode='constant', constant_values=0)  # (480, 640, 3)

        # Convert to tensor: (C, H, W)
        return torch.FloatTensor(rgb).permute(2, 0, 1)
    
    def get_depth(self, filename):
        f = open(filename,'rb')
        check = np.fromfile(f,dtype=np.float32,count=1)[0]
        assert check == TAG_FLOAT, ' depth_read:: Wrong tag in flow file (should be: {0}, is: {1}). Big-endian machine? '.format(TAG_FLOAT,check)
        width = np.fromfile(f,dtype=np.int32,count=1)[0]
        height = np.fromfile(f,dtype=np.int32,count=1)[0]
        size = width*height
        assert width > 0 and height > 0 and size > 1 and size < 100000000, ' depth_read:: Wrong input size (width = {0}, height = {1}).'.format(width,height)
        depth = np.fromfile(f,dtype=np.float32,count=-1).reshape((height,width))

        # print(depth.shape)
        #
        x_start = (1024 - 640) // 2
        depth = depth[:, x_start:x_start + 640]
        depth = np.pad(depth, ((22, 22), (0, 0)), mode='constant', constant_values=0)
        # print(depth.shape)





        depth = torch.FloatTensor(depth)
        edge_mask = self.sobel_filter(depth, thickness=1.0)
        edge_mask = torch.from_numpy(edge_mask).bool().to(depth.device)
        
        

        mask = torch.rand_like(depth) < 0.5
        # edge_mask = edge_image > 0.0

        # non_edge_mask = (torch.from_numpy(edge_image > 0.0).to(depth.device))
        # masked_depth = depth.clone()
        # masked_depth[~non_edge_mask] = 0  # same effect



        masked_depth = depth.clone()
        masked_depth[edge_mask] = 0
        
        return depth, masked_depth


