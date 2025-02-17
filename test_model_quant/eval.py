#load images from our_bins and gt_bins
#compute a bunch of different losses on each image pair loss(our_bin[i], gt_bins[i])
#take average loss across all samples, also note highest loss?
#save these into a file?
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from test_data_loader import TEST_DataLoader_NYU
from PIL import Image
import glob
import os
import re

def extract_number(filename):
    # Use a regular expression to find numeric parts in the filename
    match = re.search(r'\d+', os.path.basename(filename))
    if match:
        return int(match.group())  # Convert the matched number to an integer
    return float('inf')  # If no number is found, sort such files last


class Evaluator():
    def __init__(self, bin_path, height=480, width=640):
        self.gt_bin_path = "./gt_bin"
        self.our_bin_path = bin_path
        self.height = height
        self.width = width

        self.our_depth_list = list(sorted(glob.glob(os.path.join(self.our_bin_path, "*")), key=extract_number))
        self.gt_depth_list = list(sorted(glob.glob(os.path.join(self.gt_bin_path, "*")), key=extract_number))

        assert len(self.our_depth_list) == len(self.gt_depth_list), "Mismatched size between given binaries and ground truths!"

        self.our_images = []
        self.gt_images = []
        self.output_dir = os.path.join(os.getcwd(), "temporary_eval_output")
        os.makedirs(self.output_dir, exist_ok=True)  # Create directory if it doesn't exist
        
        self.init_images()

    def init_images(self):
            for i in range(len(self.our_depth_list)):
                # Load and save "our images"
                self.our_images.append(self.load_depth_from_binary(self.our_depth_list[i]))
                plt.imshow(self.our_images[-1])
                our_image_path = os.path.join(self.output_dir, f"our_image_{i}.png")
                plt.savefig(our_image_path)
                plt.close()

                #needs cropping to match BP_Export output
                self.gt_images.append(self.crop_image(self.load_depth_from_binary(self.gt_depth_list[i])))
                plt.imshow(self.gt_images[-1])
                gt_image_path = os.path.join(self.output_dir, f"gt_image_{i}.png")
                plt.savefig(gt_image_path)
                plt.close()
    
    def crop_image(self, img):
        img[:45, :] = 0
        img[-45:, :] = 0
        img[:, :20] = 0
        return img

    def calculate_loss(self):
        l1_loss_fn = nn.L1Loss()
        avg_l1_loss = 0
        highest_l1_loss = 0

        mse_loss_fn = nn.MSELoss()
        avg_mse_loss = 0
        highest_mse_loss = 0

        avg_rmse_loss = 0
        highest_rmse_loss = 0

        avg_irmse_loss = 0
        highest_irmse_loss = 0

        avg_imae_loss = 0
        highest_imae_loss = 0

        #TODO: Remove Outliers
        #TODO: Convert units to mm



        for i in range(len(self.our_images)):
            our_image = torch.tensor(self.our_images[i] * 1000, dtype=torch.float32)
            gt_image = torch.tensor(self.gt_images[i] * 1000, dtype=torch.float32)
            # inv_our_image = 1.0 / (our_image + 1e-6)
            # inv_gt_image = 1.0 / (gt_image + 1e-6)

            inv_our_image = 1e+6 / (our_image + 1e-6)
            inv_gt_image = 1e+6 / (gt_image + 1e-6)    




            loss = l1_loss_fn(our_image,gt_image)
            mse_loss = mse_loss_fn(our_image,gt_image)
            rmse_loss = torch.sqrt(mse_loss)
            imae_loss = l1_loss_fn(inv_our_image, inv_gt_image)
            imse_loss = mse_loss_fn(inv_our_image,inv_gt_image)
            irmse_loss = torch.sqrt(imse_loss)

            avg_l1_loss += loss
            avg_mse_loss += mse_loss
            avg_rmse_loss += rmse_loss
            avg_imae_loss += imae_loss
            avg_irmse_loss += irmse_loss
            

            if loss > highest_l1_loss:
                highest_l1_loss = loss

            if mse_loss > highest_mse_loss:
                highest_mse_loss = mse_loss

            if rmse_loss > highest_rmse_loss:
                highest_rmse_loss = rmse_loss

            if imae_loss > highest_imae_loss:
                highest_imae_loss = imae_loss

            if irmse_loss > highest_irmse_loss:
                highest_irmse_loss = irmse_loss
            
            print("Current sample:{}, L1 loss:{:.4f}, MSE loss:{:.4f}, RMSE loss:{:.4f}, iMAE loss:{:.4f}, iRMSE loss:{:.4f}".format(i, loss.item(), mse_loss.item(), rmse_loss.item(), imae_loss.item(), irmse_loss.item()))

        avg_l1_loss /= len(self.our_images)
        avg_mse_loss /= len(self.our_images)
        avg_rmse_loss /= len(self.our_images)
        avg_imae_loss /= len(self.our_images)
        avg_irmse_loss /= len(self.our_images)

        # Print overall losses
        print('Loss for curr samples:')
        print("Total samples:{}, Avg L1 loss:{:.4f}, Highest L1 loss:{:.4f}".format(len(self.our_images), avg_l1_loss, highest_l1_loss))
        print("Total samples:{}, Avg MSE loss:{:.4f}, Highest MSE loss:{:.4f}".format(len(self.our_images), avg_mse_loss, highest_mse_loss))
        print("Total samples:{}, Avg RMSE loss:{:.4f}, Highest RMSE loss:{:.4f}".format(len(self.our_images), avg_rmse_loss, highest_rmse_loss))
        print("Total samples:{}, Avg iMAE loss:{:.4f}, Highest iMAE loss:{:.4f}".format(len(self.our_images), avg_imae_loss, highest_imae_loss))
        print("Total samples:{}, Avg iRMSE loss:{:.4f}, Highest iRMSE loss:{:.4f}".format(len(self.our_images), avg_irmse_loss, highest_irmse_loss))

        return avg_l1_loss, highest_l1_loss, avg_mse_loss, highest_mse_loss, avg_rmse_loss, highest_rmse_loss, avg_imae_loss, highest_imae_loss, avg_irmse_loss, highest_irmse_loss
    
    def load_depth_from_binary(self, file_path):
        """
        Load a depth map from a binary file.

        Args:
            file_path (str): Path to the binary file.

        Returns:
            np.ndarray: 2D depth map.
        """
        try:
            with open(file_path, 'rb') as f:
                # Read the first 4 bytes to get the data length (int32)
                length_bytes = f.read(4)
                if len(length_bytes) < 4:
                    raise ValueError(f'File {file_path} is too short to contain a valid header.')

                data_length = np.frombuffer(length_bytes, dtype=np.int32)[0]

                # Expected data length
                expected_size = self.width * self.height

                if data_length != expected_size:
                    raise ValueError(
                        f'File {file_path} has data length {data_length}, expected {expected_size}.'
                    )

                # Read the float32 depth data
                float_data = np.fromfile(f, dtype=np.float32, count=data_length)

                if len(float_data) != expected_size:
                    raise ValueError(
                        f'File {file_path} contains {len(float_data)} float values, expected {expected_size}.'
                    )

                # Reshape to 2D depth map
                depth_map = float_data.reshape((self.height, self.width))
                return depth_map
        except Exception as e:
            print(f"Error loading depth map: {e}")
            return Image.fromarray(np.zeros((self.height, self.width), dtype='float32'), mode='F')







