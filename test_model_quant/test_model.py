from models.without_step1.step2 import SETP2_BP_EXPORT as step2_without_step1
from models.without_masks.step2 import SETP2_BP_EXPORT as step2_without_mask
from test_data_loader import TEST_DataLoader_NYU
from torch.utils.data import Dataset, DataLoader
from eval import Evaluator
import torch
from torch import nn
import numpy as np
import torch.optim as optimizer
import cv2
import torch.nn.functional as F
import time
import copy
import os
from scipy.ndimage import median_filter
from models.without_step1.step1 import SETP1_NCONV as step_1
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

#Modify this
checkpoint_path = "master_baseline"
output_folder = "ours"

test_dataset = TEST_DataLoader_NYU('/oscar/data/jtompki1/cli277/new_spot_data', '1')
batch_to_original_data_idx = test_dataset.get_original_data_idx()
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# model = SETP2_BP_EXPORT()
if checkpoint_path == "without_step1":
    model = step2_without_step1()

elif checkpoint_path == "without_mask" or checkpoint_path == "baseline2" or checkpoint_path == "master_baseline"  :
    model = step2_without_mask()

elif checkpoint_path == "Test":
    model = step_1()

else:
    print('Error: no corresponding model found')

# model = step2_without_step1()
model.eval()
checkpoint = torch.load("./model_checkpoints/{}.pth.tar".format(checkpoint_path))

state_dict = checkpoint["state_dict"]
new_state_dict = {}
for k, v in state_dict.items():
    name = k[7:] if k.startswith("module.") else k
    new_state_dict[name] = v
model.load_state_dict(new_state_dict, strict=False)

device_str = 'cuda'
device = torch.device('cuda')
model.to(device)

#Important for normalizations (ensuring all our images have a consistent color range)
def save_depth(depth_data):
    depth_data_normalized = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX)
    depth_data_8bit = depth_data_normalized.astype(np.uint8)
    
    return cv2.applyColorMap(depth_data_8bit, cv2.COLORMAP_INFERNO)

def save_all(output_folder, batch, estimated_depths, depth, gt, rgb):
    img_estimated = save_depth((estimated_depths[0, 0, :, :]).detach().cpu().numpy())
    img_sparse = save_depth((depth[0, 0, :, :]).detach().cpu().numpy())
    img_gt = save_depth((gt[0, 0, :, :]).detach().cpu().numpy())
    img_rgb = rgb[0, :, :, :].permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)

    combined_image = np.hstack((img_gt, img_rgb, img_sparse, img_estimated))
    
    combined_path = "./{}/combined_depth{}.png".format(output_folder, str(batch))
    cv2.imwrite(combined_path, combined_image)

    return combined_path

for batch, data in enumerate(test_loader):
    rgb = data['rgb'].to(device)
    depth = data['depth'].to(device)
    gt = data['gt'].to(device)

    if checkpoint_path == "Test":
        estimated_depths = model(depth)
        print('just using step1')


    else: 
        estimated_depths, _ = model(rgb, depth, rgb, depth)
        print('using step1 and step2')
    

    depth_data = (estimated_depths[0, 0, :, :]).detach().cpu().numpy()

    depth_data_flat = depth_data.flatten()

    # To match bin files with gt_bin? 
    # print(str(batch))
    # new_file_path = "./{}/{}".format(output_folder, str(batch))
    new_file_path = "./{}/{}/{}".format(output_folder, checkpoint_path, str(batch_to_original_data_idx[batch]))
    # Save to binary file
    with open(new_file_path, 'wb') as f:
        # Write the length of the data as a 4-byte integer (header)
        data_length = np.array([len(depth_data_flat)], dtype=np.int32)
        f.write(data_length.tobytes())

        # Write the depth data as float32 values
        f.write(depth_data_flat.astype(np.float32).tobytes())

bin_path = "./{}/{}/".format(output_folder, checkpoint_path)
eval = Evaluator(bin_path)
l1_losses, mse_losses, rmse_losses, imae_losses, irmse_losses = eval.calculate_loss()
avg_l1_loss, highest_l1_loss, avg_mse_loss, highest_mse_loss, avg_rmse_loss, highest_rmse_loss, avg_imae_loss, highest_imae_loss, avg_irmse_loss, highest_irmse_loss = eval.avg_metrics(l1_losses, mse_losses, rmse_losses, imae_losses, irmse_losses)
model_name = f"{checkpoint_path}_losses.txt"

# Prepare the loss data as a string
loss_data = f"""
Model: {checkpoint_path}
Avg L1 loss: {avg_l1_loss:.4f}, Highest L1 loss: {highest_l1_loss:.4f}
Avg MSE loss: {avg_mse_loss:.4f}, Highest MSE loss: {highest_mse_loss:.4f}
Avg RMSE loss: {avg_rmse_loss:.4f}, Highest RMSE loss: {highest_rmse_loss:.4f}
Avg iMAE loss: {avg_imae_loss:.4f}, Highest iMAE loss: {highest_imae_loss:.4f}
Avg iRMSE loss: {avg_irmse_loss:.4f}, Highest iRMSE loss: {highest_irmse_loss:.4f}
"""

# Save the losses to a text file in the root folder
file_path = os.path.join(".", model_name)  # Saves to the root folder
with open(file_path, "w") as file:
    file.write(loss_data)

print(f"Loss data saved to: {file_path}")

l1_i = eval.highest_outliers(l1_losses, 10)
# mse_i = eval.highest_outliers(mse_losses)
# rmse_i = eval.highest_outliers(rmse_losses)
# imae_i = eval.highest_outliers(imae_losses)
# irmse_i = eval.highest_outliers(irmse_losses)

def visualize_comparison(gt_path, pred_path, loss_value, save_path):
    gt = plt.imread(gt_path)
    pred = plt.imread(pred_path)
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(gt, cmap='inferno')
    axes[0].set_title('Ground Truth')
    axes[0].axis('off')

    axes[1].imshow(pred, cmap='inferno')
    axes[1].set_title('Prediction')
    axes[1].axis('off')

    plt.suptitle(f'Loss: {loss_value:.4f}')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

output_dir = "loss_debug_visuals"
os.makedirs(output_dir, exist_ok=True)
comparison_images = []

for idx in l1_i:
    gt_image = f"temporary_eval_output/gt_image_{idx}.png"
    pred_image = f"temporary_eval_output/our_image_{idx}.png"
    save_image = os.path.join(output_dir, f"comparison_{idx}.png")

    visualize_comparison(gt_image, pred_image, rmse_losses[idx], save_image)
    comparison_images.append((rmse_losses[idx], save_image))


comparison_images.sort(key=lambda x: x[0], reverse=True)

# Then generate the PDF
pdf_path = "loss_debug_report.pdf"
c = canvas.Canvas(pdf_path, pagesize=letter)

for loss, img_path in comparison_images:
    c.setFont("Helvetica", 12)
    c.drawString(30, 750, f"Loss: {loss:.4f}")
    c.drawImage(img_path, 30, 100, width=500, height=600, preserveAspectRatio=True)
    c.showPage()

c.save()