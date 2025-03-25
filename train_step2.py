import torch
import copy
import time
import os
from torch import nn
import torch.nn.functional as F
import numpy as np

from dataset.nyuloader import DataLoader_NYU
from utils import (
    get_optimizer, 
    calculate_loss_multi_resolution, 
    get_performance_multi_resolution, 
    save_checkpoint,
    save_depth,
    save_rgb
)

# Hyperparameters
output_name = "baseline2_single"
step1_checkpoint_name = "step1-rmse-less-than-0.18"
num_train_epoch = 50
learning_rate = [1e-2]
weight_decay = [1e-7]
patience = 5
use_plateau_lr_sched = True

use_gradient_loss = False

def train_model(model, train_loader, val_loader, num_epoch, parameter, patience, device_str):
    device = torch.device(device_str if (device_str == 'cuda' and torch.cuda.is_available()) else 'cpu')
    model.to(device)
    model.train()

    best_val_loss = float('inf')
    best_model = None
    num_bad_epoch = 0

    # For logging:
    train_losses = []
    val_losses   = []

    # Optimizer & Scheduler
    optim = get_optimizer(model, parameter["optim_type"], parameter["lr"], parameter["weight_decay"])
    if use_plateau_lr_sched:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode="min", factor=0.1, patience=patience)
    else:
        scheduler = torch.optim.lr_scheduler.LinearLR(optim, start_factor=1.0, end_factor=0, total_iters=num_epoch)


    print('----- Start Training -----')
    t_start = time.time()

    torch.autograd.set_detect_anomaly(True)
    for epoch in range(num_epoch):
        if (epoch == 10):
            print("Thawing step1")
            model.freeze_step1(False)

        model.train()
        batch_losses = []

        # =========== TRAIN LOOP ===========
        for batch_idx, data in enumerate(train_loader):
            rgb   = data['rgb'].to(device)     # shape [B,3,480,640] (if not changed)
            depth = data['depth'].to(device)   # shape [B,1,480,640]
            gt    = data['gt'].to(device)

            optim.zero_grad()
            # Single call to forward
            estimated_depths = model(rgb, depth, rgb, depth)

            loss = calculate_loss_multi_resolution(estimated_depths, gt, use_gradient_loss)
            loss.backward()
            optim.step()

            batch_losses.append(loss.item())

            # Debug prints/images at lower frequency
            if batch_idx % 10 == 0 and batch_idx != 0:
                print(f"[Epoch {epoch+1}, Batch {batch_idx}] loss: {loss.item():.4f}")
                save_depth(estimated_depths[-1][-1][0].detach().cpu().numpy(), 'tmp/color_output.png')
                np.save('tmp/depth_output.npy', estimated_depths[-1][-1][0].detach().cpu().numpy())
                save_depth((depth[-1, 0, :, :]).detach().cpu().numpy(), 'tmp/color_sparse.png')
                save_rgb(rgb[-1].detach().cpu().numpy(), 'tmp/color_rgb.png')


        # Average epoch loss
        epoch_train_loss = sum(batch_losses) / len(batch_losses)
        train_losses.append(epoch_train_loss)

        # =========== VALIDATION ===========
        model.eval()
        with torch.no_grad():
            val_loss = get_performance_multi_resolution(model, val_loader, device_str, use_gradient_loss)

        val_losses.append(val_loss)
        print(f"Epoch {epoch+1}/{num_epoch} - Train Loss: {epoch_train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)
            num_bad_epoch = 0
        else:
            num_bad_epoch += 1

        if num_bad_epoch >= patience*3:
            print(f"Early stopping at epoch {epoch+1}. No improvement in {patience*3} epochs.")
            break

        # Scheduler step
        if use_plateau_lr_sched:
            scheduler.step(val_loss)
        else:
            scheduler.step()

    t_end = time.time()
    print(f"Training took {(t_end - t_start)/60:.2f} minutes.")
    print(f"Best val loss: {best_val_loss:.4f}")
    print('----- Training Done -----')

    # Return best model + logs
    stats = {
        'train_losses': train_losses,
        'val_losses': val_losses,
    }

    return best_model, best_val_loss, stats

# Main driver code
def main():
    # 1) Create dataset/loader ONCE, outside hyperparam loops
    train_dataset = DataLoader_NYU('../datasets/nyuv2', 'train', use_mask=True, add_noise=False)
    val_dataset   = DataLoader_NYU('../datasets/nyuv2', 'val',   use_mask=True, add_noise=False)

    # Try a larger batch size if memory allows:
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader   = torch.utils.data.DataLoader(val_dataset,   batch_size=1, shuffle=False)

    # 2) Hyperparameter search (learning_rate, weight_decay)
    best_val_loss = float('inf')
    best_model = None
    best_lr = 0
    best_wd = 0
    final_stats = {}

    for lr in learning_rate:
        for wd in weight_decay:
            print("-------------------------------------------")
            print(f"Learning Rate: {lr}, Weight Decay: {wd}")

            # Reinit model for each hyperparameter set
            from models.step2 import SETP2_BP_TRAIN
            model = SETP2_BP_TRAIN(step1_checkpoint_name)

            # Prepare hyperparams
            param_dict = {
                "optim_type": 'adam',
                "lr": lr,
                "weight_decay": wd,
            }

            # Train
            new_model, val_loss, stats = train_model(
                model, train_loader, val_loader,
                num_epoch=num_train_epoch,
                parameter=param_dict,
                patience=5,
                device_str='cuda'
            )

            # Track best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = copy.deepcopy(new_model)
                best_lr = lr
                best_wd = wd
                final_stats = stats

    print("===== Hyperparam Search Done =====")
    print(f"Best Val Loss: {best_val_loss:.4f}")
    print(f"Best LR: {best_lr}, Best WD: {best_wd}")

    # 3) Save the best model
    save_checkpoint(best_model, num_train_epoch, "./checkpoints", final_stats, output_name)

if __name__ == "__main__":
    main()
