from dataset.nyuloader import DataLoader_NYU
from models.step1 import SETP1_NCONV
from utils import *
from torch import nn
import numpy as np
import torch.optim as optimizer
import cv2
import torch
import torch.nn.functional as F
import time
import copy
import matplotlib.pyplot as plt

output_name = "Test"
num_train_epoch = 40
learning_rate = [1e-2]
weight_decay = [1e-7]
apply_mask = True
add_noise = False
use_gradient_loss = True
use_plateau_lr_sched = True
early_stopping = False

def train_model(model, train_loader, val_loader, num_epoch, parameter, patience, device_str):
    device = torch.device(device_str if device_str == 'cuda' and torch.cuda.is_available() else 'cpu')
    model.to(device)
    #model = torch.compile(model)

    loss_all, loss_index = [], []
    num_itration = 0
    best_model, best_val_loss = model, float('inf')
    num_bad_epoch = 0

    optim = get_optimizer(model, parameter["optim_type"], parameter["lr"], parameter["weight_decay"])
    if use_plateau_lr_sched:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode="min", factor=0.1, patience=patience)
    else:
        scheduler = torch.optim.lr_scheduler.LinearLR(optim, start_factor=1.0, end_factor=0, total_iters=num_epoch)

    print('------------------------ Start Training ------------------------')
    t_start = time.time()
    t_step  = t_start
    loss = 0
    loss_train = []
    for epoch in range(num_epoch):
        model.train()
        loss_train = []
        for batch, data in enumerate(train_loader):
            # if (batch > 600):
            #     break

            #rgb = data['rgb'].to(device, non_blocking=True)
            depth = data['depth'].to(device, non_blocking=True)
            gt = data['gt'].to(device, non_blocking=True)
            #k = data['k'].to(device, non_blocking=True)

            num_itration += 1

            model.train()
            optim.zero_grad()
            estimated_depth = model(depth)
            
            loss = calculate_loss(estimated_depth, gt, use_gradient_loss)
            loss.requires_grad_().backward()
            optim.step()

            # loss_all.append(np.sqrt(loss.item()))
            # loss_train.append(np.sqrt(loss.item()))
            loss_all.append(loss.item())
            loss_train.append(loss.item())
            loss_index.append(num_itration)
            
            if (batch % (100 // train_loader.batch_size) == 0 and batch != 0):
                print('Batch No. {0}'.format(batch))
                t_end = time.time()
                print('Delta time {0:.4f} seconds'.format(t_end - t_step))
                t_step = time.time()
                save_depth((estimated_depth[0, 0, :, :]).detach().cpu().numpy(), 'tmp/color_output.png')
                save_depth((depth[0, 0, :, :]).detach().cpu().numpy(), 'tmp/color_sparse.png')
                save_depth((gt[0, 0, :, :]).detach().cpu().numpy(), 'tmp/color_gt.png')
                # save_depth((confidence[0, 0, :, :]).detach().cpu().numpy(), 'tmp/color_confidence.png')


        print('Epoch No. {0} -- loss = {1:.4f}'.format(
            epoch + 1,
            loss.item(),
        ))

        # Validation:
        print('Validation')
        val_loss = get_performance(model, val_loader, device_str, use_gradient_loss)
        #sqrt_loss = np.sqrt(val_loss)
        print("Validation loss: {:.4f}".format(val_loss))
        # val_loss = sum(loss_train) / len(loss_train)
        # print("Train loss: {:.4f}".format(val_loss))

        if val_loss < best_val_loss:
            best_model = copy.deepcopy(model)
            best_val_loss = val_loss
            num_bad_epoch = 0
        else:
            num_bad_epoch += 1

        # early stopping
        if early_stopping and (num_bad_epoch >= (patience+3)):
            break

        # learning rate scheduler
        if use_plateau_lr_sched:
            scheduler.step(val_loss)
        else:
            scheduler.step()

        print("Current learning rate: {:.9f}".format(scheduler.get_last_lr()[0]))


    t_end = time.time()
    print('Training lasted {0:.2f} minutes'.format((t_end - t_start) / 60))
    print("Best validation loss: {:.4f}".format(best_val_loss))
    print('------------------------ Training Done ------------------------')
    stats = {'loss': loss_all,
             'loss_ind': loss_index,
             }

    return best_model, best_val_loss, stats
    
def get_hyper_parameters(lr, wd):
    _para_list = [{"optim_type": 'adam', 'lr': lr, "weight_decay": wd, "store_img_training": True}]
    _num_epoch = num_train_epoch
    _patience = 2
    _device = 'cuda'
    return _para_list, _num_epoch, _patience, _device


best_val_loss = float('inf')
best_model = SETP1_NCONV()
best_lr = 0
best_wd = 0
final_stats = {}
for lr in learning_rate:
    for wd in weight_decay:
        train_dataset = DataLoader_NYU('/oscar/data/jtompki1/cli277/nyuv2/nyuv2', 'train', apply_mask, add_noise)
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, pin_memory=True)
        val_dataset = DataLoader_NYU('/oscar/data/jtompki1/cli277/nyuv2/nyuv2', 'val', apply_mask, add_noise)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True, pin_memory=True)

        print('Train size: ' + str(len(train_loader)))
        print('Val size: ' + str(len(val_loader)))  
        print('Learning Rate: ' + str(lr))
        print('Weight Decay: ' + str(wd))  

        model = SETP1_NCONV()
        model = nn.DataParallel(model)
        para_list, num_epoch, patience, device_str = get_hyper_parameters(lr, wd)

        new_model, val_loss, stats = train_model(model, train_loader, val_loader, num_epoch, para_list[0], patience, device_str)

        if (val_loss < best_val_loss):
            best_model = copy.deepcopy(new_model)
            best_val_loss = val_loss
            best_lr = lr
            best_wd = wd
            final_stats = stats

print('------------------------ Training Done ------------------------')
print('---------------------------------------------------------------')
print("Best validation loss(ALL): {:.4f}".format(best_val_loss))
print("Best learning rate(ALL): {:.4f}".format(best_lr))
print("Best weight decay(ALL): {:.4f}".format(best_wd))
print('---------------------------------------------------------------')
print('------------------------ Training Done ------------------------')
save_checkpoint(best_model, num_train_epoch, "./checkpoints", final_stats, output_name)
