
import pandas as pd
import torch
import copy
import math
import random
import numpy as np
from torch.backends import cudnn
import os

def init_seeds(seed=42, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True


def setmjpconfig(use_unk,use_drloc):
    if use_unk == 1 and use_drloc == 1:
        useunk = True
        usedrloc = True
    elif use_unk == 1 and use_drloc == 0:
        useunk = True
        usedrloc = False
    elif use_unk == 0 and use_drloc == 0:
        useunk = False
        usedrloc = False
    elif use_unk == 0 and use_drloc == 1:
        useunk = False
        usedrloc = True
    return useunk,usedrloc



# consine schedule for DRLOC
# def cosine_scheduler_drloc(max_value, base_value, epochs, warmup_epochs=0, start_warmup_value=0):
#     warmup_schedule = np.array([])
#     warmup_iters = warmup_epochs #* niter_per_ep
#     if warmup_epochs > 0:
#         warmup_schedule = np.linspace(start_warmup_value, max_value, warmup_iters)

#     iters = np.arange(epochs - warmup_iters)
#     schedule = base_value + 0.5 * (max_value - base_value) * (1 + np.cos(np.pi * iters / len(iters)))

#     schedule = np.concatenate((warmup_schedule, schedule))
#     assert len(schedule) == epochs
#     return schedule



# def cosine_scheduler_drloc(max_value, base_value, total_steps, warmup_steps=0, start_warmup_value=0):
#     warmup_schedule = np.linspace(start_warmup_value, max_value, warmup_steps) if warmup_steps > 0 else np.array([])
#     cosine_schedule = base_value + 0.5 * (max_value - base_value) * (1 + np.cos(np.pi * np.arange(total_steps - warmup_steps) / (total_steps - warmup_steps)))
#     schedule = np.concatenate((warmup_schedule, cosine_schedule))
#     assert len(schedule) == total_steps
#     print("--------",schedule.shape)
#     exit(0)
#     return schedule

def cosine_scheduler_drloc(max_value, base_value, total_steps, warmup_steps=0, start_warmup_value=0):
    # 计算余弦退火调度数组
    cosine_schedule = base_value + 0.5 * (max_value - base_value) * (1 + np.cos(np.pi * np.arange(total_steps - warmup_steps) / (total_steps - warmup_steps)))
    warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_steps) if warmup_steps > 0 else np.array([])
    schedule = np.concatenate((warmup_schedule, cosine_schedule))
    
    assert len(schedule) == total_steps
    
    return schedule

# consine with warmup for LR
def cosine_warmup_scheduler(base_lr, max_lr, total_steps, warmup_steps):
    def scheduler(step):
        if step < warmup_steps:
            return base_lr + (max_lr - base_lr) * step / warmup_steps
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return base_lr + 0.5 * (max_lr - base_lr) * (1 + math.cos(math.pi * progress))
    
    return scheduler


def save_checkpoint(config, epoch, model, max_accuracy, optimizer, lr_scheduler,save_path):
    save_state = {'model': model.module.state_dict(),        # 单机多卡
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'max_accuracy': max_accuracy,
                  'epoch': epoch,
                  'config': config}
    save_path = os.path.join(config.OUTPUT, f'ckpt_epoch_{epoch}.pth')
    torch.save(save_state, save_path)

    
  
def save_checkpoint_best(config, epoch, model, max_accuracy, save_res_dir, SEED, shuffle_ratio, minlen, maxlen):
    save_state = {'model': model.module.state_dict(),
                  'max_accuracy': max_accuracy,
                  'epoch': epoch,
                  'config': config}
    
    save_path = os.path.join(save_res_dir, f'best_network_{SEED}_{shuffle_ratio}_{minlen}_{maxlen}.pth')
    torch.save(save_state, save_path)
    