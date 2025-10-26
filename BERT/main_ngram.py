import os

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import time
import copy
import math
# from torch.optim.lr_scheduler import EarlyStopping
from position_loss import pos_fun_loss
import torch.distributed as dist
# import wandb
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from multiprocessing import Pool
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import multiprocessing
from transformers import AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
from tokenizers import Tokenizer, models, pre_tokenizers

from build_model import build_model_transformer,build_model_ngram
from shuffle_process import shuffle_token_ngram
from config import args
from toolset import setmjpconfig,init_seeds,cosine_warmup_scheduler,cosine_scheduler_drloc,save_checkpoint_best

from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from transformers import BertTokenizer
import torch.nn.functional as F

from datasetprocessing import CustomDataset
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.backends.cudnn.benchmark = False

starttime = time.strftime("%Y-%m-%d_%H_%M")  # 将时间格式改为精确到分钟
if args.model_type == 'bert':
    writer = SummaryWriter(f"./log/logs_BERTngram/{args.data_type}/seed_{args.seeds}/{args.train_type}_{args.shuffle_ratio}_{args.minlen}")
else:
    writer = SummaryWriter(f"./log/logs_TransformerParameter/{args.data_type}/seed_{args.seeds}/{args.train_type}_{args.shuffle_ratio}_{args.minlen}")

# early_stopping = EarlyStopping(patience=5, verbose=True)


class WarmupLearninngRate:
    def __init__(self, optimizer, warmup_steps = 16000, init_lr=0.01, end_lr=0.15, decay_factor=0.9):

        # linearly warmup for the first args.warmup_updates
        self.init_lr = init_lr
        self.end_lr = end_lr
        self.warmup_steps = warmup_steps
        self.lr_step = (end_lr - init_lr) / warmup_steps

        # then, decay prop. to the inverse square root of the update number
        self.decay_factor = end_lr * warmup_steps**0.5

        # initial learning rate
        self.lr = init_lr
        self.optimizer = optimizer

        self.optimizer.param_groups[0]['lr'] = self.lr
        self.num_updates = 0

        self.decay_factor = decay_factor
        self.step = self.step_warmup_decay

    def step_warmup_decay(self, epoch=False):
        """Update the learning rate after each update."""
        if self.num_updates < self.warmup_steps:
            self.lr = self.init_lr + self.num_updates*self.lr_step
        else:
            if epoch:
                self.lr = self.lr * self.decay_factor

        self.num_updates += 1
        self.optimizer.param_groups[0]['lr'] = self.lr

        return self.lr

    def step_warmup_exp(self, epoch=False):
        """Update the learning rate after each update."""
        if self.num_updates < self.warmup_steps:
            self.lr = self.init_lr + self.num_updates*self.lr_step
        else:
            self.lr = self.decay_factor * self.num_updates**-0.5

        self.num_updates += 1
        self.optimizer.param_groups[0]['lr'] = self.lr

        return self.lr


isddp = True
if isddp:
    # Configuring the GPU for each process
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    dist.init_process_group(backend='nccl', init_method='env://')
    local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    init_seeds(args.seeds + local_rank)
else:
    init_seeds(args.seeds)
    device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")

use_unk, use_drloc = setmjpconfig(args.use_unk, args.use_drloc)

if dist.get_rank() == 0:
    print("\n\n->Configurations are:")
    print(f"->use_unk: {use_unk}, use_drloc: {use_drloc}")
    [print(k, ": ", v) for (k, v) in vars(args).items()]

base_path = f"/opt/data/private/ywx/transformerattack/BERT/savemodel/{args.data_type}/subsetngram/"
model_type_dir = f"{args.model_type}_seed{args.seeds}_win{args.window_size}"
train_type_dir = args.train_type.replace("_", "")  # 去除下划线
save_res_dir = os.path.join(base_path, model_type_dir, train_type_dir)
os.makedirs(save_res_dir, exist_ok=True)
train_log_filepath = os.path.join(save_res_dir, f"train_log{args.minlen}_{args.maxlen}.txt")

tokenizer = BertTokenizer.from_pretrained("/opt/data/private/ywx/transformerattack/bert-base-uncased")
train_df = pd.read_csv(f"/opt/data/private/ywx/transformerattack/data/{args.data_type}/train_pro{args.minlen}_{args.maxlen}.csv")
val_df = pd.read_csv(f"/opt/data/private/ywx/transformerattack/data/{args.data_type}/val_pro{args.minlen}_{args.maxlen}.csv")

train_dataset = CustomDataset(train_df, tokenizer)
val_dataset = CustomDataset(val_df, tokenizer)

# 设置数据并行
train_sampler = None
if isddp:
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)  # 这个sampler会自动分配数据到各个gpu上
    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=4)
else:
    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
val_loader = data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

if args.model_type == "bert":
    model = build_model_ngram(device, args.train_type, use_unk, use_drloc, args.window_size)
else:
    model = build_model_transformer(device, args.train_type, use_unk, use_drloc, args.window_size)

from torch.optim.lr_scheduler import ReduceLROnPlateau
# 设置模型并行
if isddp:
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)  # 多个gpu的BN同步
    model = torch.nn.parallel.DistributedDataParallel(model.cuda())  # find_unused_parameters=False
else:
    model.to(device)

if args.model_type == 'bert':
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay) and 'classifier' not in n],
        'lr': args.learning_rate, 'weight_decay': 1e-3},
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay) and 'classifier' in n],
        'lr': 5e-3, 'weight_decay': 1e-3},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay) and 'classifier' not in n],
        'lr': args.learning_rate, 'weight_decay': 0.0},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay) and 'classifier' in n],
        'lr': 5e-3, 'weight_decay': 0.0}
    ]
    base_lr, max_lr = 3e-5, 5e-5
    optimizer = optim.AdamW(optimizer_parameters, lr=args.learning_rate, eps=1e-8)  # epsilon    # best:5e-5
    total_steps = len(train_loader) * args.num_epochs  
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = total_steps*0.1, num_training_steps = total_steps)
    criterion = nn.CrossEntropyLoss()
elif args.model_type == 'transformer':
    optimizer = optim.AdamW(model.parameters(), betas=(0.9, 0.999), lr=args.learning_rate, eps=1e-8, weight_decay=0.01)  # epsilon    # best:5e-3
    total_steps = len(train_loader) * args.num_epochs  
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = total_steps*0.1, num_training_steps = total_steps)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.007) # torch.optim.Adam(model.parameters(), lr=0.001)
    # lr_scheduler = WarmupLearninngRate(optimizer, warmup_steps = 10000, init_lr=0.001)
    review_weight = torch.Tensor(np.array([ 0.8, 1.4, 1.4, 1.2, 0.8 ]))
    criterion = nn.CrossEntropyLoss(review_weight).to(device)
    
    
# scheduler = cosine_warmup_scheduler(base_lr, max_lr, total_steps, args.warmup_steps) 
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=0, last_epoch=-1)
# optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.999), lr=args.learning_rate, weight_decay=0.01)
# optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
# optim_schedule = ScheduledOptim(optimizer, 512, n_warmup_steps=5000)
# lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=0)




def train(model, train_loader, optimizer, criterion, epoch):

    model.train()
    total_loss = 0.0
    correct_predictions,total_predictions = 0, 0
    for idx, batch in enumerate(train_loader):
        total_step = (epoch * len(train_loader) + idx + 1)
        input_ids = batch['input_ids'].to(device)
        labels = batch['label'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)

        optimizer.zero_grad()

        shuffle_input_ids, unk_mask = shuffle_token_ngram(copy.deepcopy(input_ids), args.shuffle_ratio, args.window_size)

        if args.model_type=="bert":
            logits, real_pos, pred_pos = model(shuffle_input_ids, unk_mask=unk_mask, attention_mask=attention_mask, token_type_ids=token_type_ids)
        else:
            logits, real_pos, pred_pos = model(shuffle_input_ids, unk_mask=unk_mask)
        
        loss_sentiment = criterion(logits, labels)
        # t_lambda_dlocr = lambda_dlocr_schedule[epoch * len(train_loader) + idx]
        loss_pos = pos_fun_loss(real_pos, pred_pos) if args.train_type not in ["shuffle", "mjp_unk", "wope"] else 0
        senti_pos_loss = loss_sentiment + loss_pos * 0.01
              
        senti_pos_loss.backward()  
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # avoid exploding vanishing gradients problem, clip the norm of the gradients to 1.0
        optimizer.step()
        lr_scheduler.step()

        total_loss += senti_pos_loss.item()

        predicted_labels = torch.argmax(logits, dim=1)
        correct_predictions += torch.sum(predicted_labels == labels).item()
        total_predictions += labels.size(0)
        

        if dist.get_rank() == 0:
            writer.add_scalar(f"loss_sentiment/Train_{args.model_type}_{args.train_type}_{args.shuffle_ratio}", loss_sentiment, total_step) 
            # writer.add_scalar(f"Total_loss/Train_{args.model_type}_{args.train_type}_{args.shuffle_ratio}", total_loss / (idx+1) , total_step)
            writer.add_scalar(f"Total_loss/Train_{args.model_type}_{args.train_type}_{args.shuffle_ratio}", senti_pos_loss.item() , total_step)
            if args.train_type == "mjp_unk_drloc":
                writer.add_scalar(f"pos_loss/Train_{args.model_type}_{args.train_type}_{args.shuffle_ratio}",loss_pos, total_step)
                

    train_acc = correct_predictions / total_predictions
    train_loss = total_loss / len(train_loader)
    if dist.get_rank() == 0:
        writer.add_scalar(f"ACC/Train_{args.model_type}_{args.train_type}_{args.shuffle_ratio}", train_acc, epoch + 1)
        writer.add_scalar(f"Epochloss/Train_{args.model_type}_{args.train_type}_{args.shuffle_ratio}", train_loss, epoch + 1)
  
    return train_acc, train_loss


def evaluate(model, val_loader, criterion, epoch):
    model.eval()
    total_loss, val_correct, total_predictions = 0.0, 0, 0
    # Don't track gradient.
    with torch.no_grad():
        for idx, batch in enumerate(val_loader):
            total_step = (epoch * len(train_loader) + idx + 1)
            input_ids = batch['input_ids'].to(device)
            labels = batch['label'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)

            if args.model_type == "bert":
                logits, real_pos, pred_pos = model(input_ids, attention_mask=attention_mask, unk_mask=None, token_type_ids=token_type_ids)
            else:
                logits, real_pos, pred_pos = model(input_ids, unk_mask=None)
            loss_sentiment = criterion(logits, labels)
            # t_lambda_dlocr = lambda_dlocr_schedule[epoch * len(train_loader) + idx]
            loss_pos = pos_fun_loss(real_pos, pred_pos) if args.train_type not in ["shuffle", "mjp_unk", "wope"] else 0

            senti_pos_loss = loss_sentiment + loss_pos * 0.01 
            val_correct += torch.sum(torch.argmax(logits, dim=1) == labels)
            total_predictions += labels.size(0)
            
            total_loss += senti_pos_loss.item()
            
            writer.add_scalar(f"loss_sentiment/Val_{args.model_type}_{args.train_type}_{args.shuffle_ratio}",loss_sentiment , total_step) 
            # writer.add_scalar(f"Total_loss/Val_{args.model_type}_{args.train_type}_{args.shuffle_ratio}",total_loss /(idx+1), total_step)
            writer.add_scalar(f"Total_loss/Val_{args.model_type}_{args.train_type}_{args.shuffle_ratio}",senti_pos_loss.item(), total_step)
            if args.train_type == "mjp_unk_drloc":
                writer.add_scalar(f"pos_loss/Val_{args.model_type}_{args.train_type}_{args.shuffle_ratio}",loss_pos, total_step)

    # Calculate average losses and accuracies
    loss = total_loss / len(val_loader)
    accuracy = val_correct / total_predictions
    writer.add_scalar(f"Epochloss/Val_{args.model_type}_{args.train_type}_{args.shuffle_ratio}", loss, epoch + 1)
    writer.add_scalar(f"ACC/Val_{args.model_type}_{args.train_type}_{args.shuffle_ratio}", accuracy, epoch + 1)
    return accuracy.item(), loss

# warmup math.ceil(total_steps*(args.warmup_steps))
# lambda_dlocr_schedule = cosine_scheduler_drloc(
#                 max_value=args.lambda_dlocr,
#                 base_value=args.lambda_dlocr*0.5,
#                 total_steps=total_steps,
#                 warmup_steps=0,
#                 start_warmup_value=0)

best_accuracy = 0
best_train_acc = 0
for epoch in range(args.num_epochs):

    time.sleep(1)
    start_time = time.time()
    if isddp:
        train_sampler.set_epoch(epoch)  # shuffle数据
    train_acc, train_loss = train(model, train_loader, optimizer, criterion, epoch)

    # scheduler.step()   # 学习率调度程序在每个epoch后更新学习率
    if dist.get_rank() == 0:
        val_acc, val_loss = evaluate(model, val_loader, criterion, epoch)

        epoch_time = time.time() - start_time
        print(
            f"Epoch {epoch}  train Accuracy : {train_acc*100:.2f}, Train Loss : {train_loss:.4f}, Time : {epoch_time:.2f}")
        print(f"Epoch {epoch}  Valid Accuracy : {val_acc*100:.2f}, Valid Loss : {val_loss:.4f}")

        train_log_txt_formatter = f"{starttime} [Epoch] {epoch:03d} [val_acc] {val_acc:.4f} [val_loss] {val_loss:.4f}\n"
        to_write = train_log_txt_formatter.format(time_str=time.strftime("%Y_%m_%d_%H:%M:%S"),
                                                  epoch=args.num_epochs,
                                                  loss_str1=" ".join(["{}".format(val_acc)]),
                                                  loss_str2=" ".join(["{}".format(val_loss)]))

        with open(train_log_filepath, "a") as f:
            f.write(to_write)

        if val_acc > best_accuracy:  # current acc > best acc
            best_accuracy = val_acc
            best_train_acc = train_acc
            save_checkpoint_best(args, epoch, model, best_accuracy, save_res_dir, args.seeds, args.shuffle_ratio, args.minlen, args.maxlen)

        #  # early stopping
        # early_stopping(val_loss, model)
        # if early_stopping.early_stop:
        #     print("Early stopping")
        #     break
        
if dist.get_rank() == 0:
    print(f"corresponding train acc: {best_train_acc*100:.2f}, Best validation accuracy: {best_accuracy*100:.2f}")

    # save best log
    train_log_filename1 = f"best_log_{args.seeds}_{args.minlen}_{args.maxlen}.txt"
    best_log_file_path = os.path.join(save_res_dir, train_log_filename1)
    best_log_txt_formatter = f"[lr]{args.learning_rate} [BS]{args.batch_size} [shuffle_ratio]{args.shuffle_ratio}[Epoch]{epoch:03d} [best_train_acc] {best_train_acc*100:.2f} [best_val_acc] {best_accuracy*100:.2f} "
    best_to_write = best_log_txt_formatter.format(shuffle_ratio=args.shuffle_ratio,
                                                  time_str1=time.strftime("%Y_%m_%d_%H:%M:%S"),
                                                  epoch1=args.num_epochs + 1,
                                                  loss_str1=" ".join(["{}".format(best_train_acc)]),
                                                  loss_str2=" ".join(["{}".format(best_accuracy)]))
    with open(best_log_file_path, "a") as f1:  # 模式a
        f1.write(best_to_write + "\n")

writer.close()
