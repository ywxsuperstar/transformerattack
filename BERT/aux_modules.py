import numpy as np
import torch
from torch import nn
import torch.nn.init as init
import random



# find posion "0"  to drloc
def collect_pos_indexes(mask):
    """
        mask: tensor->[b, len]  1:unk,0:origin   
        unshuffleidx: list->[b,len_nonzero]
    """  
    unshuffleidx = [torch.nonzero(mask[i] == 0) for i in range(mask.shape[0])]
    return unshuffleidx

class DenseRelativeLoc_ngram(nn.Module):
    def __init__(self, in_dim, out_dim=1, drloc_mode="l1", use_abs=False):
        super(DenseRelativeLoc_ngram, self).__init__()
        self.in_dim = in_dim
        self.drloc_mode = drloc_mode
        self.use_abs = use_abs
        self.out_dim = out_dim

        self.layers = nn.Sequential(
            nn.Linear(in_dim, in_dim),  
            nn.ReLU(),
            nn.Linear(in_dim, self.out_dim),        
        )

        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    init.constant_(layer.bias, 0)

    def forward(self, encode, mask, window_size, normalize=True):

        # encode->position embeddings [max_pos+1, dim]
        # mask->torch.Tensor [b, seq_len]

        xy = collect_pos_indexes(mask)  # list [b,len_nonzero]  找到未打乱的位置xy(0)来增强
         
        pos_real = []
        pos_pred = []
        for i in range(len(xy)):    # 遍历每个句子xy[i]
            
            origin_xy = torch.squeeze(xy[i].clone(), dim=-1)
            # num_groups = (max(xy[i]) + 1) // window_size + 1  # 计算组数
            # groups = [[] for _ in range(num_groups)]  # 初始化二维数组
            groups = [[] for _ in range(513 // window_size)]  # 初始化二维数组
            
            for num in xy[i]:
                group_idx = num // window_size
                groups[group_idx].append(num)
                                      
            for group in groups:
                group_len = len(group)
                if group_len > 1:
                    shuffle_indices = torch.randperm(group_len)
                    shuffled_group = torch.tensor(group)[shuffle_indices].tolist()  # 生成随机排列的索引序列并打乱分组内的索引
                    group[:] = shuffled_group 
                              
            group_tensors = [torch.tensor(group, dtype=torch.long) for group in groups]
                     
            encode_origin_xy = encode(origin_xy)
            encode_permuted_xy = encode((torch.cat(group_tensors, dim=0)).to(xy[0].device))
            
            # 计算位置偏移并预测
            pos_t = torch.squeeze(self.layers(encode_origin_xy - encode_permuted_xy), dim=-1)
            pos_pred.append(pos_t)  # embedding后降维（预测）

            # 计算真实位置偏移
            pos_real_i = (origin_xy - (torch.cat(group_tensors, dim=0)).to(xy[0].device))
            pos_real.append(pos_real_i)
                        
        return pos_real, pos_pred
    
    
class DenseAbsoluteLoc_ngram(nn.Module):
    def __init__(self, in_dim, out_dim=1, drloc_mode="l1", use_abs=False):
        super(DenseAbsoluteLoc_ngram, self).__init__()
        self.in_dim = in_dim
        self.drloc_mode = drloc_mode
        self.use_abs = use_abs
        self.out_dim = out_dim

        self.layers = nn.Sequential(
            nn.Linear(in_dim, in_dim),  
            nn.ReLU(),
            nn.Linear(in_dim, self.out_dim),        
        )

        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    init.constant_(layer.bias, 0)

    def forward(self, encode, mask, window_size, normalize=True):

        # encode->position embeddings [max_pos+1, dim]
        # mask->torch.Tensor [b, seq_len]

        xy = collect_pos_indexes(mask)  # list [b,len_nonzero]  找到未打乱的位置xy(0)来增强
         
        pos_real = []
        pos_pred = []
        for i in range(len(xy)):    # 遍历每个句子xy[i]
            
            origin_xy = torch.squeeze(xy[i].clone(), dim=-1)
            # num_groups = (max(xy[i]) + 1) // window_size + 1  # 计算组数
            # groups = [[] for _ in range(num_groups)]  # 初始化二维数组
            groups = [[] for _ in range(513 // window_size)]  # 初始化二维数组
            
            for num in xy[i]:
                group_idx = num // window_size
                groups[group_idx].append(num)
                                      
            for group in groups:
                group_len = len(group)
                if group_len > 1:
                    shuffle_indices = torch.randperm(group_len)
                    shuffled_group = torch.tensor(group)[shuffle_indices].tolist()  # 生成随机排列的索引序列并打乱分组内的索引
                    group[:] = shuffled_group 
                              
            group_tensors = [torch.tensor(group, dtype=torch.long) for group in groups]
            encode_permuted_xy = encode((torch.cat(group_tensors, dim=0)).to(xy[0].device))
            
            pos_t = torch.squeeze(self.layers(encode_permuted_xy), dim=-1)
            pos_pred.append(pos_t)  # embedding后降维（预测）

            pos_real_i = (torch.cat(group_tensors, dim=0)).to(xy[0].device)
            pos_real.append(pos_real_i)
                        
        return pos_real, pos_pred






