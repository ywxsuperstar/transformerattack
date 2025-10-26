import random

import pandas as pd
import torch
import copy
import math
import random
import numpy as np
from torch.backends import cudnn


     def shuffle_token_ngram(input_ids, percent_to_shuffle, window_size):
    '''
        input_ids: torch.Tensor, [N, seq_len=512]
        retunr: unk_mask torch.Tensor, [N, seq_len]
    '''
    batch_size, _ = input_ids.shape  
    unk_mask = torch.where(input_ids != 0, torch.zeros_like(input_ids), torch.full_like(input_ids, -1))  # 标志PAD为-1，PAD不用增强
    
    # i: 第几条句子， j: 每个元素的位置
    for i in range(batch_size):
        sentence = input_ids[i] 
        effective_length = torch.nonzero(sentence).size(0)
        
        for j in range(1, effective_length-1, window_size):
            
            if j + window_size >= effective_length:  # 检查窗口是否超出了序列的长度
                window_size = effective_length - j - 1  # 如果超出了，就将窗口大小调整为剩余部分的长度   防止与0打乱
                     
            window = copy.deepcopy(sentence[j:j+window_size])
            num_elements_to_shuffle = math.ceil(percent_to_shuffle * len(window))  # 计算应该打乱的元素数量并向上取整

            if window_size == 1 or num_elements_to_shuffle == 1:
                continue
            
            shuffle_indices = []
            if num_elements_to_shuffle > 0:
                shuffle_indices = torch.randperm(window_size)[:num_elements_to_shuffle]
                t_shuff_idx = shuffle_indices[torch.randperm(num_elements_to_shuffle)]
                unk_mask[i, j:j+window_size][t_shuff_idx] = 1

                for swi, swj in zip(shuffle_indices, t_shuff_idx):
                        input_ids[i, j+swi] = window[swj] 

    return input_ids, unk_mask

