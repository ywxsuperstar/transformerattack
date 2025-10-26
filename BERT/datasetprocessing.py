# 定义数据集类
import random

import torch
from torch.utils import data
import torch
import ast
from torch.utils.data import Dataset

class CustomDataset(data.Dataset):
    def __init__(self, df, tokenizer):

        self.texts = df['sentence']
        self.labels = df['label']
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        label = self.labels[index]

        encoding = self.tokenizer(text, padding='max_length', max_length=512,
                                  truncation=True, return_tensors='pt')
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'token_type_ids': encoding['token_type_ids'].squeeze(),
            'label': label
        }