import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from collections import OrderedDict
from torch.nn import TransformerEncoder, TransformerEncoderLayer


# Temporarily leave PositionalEmbedding module here. Will be moved somewhere else.

# class PositionalEmbedding(nn.Module):
#     r"""Inject some information about the relative or absolute position of the tokens
#         in the sequence. The positional encodings have the same dimension as
#         the embeddings, so that the two can be summed. Here, we use sine and cosine
#         functions of different frequencies.
#     .. math::
#         \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
#         \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
#         \text{where pos is the word position and i is the embed idx)
#     Args:
#         d_model: the embed dim (required).
#         dropout: the dropout value (default=0.1).
#         max_len: the max. length of the incoming sequence (default=5000).
#     Examples:
#         >>> pos_encoder = PositionalEmbedding(d_model)
#     """
#
#     def __init__(self, d_model, dropout=0.1, max_len=5000):
#         super(PositionalEmbedding, self).__init__()
#         self.dropout = nn.Dropout(p=dropout)
#
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#
#         pe = pe.unsqueeze(0)
#         self.register_buffer("pe", pe)
#
#     def forward(self, x):
#         r"""Inputs of forward function
#         Args:
#             x: the sequence fed to the positional encoder model (required).
#         Shape:
#             x: [batch size, sequence length, embed dim]
#             output: [batch size, sequence length, embed dim]
#         Examples:
#             >>> output = pos_encoder(x)
#         """
#         x = x + self.pe[:, : x.shape[1], :]
#         return self.dropout(x)
#
class LearnablePositionalEmbedding(torch.nn.Module):
    """Shorthand for a learnable embedding."""

    def __init__(self, embed_dim, max_position_embeddings=1024, dropout=0.0):
        super().__init__()
        self.embedding = torch.nn.Embedding(max_position_embeddings, embed_dim)
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, input_embeddings):
        """This is a batch-first implementation"""
        position_ids = torch.arange(input_embeddings.shape[1], device=self.embedding.weight.device)
        position_embeddings = self.embedding(position_ids[None, :])
        return self.dropout(input_embeddings + position_embeddings)


# class LearnablePositionalEmbedding(torch.nn.Module):
#     """Shorthand for a learnable embedding."""
#
#     def __init__(self, embed_dim, max_position_embeddings=1024, dropout=0.0):
#         super().__init__()
#         self.embedding = torch.nn.Embedding(max_position_embeddings, embed_dim)
#         self.dropout = torch.nn.Dropout(p=dropout)
#
#     def forward(self, input_embeddings, train_type):
#         """This is a batch-first implementation"""
#         if train_type == "shuffle_ratio" or train_type == "shuffle_all":
#             position_embeddings = self.embedding(input_embeddings)
#             return self.dropout(position_embeddings)
#         else:
#             position_ids = torch.arange(input_embeddings.shape[1], device=self.embedding.weight.device)
#             position_embeddings = self.embedding(position_ids[None, :])
#             return self.dropout(input_embeddings + position_embeddings)

from aux_modules import DenseRelativeLoc_ngram


class TransformerSentiment(nn.Module):
    def __init__(self, vocab_size, emb_dim, num_hid, num_heads, num_layers, num_classes,
                 dropout=0.2, train_type='mjp_unk_drloc', positional_embedding=True,
                 max_position_embeddings=512, use_drloc=False, drloc_mode='l1', use_abs=True, use_unk=False,window_size=32):
        super(TransformerSentiment, self).__init__()
        self.emb_dim = emb_dim
        self.train_type = train_type
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.use_unk = use_unk
        self.use_drloc = use_drloc
        self.window_size = window_size

        if self.use_unk:
            num_unk = 1
            self.pos_embed = nn.Embedding(max_position_embeddings + num_unk, emb_dim)
            self.unk_embed_index = max_position_embeddings
        else:
            self.pos_embed = nn.Embedding(max_position_embeddings, emb_dim)

        encoder_layers = TransformerEncoderLayer(emb_dim, num_heads, num_hid, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        
        self.fc = nn.Linear(emb_dim, num_classes)

        if self.use_drloc:
            self.drloc = DenseRelativeLoc_ngram(
                in_dim=self.emb_dim,
                use_abs=True
            )
        
        self.init_weights()

    # def init_weights(self):

    #     init.xavier_normal_(self.embedding.weight, gain=1)
    #     init.xavier_normal_(self.fc.weight, gain=1)
    #     init.xavier_normal_(self.pos_embed.weight, gain=1)
    
    
    def init_weights(self):
        init.kaiming_normal_(self.embedding.weight, nonlinearity='relu')
        init.kaiming_normal_(self.fc.weight, nonlinearity='relu')
        init.kaiming_normal_(self.pos_embed.weight, nonlinearity='relu')
    

    def forward_features(self, input_ids, unk_mask):

        input_embed = self.embedding(input_ids)

        src_key_padding_mask = (input_ids == 0)  # pad attention mask

        xy, predxy = None, None
        if self.use_drloc:
            if unk_mask is not None:
                xy, predxy = self.drloc(self.pos_embed, unk_mask, self.window_size, normalize=True)
            else:
                unk_mask = torch.where(input_ids != 0, torch.zeros_like(input_ids), torch.full_like(input_ids, -1)) 
                xy, predxy = self.drloc(self.pos_embed, unk_mask, self.window_size, normalize=True)
   
        if self.use_unk: 
            seq_ord = torch.arange(input_ids.shape[1], device=input_ids.device)
            if unk_mask is None:
                unk_mask = torch.where(input_ids != 0, torch.zeros_like(input_ids), torch.full_like(input_ids, -1)) 
            unk_mask[unk_mask == -1] = 0    # PAD为-1的idx置为0
            seq_ord = seq_ord * (1 - unk_mask) + unk_mask * self.unk_embed_index  # 1:unk 0:orin
            pos_embed = self.pos_embed(seq_ord)
        else:
            position_ids = torch.arange(input_ids.shape[1], device=input_ids.device)
            pos_embed = self.pos_embed(position_ids[None, :])

        input_embed = input_embed + pos_embed
        input_embed = self.dropout(input_embed)
        input_embed = self.transformer_encoder(input_embed, src_key_padding_mask=src_key_padding_mask)
        input_embed = self.dropout(input_embed)

        return input_embed[:, 0, :], xy, predxy  # 取的就是head的前一层

    def forward(self, input_ids, unk_mask):

        input_embed, xy, predxy = self.forward_features(input_ids, unk_mask)  # drloc_feats:pre, deltaxy: real
        sentiment_output = self.fc(input_embed)  # get first token for predict, [b,emb_d]

        return sentiment_output, xy, predxy


class TransformerSentimentPE(nn.Module):
    def __init__(self, vocab_size, emb_dim, num_hid, num_heads, num_layers, num_classes,
                 dropout=0.0, train_type='mjp_unk_drloc', shuffle_ratio=1.0,
                 use_drloc=False, drloc_mode='l1', use_abs=True, use_unk=False):
        super(TransformerSentimentPE, self).__init__()
        self.emb_dim = emb_dim
        self.train_type = train_type
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.shuffle_ratio = shuffle_ratio
        self.use_unk = use_unk
        self.use_drloc = use_drloc

        encoder_layers = TransformerEncoderLayer(emb_dim, num_heads, num_hid, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)

        self.fc = nn.Linear(emb_dim, num_classes)

        self.init_weights()

    def init_weights(self):
        init.xavier_normal_(self.embedding.weight, gain=1)
        init.xavier_normal_(self.fc.weight, gain=1)

    def forward_features(self, input_ids, unk_mask):
        input_embed = self.embedding(input_ids)

        src_key_padding_mask = (input_ids == 0)  # pad attention

        input_embed = self.dropout(input_embed)
        input_embed = self.transformer_encoder(input_embed,src_key_padding_mask=src_key_padding_mask)

        return input_embed[:, 0, :], None, None  # 取的就是head的前一层

    def forward(self, input_ids, unk_mask):
        input_embed, _, _ = self.forward_features(input_ids, unk_mask)
        sentiment_output = self.fc(input_embed)  # get first token for predict, [b,emb_d]

        return sentiment_output, None, None
