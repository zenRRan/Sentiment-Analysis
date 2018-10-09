#!/usr/bin/env python
# encoding: utf-8

"""
@version: python3.6
@author: 'zenRRan'
@license: Apache Licence 
@contact: zenrran@qq.com
@software: PyCharm
@file: Pooling.py
@time: 2018/10/7 15:50
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.Embedding as Embedding

class Pooling(nn.Module):
    def __init__(self, args):
        super(Pooling, self).__init__()
        self.embedings = nn.Embedding(args.wordNum, args.EmbedSize)
        if args.using_pred_emb:
            emb_text = Embedding.load_predtrained_emb_avg(args.pred_embedding_50_path,
                                                                          args.wordAlpha.string2id)
            self.embeddings.weight.data.copy_(emb_text)
        self.linear = nn.Linear(len(args.kernelSizes)*args.kernelNum, args.labelSize)
        self.embed_dropout = nn.Dropout(args.embed_dropout)

    def forward(self, input):
        out = self.embedings(input)
        out = self.embed_dropout(out)
        out = F.tanh(out)
        out = self.dropout(out)
        out = self.linear(out)
        return out