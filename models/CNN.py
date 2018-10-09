#!/usr/bin/env python
# encoding: utf-8

"""
@version: python3.6
@author: 'zenRRan'
@license: Apache Licence 
@contact: zenrran@qq.com
@software: PyCharm
@file: CNN.py
@time: 2018/10/7 15:51
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import utils.Embedding as Embedding

class CNN(nn.Module):

    def __init__(self, args):
        super(CNN, self).__init__()
        self.embedings = nn.Embedding(args.wordNum, args.EmbedSize)
        if args.using_pred_emb:
            emb_text = Embedding.load_predtrained_emb_avg(args.pred_embedding_50_path,
                                                                          args.wordAlpha.string2id)
            self.embeddings.weight.data.copy_(emb_text)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, args.kernelNum, (K, args.EmbedSize), padding=(K // 2, 0)) for K in args.kernelSizes])
        self.linear = nn.Linear(len(args.kernelSizes)*args.kernelNum, args.labelSize)
        self.embed_dropout = nn.Dropout(args.embed_dropout)

    def forward(self, input):
        out = self.embedings(input)
        out = self.embed_dropout(out)
        out = F.tanh(out)
        l = []
        out = out.unsqueeze(1)
        for conv in self.convs:
            l.append(F.tanh(conv(out)).squeeze(3))
        out = l
        l = []
        for i in out:
            l.append(F.max_pool1d(i, kernel_size=i.size(2)).squeeze(2))
        out = torch.cat(l, 1)
        out = self.dropout(out)
        out = self.linear(out)
        return out