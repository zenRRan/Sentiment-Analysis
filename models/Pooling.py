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
import torch.nn.init as init

class Pooling(nn.Module):
    def __init__(self, opts):
        super(Pooling, self).__init__()
        self.embedings = nn.Embedding(opts.wordNum, opts.EmbedSize)
        if opts.using_pred_emb:
            emb_text = Embedding.load_predtrained_emb_avg(opts.pred_embedding_50_path,
                                                          opts.wordAlpha.string2id)
            self.embeddings.weight.data.copy_(emb_text)
        else:
            nn.init.uniform(self.embedings.weight.data, -opts.embed_uniform_init, opts.embed_uniform_init)
        self.linear = nn.Linear(len(opts.kernelSizes)*opts.kernelNum, opts.labelSize)
        self.embed_dropout = nn.Dropout(opts.embed_dropout)

    def forward(self, input):
        out = self.embedings(input)
        out = self.embed_dropout(out)
        out = F.tanh(out)
        out = self.dropout(out)
        out = self.linear(out)
        return out