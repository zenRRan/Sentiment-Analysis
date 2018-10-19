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
    def __init__(self, opts, vocab, label_vocab):
        super(Pooling, self).__init__()
        self.embeddings = nn.Embedding(vocab.m_size, opts.embed_size)
        if opts.pre_embed_path != '':
            embedding = Embedding.load_predtrained_emb_avg(opts.pre_embed_path, vocab.string2id)
            self.embeddings.weight.data.copy_(embedding)
        else:
            nn.init.uniform_(self.embeddings.weight.data, -opts.embed_uniform_init, opts.embed_uniform_init)
        self.linear = nn.Linear(opts.embed_size, label_vocab.m_size)
        self.embed_dropout = nn.Dropout(opts.embed_dropout)
        self.fc_dropout = nn.Dropout(opts.fc_dropout)

    def forward(self, input):
        out = self.embeddings(input)
        out = self.embed_dropout(out)
        out = torch.tanh(out)
        # print(out.size())
        out = torch.transpose(out, 1, 2)
        out = F.max_pool1d(out, out.size(2))
        # print(out.size())
        out = out.squeeze(2)
        # out = self.dropout(out)
        out = self.linear(out)
        return out