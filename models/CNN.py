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

    def __init__(self, opts, vocab, label_vocab):
        super(CNN, self).__init__()
        self.embeddings = nn.Embedding(vocab.m_size, opts.embed_size)
        if opts.pre_embed_path != '':
            embedding = Embedding.load_predtrained_emb_avg(opts.pre_embed_path, vocab.string2id)
            self.embeddings.weight.data.copy_(embedding)
        else:
            nn.init.uniform_(self.embeddings.weight.data, -opts.embed_uniform_init, opts.embed_uniform_init)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, opts.kernel_num, (K, opts.embed_size), padding=(K // 2, 0)) for K in opts.kernel_size])
        self.linear = nn.Linear(len(opts.kernel_size)*opts.kernel_num, label_vocab.m_size)
        self.embed_dropout = nn.Dropout(opts.embed_dropout)
        self.fc_dropout = nn.Dropout(opts.fc_dropout)

    def forward(self, input):
        out = self.embeddings(input)
        out = self.embed_dropout(out)
        out = torch.tanh(out)
        l = []
        out = out.unsqueeze(1)
        for conv in self.convs:
            l.append(torch.tanh(conv(out)).squeeze(3))
        out = l
        l = []
        for i in out:
            l.append(F.max_pool1d(i, kernel_size=i.size(2)).squeeze(2))
        out = torch.cat(l, 1)
        out = self.fc_dropout(out)
        out = self.linear(out)
        return out