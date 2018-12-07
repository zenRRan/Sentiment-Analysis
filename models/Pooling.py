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

import random

class Pooling(nn.Module):
    def __init__(self, opts, vocab, label_vocab):
        super(Pooling, self).__init__()

        random.seed(opts.seed)
        torch.manual_seed(opts.seed)
        torch.cuda.manual_seed(opts.seed)

        self.embed_dim = opts.embed_size
        self.word_num = vocab.m_size
        self.pre_embed_path = opts.pre_embed_path
        self.string2id = vocab.string2id
        self.embed_uniform_init = opts.embed_uniform_init
        self.label_num = label_vocab.m_size
        self.embed_dropout = opts.embed_dropout
        self.fc_dropout = opts.fc_dropout

        self.embeddings = nn.Embedding(self.word_num, self.embed_dim)
        if opts.pre_embed_path != '':
            embedding = Embedding.load_predtrained_emb_zero(self.pre_embed_path, self.string2id)
            self.embeddings.weight.data.copy_(embedding)
        else:
            nn.init.uniform_(self.embeddings.weight.data, -self.embed_uniform_init, self.embed_uniform_init)

        self.linear1 = nn.Linear(self.embed_dim, self.embed_dim // 2)
        self.linear2 = nn.Linear(self.embed_dim // 2, self.label_num)

        self.embed_dropout = nn.Dropout(self.embed_dropout)
        self.fc_dropout = nn.Dropout(self.fc_dropout)

    def forward(self, input):
        out = self.embeddings(input)
        out = self.embed_dropout(out)
        out = torch.tanh(out)
        out = torch.transpose(out, 1, 2)
        out = F.max_pool1d(out, out.size(2))
        out = out.squeeze(2)
        out = self.fc_dropout(out)
        out = self.linear1(F.relu(out))
        out = self.linear2(F.relu(out))
        return out