#!/usr/bin/env python
# encoding: utf-8

"""
@version: python3.6
@author: 'zenRRan'
@license: Apache Licence 
@contact: zenrran@qq.com
@software: PyCharm
@file: LSTM.py
@time: 2018/10/7 15:51
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.Embedding as Embedding

class LSTM(nn.Module):
    def __init__(self, opts, vocab, label_vocab):
        super(LSTM, self).__init__()
        self.embeddings = nn.Embedding(vocab.m_size, opts.embed_size)
        if opts.pre_embed_path != '':
            embedding = Embedding.load_predtrained_emb_avg(opts.pre_embed_path, vocab.string2id)
            self.embeddings.weight.data.copy_(embedding)
        else:
            nn.init.uniform_(self.embeddings.weight.data, -opts.embed_uniform_init, opts.embed_uniform_init)

        self.biLSTM = nn.LSTM(
            opts.embed_size,
            opts.hidden_size,
            dropout=opts.hidden_dropout,
            num_layers=opts.hidden_num,
            batch_first=True,
            bidirectional=opts.bidirectional
        )
        self.embed_dropout = nn.Dropout(opts.embed_dropout)
        self.linear1 = nn.Linear(opts.hidden_size * 2, opts.hidden_size // 2)
        self.linear2 = nn.Linear(opts.hidden_size // 2, label_vocab.m_size)

    def forward(self, input):
        out = self.embeddings(input)
        out = self.embed_dropout(out)
        out, _ = self.biLSTM(out)   #[1, 1, 200]

        out = torch.transpose(out, 1, 2)

        out = torch.tanh(out)

        out = F.max_pool1d(out, out.size(2))  #[1, 200, 1]

        out = out.squeeze(2)          #[1, 400]

        out = self.linear1(out)
        out = F.relu(out)
        output = self.linear2(out)

        return output






