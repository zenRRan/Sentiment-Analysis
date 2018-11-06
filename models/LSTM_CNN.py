# Version python3.6
# -*- coding: utf-8 -*-
# @Time    : 2018/10/27 7:51 PM
# @Author  : zenRRan
# @Email   : zenrran@qq.com
# @File    : LSTM_CNN.py
# @Software: PyCharm Community Edition


import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.Embedding as Embedding

import random

class LSTM_CNN(nn.Module):

    def __init__(self, opts, vocab, label_vocab):
        super(LSTM_CNN, self).__init__()

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

        # CNN
        self.stride = opts.stride
        self.kernel_size = opts.kernel_size
        self.kernel_num = opts.kernel_num

        # RNN
        self.hidden_num = opts.hidden_num
        self.hidden_size = opts.hidden_size
        self.hidden_dropout = opts.hidden_dropout
        self.bidirectional = opts.bidirectional

        self.flag = 2 if self.bidirectional else 1

        self.embeddings = nn.Embedding(self.word_num, self.embed_dim)
        if opts.pre_embed_path != '':
            embedding = Embedding.load_predtrained_emb_zero(self.pre_embed_path, self.string2id)
            self.embeddings.weight.data.copy_(embedding)
        else:
            nn.init.uniform_(self.embeddings.weight.data, -self.embed_uniform_init, self.embed_uniform_init)

        self.convs = nn.ModuleList(
            [nn.Conv2d(1,
                       self.kernel_num,
                       (K, self.hidden_size * self.flag),
                       stride=self.stride,
                       padding=(K // 2, 0)) for K in self.kernel_size])

        self.lstm = nn.LSTM(
            self.embed_dim,
            self.hidden_size,
            dropout=self.hidden_dropout,
            num_layers=self.hidden_num,
            batch_first=True,
            bidirectional=self.bidirectional)

        in_fea = len(self.kernel_size) * self.kernel_num

        self.linear1 = nn.Linear(in_fea, in_fea // 2)
        self.linear2 = nn.Linear(in_fea // 2, self.label_num)

        self.embed_dropout = nn.Dropout(self.embed_dropout)
        self.fc_dropout = nn.Dropout(self.fc_dropout)

    def forward(self, input):
        out = self.embeddings(input)
        out = self.embed_dropout(out)

        #lstm
        out, _ = self.lstm(out)  # torch.Size([64, 39, 256])
        # print(out.size())

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
        out = self.linear1(F.relu(out))
        out = self.linear2(F.relu(out))
        return out