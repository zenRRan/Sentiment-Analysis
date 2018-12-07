# Version python3.6
# -*- coding: utf-8 -*-
# @Time    : 2018/11/6 4:40 PM
# @Author  : zenRRan
# @Email   : zenrran@qq.com
# @File    : GRU.py
# @Software: PyCharm Community Edition


import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.Embedding as Embedding

import random

class GRU(nn.Module):
    def __init__(self, opts, vocab, label_vocab):
        super(GRU, self).__init__()

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
        self.hidden_num = opts.hidden_num
        self.hidden_size = opts.hidden_size
        self.hidden_dropout = opts.hidden_dropout
        self.bidirectional = opts.bidirectional

        self.embeddings = nn.Embedding(self.word_num, self.embed_dim)
        if opts.pre_embed_path != '':
            embedding = Embedding.load_predtrained_emb_zero(self.pre_embed_path, self.string2id)
            self.embeddings.weight.data.copy_(embedding)
        else:
            nn.init.uniform_(self.embeddings.weight.data, -self.embed_uniform_init, self.embed_uniform_init)

        self.gru = nn.GRU(
            self.embed_dim,
            self.hidden_size,
            dropout=self.hidden_dropout,
            num_layers=self.hidden_num,
            batch_first=True,
            bidirectional=self.bidirectional
        )
        self.embed_dropout = nn.Dropout(self.embed_dropout)
        self.fc_dropout = nn.Dropout(self.fc_dropout)
        self.linear1 = nn.Linear(self.hidden_size * 2, self.hidden_size // 2)
        self.linear2 = nn.Linear(self.hidden_size // 2, self.label_num)

    def forward(self, input):
        out = self.embeddings(input)
        out = self.embed_dropout(out)
        out, _ = self.gru(out)   #[1, 1, 200]

        out = torch.transpose(out, 1, 2)

        out = torch.tanh(out)

        out = F.max_pool1d(out, out.size(2))  #[1, 200, 1]

        out = out.squeeze(2)          #[1, 400]

        out = self.fc_dropout(out)
        out = self.linear1(F.relu(out))
        output = self.linear2(F.relu(out))

        return output
