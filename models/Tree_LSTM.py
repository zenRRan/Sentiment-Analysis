# Version python3.6
# -*- coding: utf-8 -*-
# @Time    : 2018/11/5 1:54 PM
# @Author  : zenRRan
# @Email   : zenrran@qq.com
# @File    : Tree_LSTM.py
# @Software: PyCharm Community Edition


import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.Embedding as Embedding

import random

class Child_Sum_Tree_LSTM(nn.Module):
    def __init__(self, opts, vocab, label_vocab):
        super(Child_Sum_Tree_LSTM, self).__init__()

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





    def forward(self, input):
        # TODO
        return None