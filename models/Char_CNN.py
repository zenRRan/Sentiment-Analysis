# Version python3.6
# -*- coding: utf-8 -*-
# @Time    : 2018/10/27 7:53 PM
# @Author  : zenRRan
# @Email   : zenrran@qq.com
# @File    : Char_CNN.py
# @Software: PyCharm Community Edition


import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.Embedding as Embedding

class Char_CNN(nn.Module):

    def __init__(self, opts, vocab, char_vocab, label_vocab):
        super(Char_CNN, self).__init__()

        self.embed_dim = opts.embed_size
        self.char_embed_dim = opts.char_embed_size
        self.word_num = vocab.m_size
        self.char_num = char_vocab.m_size
        self.pre_embed_path = opts.pre_embed_path
        self.string2id = vocab.string2id
        self.char_string2id = char_vocab.string2id
        self.embed_uniform_init = opts.embed_uniform_init
        self.stride = opts.stride
        self.kernel_size = opts.kernel_size
        self.kernel_num = opts.kernel_num
        self.label_num = label_vocab.m_size
        self.embed_dropout = opts.embed_dropout
        self.fc_dropout = opts.fc_dropout

        self.word_embeddings = nn.Embedding(self.word_num, self.embed_dim)
        self.char_embeddings = nn.Embedding(self.char_num, self.char_embed_dim)

        if opts.pre_embed_path != '':
            embedding = Embedding.load_predtrained_emb_zero(self.pre_embed_path, self.string2id)
            self.word_embeddings.weight.data.copy_(embedding)
        else:
            nn.init.uniform_(self.word_embeddings.weight.data, -self.embed_uniform_init, self.embed_uniform_init)
        nn.init.uniform_(self.char_embeddings.weight.data, -self.embed_uniform_init, self.embed_uniform_init)

        self.word_convs = nn.ModuleList(
            [nn.Conv2d(1, self.embed_dim, (K, self.embed_dim), stride=self.stride, padding=(K // 2, 0)) for K in
             self.kernel_size])
        self.char_convs = nn.ModuleList(
            [nn.Conv2d(1, self.char_embed_dim, (K, self.char_embed_dim), stride=self.stride, padding=(K // 2, 0)) for K in
             self.kernel_size])

        in_fea = len(self.kernel_size) * (self.embed_dim + self.char_embed_dim)

        self.linear1 = nn.Linear(in_fea, in_fea // 2)
        self.linear2 = nn.Linear(in_fea // 2, self.label_num)

        self.embed_dropout = nn.Dropout(self.embed_dropout)
        self.fc_dropout = nn.Dropout(self.fc_dropout)

    def forward(self, word, char):
        word = self.word_embeddings(word)
        char = self.char_embeddings(char)
        word = self.embed_dropout(word)
        char = self.embed_dropout(char)

        word_l = []
        word = word.unsqueeze(1)
        for conv in self.word_convs:
            word_l.append(torch.tanh(conv(word)).squeeze(3))

        word_out = []
        for i in word_l:
            word_out.append(F.max_pool1d(i, kernel_size=i.size(2)).squeeze(2))
        word_out = torch.cat(word_out, 1)  # torch.Size([64, 300])

        char_l = []
        char = char.unsqueeze(1)
        for conv in self.char_convs:
            char_l.append(torch.tanh(conv(char)).squeeze(3))

        char_out = []
        for i in char_l:
            char_out.append(F.max_pool1d(i, kernel_size=i.size(2)).squeeze(2))
        char_out = torch.cat(char_out, 1)  # torch.Size([64, 150])

        out = torch.cat((word_out, char_out), 1)  # torch.Size([64, 450])

        out = self.fc_dropout(out)
        out = self.linear1(F.relu(out))
        out = self.linear2(F.relu(out))
        return out