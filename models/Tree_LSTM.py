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
from torch.autograd import Variable

import random

class ChildSumTreeLSTM(nn.Module):
    def __init__(self, opts, vocab, label_vocab):
        super(ChildSumTreeLSTM, self).__init__()

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
        self.hidden_size = opts.hidden_size
        self.use_cuda = opts.use_cuda


        self.embeddings = nn.Embedding(self.word_num, self.embed_dim)
        if opts.pre_embed_path != '':
            embedding = Embedding.load_predtrained_emb_zero(self.pre_embed_path, self.string2id)
            self.embeddings.weight.data.copy_(embedding)

        # build lstm
        self.ix = nn.Linear(self.embed_dim, self.hidden_size)
        self.ih = nn.Linear(self.hidden_size, self.hidden_size)

        self.fx = nn.Linear(self.embed_dim, self.hidden_size)
        self.fh = nn.Linear(self.hidden_size, self.hidden_size)

        self.ox = nn.Linear(self.embed_dim, self.hidden_size)
        self.oh = nn.Linear(self.hidden_size, self.hidden_size)

        self.ux = nn.Linear(self.embed_dim, self.hidden_size)
        self.uh = nn.Linear(self.hidden_size, self.hidden_size)

        self.out = nn.Linear(self.hidden_size, self.label_num)

        self.softmax = nn.LogSoftmax(dim=1)
        self.loss = nn.NLLLoss()
        self.embed_dropout = nn.Dropout(self.embed_dropout)
        self.fc_dropout = nn.Dropout(self.fc_dropout)

        if self.use_cuda:
            self.loss = self.loss.cuda()

    def node_forward(self, input, child_c, child_h):
        child_h_sum = torch.sum(torch.squeeze(child_h, 1), 0)

        i = F.sigmoid(self.ix(input) + self.ih(child_h_sum))
        o = F.sigmoid(self.fx(input) + self.fh(child_h_sum))
        u = F.sigmoid(self.ux(input) + self.uh(child_h_sum))

        fx = torch.unsqueeze(self.fx(input), 1)
        f = torch.cat([self.fh(child_i) + fx for child_i in child_h])
        f = torch.sigmoid(f)

        fc = F.torch.squeeze(F.torch.mul(f, child_c), 1)

        c = F.torch.mul(i, u) + F.torch.sum(fc, 0)
        h = F.torch.mul(o, F.tanh(c))

        return c, h

    def forward(self, tree, x):
        loss = Variable(torch.Tensor([0]))
        if self.use_cuda:
            loss = loss.cuda()
        for child in tree.children:
            output, child_loss = self.forward(child, x)
            loss += child_loss
        child_c, child_h = self.get_child_states(tree)
        tree.c, tree.h = self.node_forward(x[tree.word_idx], child_c, child_h)
        h = self.fc_dropout(tree.h)
        output = self.out(h)
        output = self.softmax(output)
        if tree.label is not None:
            label = Variable(torch.LongTensor([tree.label]))
            if self.use_cuda:
                label = label.cuda()
            loss += self.loss(output, label)

        return output, loss

    def get_child_states(self, tree):
        '''
        get c and h of all children
        :param tree:
        :return:
        '''

        children_num = len(tree.children_list)

        if children_num == 0:
            c = Variable(torch.zeros((1, 1, self.hidden_size)))
            h = Variable(torch.zeros((1, 1, self.hidden_size)))

        else:
            c = Variable(torch.zeros(children_num, 1, self.hidden_size))
            h = Variable(torch.zeros(children_num, 1, self.hidden_size))
            for idx, child in enumerate(tree.children_list):
                c[idx] = child.c
                h[idx] = child.h

        if self.use_cuda:
            c = c.cuda()
            h = h.cuda()
        return c, h




