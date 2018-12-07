#!/usr/bin/env python
# encoding: utf-8

"""
@version: python3.6
@author: 'zenRRan'
@license: Apache Licence 
@contact: zenrran@qq.com
@software: PyCharm
@file: LSTM_TreeLSTM_rels.py
@time: 2018/12/3 20:21
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as Var
import utils.Embedding as Embedding
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utils.tree import *
import numpy as np

import random

class LSTM_ChildSumTreeLSTM_rel(nn.Module):
    def __init__(self, opts, vocab, label_vocab, rel_vocab):
        super(LSTM_ChildSumTreeLSTM_rel, self).__init__()

        random.seed(opts.seed)
        torch.manual_seed(opts.seed)
        torch.cuda.manual_seed(opts.seed)

        self.embed_dim = opts.embed_size
        self.word_num = vocab.m_size
        self.pre_embed_path = opts.pre_embed_path
        self.string2id = vocab.string2id
        self.embed_uniform_init = opts.embed_uniform_init
        self.label_num = label_vocab.m_size
        self.rel_num = rel_vocab.m_size
        self.dropout = opts.dropout
        self.hidden_size = opts.hidden_size
        self.hidden_num = opts.hidden_num
        self.bidirectional = opts.bidirectional
        self.use_cuda = opts.use_cuda
        self.debug = False

        self.embeddings = nn.Embedding(self.word_num, self.embed_dim)
        self.rel_embeddings = nn.Embedding(self.rel_num, self.embed_dim)
        self.dropout = nn.Dropout(self.dropout)
        self.rnn = nn.LSTM(
            self.embed_dim,
            self.hidden_size,
            dropout=opts.dropout,
            num_layers=self.hidden_num,
            batch_first=True)
        if opts.pre_embed_path != '':
            embedding = Embedding.load_predtrained_emb_zero(self.pre_embed_path, self.string2id)
            self.embeddings.weight.data.copy_(embedding)

        self.dt_tree = DTTreeLSTM(self.embed_dim * 2, self.hidden_size)
        self.td_tree = TDTreeLSTM(self.embed_dim * 2, self.hidden_size)

        self.linear1 = nn.Linear(self.hidden_size * 2, self.hidden_size // 2)
        self.linear2 = nn.Linear(self.hidden_size // 2, self.label_num)

    def forward(self, xs, rels, heads, xlengths):

        emb = self.embeddings(xs)
        emb = self.dropout(emb)
        input_packed = pack_padded_sequence(emb, lengths=np.array(xlengths), batch_first=True)
        out_packed, h_n = self.rnn(input_packed)
        out = pad_packed_sequence(out_packed, batch_first=True)[0]
        out = out.contiguous()

        rel_emb = self.rel_embeddings(rels)
        outputs = torch.cat([out, rel_emb], 2)
        outputs = outputs.transpose(0, 1)

        max_length, batch_size, input_dim = outputs.size()

        trees = []
        indexes = np.zeros((max_length, batch_size), dtype=np.int32)
        for b, head in enumerate(heads):
            root, tree = createTree(head)
            root.traverse()
            for step, index in enumerate(root.order):
                indexes[step, b] = index
            trees.append(tree)

        dt_outputs = self.dt_tree(outputs, indexes, trees, xlengths)
        td_outputs = self.td_tree(outputs, indexes, trees, xlengths)

        out = torch.cat([dt_outputs, td_outputs], dim=2)
        out = self.dropout(out)
        out = torch.transpose(out, 1, 2)
        out = torch.tanh(out)
        out = F.max_pool1d(out, out.size(2))
        out = out.squeeze(2)
        out = self.linear1(F.relu(out))
        out = self.linear2(F.relu(out))
        return out



class DTTreeLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        """
        """
        super(DTTreeLSTM, self).__init__()
        self._input_size = input_size
        self._hidden_size = hidden_size

        # linear parameters for transformation from input to hidden state
        # LSTM
        self.i_x = nn.Linear(in_features=input_size, out_features=hidden_size, bias=True)
        self.i_h = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=True)
        self.f_x = nn.Linear(in_features=input_size, out_features=hidden_size, bias=True)
        self.f_h = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=True)
        self.o_x = nn.Linear(in_features=input_size, out_features=hidden_size, bias=False)
        self.o_h = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=False)
        self.u_x = nn.Linear(in_features=input_size, out_features=hidden_size, bias=False)
        self.u_h = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=False)

    def forward(self, inputs, indexes, trees, lengths):
        """
        :param inputs: batch first
        :param tree:
        :return: output, h_n
        """
        # print('inputs.size():', inputs.size())
        max_length, batch_size, input_dim = inputs.size()
        dt_state_h = []
        dt_state_c = []
        degree = np.zeros((batch_size, max_length), dtype=np.int32)
        last_indexes = np.zeros((batch_size), dtype=np.int32)
        for b, tree in enumerate(trees):
            dt_state_h.append({})
            dt_state_c.append({})
            for index in range(lengths[b]):
                degree[b, index] = tree[index].left_num + tree[index].right_num

        zeros = Var(inputs.data.new(self._hidden_size).fill_(0.))
        # print('zeros.size():', zeros.size())
        for step in range(max_length):
            step_inputs, left_child_hs, right_child_hs, compute_indexes = [], [], [], []
            left_child_cs, right_child_cs = [], []
            for b, tree in enumerate(trees):
                last_index = last_indexes[b]
                for idx in range(last_index, lengths[b]):
                    cur_index = indexes[idx, b]
                    if degree[b, cur_index] > 0:
                        break
                    last_indexes[b] += 1

                    compute_indexes.append((b, cur_index))
                    step_inputs.append(inputs[cur_index, b])
                    if tree[cur_index].left_num == 0:
                        left_child_h = [zeros]
                        left_child_c = [zeros]
                    else:
                        left_child_h = [dt_state_h[b][child.index] for child in tree[cur_index].left_children]
                        left_child_c = [dt_state_c[b][child.index] for child in tree[cur_index].left_children]



                    if tree[cur_index].right_num == 0:
                        right_child_h = [zeros]
                        right_child_c = [zeros]
                    else:
                        right_child_h = [dt_state_h[b][child.index] for child in tree[cur_index].right_children]
                        right_child_c = [dt_state_c[b][child.index] for child in tree[cur_index].right_children]

                    left_child_hs.append(left_child_h)
                    right_child_hs.append(right_child_h)
                    left_child_cs.append(left_child_c)
                    right_child_cs.append(right_child_c)



            if len(compute_indexes) == 0:
                for b, last_index in enumerate(last_indexes):
                    if last_index != lengths[b]:
                        print('bug exists: some nodes are not completed')
                break

            # add by zenRRan

            assert len(left_child_hs) == len(right_child_hs)
            assert len(left_child_cs) == len(right_child_cs)
            assert len(left_child_hs) == len(left_child_cs)

            child_hs = []
            child_cs = []
            for i in range(len(left_child_hs)):
                child_h = []
                child_h.extend(left_child_hs[i])
                child_h.extend(right_child_hs[i])
                child_c = []
                child_c.extend(left_child_cs[i])
                child_c.extend(right_child_cs[i])
                child_hs.append(child_h)
                child_cs.append(child_c)
            max_child_num = max([len(child_h) for child_h in child_hs])
            for i in range(len(child_hs)):
                child_hs[i].extend((max_child_num - len(child_hs[i])) * [zeros])
                child_cs[i].extend((max_child_num - len(child_cs[i])) * [zeros])
                child_hs[i] = torch.stack(child_hs[i], 0)
                child_cs[i] = torch.stack(child_cs[i], 0)

            #######

            step_inputs = torch.stack(step_inputs, 0)
            child_hs = torch.stack(child_hs, 0)
            child_cs = torch.stack(child_cs, 0)


            h, c = self.node_forward(step_inputs, child_hs, child_cs)
            for idx, (b, cur_index) in enumerate(compute_indexes):
                dt_state_h[b][cur_index] = h[idx]
                dt_state_c[b][cur_index] = c[idx]
                if trees[b][cur_index].parent is not None:
                    parent_index = trees[b][cur_index].parent.index
                    degree[b, parent_index] -= 1
                    if degree[b, parent_index] < 0:
                        print('strange bug')

        outputs, output_t = [], []

        for b in range(batch_size):
            output = [dt_state_h[b][idx] for idx in range(0, lengths[b])] \
                     + [zeros for idx in range(lengths[b], max_length)]
            outputs.append(torch.stack(output, 0))

        return torch.stack(outputs, 0)

    def node_forward(self, input, child_hs, child_cs):

        h_sum = torch.sum(child_hs, 1)

        i = self.i_x(input) + self.i_h(h_sum)
        i = torch.sigmoid(i)

        fx = self.f_x(input)
        fx = fx.unsqueeze(1)
        fx = fx.view(fx.size(0), 1, fx.size(2)).expand(fx.size(0), child_hs.size(1), fx.size(2))
        f = self.f_h(child_hs) + fx
        f = torch.sigmoid(f)

        fc = f * child_cs

        o = self.o_x(input) + self.o_h(h_sum)
        o = torch.sigmoid(o)

        u = self.u_x(input) + self.u_h(h_sum)
        u = torch.tanh(u)

        c = i * u + torch.sum(fc, 1)
        h = o * torch.tanh(c)

        return h, c

    def node_forward_1(self, input, left_child_h, right_child_h, left_child_c, right_child_c):
        hidden = left_child_h + right_child_h
        i = self.i_x(input) + self.i_h(hidden)
        i = torch.sigmoid(i)

        fl = self.f_xl(input) + self.f_hl(left_child_h)
        fl = torch.sigmoid(fl)

        fr = self.f_xr(input) + self.f_hr(right_child_h)
        fr = torch.sigmoid(fr)

        o = self.o_x(input) + self.o_h(hidden)
        o = torch.sigmoid(o)

        u = self.u_x(input) + self.u_h(hidden)
        u = torch.tanh(u)
        c = i * u + fl * left_child_c + fr * right_child_c
        h = o * torch.tanh(c)

        return h, c



class TDTreeLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        """
        """
        super(TDTreeLSTM, self).__init__()
        self._input_size = input_size
        self._hidden_size = hidden_size

        self.i_x = nn.Linear(in_features=input_size, out_features=hidden_size, bias=True)
        self.i_h = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=True)
        self.f_x = nn.Linear(in_features=input_size, out_features=hidden_size, bias=True)
        self.f_h = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=True)
        self.o_x = nn.Linear(in_features=input_size, out_features=hidden_size, bias=False)
        self.o_h = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=False)
        self.u_x = nn.Linear(in_features=input_size, out_features=hidden_size, bias=False)
        self.u_h = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=False)

    def forward(self, inputs, indexes, trees, lengths):
        """
        :param inputs:
        :param tree:
        :return: output, h_n
        """
        max_length, batch_size, input_dim = inputs.size()
        degree = np.ones((batch_size, max_length), dtype=np.int32)
        last_indexes = max_length * np.ones((batch_size), dtype=np.int32)
        td_state_h = []
        td_state_c = []
        for b in range(batch_size):
            td_state_h.append({})
            td_state_c.append({})
            root_index = indexes[lengths[b] - 1, b]
            degree[b, root_index] = 0
            last_indexes[b] = lengths[b]

        zeros = Var(inputs.data.new(self._hidden_size).fill_(0.))
        for step in range(max_length):
            step_inputs, left_parent_hs, right_parent_hs, compute_indexes = [], [], [], []
            left_parent_cs, right_parent_cs = [], []
            for b, tree in enumerate(trees):
                last_index = last_indexes[b]
                for idx in reversed(range(last_index)):
                    cur_index = indexes[idx, b]
                    if degree[b, cur_index] > 0:
                        break
                    last_indexes[b] -= 1
                    compute_indexes.append((b, cur_index))
                    step_inputs.append(inputs[cur_index, b])
                    parent_h = zeros
                    parent_c = zeros
                    if tree[cur_index].parent is None:
                        left_parent_hs.append(parent_h)
                        right_parent_hs.append(parent_h)
                        left_parent_cs.append(parent_c)
                        right_parent_cs.append(parent_c)
                    else:
                        valid_parent_h = td_state_h[b][tree[cur_index].parent.index]
                        valid_parent_c = td_state_c[b][tree[cur_index].parent.index]
                        if tree[cur_index].is_left:
                            left_parent_hs.append(valid_parent_h)
                            right_parent_hs.append(parent_h)
                            left_parent_cs.append(valid_parent_c)
                            right_parent_cs.append(parent_c)
                        else:
                            left_parent_hs.append(parent_h)
                            right_parent_hs.append(valid_parent_h)
                            left_parent_cs.append(parent_c)
                            right_parent_cs.append(valid_parent_c)

            if len(compute_indexes) == 0:
                for last_index in last_indexes:
                    if last_index != 0:
                        print('bug exists: some nodes are not completed')
                break

            step_inputs = torch.stack(step_inputs, 0)
            left_parent_hs = torch.stack(left_parent_hs, 0)
            right_parent_hs = torch.stack(right_parent_hs, 0)
            parent_hs = left_parent_hs + right_parent_hs
            left_parent_cs = torch.stack(left_parent_cs, 0)
            right_parent_cs = torch.stack(right_parent_cs, 0)
            parent_cs = left_parent_cs + right_parent_cs

            h, c = self.node_forward(step_inputs, parent_hs, parent_cs)
            for idx, (b, cur_index) in enumerate(compute_indexes):
                td_state_h[b][cur_index] = h[idx]
                td_state_c[b][cur_index] = c[idx]
                for child in trees[b][cur_index].left_children:
                    degree[b, child.index] -= 1
                    if degree[b, child.index] < 0:
                        print('strange bug')
                for child in trees[b][cur_index].right_children:
                    degree[b, child.index] -= 1
                    if degree[b, child.index] < 0:
                        print('strange bug')

        outputs, output_t = [], []
        for b in range(batch_size):
            output = [td_state_h[b][idx] for idx in range(0, lengths[b])] \
                     + [zeros for idx in range(lengths[b], max_length)]
            outputs.append(torch.stack(output, 0))

        return torch.stack(outputs, 0)

    def node_forward(self, input, parent_hs, parent_cs):

        i = self.i_x(input) + self.i_h(parent_hs)
        i = torch.sigmoid(i)

        f = self.f_x(input) + self.f_h(parent_hs)
        f = torch.sigmoid(f)

        o = self.o_x(input) + self.o_h(parent_hs)
        o = torch.sigmoid(o)

        u = self.u_x(input) + self.u_h(parent_hs)
        u = torch.tanh(u)

        c = i * u + f * parent_cs
        h = o * torch.tanh(c)

        return h, c

    def node_forward_1(self, input, left_parent_hs, right_parents_hs, left_parent_cs, right_parent_cs):
        hidden = left_parent_hs + right_parents_hs
        i = self.i_x(input) + self.i_h(hidden)
        i = F.sigmoid(i)

        f = self.f_x(input) + self.f_hl(left_parent_hs) + self.f_hr(right_parents_hs)
        f = F.sigmoid(f)

        o = self.o_x(input) + self.o_h(hidden)
        o = F.sigmoid(o)

        u = self.u_x(input) + self.u_h(hidden)
        u = F.tanh(u)

        c = i * u + f * (left_parent_cs + right_parent_cs)
        h = o * F.tanh(c)

        return h, c
