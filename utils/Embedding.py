# Version python3.6
# -*- coding: utf-8 -*-
# @Time    : 2018/2/8 下午8:22
# @Author  : zenRRan
# @Email   : zenrran@qq.com
# @File    : Embedding.py
# @Software: PyCharm Community Edition

import torch
from Common import padding_key
import numpy as np
import torch.nn as nn
import random
random.seed(23)
torch.manual_seed(23)

class Embedding(nn.Embedding):
    def reset_parameters(self):
        print("Use uniform to initialize the embedding")
        self.weight.data.uniform_(-0.01, 0.01)

        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)

class ConstEmbedding(nn.Module):
    def __init__(self, pretrained_embedding, padding_idx=0):
        super(ConstEmbedding, self).__init__()
        self.vocab_size = pretrained_embedding.size(0)
        self.embedding_size = pretrained_embedding.size(1)
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=padding_idx, sparse=True)
        self.embedding.weight = nn.Parameter(pretrained_embedding, requires_grad=False)

    def cuda(self, device_id=None):
        """
           The weights should be always on cpu
       """
        return self._apply(lambda t: t.cpu())

    def forward(self, input):
        """
           return cpu tensor
       """
        # is_cuda = next(input).is_cuda
        is_cuda = input.is_cuda
        if is_cuda: input = input.cpu()
        self.embedding._apply(lambda t: t.cpu())

        x = self.embedding(input)
        if is_cuda: x = x.cuda()

        return x

class VarEmbeddingCuda(nn.Module):
    def __init__(self, pretrained_embedding, padding_idx=0):
        super(VarEmbeddingCuda, self).__init__()
        self.vocab_size = pretrained_embedding.size(0)
        self.embedding_size = pretrained_embedding.size(1)
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=padding_idx)
        self.embedding.weight = nn.Parameter(pretrained_embedding, requires_grad=True)
        # self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embedding))
        # self.embedding.weight.requires_grad = True

    def forward(self, input):
        x = self.embedding(input)
        return x

class VarEmbeddingCPU(nn.Module):
    def __init__(self, pretrained_embedding, padding_idx=0):
        super(VarEmbeddingCPU, self).__init__()
        self.vocab_size = pretrained_embedding.size(0)
        self.embedding_size = pretrained_embedding.size(1)
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=padding_idx)
        self.embedding.weight = nn.Parameter(pretrained_embedding, requires_grad=True)
        # self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embedding))
        # self.embedding.weight.requires_grad = True

    def forward(self, input):
        is_cuda = input.is_cuda
        if is_cuda: input = input.cpu()
        self.embedding._apply(lambda t: t.cpu())

        x = self.embedding(input)
        if is_cuda: x = x.cuda()
        return x


class LSTM(nn.LSTM):
    def reset_parameters(self):
        for name, param in self.named_parameters():
            if "weight" in name:
                for i in range(4):
                    nn.init.orthogonal(self.__getattr__(name)[self.hidden_size*i:self.hidden_size*(i+1),:])
            if "bias" in name:
                nn.init.constant(self.__getattr__(name), 0)


def load_predtrained_emb_zero(path, words_dic, padding=False):
    print("start load predtrained embedding...")
    if padding:
        padID = words_dic[padding_key]
    embeding_dim = -1
    with open(path, encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip().split(" ")
            if len(line) <= 1:
                print("load_predtrained_embedding text is wrong!  -> len(line) <= 1")
                break
            else:
                embeding_dim = len(line) - 1
                break
    word_size = len(words_dic)
    print("The word size is ", word_size)
    print("The dim of predtrained embedding is ", embeding_dim, "\n")

    embedding = np.zeros((word_size, embeding_dim))
    in_word_list = []
    with open(path, encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip().split(' ')
            index = words_dic.get(line[0])
            if index:
                vector = np.array(line[1:], dtype='float32')
                embedding[index] = vector
                in_word_list.append(index)
    print("done")
    print(embedding)
    return torch.from_numpy(embedding).float()


def load_predtrained_emb_avg(path, words_dic, padding=False, save=''):
    print("start load predtrained embedding...")
    if padding:
        padID = words_dic[padding_key]
    embeding_dim = -1
    with open(path, encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip().split(" ")
            if len(line) <= 1:
                print("load_predtrained_embedding text is wrong!  -> len(line) <= 1")
                break
            else:
                embeding_dim = len(line) - 1
                break
    word_size = len(words_dic)
    print("The word size is ", word_size)
    print("The dim of predtrained embedding is ", embeding_dim, "\n")

    lines = []
    embedding = np.zeros((word_size, embeding_dim))
    in_word_list = []
    with open(path, encoding='utf-8') as f:
        for line in f.readlines():
            rawline = line
            line = line.strip().split(' ')
            index = words_dic.get(line[0])
            if index:
                lines.append(rawline)
                vector = np.array(line[1:], dtype='float32')
                embedding[index] = vector
                in_word_list.append(index)

    embedding = np.zeros((word_size, embeding_dim))
    avg_col = np.sum(embedding, axis=0) / len(in_word_list)
    for i in range(word_size):
        if not padding:
            if i in in_word_list:
                embedding[i] = avg_col
        elif i in in_word_list and i != padID:
            embedding[i] = avg_col
    print("done")

    '''
        save
    '''
    if save != '':
        with open(save, 'a') as f:
            for line in lines:
                line = line.strip()
                f.write(line+'\n')
            print("save successful! path=", save)
    return torch.from_numpy(embedding).float()















