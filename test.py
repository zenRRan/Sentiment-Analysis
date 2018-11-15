#!/usr/bin/env python
# encoding: utf-8

"""
@version: python3.6
@author: 'zenRRan'
@license: Apache Licence 
@contact: zenrran@qq.com
@software: PyCharm
@file: test.py
@time: 2018/10/15 8:14
"""

import torch
import torch.nn as nn
import utils.opts as opts
import argparse
import time
from torch.autograd import Variable
import random

random.seed(23)

class Decoder:
    def __init__(self, opts):
        self.opts = opts
        self.model = torch.load(self.opts.model_path)
        self.vocab = torch.load(self.opts.data_dir + 'vocab.sst')
        self.label_vocab = torch.load(self.opts.data_dir + 'label_vocab.sst')

    def id2str(self):
        #TODO
        pass


if __name__ == '__main__':

    rels = torch.Tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    # rels = torch.unsqueeze(rels, 1)
    # print(rels.size())
    # print(rels.size(0))
    # print(rels.size(2))

    rels_broadcast = rels.unsqueeze(1).expand(rels.size(0), 100, rels.size(1))

    print(rels_broadcast)
    # l = [1, 2, 3, 4, 5]
    # print(l[2:])

    # conll_path = 'data/MR/mr.train.txt.conll.out'
    # save_path = 'train.conll.out.txt'
    # out = []
    # words = []
    # with open(conll_path, 'r', encoding='utf8') as f:
    #     words = []
    #     for line in f.readlines():
    #         line = line.strip().split()
    #         if len(line) == 0:
    #             out.append(' '.join(words))
    #             words = []
    #         else:
    #             words.append(line[1])
    # with open(save_path, 'w', encoding='utf8') as f:
    #     for line in out:
    #         f.write(line + '\n')

# parser = argparse.ArgumentParser('decoder opts')
    # parser = opts.decoder_opts(parser)
    # parser = parser.parse_args()
    #
    #
    # decoder = Decoder(opts=opts, )

    # path = '/Users/zhenranran/Desktop/law_research_cup/corpus/cail2018_small/good/data_valid.json'







