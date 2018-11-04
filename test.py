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
import random

class Decoder:
    def __init__(self, opts):
        self.opts = opts
        self.model = torch.load(self.opts.model_path)
        self.vocab = torch.load(self.opts.data_dir + 'vocab.sst')
        self.label_vocab = torch.load(self.opts.data_dir + 'label_vocab.sst')

    def id2str(self):
        #TODO
        pass


# class Node:
#     def __init__(self, id):
#         self.id = id
#         self.next_node = None

if __name__ == '__main__':
    l1 = torch.Tensor([[1, 2]])
    l2 = torch.Tensor([[3, 4]])

    # input = torch.randn(20, 1, 50, 100)
    # # With square kernels and equal stride
    # n = nn.Conv2d(in_channels=1, out_channels=100, kernel_size=(3, 100), stride=1, padding=(2, 0))
    # m = nn.Conv2d(in_channels=1, out_channels=100, kernel_size=(3, 100), stride=1, padding=(2, 0))
    # output = m(input)
    # print(output.size())
    # output = n(input)
    # print(output.size())


    # a = torch.randn(1, 2, 10)
    # b = torch.randn((1, 10))
    # print(a.size())
    # print(b.size())
    # print(torch.stack((a, b)))

    a = [1, 2, 3]

    # # non-square kernels and unequal stride and with padding
    # m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
    # output = m(input)
    # print(output.size())
    #
    # # non-square kernels and unequal stride and with padding and dilation
    # m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
    # output = m(input)
    # print(output.size())


# parser = argparse.ArgumentParser('decoder opts')
    # parser = opts.decoder_opts(parser)
    # parser = parser.parse_args()
    #
    #
    # decoder = Decoder(opts=opts, )

    # path = '/Users/zhenranran/Desktop/law_research_cup/corpus/cail2018_small/good/data_valid.json'







