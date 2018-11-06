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
    l1 = torch.Tensor([[1, 2]])
    l2 = torch.Tensor([[3, 4]])

    a = ['5', '2', '3', '4', '5']
    b = [1, 2, 3, 4, 5]
    c = 0

    print(list(map(int, a)))


# parser = argparse.ArgumentParser('decoder opts')
    # parser = opts.decoder_opts(parser)
    # parser = parser.parse_args()
    #
    #
    # decoder = Decoder(opts=opts, )

    # path = '/Users/zhenranran/Desktop/law_research_cup/corpus/cail2018_small/good/data_valid.json'







