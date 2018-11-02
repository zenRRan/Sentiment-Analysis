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
    n = int(input('input a number:'))
    node_list = []
    for i in range(n):
        new_node = i+1
        node_list.append(new_node)

    cnt = 1

    delete_node_list = []
    node = None

    while len(node_list) != 0:
        for i in range(len(node_list)):
            node = node_list[i]
            if cnt == 3:
                delete_node_list.append(node_list[i])
                cnt = 1
            else:
                cnt += 1
        if len(delete_node_list) != 0:
            for node in delete_node_list:
                node_list.remove(node)
            delete_node_list = []
        if len(node_list) == 1:
            print(node_list[0])
            break



# parser = argparse.ArgumentParser('decoder opts')
    # parser = opts.decoder_opts(parser)
    # parser = parser.parse_args()
    #
    #
    # decoder = Decoder(opts=opts, )

    # path = '/Users/zhenranran/Desktop/law_research_cup/corpus/cail2018_small/good/data_valid.json'







