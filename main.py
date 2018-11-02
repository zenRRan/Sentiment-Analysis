#!/usr/bin/env python
# encoding: utf-8

"""
@version: python3.6
@author: 'zenRRan'
@license: Apache Licence 
@contact: zenrran@qq.com
@software: PyCharm
@file: main.py
@time: 2018/10/7 15:49
"""


import argparse
import utils.opts as opts
import torch
from utils.Feature import Feature
from utils.Common import unk_key, padding_key
from utils.trainer import Trainer
from utils.build_batch import Build_Batch

if __name__ == '__main__':

    # get the train opts
    parser = argparse.ArgumentParser('Train opts')
    parser = opts.trainer_opts(parser)
    opts = parser.parse_args()

    # load the data
    train_features_list = torch.load(opts.data_dir + '/train.sst')
    dev_features_list = torch.load(opts.data_dir + '/dev.sst')
    test_features_list = torch.load(opts.data_dir + '/test.sst')

    # load word-level vocab
    vocab = torch.load(opts.data_dir + '/vocab.sst')

    # load char-level vocab
    char_vocab = torch.load(opts.data_dir + '/char_vocab.sst')

    label_vocab = torch.load(opts.data_dir + '/label_vocab.sst')
    train_dev_test = (train_features_list, dev_features_list, test_features_list)
    #build batch
    # build_batcher = Build_Batch(features=train_features_list, opts=opts, pad_idx=vocab)

    vocab = (vocab, char_vocab)

    train = Trainer(train_dev_test, opts, vocab, label_vocab)
    train.train()




