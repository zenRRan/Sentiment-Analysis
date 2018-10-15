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

if __name__ == '__main__':

    # get the train opts
    parser = argparse.ArgumentParser('Train opts')
    parser = opts.trainer_opts(parser)
    opts = parser.parse_args()

    # load the data
    train_features_list = torch.load(opts.data.dir + 'train.sst')
    dev_features_list = torch.load(opts.data.dir + 'dev.sst')
    dev_features_list = torch.load(opts.data.dir + 'test.sst')



