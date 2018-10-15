#!/usr/bin/env python
# encoding: utf-8

"""
@version: python3.6
@author: 'zenRRan'
@license: Apache Licence 
@contact: zenrran@qq.com
@software: PyCharm
@file: build_batch.py
@time: 2018/10/15 10:44
"""

import utils.Common
import random


class Build_Batch:

    def __init__(self, features, opts):
        self.batch_size = opts.batch_size
        self.batch_num = 0
        self.features = features
        self.shuffer = opts.shuffer
        self.sort = opts.sort
        self.batch_features = []

        random.seed(opts.seed)

    def create_same_sents_length_one_batch(self):
        '''
        :return:[[[x x x x]
                  [x x x x]]
                 [[x x x o]
                  [x x x o]
                  [x x x o]]]
        '''
        pass

    def create_sorted_batch(self):
        '''
        :return: [[[x x o o]
                   [x x x o]
                   [x x x o]]
                  [[x x x o]
                   [x x x x]]]
        '''
        pass

    def sort_batch(self, features):
        if self.sort:
            features = sorted(features, key= lambda feature: feature.length)
        return features

    def shuffer_data(self, data):
        if self.shuffer:
            data = random.shuffle(data)
        return data
