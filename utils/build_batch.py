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

import random


class Build_Batch:

    def __init__(self, features, opts, batch_size, pad_idx):

        self.batch_size = batch_size
        self.features = features
        self.shuffer = opts.shuffer
        self.sort = opts.sort

        self.batch_num = 0
        self.batch_features = []
        self.data_batchs = [] # [(data, label)]

        self.PAD = pad_idx

        random.seed(opts.seed)

    def create_same_sents_length_one_batch(self):
        '''
        :return:[[[x x x x]
                  [x x x x]]
                 [[x x x o]
                  [x x x o]
                  [x x x o]]]
        '''

        self.features = self.sort_features(self.features)
        new_list = []
        self.batch_features = []
        self.data_batchs = []
        same_len = True
        for feature in self.features:
            if same_len and len(new_list) < self.batch_size:
                new_list.append(feature)
            else:
                new_list = self.shuffer_data(new_list)
                self.batch_features.append(new_list)
                ids, labels = self.choose_data_from_features(new_list)
                ids = self.add_pad(ids)
                self.data_batchs.append((ids, labels))
                new_list = []
        self.batch_features = self.shuffer_data(self.batch_features)
        return self.batch_features, self.data_batchs

    def create_sorted_normal_batch(self):
        '''
        :return: [[[x x o o]
                   [x x x o]
                   [x x x o]]
                  [[x x x o]
                   [x x x x]]]
        '''

        self.features = self.sort_features(self.features)
        new_list = []
        self.batch_features = []
        self.data_batchs = []

        for feature in self.features:
            if len(new_list) < self.batch_size:
                new_list.append(feature)
            else:
                self.batch_num += 1

                new_list = self.shuffer_data(new_list)
                self.batch_features.append(new_list)
                ids, labels = self.choose_data_from_features(new_list)
                ids = self.add_pad(ids)
                self.data_batchs.append((ids, labels))
                new_list = []
        self.batch_features = self.shuffer_data(self.batch_features)
        return self.batch_features, self.data_batchs

    def choose_data_from_features(self, features):
        ids = []
        labels = []
        for feature in features:
            ids.append(feature.ids)
            labels.append(feature.label)

        return ids, labels


    def add_pad(self, data_list):
        '''
        :param data_list: [[x x x], [x x x x],...]
        :return: [[x x x o o], [x x x x o],...]
        '''
        max_len = 0
        for data in data_list:
            max_len = max(max_len, len(data))
        for data in data_list:
            data.extend([self.PAD]*(max_len - len(data)))

        return data_list

    def sort_features(self, features):
        if self.sort:
            features = sorted(features, key=lambda feature: feature.length)
        return features

    def shuffer_data(self, data):
        if self.shuffer:
            data = random.shuffle(data)
        return data
