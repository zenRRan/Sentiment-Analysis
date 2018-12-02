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

    def __init__(self, features, opts, batch_size, pad_idx, char_padding_id):

        self.batch_size = batch_size
        self.features = features
        self.shuffle = opts.shuffle
        self.sort = opts.sort

        self.batch_num = 0
        self.batch_features = []
        self.data_batchs = [] # [(data, label)]

        self.PAD = pad_idx
        self.CPAD = char_padding_id

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
            if len(new_list) != 0 and len(feature.words) != len(new_list[-1].words):
                same_len = False
            if same_len and len(new_list) < self.batch_size:
                new_list.append(feature)
            else:
                new_list = self.shuffle_data(new_list)
                self.batch_features.append(new_list)
                ids, char_ids, labels, forest, heads, children_batch_list, tag_rels = self.choose_data_from_features(new_list)
                ids_lengths = [len(id) for id in ids]
                ids = self.add_pad(ids, self.PAD)
                tag_rels = self.add_pad(tag_rels, self.PAD)
                char_ids = self.add_char_pad(char_ids, ids, self.CPAD)
                self.data_batchs.append((ids, labels, char_ids, forest, heads, children_batch_list, ids_lengths, tag_rels))
                new_list = []
                same_len = True
                new_list.append(feature)
        self.batch_features = self.shuffle_data(self.batch_features)
        self.data_batchs = self.shuffle_data(self.data_batchs)
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

                new_list = self.shuffle_data(new_list)
                self.batch_features.append(new_list)
                ids, char_ids, labels, forest, heads, children_batch_list, tag_rels = self.choose_data_from_features(new_list)
                ids_lengths = [len(id) for id in ids]
                ids = self.add_pad(ids, self.PAD)
                tag_rels = self.add_pad(tag_rels, self.PAD)
                char_ids = self.add_char_pad(char_ids, ids, self.CPAD)
                self.data_batchs.append((ids, labels, char_ids, forest, heads, children_batch_list, ids_lengths, tag_rels))
                new_list = []
                new_list.append(feature)
        self.batch_features = self.shuffle_data(self.batch_features)
        self.data_batchs = self.shuffle_data(self.data_batchs)
        return self.batch_features, self.data_batchs

    def choose_data_from_features(self, features):
        ids = []
        char_ids = []
        labels = []
        heads = []
        forest = []
        # bfs_batch_list = []
        children_batch_list = []
        tag_rels = []

        for feature in features:
            ids.append(feature.ids)
            char_ids.append(feature.char_ids)
            labels.append(feature.label)
            heads.append(feature.heads)
            forest.append(feature.root)
            # bfs_batch_list.append(feature.bfs_list)
            tag_rels.append(feature.rels_ids)
            rel = [tree.children_index_list for tree in feature.forest]
            max_len = feature.length
            new_rel = [[0 for _ in range(max_len)] for _ in range(max_len)]
            for i, each in enumerate(rel):
                for j, index in enumerate(each):
                    new_rel[i][index] = 1
            children_batch_list.append(new_rel)
        return ids, char_ids, labels, forest, heads, children_batch_list, tag_rels

    def add_char_pad(self, data_list, sents_ids_list, PAD):
        '''
        :param data_list:[[[x x], [x x x],...],[[x], [x x],...]]
        :param PAD: PAD id
        :return: [[[x x o], [x x x],...],[[x o], [x x],...]]
        '''
        new_data_list = []
        for sent_list, sent in zip(data_list, sents_ids_list):
            word_len = len(sent)
            max_len = 0
            new_sent_list = []
            for word_list in sent_list:
                max_len = max(max_len, len(word_list))
            for word_list in sent_list:
                new_sent_list.append(word_list + [PAD] * (max_len - len(word_list)))
            new_data_list.append(new_sent_list + [[PAD] * max_len] * (word_len - len(new_sent_list)))
        return new_data_list

    def add_pad(self, data_list, PAD):
        '''
        :param data_list: [[x x x], [x x x x],...]
        :return: [[x x x o o], [x x x x o],...]
        '''
        max_len = 0
        new_data_list = []
        for data in data_list:
            max_len = max(max_len, len(data))
        for data in data_list:
            new_data_list.append(data + [PAD]*(max_len - len(data)))

        return new_data_list

    def sort_features(self, features):
        if self.sort:
            features = sorted(features, key=lambda feature: feature.length)
        return features

    def shuffle_data(self, data):
        if self.shuffle:
            random.shuffle(data)
        return data
