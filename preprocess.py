#!/usr/bin/env python
# encoding: utf-8

"""
@version: python3.6
@author: 'zenRRan'
@license: Apache Licence 
@contact: zenrran@qq.com
@software: PyCharm
@file: preprocess.py
@time: 2018/10/9 9:01
"""

import os
import argparse
import utils.opts as opts
from utils.Feature import Feature
from utils.Alphabet import Alphabet
import collections
import torch
from utils.Common import unk_key, padding_key


def read_file2list(fpath):
    '''
    :param fpath: data's path
    :return: sents_list -> ['0 i like it .', '3 no way .', ...]
    '''
    sents = []
    with open(fpath, 'r', encoding='utf8') as f:
        for line in f.readlines():
            line = line.strip().split()
            sent = line[1:]
            label = line[0]
            sents.append((sent, label))
    return sents

def build_dict(sents_list):
    '''
    :param sents_list: [('i like it .', 0), ('no way .', 3), ...]
    :return: OrderedDict() -> freq:word
    '''
    dict = collections.OrderedDict()
    for t in sents_list:
        words = t[0]
        for word in words:
            if word not in dict:
                dict[word] = 1
            else:
                dict[word] += 1
    return dict

def build_vab(dict, cutoff, vcb_size):
    '''
    :param dict: OrderedDict() -> freq:word
    :param cutoff: frequence's smaller than cutoff will be deleted.
    :return: alphabet class
    '''

    alpha = Alphabet(cutoff=cutoff, max_cap=vcb_size)
    alpha.initial(dict)
    alpha.from_string(unk_key)
    alpha.from_string(padding_key)
    alpha.m_b_fixed = True

    return alpha

def get_idx(words, alpha):
    '''
    :param words: [i like it .]
    :param alpha: Alphabet()
    :return: indexs -> [23, 65, 7]
    '''
    indexs = []
    for word in words:
        indexs.append(alpha.from_string(word))
    return indexs

def build_features(sents_list, alpha):
    '''
    :param fpath: data's path
    :param alpha: Alphabet()
    :return: Features -> [class Feature, class Feature, ...]
    '''
    features = []
    for t in sents_list:
        feature = Feature()
        words = t[1]
        label = t[0]
        feature.words = words
        feature.length = len(words)
        feature.label = label
        feature.ids = get_idx(words, alpha)
        features.append(feature)

    return features


if __name__ == '__main__':

    #init args
    parser = argparse.ArgumentParser('data opts')
    parser = opts.preprocesser_opts(parser)
    parser = parser.parse_args()

    #get sents list
    train_sents_list = read_file2list(parser.raw_train_path)
    dev_sents_list = read_file2list(parser.raw_dev_path)
    test_sents_list = read_file2list(parser.raw_test_path)

    #build dict and get the features
    dict = build_dict(train_sents_list)
    alpha = build_vab(dict=dict, cutoff=parser.freq_vocab, vcb_size=parser.vcb_size)
    train_features = build_features(train_sents_list, alpha)
    dev_features = build_features(dev_sents_list, alpha)
    test_features = build_features(test_sents_list, alpha)

    #save features
    if not os.path.isdir(parser.save_dir):
        os.mkdir(parser.save_dir)
    torch.save(train_features, parser.save_dir + '/train.sst')
    torch.save(dev_features, parser.save_dir + '/dev.sst')
    torch.save(test_features, parser.save_dir + '/test.sst')
