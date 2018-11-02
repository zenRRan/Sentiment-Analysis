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

    test: 1 ||| xxx xx x xx

    :param fpath: data's path
    :return: sents_list -> ['0 i like it .', '3 no way .', ...]
    '''
    sents = []
    with open(fpath, 'r', encoding='utf8') as f:
        for line in f.readlines():
            line = line.strip().split()
            sent = line[2:]
            label = line[0]
            sents.append((sent, label))
    return sents

def build_dict(sents_list):
    '''
    :param sents_list: [('i like it .', 0), ('no way .', 3), ...]
    :return: OrderedDict() -> freq:word   char-vocab.sst
    '''
    dict = collections.OrderedDict()
    char_dict = collections.OrderedDict()
    label_dict = collections.OrderedDict()
    for t in sents_list:
        words = t[0]
        for word in words:
            if word not in dict:
                dict[word] = 1
            else:
                dict[word] += 1
            for char in word:
                if char not in char_dict:
                    char_dict[char] = 1
                else:
                    char_dict[char] += 1
        label = t[1]
        if label not in label_dict:
            label_dict[label] = 1
        else:
            label_dict[label] += 1
    return dict, char_dict, label_dict

def build_vab(dict, char_dict=None, cutoff=0, vcb_size=30000):
    '''
    :param dict: OrderedDict() -> freq:word
    :param cutoff: frequence's smaller than cutoff will be deleted.
    :return: alphabet class
    '''

    dict[unk_key] = 100
    dict[padding_key] = 100
    alpha = Alphabet(cutoff=cutoff, max_cap=vcb_size)
    alpha.initial(dict)
    alpha.m_b_fixed = True

    char_alpha = None
    if char_dict != None:
        char_dict[unk_key] = 100
        char_dict[padding_key] = 100
        char_alpha = Alphabet(cutoff=cutoff, max_cap=vcb_size)
        char_alpha.initial(char_dict)
        char_alpha.m_b_fixed = True


    return alpha, char_alpha

def get_idx(words, alpha):
    '''
    :param words: [i like it .]
    :param alpha: Alphabet()
    :return: indexs -> [23, 65, 7]
    '''
    indexs = []
    for word in words:
        idx = alpha.from_string(word)
        if idx == -1:
            idx = alpha.from_string(unk_key)
        indexs.append(idx)
    return indexs

def build_features(sents_list, alphabet, char_alphabet, label_alphabet):
    '''
    :param fpath: data's path
    :param alpha: Alphabet()
    :return: Features -> [class Feature, class Feature, ...]
    '''
    features = []
    for t in sents_list:
        feature = Feature()
        words = t[0]
        chars = list(' '.join(words))
        label = t[1]

        feature.words = words
        feature.chars = chars

        feature.length = len(words)
        feature.label = label_alphabet.string2id[label]

        feature.ids = get_idx(words, alphabet)
        feature.char_ids = get_idx(chars, char_alphabet)

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
    data_dict, char_dict, label_dict = build_dict(train_sents_list)
    alphabet, char_alphabet = build_vab(dict=data_dict, char_dict=char_dict, cutoff=parser.freq_vocab, vcb_size=parser.vcb_size)
    label_alphabet, _ = build_vab(dict=label_dict)
    train_features = build_features(train_sents_list, alphabet, char_alphabet, label_alphabet=label_alphabet)
    dev_features = build_features(dev_sents_list, alphabet, char_alphabet, label_alphabet=label_alphabet)
    test_features = build_features(test_sents_list, alphabet, char_alphabet, label_alphabet=label_alphabet)

    #save features
    if not os.path.isdir(parser.save_dir):
        os.mkdir(parser.save_dir)
    torch.save(train_features, parser.save_dir + '/train.sst')
    torch.save(dev_features, parser.save_dir + '/dev.sst')
    torch.save(test_features, parser.save_dir + '/test.sst')
    torch.save(alphabet, parser.save_dir + '/vocab.sst')
    torch.save(char_alphabet, parser.save_dir + '/char_vocab.sst')
    torch.save(label_alphabet, parser.save_dir + '/label_vocab.sst')
