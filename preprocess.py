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
import re
import argparse
import utils.opts as opts
from utils.Feature import Feature
from utils.Alphabet import Alphabet
import collections
import torch
from utils.tree import *
from utils.Common import unk_key, padding_key

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`|]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)

    return string.strip()

def read_file2list(fpath):
    '''

    test: 1 ||| xxx xx x xx

    :param fpath: data's path
    :return: sents_list -> ['0 i like it .', '3 no way .', ...]
    '''
    sents = []
    idx = 1
    with open(fpath, 'r', encoding='utf8') as f:
        for line in f.readlines():
            # print(line)
            line = line.strip().split()
            sent = clean_str(' '.join(line[2:]))
            if len(sent) == 0:
                print(idx)
            sent = sent.split()
            label = line[0]
            sents.append((sent, label))
            idx += 1
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

def build_features(sents_list, alphabet, char_alphabet, label_alphabet, conll_list=None):
    '''
    :param fpath: data's path
    :param alpha: Alphabet()
    :return: Features -> [class Feature, class Feature, ...]
    '''

    features = []

    (conll_list, rel_alpha) = conll_list
    if conll_list is not None:
        assert len(conll_list) == len(sents_list)

    for idx, t in enumerate(sents_list):
        feature = Feature()
        words = t[0]
        # chars = list(' '.join(words))
        chars_list = []
        for word in words:
            chars_list.append(list(word))
        label = t[1]

        feature.words = words
        feature.chars = chars_list

        feature.length = len(words)
        feature.label = label_alphabet.string2id[label]

        feature.ids = get_idx(words, alphabet)
        chars_ids = []
        for chars in chars_list:
            chars_ids.append(get_idx(chars, char_alphabet))
        feature.char_ids = chars_ids

        if conll_list is not None:
            feature.heads = conll_list[idx][0]
            feature.root = conll_list[idx][1]
            feature.forest = conll_list[idx][2]
            feature.rels = conll_list[idx][3]
            feature.rels_ids = get_idx(feature.rels, rel_alpha)

        features.append(feature)

    return features

def read_conll(conll_path):

    heads_root_forest_rels_list = []
    dict = collections.OrderedDict()
    with open(conll_path, 'r', encoding='utf8') as f:
        sent = []
        idx = 0
        for line in f.readlines():
            line = line.strip().split()
            if len(line) == 0:
                heads, root, forest, rels = conll2word_heads_root_forest(sent)
                heads_root_forest_rels_list.append((heads, root, forest, rels))
                sent = []
                idx += 1
                for rel in rels:
                    if rel not in dict:
                        dict[rel] = 1
                    else:
                        dict[rel] += 1
            else:
                sent.append(line)
        rel_alpha, _ = build_vab(dict=dict)

    return heads_root_forest_rels_list, rel_alpha

def conll2word_heads_root_forest(conll_sent):
    '''
        1	a	_	NN	_	_	3	det	_	_
        2	technical	_	NN	_	_	3	amod	_	_
        3	triumph	_	NN	_	_	0	root	_	_
        4	and	_	NN	_	_	3	cc	_	_
        5	an	_	NN	_	_	7	det	_	_
        6	extraordinary	_	NN	_	_	7	amod	_	_
        7	bore	_	NN	_	_	3	conj	_	_
        8	.	_	NN	_	_	3	punct	_	_
    :param conll_sent:
    :return:
    '''

    heads, root, forest, rels = [], None, [], []

    for elem in conll_sent:
        assert type(elem) is list
        assert len(elem) == 10
        heads.append(int(elem[6]) - 1)
        rels.append(elem[7])

    root, forest = createTree(heads)

    return heads, root, forest, rels

def tree_add_label(feature_list):
    for feature in feature_list:
        label = feature.label
        feature.root.label = label

def tree_add_bfs(feature_list):
    for feature in feature_list:
        bfs = []
        depth = 0
        while len(bfs) != len(feature.forest):
            for child in feature.forest:
                if child.depth() == depth:
                    bfs.append(child.index)
            depth += 1
        feature.bfs_list = bfs

if __name__ == '__main__':

    # init args
    parser = argparse.ArgumentParser('data opts')
    parser = opts.preprocesser_opts(parser)
    parser = parser.parse_args()

    # get sents list
    train_sents_list = read_file2list(parser.raw_train_path)
    dev_sents_list = read_file2list(parser.raw_dev_path)
    test_sents_list = read_file2list(parser.raw_test_path)

    # get conll list(heads, root, forest)
    use_tree = False
    train_conll_list = None
    dev_conll_list = None
    test_conll_list = None
    if parser.train_conll_path != '' and parser.dev_conll_path != '' and parser.test_conll_path != '':
        use_tree = True
        train_conll_list = read_conll(parser.train_conll_path)
        dev_conll_list = read_conll(parser.dev_conll_path)
        test_conll_list = read_conll(parser.test_conll_path)

    # build dict and get the features
    data_dict, char_dict, label_dict = build_dict(train_sents_list)
    alphabet, char_alphabet = build_vab(dict=data_dict,
                                        char_dict=char_dict,
                                        cutoff=parser.freq_vocab,
                                        vcb_size=parser.vcb_size)
    label_alphabet, _ = build_vab(dict=label_dict)
    rel_alphabet = None
    if use_tree:
        rel_alphabet = train_conll_list[1]

    train_features = build_features(train_sents_list,
                                    alphabet,
                                    char_alphabet,
                                    label_alphabet=label_alphabet,
                                    conll_list=train_conll_list)
    dev_features = build_features(dev_sents_list,
                                  alphabet,
                                  char_alphabet,
                                  label_alphabet=label_alphabet,
                                  conll_list=dev_conll_list)
    test_features = build_features(test_sents_list,
                                   alphabet,
                                   char_alphabet,
                                   label_alphabet=label_alphabet,
                                   conll_list=test_conll_list)

    if use_tree:
        # add label
        tree_add_label(train_features)
        tree_add_label(dev_features)
        tree_add_label(test_features)

        # add bfs
        tree_add_bfs(train_features)
        tree_add_bfs(dev_features)
        tree_add_bfs(test_features)

    # save features
    if not os.path.isdir(parser.save_dir):
        os.mkdir(parser.save_dir)
    torch.save(train_features, parser.save_dir + '/train.sst')
    torch.save(dev_features, parser.save_dir + '/dev.sst')
    torch.save(test_features, parser.save_dir + '/test.sst')
    torch.save(alphabet, parser.save_dir + '/vocab.sst')
    torch.save(char_alphabet, parser.save_dir + '/char_vocab.sst')
    torch.save(label_alphabet, parser.save_dir + '/label_vocab.sst')
    torch.save(rel_alphabet, parser.save_dir + '/rel_vocab.sst')
