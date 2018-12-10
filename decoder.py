#!/usr/bin/env python
# encoding: utf-8

"""
@version: python3.6
@author: 'zenRRan'
@license: Apache Licence 
@contact: zenrran@qq.com
@software: PyCharm
@file: decoder.py
@time: 2018/12/9 19:21
"""


import torch
from utils.Common import *
from utils.build_batch import Build_Batch
import argparse
from utils.opts import *
from torch.autograd import Variable

class Decoder:
    def __init__(self, opts):
        self.opts = opts
        self.model = torch.load(self.opts.model_path)
        self.features_list, self.vocab, self.char_vocab, self.label_vocab, self.rel_vocab \
            = None, None, None, None, None
        self.batch_size = self.opts.batch_size
        self.save_path = self.opts.save_path
        self.load_data(self.opts.dir, self.opts.type)
        self.decoder()

    def load_data(self, data_dir, type):
        self.features_list = torch.load(data_dir + '/'+ type +'.sst')
        self.vocab = torch.load(data_dir + '/vocab.sst')
        self.char_vocab = torch.load(data_dir + '/char_vocab.sst')
        self.label_vocab = torch.load(data_dir + '/label_vocab.sst')
        self.rel_vocab = torch.load(data_dir + '/rel_vocab.sst')

    def decoder(self):
        '''
        build train dev test batches
        '''
        padding_id = self.vocab.from_string(padding_key)
        char_padding_id = self.char_vocab.from_string(padding_key)
        rel_padding_id = None
        if self.rel_vocab is not None:
            rel_padding_id = self.rel_vocab.from_string(padding_key)
        self.build_batch = Build_Batch(features=self.features_list,
                                             batch_size=self.batch_size,
                                             opts=self.opts, pad_idx=padding_id, char_padding_id=char_padding_id,
                                             rel_padding_id=rel_padding_id)
        self.batch_features, self.data_batchs = self.build_batch.create_sorted_normal_batch()

        # correct_num = 0
        data_batchs = self.data_batchs
        wrongs = []
        for batch in data_batchs:
            self.model.eval()
            if 'tree' in self.opts.model:
                sents = Variable(torch.LongTensor(batch[0]), requires_grad=False)
                label = Variable(torch.LongTensor(batch[1]), requires_grad=False)
                heads = batch[4]
                xlength = batch[6]
                tag_rels = Variable(torch.LongTensor(batch[7]), requires_grad=False)
                if self.opts.use_cuda:
                    sents = sents.cuda()
                    label = label.cuda()
                    tag_rels = tag_rels.cuda()
                if self.opts.model in ['treelstm', 'bitreelstm']:
                    pred = self.model(sents, heads, xlength)
                if self.opts.model in ['lstm_treelstm_rel', 'treelstm_rel', 'bitreelstm_rel']:
                    pred = self.model(sents, tag_rels, heads, xlength)
            else:
                sents = Variable(torch.LongTensor(batch[0]))
                label = Variable(torch.LongTensor(batch[1]))

                char_data = []
                if 'Char' in self.opts.model:
                    for char_list in batch[2]:
                        char_data.append(Variable(torch.LongTensor(char_list)))
                if self.opts.use_cuda:
                    sents = sents.cuda()
                    label = label.cuda()
                    new_char_data = []
                    for data in char_data:
                        new_char_data.append(data.cuda())
                    char_data = new_char_data
                if 'Char' in self.opts.model:
                    pred = self.model(sents, char_data)
                else:
                    pred = self.model(sents)

            # correct_num += (torch.max(pred, 1)[1].view(label.size()).data == label.data).sum()
            pred_index = torch.max(pred, 1)[1].view(label.size()).data.tolist()
            sents = batch[0]
            label = batch[1]
            for index, (t, p) in enumerate(zip(label, pred_index)):
                if t != p:
                    wrong_sent = self.get_sent(sents[index])
                    wrong_label = self.get_label(p)
                    wrongs.append((wrong_sent, wrong_label, t))
        self.write(wrongs)


    def write(self, wrongs):
        with open(self.save_path, 'w', encoding='utf8') as f:
            for wrong in wrongs:
                f.write('pred: ' + str(wrong[1]) + ' right: ' + str(wrong[2]) +' sent: ' + wrong[0] + '\n')

    def get_sent(self, idx):
        sent = []
        for id in idx:
            word = self.vocab.id2string[id]
            if word != padding_key:
                sent.append(word)
        return ' '.join(sent)

    def get_label(self, id):
        return self.label_vocab.id2string[id]


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train opts')
    parser = decoder_opts(parser)
    opts = parser.parse_args()
    print(opts)
    Decoder(opts)