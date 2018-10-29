# Version python3.6
# -*- coding: utf-8 -*-
# @Time    : 2018/10/16 10:49 AM
# @Author  : zenRRan
# @Email   : zenrran@qq.com
# @File    : trainer.py
# @Software: PyCharm Community Edition

from utils.build_batch import Build_Batch
from models.CNN import CNN
from models.LSTM import LSTM
from models.Multi_layer_CNN import Multi_Layer_CNN
from models.multi_channel_CNN import Multi_Channel_CNN
from models.Char_CNN import Char_CNN
from models.LSTM_CNN import LSTM_CNN
from models.Pooling import Pooling
from utils.Common import padding_key

import torch
import torch.nn.functional as F
import torch.optim as optim

from utils.log import Log

import os
import time

class Trainer:
    def __init__(self, train_dev_test, opts, vocab, label_vocab):
        self.train_features_list = train_dev_test[0]
        self.dev_features_list = train_dev_test[1]
        self.test_features_list = train_dev_test[2]
        self.opts = opts
        self.vocab = vocab[0]
        self.char_vocab = vocab[1]
        self.label_vocab = label_vocab
        self.epoch = opts.epoch
        self.model = None

        self.best_dev = 0
        self.best_dev_test = 0
        self.best_dev_epoch = 0

        self.char = False

        self.build_batch()
        self.init_model()
        self.init_optim()

        self.print_log = Log(opts)

        #save model switch
        self.save_model_switch = False

    def build_batch(self):
        '''
        build train dev test batches
        '''
        padding_id = self.vocab.from_string(padding_key)
        char_padding_id = self.char_vocab.from_string(padding_key)
        self.train_build_batch = Build_Batch(features=self.train_features_list, batch_size=self.opts.train_batch_size,
                                             opts=self.opts, pad_idx=padding_id, char_padding_id=char_padding_id)
        self.dev_build_batch = Build_Batch(features=self.dev_features_list, batch_size=self.opts.dev_batch_size,
                                           opts=self.opts, pad_idx=padding_id, char_padding_id=char_padding_id)
        self.test_build_batch = Build_Batch(features=self.test_features_list, batch_size=self.opts.test_batch_size,
                                            opts=self.opts, pad_idx=padding_id, char_padding_id=char_padding_id)

        if self.opts.train_batch_type == 'normal':
            self.train_batch_features, self.train_data_batchs = self.train_build_batch.create_sorted_normal_batch()
        elif self.opts.train_batch_type == 'same':
            self.train_batch_features, self.train_data_batchs = self.train_build_batch.create_same_sents_length_one_batch()
        else:
            raise RuntimeError('not normal or same')

        if self.opts.dev_batch_type == 'normal':
            self.dev_batch_features, self.dev_data_batchs = self.dev_build_batch.create_sorted_normal_batch()
        elif self.opts.dev_batch_type == 'same':
            self.dev_batch_features, self.dev_data_batchs = self.dev_build_batch.create_same_sents_length_one_batch()
        else:
            raise RuntimeError('not normal or same')

        if self.opts.test_batch_type == 'normal':
            self.test_batch_features, self.test_data_batchs = self.test_build_batch.create_sorted_normal_batch()
        elif self.opts.test_batch_type == 'same':
            self.test_batch_features, self.test_data_batchs = self.test_build_batch.create_same_sents_length_one_batch()
        else:
            raise RuntimeError('not normal or same')

    def init_model(self):
        '''
        pooling, rnn, lstm, bilstm, cnn, multi_cnn, gru
        :return:
        '''
        if self.opts.model == 'pooling':
            self.model = Pooling(opts=self.opts, vocab=self.vocab, label_vocab=self.label_vocab)
        elif self.opts.model == 'cnn':
            self.model = CNN(opts=self.opts, vocab=self.vocab, label_vocab=self.label_vocab)
        elif self.opts.model == 'multi_channel_cnn':
            self.model = Multi_Channel_CNN(opts=self.opts, vocab=self.vocab, label_vocab=self.label_vocab)
        elif self.opts.model == 'multi_layer_cnn':
            self.model = Multi_Layer_CNN(opts=self.opts, vocab=self.vocab, label_vocab=self.label_vocab)
        elif self.opts.model == 'char_cnn':
            self.char = True
            self.model = Char_CNN(opts=self.opts, vocab=self.vocab, char_vocab=self.char_vocab, label_vocab=self.label_vocab)
        elif self.opts.model == 'lstm':
            self.model = LSTM(opts=self.opts, vocab=self.vocab, label_vocab=self.label_vocab)
        elif self.opts.model == 'lstm_cnn':
            self.model = LSTM_CNN(opts=self.opts, vocab=self.vocab, label_vocab=self.label_vocab)
        # elif self.opts.model == 'bilstm':
        #     self.model = CNN(opts=self.opts)
        # elif self.opts.model == 'cnn':
        #     self.model = CNN(opts=self.opts)
        else:
            raise RuntimeError('please choose your model first!')

        if self.opts.use_cuda:
            self.model = self.model.cuda()

    def init_optim(self):
        'sgd, adam'
        if self.opts.optim == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.opts.lr, weight_decay=self.opts.weight_decay, momentum=self.opts.momentum)
        elif self.opts.optim == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.opts.lr, weight_decay=self.opts.weight_decay)

    def train(self):

        for epoch in range(self.epoch):
            totle_loss = torch.Tensor([0])
            correct_num = 0
            step = 0
            inst_num = 0
            for batch in self.train_data_batchs:

                self.model.train()
                self.optimizer.zero_grad()

                inst_num += len(batch[1])

                data = torch.LongTensor(batch[0])
                label = torch.LongTensor(batch[1])

                char_data = None
                if self.char:
                    char_data = torch.LongTensor(batch[2])

                if self.opts.use_cuda:
                    data = data.cuda()
                    label = label.cuda()
                    if self.char:
                        char_data = char_data.cuda()

                if self.char:
                    pred = self.model(data, char_data)
                else:
                    pred = self.model(data)


                loss = F.cross_entropy(pred, label)

                loss.backward()
                self.optimizer.step()

                totle_loss += loss.data

                step += 1
                correct_num += (torch.max(pred, 1)[1].view(label.size()).data == label.data).sum()
                if step % self.opts.print_every == 0:
                    avg_loss = totle_loss / inst_num
                    acc = float(correct_num) / inst_num * 100
                    time_dic = self.get_time()
                    time_str = "[{}-{:0>2d}-{:0>2d} {:0>2d}:{:0>2d}:{:0>2d}]".format(time_dic['year'], time_dic['month'], time_dic['day'], \
                                                          time_dic['hour'], time_dic['min'], time_dic['sec'])
                    log = time_str + " Epoch {} step {} acc: {:.2f}% loss: {:.6f}".format(epoch, step, acc, avg_loss.numpy()[0])
                    self.print_log.print_log(log)
                    print(log)
                    totle_loss = torch.Tensor([0])
                    inst_num = 0
                    correct_num = 0

            dev_score = self.accurcy(type='dev')
            test_score = self.accurcy(type='test')
            if dev_score > self.best_dev:
                self.best_dev = dev_score
                self.best_dev_epoch = epoch
                self.best_dev_test = test_score
                log = "Update! best test acc: {:.2f}%".format(self.best_dev_test)
                print(log)
                self.save_model(epoch)
            else:
                log = "not improved, best test acc: {:.2f}%, in epoch {}".format(self.best_dev_test, self.best_dev_epoch)
                print(log)

            self.print_log.print_log(log)

    def save_model(self, cur_epoch):
        if not os.path.isdir(self.opts.save_model_dir):
            os.mkdir(self.opts.save_model_dir)
        if self.opts.save_model_start_from <= cur_epoch:
            self.save_model_switch = True
        # if self.save_model_switch and (cur_epoch - self.opts.save_model_start_from) % self.opts.save_model_every == 0:
        if self.save_model_switch:
            time_dic = self.get_time()
            time_str = "[{}-{:0>2d}-{:0>2d}-{:0>2d}-{:0>2d}-{:0>2d}-]".format(time_dic['year'], time_dic['month'], time_dic['day'], \
                                                    time_dic['hour'], time_dic['min'], time_dic['sec'])
            fname = self.opts.save_model_dir + '/' + time_str + self.opts.model +'-model_epoch_' + str(cur_epoch) + '.pt'
            torch.save(self.model, fname)
            self.print_log.print_log('model saved succeed in ' + fname)
            print('model saved succeed in ' + fname)

    def accurcy(self, type):

        totle_loss = torch.Tensor([0])
        correct_num = 0
        inst_num = 0

        data_batchs = None
        if type == 'dev':
            data_batchs = self.dev_data_batchs
        elif type == 'test':
            data_batchs = self.test_data_batchs
        else:
            raise RuntimeError('type wrong!')

        for batch in data_batchs:

            self.model.eval()

            inst_num += len(batch[1])

            data = torch.LongTensor(batch[0])
            label = torch.LongTensor(batch[1])

            char_data = None
            if self.char:
                char_data = torch.LongTensor(batch[2])

            if self.opts.use_cuda:
                data = data.cuda()
                label = label.cuda()
                if self.char:
                    char_data = char_data.cuda()

            if self.char:
                pred = self.model(data, char_data)
            else:
                pred = self.model(data)

            loss = F.cross_entropy(pred, label)

            totle_loss += loss.data

            correct_num += (torch.max(pred, 1)[1].view(label.size()).data == label.data).sum()

        avg_loss = totle_loss / inst_num
        acc = float(correct_num) / inst_num * 100

        log = type + " acc: {:.2f}% loss: {:.6f}".format(acc, avg_loss.numpy()[0])
        self.print_log.print_log(log)
        print(log)

        return acc

    def get_time(self):
        # tm_year=2018, tm_mon=10, tm_mday=28, tm_hour=10, tm_min=32, tm_sec=14, tm_wday=6, tm_yday=301, tm_isdst=0
        cur_time = time.localtime(time.time())

        dic = dict()
        dic['year'] = cur_time.tm_year
        dic['month'] = cur_time.tm_mon
        dic['day'] = cur_time.tm_mday
        dic['hour'] = cur_time.tm_hour
        dic['min'] = cur_time.tm_min
        dic['sec'] = cur_time.tm_sec

        return dic