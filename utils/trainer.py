# Version python3.6
# -*- coding: utf-8 -*-
# @Time    : 2018/10/16 10:49 AM
# @Author  : zenRRan
# @Email   : zenrran@qq.com
# @File    : trainer.py
# @Software: PyCharm Community Edition

from utils.build_batch import Build_Batch
from models.CNN import CNN
from models.GRU import GRU
from models.LSTM import LSTM
from models.Multi_layer_CNN import Multi_Layer_CNN
from models.multi_channel_CNN import Multi_Channel_CNN
from models.Char_CNN import Char_CNN
from models.LSTM_CNN import LSTM_CNN
from models.Pooling import Pooling
from models.Tree_LSTM import BatchChildSumTreeLSTM
from models.TreeLSTM import ChildSumTreeLSTM
from models.biTreeLSTM import biChildSumTreeLSTM
from models.TreeLSTM_rel import ChildSumTreeLSTM_rel
from models.biTreeLSTM_rel import biChildSumTreeLSTM_rel
from models.LSTM_TreeLSTM_rels import LSTM_ChildSumTreeLSTM_rel
from models.CNN_TreeLSTM import CNN_TreeLSTM
from models.LSTM_TreeLSTM import LSTM_TreeLSTM
from utils.Common import padding_key

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils as utils
from torch.autograd import Variable, gradcheck

from utils.log import Log

import os
import time
import random


class Trainer:
    def __init__(self, train_dev_test, opts, vocab, label_vocab, rel_vocab):
        self.train_features_list = train_dev_test[0]
        self.dev_features_list = train_dev_test[1]
        self.test_features_list = train_dev_test[2]
        self.opts = opts
        self.vocab = vocab[0]
        self.char_vocab = vocab[1]
        self.label_vocab = label_vocab
        self.rels_vocab = rel_vocab
        self.epoch = opts.epoch
        self.shuffle = opts.shuffle
        self.model = None

        self.best_dev = 0
        self.best_dev_test = 0
        self.best_dev_epoch = 0

        self.char = False
        self.tree = False

        self.lr = self.opts.lr

        self.build_batch()
        self.init_model()
        self.init_optim()

        self.print_log = Log(opts)

        #save model switch
        self.save_model_switch = False

        random.seed(opts.seed)
        torch.manual_seed(opts.seed)

        if self.opts.use_cuda:
            # torch.backends.cudnn.enabled = True
            torch.backends.cudnn.deterministic = True
            torch.cuda.set_device(self.opts.gpu_device)
            torch.cuda.manual_seed(self.opts.seed)
            log = 'use CUDA!'
            self.print_log.print_log(log)
            print(log)

    def build_batch(self):
        '''
        build train dev test batches
        '''
        padding_id = self.vocab.from_string(padding_key)
        char_padding_id = self.char_vocab.from_string(padding_key)
        rel_padding_id = None
        if self.rels_vocab is not None:
            rel_padding_id = self.rels_vocab.from_string(padding_key)
        self.train_build_batch = Build_Batch(features=self.train_features_list, batch_size=self.opts.train_batch_size,
                                             opts=self.opts, pad_idx=padding_id, char_padding_id=char_padding_id,
                                             rel_padding_id=rel_padding_id)
        self.dev_build_batch = Build_Batch(features=self.dev_features_list, batch_size=self.opts.dev_batch_size,
                                           opts=self.opts, pad_idx=padding_id, char_padding_id=char_padding_id,
                                           rel_padding_id=rel_padding_id)
        self.test_build_batch = Build_Batch(features=self.test_features_list, batch_size=self.opts.test_batch_size,
                                            opts=self.opts, pad_idx=padding_id, char_padding_id=char_padding_id,
                                            rel_padding_id=rel_padding_id)

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
        elif self.opts.model == 'gru':
            self.model = GRU(opts=self.opts, vocab=self.vocab, label_vocab=self.label_vocab)
        elif self.opts.model == 'lstm_cnn':
            self.model = LSTM_CNN(opts=self.opts, vocab=self.vocab, label_vocab=self.label_vocab)
        elif self.opts.model == 'treelstm':
            self.tree = True
            self.model = ChildSumTreeLSTM(opts=self.opts, vocab=self.vocab, label_vocab=self.label_vocab,
                                          rel_vocab=self.rels_vocab)
        elif self.opts.model == 'bitreelstm':
            self.tree = True
            self.model = biChildSumTreeLSTM(opts=self.opts, vocab=self.vocab, label_vocab=self.label_vocab,
                                          rel_vocab=self.rels_vocab)
        elif self.opts.model == 'treelstm_rel':
            self.tree = True
            self.model = ChildSumTreeLSTM_rel(opts=self.opts, vocab=self.vocab, label_vocab=self.label_vocab,
                                                   rel_vocab=self.rels_vocab)
        elif self.opts.model == 'bitreelstm_rel':
            self.tree = True
            self.model = biChildSumTreeLSTM_rel(opts=self.opts, vocab=self.vocab, label_vocab=self.label_vocab,
                                                   rel_vocab=self.rels_vocab)
        elif self.opts.model == 'cnn_treelstm':
            self.tree = True
            self.model = CNN_TreeLSTM(opts=self.opts, vocab=self.vocab, label_vocab=self.label_vocab)
        elif self.opts.model == 'lstm_treelstm':
            self.tree = True
            self.model = LSTM_TreeLSTM(opts=self.opts, vocab=self.vocab, label_vocab=self.label_vocab)
        elif self.opts.model == 'lstm_treelstm_rel':
            self.tree = True
            self.model = LSTM_ChildSumTreeLSTM_rel(opts=self.opts, vocab=self.vocab, label_vocab=self.label_vocab,
                                                   rel_vocab=self.rels_vocab)
        else:
            raise RuntimeError('please choose your model first!')

        print(self.model)

        if self.opts.use_cuda:
            self.model = self.model.cuda()

    def init_optim(self):
        'sgd, adam'
        if self.opts.optim == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.opts.lr, weight_decay=self.opts.weight_decay, momentum=self.opts.momentum)
        elif self.opts.optim == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.opts.lr, weight_decay=self.opts.weight_decay)

    def train(self):

        early_stop_count = 1
        lr_decay_count = 1

        for epoch in range(self.epoch):
            totle_loss = torch.Tensor([0])
            correct_num = 0
            step = 1
            inst_num = 0
            totle_step = len(self.train_data_batchs)
            if self.shuffle:
                random.shuffle(self.train_data_batchs)
                log = 'data has shuffled!'
                print(log)
                self.print_log.print_log(log)
            for batch in self.train_data_batchs:


                self.model.train()
                self.optimizer.zero_grad()

                inst_num += len(batch[1])

                if self.tree:
                    sents = Variable(torch.LongTensor(batch[0]), requires_grad=False)
                    # tree = batch[3][0]
                    label = Variable(torch.LongTensor(batch[1]), requires_grad=False)
                    heads = batch[4]
                    # bfs_tensor = Variable(torch.LongTensor(batch[4]), requires_grad=False)
                    # children_batch_list = Variable(torch.LongTensor(batch[5]), requires_grad=False)
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
                    if self.char:
                        for char_list in batch[2]:
                            char_data.append(Variable(torch.LongTensor(char_list)))
                    if self.opts.use_cuda:
                        sents = sents.cuda()
                        label = label.cuda()
                        new_char_data = []
                        for data in char_data:
                            new_char_data.append(data.cuda())
                        char_data = new_char_data
                    if self.char:
                        pred = self.model(sents, char_data)
                    else:
                        pred = self.model(sents)

                loss = F.cross_entropy(pred, label)

                loss.backward()

                # print("gradCheck :", gradcheck(self.model, (self.model.embeddings,)))

                if self.opts.init_clip_max_norm is not None:
                    utils.clip_grad_norm_(self.model.parameters(), max_norm=self.opts.init_clip_max_norm)

                self.optimizer.step()

                loss = loss.cpu()
                totle_loss += loss.data

                correct_num += (torch.max(pred, 1)[1].view(label.size()).data == label.data).sum()
                if step % self.opts.print_every == 0:
                    avg_loss = totle_loss / inst_num
                    acc = float(correct_num) / inst_num * 100
                    time_dic = self.get_time()
                    time_str = "[{}-{:0>2d}-{:0>2d} {:0>2d}:{:0>2d}:{:0>2d}]".format(time_dic['year'], time_dic['month'], time_dic['day'], \
                                                          time_dic['hour'], time_dic['min'], time_dic['sec'])
                    log = time_str + " Epoch {} step [{}|{}] lr={:.8f} acc: {:.2f}% loss: {:.6f}".format(epoch, step, totle_step, self.lr, acc, avg_loss.numpy()[0])
                    self.print_log.print_log(log)
                    print(log)
                    totle_loss = torch.Tensor([0])
                    inst_num = 0
                    correct_num = 0

                step += 1

            dev_score = self.accurcy(type='dev')
            test_score = self.accurcy(type='test')
            if dev_score > self.best_dev and test_score > self.best_dev_test:
                early_stop_count = 0
                lr_decay_count = 0
                self.best_dev = dev_score
                self.best_dev_epoch = epoch
                self.best_dev_test = test_score
                log = "Update! best test acc: {:.2f}%".format(self.best_dev_test)
                print(log)
                self.save_model(epoch)
            elif dev_score > self.best_dev:
                self.best_dev = dev_score
                self.best_dev_epoch = epoch
                log = "not improved, best test acc: {:.2f}%, in epoch {}".format(self.best_dev_test,
                                                                                 self.best_dev_epoch)
            else:
                early_stop_count += 1
                lr_decay_count += 1
                log = "not improved, best test acc: {:.2f}%, in epoch {}".format(self.best_dev_test, self.best_dev_epoch)
                print(log)

            self.print_log.print_log(log)

            if early_stop_count == self.opts.early_stop:
                log = "{} epoch have not improved, so early stop the train!".format(early_stop_count)
                self.print_log.print_log(log)
                print(log)
                return

            if lr_decay_count == self.opts.lr_decay_every:
                lr_decay_count = 0
                self.adjust_learning_rate(self.optimizer, self.opts.lr_decay_rate)
                log = "{} epoch have not improved, so adjust lr to {}".format(early_stop_count, self.lr)
                self.print_log.print_log(log)
                print(log)

    def save_model(self, cur_epoch):
        if not os.path.isdir(self.opts.save_model_dir):
            os.mkdir(self.opts.save_model_dir)
        if self.opts.save_model_start_from <= cur_epoch:
            self.save_model_switch = True
        # if self.save_model_switch and (cur_epoch - self.opts.save_model_start_from) % self.opts.save_model_every == 0:
        if self.save_model_switch:
            time_dic = self.get_time()
            time_str = "{}-{:0>2d}-{:0>2d}-{:0>2d}-{:0>2d}-{:0>2d}-".format(time_dic['year'], time_dic['month'], time_dic['day'], \
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

            if self.tree:
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

                # print(data)

                char_data = []
                if self.char:
                    for char_list in batch[2]:
                        char_data.append(Variable(torch.LongTensor(char_list)))
                        # print(type(char_data[0]), char_data[0].size())
                # print(char_data)
                if self.opts.use_cuda:
                    sents = sents.cuda()
                    label = label.cuda()
                    new_char_data = []
                    for data in char_data:
                        new_char_data.append(data.cuda())
                    char_data = new_char_data
                    # print(type(char_data[0]))
                if self.char:
                    pred = self.model(sents, char_data)
                else:
                    pred = self.model(sents)

            loss = F.cross_entropy(pred, label)

            loss = loss.cpu()
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

    def adjust_learning_rate(self, optim, lr_decay_rate):
        for param_group in optim.param_groups:
            param_group['lr'] = param_group['lr'] * (1 - lr_decay_rate)
            self.lr = param_group['lr']




