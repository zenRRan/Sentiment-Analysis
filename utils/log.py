# Version python3.6
# -*- coding: utf-8 -*-
# @Time    : 2018/10/18 3:09 PM
# @Author  : zenRRan
# @Email   : zenrran@qq.com
# @File    : log.py
# @Software: PyCharm Community Edition


import os

class Log:
    def __init__(self, opts):
        if opts.log_fname == '':
            raise RuntimeError('-log_fname must be given')
        self.path = opts.log_dir + '/' + opts.log_fname
        if not os.path.isdir(opts.log_dir):
            os.mkdir(opts.log_dir)
        self.opts = opts
        self.print_opts()

    def print_opts(self):

        with open(self.path, 'w', encoding='utf8') as f:

            f.write('data_dir=' + str(self.opts.data_dir) + '\n')
            f.write('\n')

            f.write('----batch----\n')
            f.write('train_batch_size=' + str(self.opts.train_batch_size) + '\n')
            f.write('train_batch_type=' + str(self.opts.train_batch_type) + '\n')
            f.write('dev_batch_size=' + str(self.opts.dev_batch_size) + '\n')
            f.write('dev_batch_type=' + str(self.opts.dev_batch_type) + '\n')
            f.write('test_batch_size=' + str(self.opts.test_batch_size) + '\n')
            f.write('test_batch_type=' + str(self.opts.test_batch_type) + '\n')
            f.write('shuffer=' + str(self.opts.shuffer) + '\n')
            f.write('sort=' + str(self.opts.sort) + '\n')
            f.write('\n')

            f.write('----embedding----\n')
            f.write('embed_size=' + str(self.opts.embed_size) + '\n')
            f.write('char_embed_size=' + str(self.opts.char_embed_size) + '\n')
            f.write('embed_uniform_init=' + str(self.opts.embed_uniform_init) + '\n')
            # f.write('embed_dropout=' + str(self.opts.embed_dropout) + '\n')
            f.write('pre_embed_path=' + str(self.opts.pre_embed_path) + '\n')
            f.write('\n')

            f.write('----model----\n')
            f.write('model=' + str(self.opts.model) + '\n')
            f.write('save_model_dir=' + str(self.opts.save_model_dir) + '\n')
            f.write('save_model_every=' + str(self.opts.save_model_every) + '\n')
            f.write('save_model_start_from=' + str(self.opts.save_model_start_from) + '\n')

            if 'cnn' in self.opts.model:
                f.write('kernel_num=' + str(self.opts.kernel_num) + '\n')
                f.write('kernel_size=' + str(self.opts.kernel_size) + '\n')
                f.write('stride=' + str(self.opts.stride) + '\n')

            elif 'lstm' in self.opts.model or 'gru' in self.opts.model:
                f.write('hidden_num=' + str(self.opts.hidden_num) + '\n')
                f.write('hidden_size=' + str(self.opts.hidden_size) + '\n')
                f.write('bidirectional=' + str(self.opts.bidirectional) + '\n')
            f.write('dropout=' + str(self.opts.dropout) + '\n')
            f.write('\n')

            f.write('----optimizer----\n')
            f.write('lr=' + str(self.opts.lr) + '\n')
            f.write('optim=' + str(self.opts.optim) + '\n')
            f.write('weight_decay=' + str(self.opts.weight_decay) + '\n')
            f.write('momentum=' + str(self.opts.momentum) + '\n')
            f.write('early_stop=' + str(self.opts.early_stop) + '\n')
            f.write('\n')

            f.write('----train----\n')
            f.write('epoch=' + str(self.opts.epoch) + '\n')
            f.write('print_every=' + str(self.opts.print_every) + '\n')
            f.write('\n')



            f.write('----GPU----\n')
            f.write('gpu_device=' + str(self.opts.gpu_device) + '\n')
            f.write('use_cuda=' + str(self.opts.use_cuda) + '\n')
            f.write('\n')

            f.write('----seed----\n')
            f.write('gpu_seed=' + str(self.opts.gpu_seed) + '\n')
            f.write('seed=' + str(self.opts.seed) + '\n')
            f.write('\n')

            f.write('----log----\n')
            f.write('log_dir=' + str(self.opts.log_dir) + '\n')
            f.write('log_fname=' + str(self.opts.log_fname) + '\n')
            f.write('\n')


    def print_log(self, text):
        with open(self.path, 'a', encoding='utf8') as f:
            f.write(text + '\n')

