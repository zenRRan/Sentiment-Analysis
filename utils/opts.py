#!/usr/bin/env python
# encoding: utf-8

"""
@version: python3.6
@author: 'zenRRan'
@license: Apache Licence 
@contact: zenrran@qq.com
@software: PyCharm
@file: parameters.py
@time: 2018/10/7 15:50
"""


def preprocesser_opts(parser):
    #data
    parser.add_argument('-raw_train_path', type=str, default='', help="raw train file's path")
    parser.add_argument('-raw_dev_path', type=str, default='', help="raw dev file's path")
    parser.add_argument('-raw_test_path', type=str, default='', help="raw test file's path")
    parser.add_argument('-freq_vocab', type=int, default=0, help='what less than that value will be deleted')
    parser.add_argument('-vcb_size', type=int, default=30000, help='what high than that value will be deleted')
    parser.add_argument('-save_dir', type=str, default='processed_data',
                        help='train.sst, dev.sst, test.sst vocab.txt those who are processed will be saved here')
    parser.add_argument('-shuffle', type=bool, default=False, help='shuffle data')
    # parser.add_argument('-', type=int, default=1, help='')

    #seed
    parser.add_argument('-seed', type=int, default=23,
                        help='cpu seed! default 23. If you want set GPU seed, please use -gpu_seed!')

    #gpu
    parser.add_argument('-gpu_device', type=int, default=0,
                        help='decide which gpu device will be selected, default device 0')
    parser.add_argument('-use_cuda', type=bool, default=False, help='if use cuda, default False')
    # parser.add_argument('-', type=int, default=1, help='')

    return parser

def trainer_opts(parser):
    #seed
    parser.add_argument('-seed', type=int, default=23,
                        help='cpu seed! default 23. If you want set GPU seed, please use -gpu_seed!')
    parser.add_argument('-shuffle', action='store_true', help='shuffle the data')
    #gpu
    parser.add_argument('-gpu_seed', type=int, default=23, help='GPU seed! default 23.')
    parser.add_argument('-gpu_device', type=int, default=0,
                        help='decide which gpu device will be selected, default device 0')
    parser.add_argument('-use_cuda', action='store_true', help='if use cuda')

    #embedding
    parser.add_argument('-embed_size', type=int, default=100, help='embedding size, default 100, \
                        recommand 100, 200, 300')
    parser.add_argument('-char_embed_size', type=int, default=50, help='char embedding size, default 50')
    parser.add_argument('-pre_embed_path', type=str, default='', help='pretrained embedding path')
    parser.add_argument('-embed_uniform_init', type=float, default=0,
                        help='nn.init.uniform(-embed_uniform_init, embed_uniform_init), default=0')

    # -pre_embed_path

    #data
    parser.add_argument('-data_dir', type=str, default='processed_data', help='code will read train [dev test vocab].sst\
                        file here ')

    #learning rate
    parser.add_argument('-lr', type=float, default=0.001, help='learning rate, recommand: sgd:0.1, adam:0.001')
    parser.add_argument('-lr_decay_rate', type=float, default=0.3, help='lr = lr * (1 - lr_decay_rate)')
    parser.add_argument('-lr_decay_every', type=int, default=8, help='if lr have not improved in lr_decay_every epoch, \
                        will do lr_decay')

    #optim
    parser.add_argument('-optim', type=str, default='adam', help='sgd, adam')
    parser.add_argument('-momentum', type=float, default=0.9, help='used in sgd')
    parser.add_argument('-weight_decay', type=float, default=1e-8, help='weight decay')
    parser.add_argument('-early_stop', type=int, default=9999999, help='if best value not change in [early_stop], \
                        the code will be stoped')
    parser.add_argument('-init_clip_max_norm', type=int, default=10, help='if the sum of the grad if high than this, \
                        would be cut')

    '''
    parameters
    '''
    #batch
    parser.add_argument('-train_batch_size', type=int, default=64, help='batch size')
    parser.add_argument('-dev_batch_size', type=int, default=64, help='dev size')
    parser.add_argument('-test_batch_size', type=int, default=64, help='test size')

    #batch_menu_choose
    parser.add_argument('-train_batch_type', type=str, default='normal', help='You can choose [normal, same]')
    parser.add_argument('-dev_batch_type', type=str, default='normal', help='You can choose [normal, same]')
    parser.add_argument('-test_batch_type', type=str, default='normal', help='You can choose [normal, same]')

    #shuffer
    parser.add_argument('-shuffer', action='store_true',
                        help='if your batch want to be shuffered, you should add [-shuffer] in your command line')

    #sort
    parser.add_argument('-sort', action='store_true',
                        help='if your batch want to be sorted, you should add [-sort] in your command line')

    parser.add_argument('-epoch', type=int, default=100, help='epochs')
    parser.add_argument('-print_every', type=int, default=10, help='every that times, print log')
    # parser.add_argument('-dev_every_step')

    #model
    parser.add_argument('-model', type=str, default='', help='select one of [pooling, rnn, lstm, gru, cnn, multi_layer_cnn,\
                        multi_channel_cnn, char_cnn, lstm_cnn]')
    parser.add_argument('-save_model_dir', type=str, default='save_models', help='save model dir')
    parser.add_argument('-save_model_every', type=int, default=1, help='save model every this epoch')
    parser.add_argument('-save_model_start_from', type=int, default=0, help='save model start from this epoch')

    #dropout
    parser.add_argument('-embed_dropout', type=float, default=0, help='embedding dropout')
    parser.add_argument('-fc_dropout', type=float, default=0, help='full connection dropout')

    #cnn
    parser.add_argument('-kernel_size', type=list, default=[1, 2, 3, 4], help="cnn's kernel size, default [3,5,7]")
    parser.add_argument('-kernel_num', type=int, default=100, help="cnn's kernel num, default 100")
    parser.add_argument('-stride', type=int, default=1, help="cnn stride, default 1")

    #rnn
    parser.add_argument('-hidden_size', type=int, default=128, help="rnn's hidden size, default 128")
    parser.add_argument('-hidden_num', type=int, default=1, help="rnn's hidden num, default 1")
    parser.add_argument('-hidden_dropout', type=float, default=0, help='rnn hidden dropout')
    parser.add_argument('-bidirectional', action='store_true', help='selected you will train birnn')

    # parser.add_argument('-', type=int, default=1, help='')

    #log
    parser.add_argument('-log_dir', type=str, default='log', help='log dir default [log]')
    parser.add_argument('-log_fname', type=str, default='', help='log file name')

    return parser

def decoder_opts(parser):
    #seed
    parser.add_argument('-seed', type=int, default=23,
                        help='cpu seed! default 23. If you want set GPU seed, please use -gpu_seed!')

    #gpu
    parser.add_argument('-gpu_seed', type=int, default=23, help='GPU seed! default 23.')
    parser.add_argument('-gpu_device', type=int, default=0,
                        help='decide which gpu device will be selected, default device 0')
    parser.add_argument('-use_cuda', type=bool, default=False, help='if use cuda, default False')

    #decoder file
    parser.add_argument('-file', type=str, default='', help='decoder this file')

    #model
    parser.add_argument('-model_path', type=str, default='', help='select your want to use model path')
    # parser.add_argument('-', type=int, default=1, help='')
    # parser.add_argument('-', type=int, default=1, help='')
    # parser.add_argument('-', type=int, default=1, help='')
    # parser.add_argument('-', type=int, default=1, help='')
    # parser.add_argument('-', type=int, default=1, help='')
    # parser.add_argument('-', type=int, default=1, help='')
    # parser.add_argument('-', type=int, default=1, help='')
    # parser.add_argument('-', type=int, default=1, help='')
    # parser.add_argument('-', type=int, default=1, help='')

    return parser
