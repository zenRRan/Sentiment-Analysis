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


import argparse

class Parameters:
    def __init__(self):
        parser = argparse.ArgumentParser(description='Set parameters to drive this code!')

        #data
        parser.add_argument('-', type=int, default=1, help='')
        parser.add_argument('-', type=int, default=1, help='')
        parser.add_argument('-', type=int, default=1, help='')
        parser.add_argument('-', type=int, default=1, help='')
        parser.add_argument('-', type=int, default=1, help='')
        parser.add_argument('-', type=int, default=1, help='')
        parser.add_argument('-', type=int, default=1, help='')

        #train
        parser.add_argument('-seed', type=int, default=23, help='cpu seed! default 23. If you want set GPU seed, please use -gpu_seed!')
        parser.add_argument('-gpu_seed', type=int, default=23, help='GPU seed! default 23.')
        parser.add_argument('-seed', type=int, default=23, help='cpu seed! If you want set GPU seed, please use -gpu_seed!')
        parser.add_argument('-', type=int, default=1, help='')
        parser.add_argument('-', type=int, default=1, help='')
        parser.add_argument('-', type=int, default=1, help='')
        parser.add_argument('-', type=int, default=1, help='')
        parser.add_argument('-', type=int, default=1, help='')
        parser.add_argument('-', type=int, default=1, help='')
        #decode
        # parser.add_argument('-', type=int, default=1, help='')
        # parser.add_argument('-', type=int, default=1, help='')
        # parser.add_argument('-', type=int, default=1, help='')
        # parser.add_argument('-', type=int, default=1, help='')
        # parser.add_argument('-', type=int, default=1, help='')
        # parser.add_argument('-', type=int, default=1, help='')
        # parser.add_argument('-', type=int, default=1, help='')
        # parser.add_argument('-', type=int, default=1, help='')
        # parser.add_argument('-', type=int, default=1, help='')

