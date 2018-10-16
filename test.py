#!/usr/bin/env python
# encoding: utf-8

"""
@version: python3.6
@author: 'zenRRan'
@license: Apache Licence 
@contact: zenrran@qq.com
@software: PyCharm
@file: test.py
@time: 2018/10/15 8:14
"""

import torch


def add_pad(data_list):
    '''
    :param data_list: [[x x x], [x x x x],...]
    :return: [[x x x o o], [x x x x o],...]
    '''
    max_len = 0
    for data in data_list:
        max_len = max(max_len, len(data))
    for data in data_list:
        data.extend([0] * (max_len - len(data)))


data_list = [[1,2,3], [2,3,4,5]]
add_pad(data_list)
print(data_list)