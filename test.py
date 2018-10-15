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

def test_args_kwargs(one, *args, **kwargs):
    print(one)
    print(args)
    print(kwargs)

print(test_args_kwargs('0', '1', name='xxx'))