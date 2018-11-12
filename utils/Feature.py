# Version python3.6
# -*- coding: utf-8 -*-
# @Time    : 2017/12/26 下午12:16
# @Author  : zenRRan
# @Email   : zenrran@qq.com
# @File    : Feature.py
# @Software: PyCharm Community Edition


import torch
import torch.nn as nn


class Feature:

    def __init__(self):

        self.words = None
        self.chars = None

        self.ids = None
        self.char_ids = None

        self.length = 0

        self.label = None

        #conll
        self.heads = None
        self.root = None
        self.forest = None



