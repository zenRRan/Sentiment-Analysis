# Version python3.6
# -*- coding: utf-8 -*-
# @Time    : 2017/12/26 下午12:16
# @Author  : zenRRan
# @Email   : zenrran@qq.com
# @File    : Feature.py
# @Software: PyCharm Community Edition


import torch
import torch.nn as nn


class extractFeature:

    def __init__(self, sentances):
        self.sentances = sentances


a = [[1,2,3]]
b = [6]
a.append(b)
print(a)

