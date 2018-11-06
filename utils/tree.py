# Version python3.6
# -*- coding: utf-8 -*-
# @Time    : 2018/11/5 2:01 PM
# @Author  : zenRRan
# @Email   : zenrran@qq.com
# @File    : tree.py
# @Software: PyCharm Community Edition



class Tree(object):

    def __init__(self):
        self.parent = None
        self.child_num = 0
        self.child_list = list()

    def add_child(self, child):
        child.parent = self
        self.child_num += 1
        self.child_list.append(child)

    def size(self):
        if getattr(self, '_size'):
            return self._size
        count = 1
        for child in self.child_list:
            count += child.size()
        self._size = count
        return self._size

    def depth(self):
        if getattr(self, '_depth'):
            return self._depth
        count = 1
        for child in self.child_list:
            child_depth = child.depth
            if child_depth > count:
                count = child_depth
        self._depth = count
        return self._depth