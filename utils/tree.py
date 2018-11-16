# Version python3.6
# -*- coding: utf-8 -*-
# @Time    : 2018/11/5 2:01 PM
# @Author  : zenRRan
# @Email   : zenrran@qq.com
# @File    : tree.py
# @Software: PyCharm Community Edition



class Tree(object):

    def __init__(self, index):
        self.parent = None
        self.c = None
        self.h = None
        self.label = None
        self.index = index
        self.child_num = 0
        self.children_list = list()
        self.children_index_list = list()
        self.left_children = list()
        self.right_children = list()

    # def add_child(self, child):
    #     child.parent = self
    #     self.child_num += 1
    #     self.left_children.append(child)

    def add_left_child(self, child):
        child.parent = self
        self.child_num += 1
        self.left_children.append(child)
        self.children_list.append(child)
        self.children_index_list.append(child.index)

    def add_right_child(self, child):
        child.parent = self
        self.child_num += 1
        self.right_children.append(child)
        self.children_list.append(child)
        self.children_index_list.append(child.index)

    def size(self):
        if hasattr(self, '_size'):
            return self._size
        count = 1
        for child in self.left_children:
            cur_count = child.size()
            if cur_count > count:
                count = cur_count
        for child in self.right_children:
            cur_count = child.size()
            if cur_count > count:
                count = cur_count
        self._size = count
        return self._size

    def depth(self):
        if hasattr(self, '_depth'):
            return self._depth
        depth = 0
        if self.child_num > 0:
            for child in self.left_children:
                cur_depth = child.depth()
                if cur_depth > depth:
                    depth = cur_depth
            for child in self.right_children:
                cur_depth = child.depth()
                if cur_depth > depth:
                    depth = cur_depth
            depth += 1
        self._depth = depth
        return self._depth

def createTree(heads):
    '''
    :param heads: eg.[2, 0, 2]
    :return: tree's head
    '''

    forest = []
    for idx in range(len(heads)):
        forest.append(Tree(idx))

    root = None
    for idx, head in enumerate(heads):
        if head == -1:
            root = forest[idx]
            continue
        if head > idx:
            forest[head].add_left_child(forest[idx])
        if head < idx:
            forest[head].add_right_child(forest[idx])
    if root is None:
        RuntimeError('not have root! please check data!')
    return root, forest

# heads = [1, -1, 1, 2]
# root, _ = createTree(heads)
# print(root.depth())
