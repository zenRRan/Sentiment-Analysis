# Version python3.6
# -*- coding: utf-8 -*-
# @Time    : 2017/12/26 上午10:51
# @Author  : zenRRan
# @Email   : zenrran@qq.com
# @File    : Evaluate.py
# @Software: PyCharm Community Edition

class Eval:

    def __init__(self, pred_right_num=0., pred_num=0., gold_num=0.):
        self.pred_right_num = float(pred_right_num)
        self.pred_num = float(pred_num)
        self.gold_num = float(gold_num)

        if self.pred_num != 0:
            self.P = self.pred_right_num/self.pred_num
        else:
            self.P = 0
        if self.gold_num != 0:
            self.R = self.pred_right_num/self.gold_num
        else:
            self.R = 0
        if self.P + self.R != 0:
            self.F1 = 2*self.P*self.R/(self.P+self.R)
        else:
            self.F1 = 0
        self.P = "%.4f" % self.P
        self.R = "%.4f" % self.R
        self.F1 = "%.4f" % self.F1
        self.P_R_F1 = [float(self.P), float(self.R), float(self.F1)]

# evalList = [[0, 0, 0], [0, 0, 0]]
# label_list = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
# pred_list = [1, 0, 1, 1, 1, 1, 1, 0, 0, 0]
# pred_right_num_idx = 0
# pred_num_idx = 1
# gold_num_idx = 2
# for i in range(len(evalList)):
#     for j in range(len(label_list)):
#         if label_list[j] == i:
#             evalList[i][gold_num_idx] += 1
#             if label_list[j] == pred_list[j]:
#                 evalList[i][pred_right_num_idx] += 1
#         if pred_list[j] == i:
#             evalList[i][pred_num_idx] += 1
# print(Eval(evalList[0][0], evalList[0][1], evalList[0][2]).P_R_F1)
# print(Eval(evalList[1][0], evalList[1][1], evalList[1][2]).P_R_F1)
