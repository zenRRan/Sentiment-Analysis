# Version python3.6
# -*- coding: utf-8 -*-
# @Time    : 2017/12/26 上午10:50
# @Author  : zenRRan
# @Email   : zenrran@qq.com
# @File    : Reader.py
# @Software: PyCharm Community Edition



import sys
import re
import argparse
from langconv import *

class reader:

    def __init__(self, filename, needFresh=True, language='chn'):
        self.post = []
        self.response = []
        self.label = []
        self.fileText = []
        self.maxLen = 0
        print("Reading " + filename)
        file = open(filename, "r", encoding='utf-8')
        self.fileText = file.readlines()
        file.close()

        if needFresh:
            fresh = refresh_chn_data(self.fileText)
            fresh = refresh_eng_data(fresh.getText())
            self.fileText = fresh.getText()

        flag = True
        for line in self.fileText:
            line = line.strip().split()
            if len(line) == 0:
                continue
            if flag:
                self.post.append(line)
                flag = False
            else:
                self.label.append(line[0])
                self.response.append(line[2:])
                flag = True
        if len(self.post) != len(self.response):
            print('len(self.post) != len(self.response) please check !')
    def getData(self):
        return self.post, self.response, self.label

    def getWholeText(self):
        # random.shuffle(self.fileText)
        return self.fileText

# data = reader('/Users/zhenranran/Desktop/zenRRan.github.com/Stance Detection/CNN/Data/dev.sd')


"""
refresh English data
"""
class refresh_eng_data:
    def __init__(self, stringlist):
        self.newlist = []
        for line in stringlist:
            self.newlist.append(self.freshData(line))
    def getText(self):
        return self.newlist
    def freshData(self, string):
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()


# string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
"""
 全角数字转半角
 全角英文字母转半角
 全角中文标点转半角
 转小写(可选)
"""
class refresh_chn_data:
    def normChar(self, istring):
        rstring = ""
        for uchar in istring:
            inside_code = ord(uchar)
            if inside_code ==  0x3000:
                inside_code = 0x0020
            else:
                inside_code -= 0xfee0
            if inside_code < 0x0020 or inside_code > 0x7e:
                rstring += uchar
            else:
                rstring += chr(inside_code)
        rstring = re.sub(r'\s+', ' ', rstring).lower()
        return rstring

    def __init__(self, stringlist):
        parser = argparse.ArgumentParser()
        parser.add_argument('-l', '--toLower', action='store_true')
        option = parser.parse_args()
        tolower = option.toLower

        self.newlist = []
        for line in stringlist:
            line = self.normChar(line.strip())
            line = Converter('zh-hans').convert(line)
            self.newlist.append(line)

    def getText(self):
        return self.newlist

# data = ['。。。...，，，,,,輸入簡體字,點下面繁體字按鈕進行在線轉換']


# 转换繁体到简体
# line = Converter('zh-hans').convert(data[0])
# line = line
# print(line)
# path = 'D:/语料/立场检测/中文/train.sd'
# textlist = reader(path, language='chn').getWholeText()
# with open(path, encoding='utf-8') as f:
#     textlist = f.readlines()[:10]
#     for i in range(len(textlist)):
#         print("".join(textlist[i]))
#         print(textlist[i])
