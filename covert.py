import sys
import re

class Instance:
    def __init__(self):
        self.words = []
    def clean_str(self, string):
        """
        Tokenization/string cleaning for all datasets except for SST.abs    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
        """
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`|]", " ", string)
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
       
        return string.strip()
            
    def conll(self):
        for idx in range(len(self.words)):
            if idx == 0:
                head = 0
                relation = 'root'
            else:
                head = 1
                relation = 'det'
            line = str(idx+1) + '\t' + \
                   self.words[idx] + '\t' + \
                   '_' + '\t' + \
                   'NN' + '\t' + \
                   'NN' + '\t' + \
                   '_' + '\t' + \
                   str(head) + '\t' + \
                   relation + '\t' + \
                   '_' + '\t' + \
                   '_'
            print(line)
        print()

def s2c(file_path):
    inst = Instance()
    with open(file_path, encoding='utf8') as input_file:
        while True:
            line = input_file.readline()
            if not line:
                break
            line = inst.clean_str(line.strip())
            info = line.split(" ")[2:]
            info = ' '.join(info).strip().split()
            inst.words = info
            inst.conll()
s2c('data/MR/mr.train.txt')
