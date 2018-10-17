#!/usr/bin/env python
# encoding: utf-8

"""
@version: python3.6
@author: 'zenRRan'
@license: Apache Licence 
@contact: zenrran@qq.com
@software: PyCharm
@file: LSTM.py
@time: 2018/10/7 15:51
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.Embedding as Embedding

class LSTM(nn.Module):
    def __init__(self, opts):
        super(LSTM, self).__init__()
        self.embedingTopic = nn.Embedding(opts.topicSize, opts.EmbedSize)
        self.embedingText = nn.Embedding(opts.wordNum, opts.EmbedSize)
        if opts.using_pred_emb:
            load_emb_text = Embedding.load_predtrained_emb_avg(opts.pred_embedding_50_path,
                                                               opts.wordAlpha.string2id)
            load_emb_topic = Embedding.load_predtrained_emb_avg(opts.pred_embedding_50_path,
                                                                opts.wordAlpha.string2id)
            self.embedingTopic.weight.data.copy_(load_emb_topic)
            self.embedingText.weight.data.copy_(load_emb_text)

        self.biLSTM = nn.LSTM(
            opts.EmbedSize,
            opts.hiddenSize,
            dropout=opts.dropout,
            num_layers=opts.hiddenNum,
            batch_first=True,
            bidirectional=True
        )
        self.linear1 = nn.Linear(opts.hiddenSize * 4, opts.hiddenSize // 2)
        self.linear2 = nn.Linear(opts.hiddenSize // 2, opts.labelSize)

    def forward(self, topic, text):
        topic = self.embedingTopic(topic)
        text = self.embedingText(text)

        topic, _ = self.biLSTM(topic)   #[1, 1, 200]
        text,  _ = self.biLSTM(text)    #[1, 17, 200]


        topic = torch.transpose(topic, 1, 2)
        text = torch.transpose(text, 1, 2)

        topic = F.tanh(topic)
        text = F.tanh(text)

        topic = F.max_pool1d(topic, topic.size(2))  #[1, 200, 1]
        text = F.max_pool1d(text, text.size(2))     #[1, 200, 1]

        topic_text = torch.cat([topic, text], 1)    #[1, 400, 1]

        topic_text = topic_text.squeeze(2)          #[1, 400]

        output = self.linear1(topic_text)
        # output = F.tanh(output)
        output = F.relu(output)
        output = self.linear2(output)

        return output






