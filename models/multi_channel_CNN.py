# Version python3.6
# -*- coding: utf-8 -*-
# @Time    : 2018/10/25 8:38 PM
# @Author  : zenRRan
# @Email   : zenrran@qq.com
# @File    : multi_channel_CNN.py
# @Software: PyCharm Community Edition


import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.Embedding as Embedding

class Multi_Channel_CNN(nn.Module):
    def __init__(self, opts, vocab, label_vocab):
        super(Multi_Channel_CNN, self).__init__()

        self.embed_dim = opts.embed_size
        self.word_num = vocab.m_size
        self.pre_embed_path = opts.pre_embed_path
        self.string2id = vocab.string2id
        self.embed_uniform_init = opts.embed_uniform_init
        self.stride = opts.stride
        self.kernel_size = opts.kernel_size
        self.kernel_num = opts.kernel_num
        self.label_num = label_vocab.m_size
        self.embed_dropout = opts.embed_dropout
        self.fc_dropout = opts.fc_dropout

        self.embeddings = nn.Embedding(self.word_num, self.embed_dim)
        self.embeddings_static = nn.Embedding(self.word_num, self.embed_dim)

        if opts.pre_embed_path != '':
            embedding = Embedding.load_predtrained_emb_zero(self.pre_embed_path, self.string2id)
            self.embeddings_static.weight.data.copy_(embedding)
        else:
            nn.init.uniform_(self.embeddings_static.weight.data, -self.embed_uniform_init, self.embed_uniform_init)

        nn.init.uniform_(self.embeddings.weight.data, -self.embed_uniform_init, self.embed_uniform_init)

        # 2 convs
        self.convs = nn.ModuleList(
            [nn.Conv2d(2, self.embed_dim, (K, self.embed_dim), stride=self.stride, padding=(K // 2, 0)) for K in self.kernel_size])

        in_fea = len(self.kernel_size)*self.kernel_num
        self.linear1 = nn.Linear(in_fea, in_fea // 2)
        self.linear2 = nn.Linear(in_fea // 2, self.label_num)
        self.embed_dropout = nn.Dropout(self.embed_dropout)
        self.fc_dropout = nn.Dropout(self.fc_dropout)

    def forward(self, input):
        # print(self.convs)
        static_embed = self.embeddings_static(input)  # torch.Size([64, 39, 100])
        embed = self.embeddings(input)  # torch.Size([64, 39, 100])
        # print(static_embed.size())
        # print(embed.size())

        x = torch.stack([static_embed, embed], 1)  # torch.Size([64, 2, 39, 100])
        # print(x.size())

        out = self.embed_dropout(x)
        # print('1:', out.size())
        l = []
        for conv in self.convs:
            l.append(F.relu(conv(out)).squeeze(3))  # torch.Size([64, 100, 39])
        # print('2:', l[0].size())
        out = l
        l = []
        for i in out:
            l.append(F.max_pool1d(i, kernel_size=i.size(2)).squeeze(2))  # torch.Size([64, 100])
        out = l
        # print('3:', out[0].size())
        out = torch.cat(l, 1)  # torch.Size([64, 300])
        # print('4:', out.size())

        out = self.fc_dropout(out)

        out = self.linear1(out)
        out = self.linear2(F.relu(out))

        return out

