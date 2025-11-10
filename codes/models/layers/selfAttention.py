"""
@File    : selfAttention.py
@Author  : wones
@Time    : 2022/10/28 20:59
@Todo    :定义selfAttention层
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self,hidden_dim,device = 'cpu'):
        super(SelfAttention, self).__init__()
        self.QLinear = nn.Linear(hidden_dim,hidden_dim,bias = False)
        self.KLinear = nn.Linear(hidden_dim,hidden_dim,bias = False)
        self.VLinear = nn.Linear(hidden_dim,hidden_dim,bias= False)
        self.device = device
    def forward(self,inputs,lens):
        size = inputs.size()
        Q = self.QLinear(inputs)
        K = self.KLinear(inputs).permute(0,2,1) #进行转置
        V = self.VLinear(inputs)

        #计算生成mask矩阵
        max_len = max(lens)
        sentence_lengths = torch.tensor(lens)
        mask = torch.arange(sentence_lengths.max().item())[None, :] < sentence_lengths[:, None]
        mask = mask.unsqueeze(dim=1)  # [batch_size, 1, max_len]
        mask = mask.expand(size[0], max_len, max_len)
        padding_num = torch.ones_like(mask)
        padding_num = -2 ** 31 * padding_num.float()
        alpha = torch.matmul(Q, K) / math.sqrt(Q.size(-1))
        mask = mask.to(self.device)
        alpha = alpha.to(self.device)
        padding_num = padding_num.to(self.device)
        # 下面开始mask
        alpha = torch.where(mask, alpha, padding_num)
        # 下面开始softmax
        alpha = F.softmax(alpha, dim=2)
        # print('\nalpha is :', alpha)

        out = torch.matmul(alpha, V)

        return out

if __name__ == '__main__':
    out = torch.rand(3,6,512)
    attn = SelfAttention(out.size(-1))
    lens = [6]
    attn_out = attn(out,lens)
    print(attn_out.shape)