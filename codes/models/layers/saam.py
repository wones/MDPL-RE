"""
@File    : saam.py
@Author  : wones
@Time    : 2022/10/28 19:25
@Todo    :定义saam结构
"""

import torch
import torch.nn as nn
from models.layers.selfAttention import SelfAttention

class SAAM(nn.Module):
    def __init__(self,in_hidden_dim,out_hidden_dim,device = 'cpu'):
        super(SAAM, self).__init__()
        self.selfAttention = SelfAttention(in_hidden_dim,device = device)
        self.linear = nn.Linear(in_hidden_dim,out_hidden_dim)

    def forward(self,inputs):
        lens = [inputs.size(1)]
        attn_out = self.selfAttention(inputs,lens)
        avg = torch.mean(attn_out,dim = 1)
        # print(avg.shape)
        out = self.linear(avg)
        return out

if __name__ == '__main__':
    saam = SAAM(768,384)
    insaam = torch.rand(3,6,768)
    outsaam = saam(insaam)
    print(outsaam)
    print(outsaam.shape)
