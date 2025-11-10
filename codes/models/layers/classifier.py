"""
@File    : classifier.py
@Author  : wones
@Time    : 2022/10/29 10:40
@Todo    :预测结果
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiNonLinearClassifier(nn.Module):
    def __init__(self,hidden_dim,num_label,inter_hidden_dim=None,dropout_rate = 0.4,act_func = 'relu'):
        super(MultiNonLinearClassifier, self).__init__()
        self.num_label = num_label
        self.hidden_dim = hidden_dim
        self.inter_hidden_dim = 2 * hidden_dim if inter_hidden_dim is None else inter_hidden_dim
        self.classifier1 = nn.Linear(hidden_dim,self.inter_hidden_dim)
        self.classifier2 = nn.Linear(self.inter_hidden_dim,self.num_label)
        self.dropout = nn.Dropout(dropout_rate)
        self.act_func = act_func
    def forward(self,inputs):
        features = self.classifier1(inputs)
        if self.act_func == "gelu":
            features = F.gelu(features)
        elif self.act_func == "relu":
            features = F.relu(features)
        elif self.act_func == "tanh":
            features = F.tanh(features)
        else:
            raise ValueError
        features = self.dropout(features)
        features = self.classifier2(features)
        return features
