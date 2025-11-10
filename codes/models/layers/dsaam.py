"""
@File    : dsaam.py
@Author  : wones
@Time    : 2022/10/28 22:18
@Todo    :定义DSAAM层
"""
import torch
import torch.nn as nn
from models.layers.saam import SAAM

class DSAAM(nn.Module):
    def __init__(self,in_hidden_dim,out_hidden_dim,device = 'cpu'):
        super(DSAAM, self).__init__()
        self.eSAAM = SAAM(in_hidden_dim,out_hidden_dim,device = device)
        self.rSAAM = SAAM(in_hidden_dim, out_hidden_dim,device = device)

    def getSpecialFeatures(self,bertOutput, specialIndex):
        specialFeatures = torch.index_select(bertOutput, 1, specialIndex)
        return specialFeatures

    def forward(self,outputs,entityIndex,relIndex,questionIndex,textIndex):
        specialEntityFeatures = self.getSpecialFeatures(outputs, entityIndex)
        specialRelFeatures = self.getSpecialFeatures(outputs, relIndex)
        specialQuestionFeatures = self.getSpecialFeatures(outputs, questionIndex)
        specialTextFeatures = self.getSpecialFeatures(outputs, textIndex)

        entityFeatures = torch.cat([specialEntityFeatures, specialQuestionFeatures, specialTextFeatures], dim=1)
        relFeatures = torch.cat([specialRelFeatures, specialQuestionFeatures, specialTextFeatures], dim=1)

        De = self.eSAAM(entityFeatures)
        Dr = self.rSAAM(relFeatures)
        Df = torch.cat([De, Dr], dim=1)

        return Df