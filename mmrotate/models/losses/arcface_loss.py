import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import pi


from ..builder import ROTATED_LOSSES

@ROTATED_LOSSES.register_module()
class ArcFaceLoss(nn.Module):
    def __init__(self, in_features, out_features, s=10.0, m=0.1):
        super(ArcFaceLoss, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    
    def forward(self, input):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        theta = torch.acos(cosine)
        top = torch.exp((torch.cos(theta + self.m)) * self.s)
        top_ = torch.exp(torch.cos(theta) * self.s)
        bottom = torch.sum(torch.exp(cosine * self.s), dim=1).view(-1,1)
        output = (top / (bottom - top_ + top)) + 1e-10

        return output

