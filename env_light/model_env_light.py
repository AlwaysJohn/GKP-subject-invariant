#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 17:23:11 2020

@author: root
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class EEGNetFE(nn.Module):
    def __init__(self, act = 'ELU'):
        super(EEGNetFE, self).__init__()
        
        if act=='ELU':
            self.act1 = nn.ELU(alpha=1.0)
            self.act2 = nn.ELU(alpha=1.0)
        elif act=='ReLU':
            self.act1 = nn.ReLU()
            self.act2 = nn.ReLU()
        elif act=='LeakyReLU':
            self.act1 = nn.LeakyReLU()
            self.act2 = nn.LeakyReLU()
        elif act=='Tanh':
            self.act1 = nn.Tanh()
            self.act2 = nn.Tanh()
        
        self.conv1 = nn.Conv2d(1, 8, kernel_size=(1, 30), stride=(1,1), padding=(0,15), bias=True)
        self.bn1 = nn.BatchNorm2d(num_features=8, eps=1e-5, momentum=0.1, affine=True, track_running_stats=False)

        self.conv2 = nn.Conv2d(8, 16, kernel_size=(5, 1), stride=(1,1), groups=8, bias=True) 
        self.bn2 = nn.BatchNorm2d(num_features=16, eps=1e-5, momentum=0.1, affine=True, track_running_stats=False)
        
        self.avgp1 = nn.AvgPool2d(kernel_size=(1,5), stride=(1,5), padding=0) #5
        self.dp1 = nn.Dropout(p=0.6)
        

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = self.bn2(self.conv2(out))
        out = self.act1(out)
        out = self.avgp1(out)
        
        out = out.view(out.size(0), -1)
        
        return out

class TaskNet(nn.Module):
    def __init__(self, num_classes = 3):
        super(TaskNet, self).__init__()

        self.conv3 = nn.Conv2d(16, 16, kernel_size=(1, 15), stride=(1,1), padding=(0,7), bias=True) #15,7
        self.bn3 = nn.BatchNorm2d(num_features=16, eps=1e-5, momentum=0.1, affine=True, track_running_stats=False)
        self.act2 = nn.LeakyReLU()        
        self.avgp2 = nn.AvgPool2d(kernel_size=(1,8), stride=(1,8), padding=0) #8

        self.lin = nn.Linear(in_features=96, out_features=96, bias=True)  #192
        self.act = nn.LeakyReLU()
        self.lin2 = nn.Linear(in_features = 96, out_features=48, bias=True)
        self.act2 = nn.LeakyReLU()
        self.lin3 = nn.Linear(in_features = 48, out_features=num_classes, bias=True)
        
    def forward(self, x):
        x = x.reshape((x.shape[0], 16, 1, -1))
        out = self.bn3(self.conv3(x))
        out = self.act2(out)
        out = self.avgp2(out)    
        
        out = out.view(out.size(0),-1)
        out = self.lin(out)
        out = self.lin2(self.act(out))
        out = self.lin3(self.act2(out))
        return out
