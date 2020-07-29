#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 20:38:38 2019

@author: john
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

'''change batchnorm2d to instancenorm2d'''

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
        
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(1, 30), stride=(1,1), padding=(0,15), bias=True) #30
        #self.bn1 = nn.BatchNorm2d(num_features=16, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.bn1 = nn.LayerNorm([16,5,251], eps=1e-5)
        
        #self.avgpnew = nn.AvgPool2d(kernel_size=(1,5), stride=(1,5), padding=0)
        #self.convNew = nn.Conv2d(16, 32, kernel_size=(1, 15), stride=(1,1), padding=(0,7), bias=True)
        
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(5, 1), stride=(1,1), groups=16, bias=True) 
        #self.bn2 = nn.BatchNorm2d(num_features=32, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.bn2 = nn.LayerNorm([32,1,251], eps=1e-5)
        
        self.avgp1 = nn.AvgPool2d(kernel_size=(1,5), stride=(1,5), padding=0) #5
        self.dp1 = nn.Dropout(p=0.6)
        

        
    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = self.bn2(self.conv2(out))
        out = self.act1(out)
        #out = self.dp1(self.avgp1(out))
        out = self.avgp1(out)
        

        
        return out

class TaskNet(nn.Module):
    def __init__(self, num_classes = 3):
        super(TaskNet, self).__init__()
        self.act2 = nn.LeakyReLU()
        
        self.conv3 = nn.Conv2d(32, 32, kernel_size=(1, 15), stride=(1,1), padding=(0,7), bias=True) #15,7
        #self.bn3 = nn.BatchNorm2d(num_features=32, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.bn3 = nn.LayerNorm([32,1,50], eps=1e-5)
        
        self.avgp2 = nn.AvgPool2d(kernel_size=(1,8), stride=(1,8), padding=0) #8
        self.dp2 = nn.Dropout(p=0.6)
        
        self.lin = nn.Linear(in_features=192, out_features=num_classes, bias=True)
        
    def forward(self, x):
        out = self.bn3(self.conv3(x))
        out = self.act2(out)
        #out = self.dp2(self.avgp2(out))
        out = self.avgp2(out)        
        
        out = out.view(out.size(0), -1)
        out = self.lin(out)
        return out
    
class EnvNet(nn.Module):
    def __init__(self, num_classes = 8):
        super(EnvNet, self).__init__()
        self.act2 = nn.LeakyReLU()
        
        self.conv3 = nn.Conv2d(32, 32, kernel_size=(1, 15), stride=(1,1), padding=(0,7), bias=True) #15,7
        #self.bn3 = nn.BatchNorm2d(num_features=32, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.bn3 = nn.LayerNorm([32,1,50], eps=1e-5)
                                  
        self.avgp2 = nn.AvgPool2d(kernel_size=(1,8), stride=(1,8), padding=0) #8
        self.dp2 = nn.Dropout(p=0.6)
        
        self.lin = nn.Linear(in_features=192, out_features=96, bias=True)
        #self.bn = nn.BatchNorm1d(num_features=96, eps=1e-5, momentum=0.1, affine=True, track_running_stats=False)
        self.bn = nn.LayerNorm([96], eps=1e-5)
        self.act = nn.LeakyReLU()
        self.lin2 = nn.Linear(in_features=96, out_features=48, bias=True)
        
    def forward(self, x):
        out = self.bn3(self.conv3(x))
        out = self.act2(out)
        #out = self.dp2(self.avgp2(out))
        out = self.avgp2(out)
        
        out = out.view(out.size(0), -1)
        out = self.act(self.bn(self.lin(out)))
        out = self.lin2(out)
        #out = self.lin2(self.lin(out))
        return out

class EnvClassNet(nn.Module):
    def __init__(self, num_classes=8):
        super(EnvClassNet, self).__init__()
        
        self.lin = nn.Linear(in_features=48, out_features=num_classes, bias=True)
        #self.lin2 = nn.Linear(in_features=24, out_features=num_classes, bias=True)
    
    def forward(self, x):
        return self.lin(x)
        #return self.lin2(self.lin(x))

class EEGNetPreAct(nn.Module):
    def __init__(self, act = 'ELU', num_classes = 3):
        super(EEGNetPreAct, self).__init__()
        
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
        
        self.bn1 = nn.BatchNorm2d(num_features=1, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(1, 30), stride=(1,1), padding=(0,15), bias=False)
        
        self.bn2 = nn.BatchNorm2d(num_features=16, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(4, 1), stride=(1,1), groups=16, bias=False)
        self.avgp1 = nn.AvgPool2d(kernel_size=(1,5), stride=(1,5), padding=0)
        self.dp1 = nn.Dropout(p=0.6)
        
        self.bn3 = nn.BatchNorm2d(num_features=32, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=(1, 15), stride=(1,1), padding=(0,7), bias=False)
        self.avgp2 = nn.AvgPool2d(kernel_size=(1,8), stride=(1,8), padding=0)
        self.dp2 = nn.Dropout(p=0.6)
        
        self.lin = nn.Linear(in_features=192, out_features=num_classes, bias=True)
        
    def forward(self, x):
        out = self.conv1(self.bn1(x))
        
        out = self.conv2(self.avgp1(self.act1(self.bn2(out))))
        out = self.dp1(out)
        
        out = self.conv3(self.avgp2(self.act2(self.bn3(out))))
        out = self.dp2(out)
        
        out = out.view(out.size(0), -1)
        #print(out.shape)
        out = self.lin(out)
        
        return out


class DCN(nn.Module):
    def __init__(self, act='ELU', num_classes=2):
        super(DCN,self).__init__()
        self.act = act
        self.ori_f = 25
        self.num_classes = num_classes
        
        self.conv1 = nn.Conv2d(1, 25, kernel_size=(1,5), bias=True)
        
        self.block1 = self.make_layer(25, (3,1))
        self.block2 = self.make_layer(50)
        self.block3 = self.make_layer(100)
        self.block4 = self.make_layer(200)
        self.lin = nn.Linear(in_features=8600, out_features=2, bias=True)

    def make_layer(self, features, k_size=(1,5)):  #self, block type, number of basic blocks in one layer, stride if is 2, then the output map size shrink to half
        layers = []
        layers.append(nn.Conv2d(self.ori_f, features, kernel_size=k_size, bias=True))
        layers.append(nn.BatchNorm2d(features, eps=1e-5,momentum=0.1))
        if self.act == 'ELU':
            layers.append(nn.ELU())
        elif self.act=='ReLU':
            layers.append(nn.ReLU())
        elif self.act=='LeakyReLU':
            layers.append(nn.LeakyReLU())
        layers.append(nn.MaxPool2d((1,2)))
        layers.append(nn.Dropout(p=0.5))
        
        self.ori_f = features
        
        return nn.Sequential(*layers)
          
    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = out.view(out.size(0), -1)
        out = self.lin(out)
        
        return out
    
class Cla(nn.Module):
    def __init__(self):
        super(Cla, self).__init__()
        self.lin = nn.Linear(in_features = 4, out_features=3, bias=True)
        #self.lin = nn.Linear(in_features = 224*2, out_features=3, bias=True)
    
    def forward(self, x):
        out = self.lin(x)
        return out

class ANN(nn.Module):
    def __init__(self, num_classes=3):
        super(ANN, self).__init__()
        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()
        self.lin1 = nn.Linear(1250, 2500, bias=True)
        self.lin2 = nn.Linear(2500,1250, bias=True)
        self.lin3 = nn.Linear(1250, num_classes, bias=True)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.act1(self.lin1(x))
        out = self.act2(self.lin2(out))
        out = self.lin3(out)
        return out
