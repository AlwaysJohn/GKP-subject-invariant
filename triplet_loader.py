#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 14:48:16 2020

@author: john
"""

import torch
from torch.utils import data
import numpy as np
import random

class TripletLoader(data.Dataset):
    def __init__(self, data, label, env, shuffle=False):
        """
        Args:
            data : The eeg signal with shape [N, 1, C, T]
            env  : The index of environment where the data was recorded
            mode : Indicate procedure status(training or testing)
        """
        self.data = data.copy()
        self.label = label.copy()
        self.env = env.copy()
        
        #self.labelnum, self.envnum = max(self.label)+1, max(self.env)+1
        self.labelnum, self.envnum = len(np.unique(self.label)), len(np.unique(self.env))
        print("There are {0} datas".format(len(self.label)))
        
        """
            iteIndex shows how many pairs in this part of triplet pairs are already taken
            triPairLen shows how many triplet paris there are in this part
            triPairIndex shows the index of the part
        """
        #self.make_triplet_pairs()
        self.dataNums = np.zeros((max(self.env)+1, max(self.label)+1)).astype(int)
        self.datalen = 0
        self.dataNumsAccum = []
        self.sub_data_A_ind = []
        self.sub_data_N_ind = []
        self.sub_data_AP_ind = []
        if shuffle:
            self.shuffle()
        self.set_length()

        self.selected = np.zeros(len(self.label))


    def __len__(self):
        """'return the size of dataset"""
        return self.dataNumsAccum[-1]
        #return self.datalen

    def __getitem__(self, index):
        '''
        for i in range(len(self.dataNumsAccum)):
            if self.dataNumsAccum[i] > index:
                envi, taski = i//self.labelnum, i%self.labelnum
                if i==0:
                    break
                index = index - self.dataNumsAccum[i-1]
                break
        
        t = len(self.sub_data_N_ind[i])
        N = self.data[self.sub_data_N_ind[i][index%t]]
        NEnv = self.env[self.sub_data_N_ind[i][index%t]]
        index = index // t
        
        t = len(self.sub_data_A_ind[i])
        Pfind = 0
        for Anums in range(t-1,0,-1):
            if Anums > index:
                break
            else:
                Pfind += 1
                index -= Anums
            
        P = self.data[self.sub_data_A_ind[i][Pfind]]
        A = self.data[self.sub_data_A_ind[i][Pfind+index+1]]
        PEnv = self.env[self.sub_data_A_ind[i][Pfind]]
        AEnv = self.env[self.sub_data_A_ind[i][Pfind+index+1]]
        
        label = taski
        
        return (torch.tensor(P, dtype=torch.float), torch.tensor(A, dtype=torch.float), torch.tensor(N, dtype=torch.float)), torch.tensor(label, dtype=torch.long), (torch.tensor([PEnv, AEnv, NEnv], dtype=torch.long))
        '''
        '''cannot apply weightedRandomSampler
        for i in range(len(self.dataNumsAccum)):
            if self.dataNumsAccum[i] > index:
                envi, taski = i//self.labelnum, i%self.labelnum
                if i==0:
                    break
                index = index - self.dataNumsAccum[i-1]
                break
        
        A = self.data[self.sub_data_A_ind[i][index]]
        AEnv = self.env[self.sub_data_A_ind[i][index]]
        
        pindrange = list(range(0,index)) + list(range(index+1, len(self.sub_data_A_ind[i])))
        pind = random.choice(pindrange)
        P = self.data[self.sub_data_A_ind[i][pind]]
        PEnv = self.env[self.sub_data_A_ind[i][pind]]
        
        nind = random.choice(range(0,len(self.sub_data_N_ind[i])))
        N = self.data[self.sub_data_N_ind[i][nind]]
        NEnv = self.env[self.sub_data_N_ind[i][nind]]
        
        label = taski
        
        '''
        
        A = self.data[index]
        AEnv = self.env[index]
        label = self.label[index]
        
        index = AEnv*self.labelnum + label
        
        pind = random.choice(range(0,len(self.sub_data_A_ind[index])))
        P = self.data[self.sub_data_A_ind[index][pind]]
        PEnv = self.env[self.sub_data_A_ind[index][pind]]
        
        nind = random.choice(range(0,len(self.sub_data_N_ind[index])))
        N = self.data[self.sub_data_N_ind[index][nind]]
        NEnv = self.env[self.sub_data_N_ind[index][nind]]
        
        
        return (torch.tensor(P, dtype=torch.float), torch.tensor(A, dtype=torch.float), torch.tensor(N, dtype=torch.float)), torch.tensor(label, dtype=torch.long), (torch.tensor([PEnv, AEnv, NEnv], dtype=torch.long))

    def make_triplet_pairs(self):
        tri_pair = []
        tri_label = []
        i=0
        
        sub_data_ind = np.where(self.label == self.label[i])[0] #the sub dataset where the data has tha same label as data[d]
        sub_env = self.env[self.label == self.label[i]]
        
        sub_data_A_ind = sub_data_ind[sub_env == self.env[i]]
        sub_data_N_ind = sub_data_ind[sub_env != self.env[i]]
        
        for j in sub_data_A_ind:
            for k in sub_data_N_ind:
                tri_pair.append((i, j, k))
                tri_label.append(self.label[i])
        
    def set_length(self):
        for i in range(self.envnum):
            for taski in range(self.labelnum):
                sub_data_ind = np.where(self.label == taski)[0]
                sub_env = self.env[self.label == taski]
                
                sub_data_A_ind = sub_data_ind[sub_env == i]
                sub_data_N_ind = sub_data_ind[sub_env != i]
                
                self.sub_data_A_ind.append(sub_data_A_ind.copy())
                self.sub_data_N_ind.append(sub_data_N_ind.copy())
                #self.dataNums[i][taski] = len(sub_data_A_ind)*(len(sub_data_A_ind)-1)*len(sub_data_N_ind)//2
                self.dataNums[i][taski] = len(sub_data_A_ind)
                
                self.datalen += len(sub_data_A_ind)
        
        self.dataNumsAccum = self.dataNums.reshape(-1)
        for i in range(1, len(self.dataNumsAccum)):
            self.dataNumsAccum[i] += self.dataNumsAccum[i-1] 
    
    def shuffle(self):
        p = np.random.permutation(len(self.label))
        self.data, self.label, self.env = self.data[p], self.label[p], self.env[p]
    
    def getTensors(self):
        return (torch.FloatTensor(self.data), torch.LongTensor(self.label), torch.LongTensor(self.env))
    
    def get(self):
        return self.dataNums