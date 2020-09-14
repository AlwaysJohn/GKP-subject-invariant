#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 00:56:17 2020

@author: root
"""
import numpy as np
import scipy
import matplotlib.pyplot as plt
import loader
from demoDataTest import BP
from whitening import zca_whitening_matrix
from sklearn.model_selection import train_test_split

leng = 250
fs = 125
chs = 5
def purify_data(data, log):
    data = data[30:-40]
    log = log[30:-40]

    v = log[0]
    i = 2
    
    while i < len(log):
        if v != log[i]:
            v = log[i]
            data = np.delete(data, range(i-2,i+3), axis=0)
            log = np.delete(log, range(i-2,i+3))
            i -= 2
        i += 1
    
    return data, log

def removeBadChWise(data, log):
    meanPower = np.mean(np.mean(np.sqrt(pow(data, 2)), axis=1), axis=0)
    stdPower = np.mean(np.std(np.sqrt(pow(data,2)), axis=1), axis=0)
    
    print('power mean : {}, power std : {}'.format(meanPower, stdPower))
    
    i, n = 0, 0
    while i < len(log):
        power = np.mean(np.sqrt(pow(data[i], 2)), axis=0)
        for ch in range(chs):
            if abs(power[ch] - meanPower[ch]) > stdPower[ch]:
                data = np.delete(data, i, axis=0)
                log = np.delete(log, i, axis=0)
                n += 1      
                break
        if ch == chs-1:
            i += 1
    print('remove {} datas'.format(n))
    return data, log

def removeBad(data, log): 
    meanPower = np.mean(np.mean(pow(data, 2), axis=(1,2)))
    stdPower = np.mean(np.std(pow(data,2), axis=(1,2)))
    
    print('power mean : {}, power std : {}'.format(meanPower, stdPower))
    
    i = 0
    n = 0
    while i < len(log):
        power = np.mean(pow(data[i], 2))
        if abs(power - meanPower) > stdPower:
            data = np.delete(data, i, axis=0)
            log = np.delete(log, i, axis=0)
            n += 1
        else:
            i += 1
    print('remove {} datas'.format(n))
    return data, log

        

def removeBaseline(data, log, debase, whiten):
    i = 0
    while log[i] == 3:
        i += 1
    
    base_data = data[0:i]
    actdata, actlog = data[i:], log[i:]
    
    if whiten:
        if i==0: #file is 0424 or after, has no baseline, using rest in first 300 data as base
            base_data = data[0:250][log[0:250]==2]
            print('get base data ', base_data.shape)
        
        base = base_data[0]
        for i in range(1,len(base_data)):
            base = np.concatenate((base, base_data[i,200:250]), axis=0)
        
        ZCAM = zca_whitening_matrix(base)
        actdata = np.dot(actdata, ZCAM)
        print('whiten')
    
    elif debase:
        if i==0: #file is 0424 or after, has no baseline, using rest in first 300 data as base
            base_data = data[0:250][log[0:250]==2]
            print('get base data ', base_data.shape)
        
        bmean = np.mean(base_data, axis=(0,1))
        bstd = np.mean(np.std(base_data, axis=1), axis=0)
        
        for i in range(len(actdata)):
            actdata[i] = (actdata[i] - bmean) / bstd
    
    return actdata, actlog
 
def getHighPower(data):
    N = data.shape[1]
    yf = scipy.fft(np.transpose(data,(0,2,1)))
    
    yf = 2.0/N * np.abs(yf[:,:,:N//2])
    high_pow = np.sqrt(np.mean(pow(yf[:,:,20:100],2), axis=(1,2))) #where frequency is 10~50 Hz
    
    return high_pow
       
def read_demo_record(files, require_bp = False, ch=5, require_env=False, debase=False, whiten=False, norm=False, lowcut = 1, highcut = 50):
    train_data, train_log, tr_env, val_data, val_log, val_env = [], [], [], [], [], []

    DemoIIR = BP() #every time call IIR_bp function, the z value reset
    envCount, fileCount = -1, []
    for fi in range(len(files)):
        if files[fi][0:4] not in fileCount:
            envCount += 1
            fileCount.append(files[fi][0:4])
        
        data = []
        log = []
        with open('data/demo/'+files[fi], 'r') as f:
            t = []
            for row in f:
                t.append(row)
    
        i = 0
        while i < len(t):
            log.append(int(t[i]))
            
            for z in range(1, 251):
                data.append([float(x) for x in t[i+z].split(' ')[:-1]])
            i = i+z+2
        
        data = np.reshape(np.array(data), (-1, leng, 5))
        if require_bp:
            data = DemoIIR.IIR_bp(data, lowcut = lowcut, highcut = highcut, ch=ch)
        log = np.array(log)
        
        #here we remove the data recorded while fine tuning the model on demo
        if('fined' in files[fi]):
            data, log = np.delete(data, range(300, 320), axis=0), np.delete(log, range(300,320), axis=0)
        
        data, log = purify_data(data, log)
        data, log = removeBadChWise(data, log)
        data, log = removeBaseline(data, log, debase, whiten)

        tr, v, trlog, vlog = train_test_split(data, log, test_size = 0.2, random_state=0)

        if norm:
            tr, v = loader.standard(tr), loader.standard(v)
        
        if fi == 0:
            train_data, val_data, train_log, val_log = tr.copy(), v.copy(), trlog.copy(), vlog.copy()
            tr_env, val_env = [0] * len(trlog), [0] * len(vlog)
        else:
            train_data, val_data = np.concatenate((train_data, tr), axis=0), np.concatenate((val_data, v), axis=0)
            train_log, val_log = np.concatenate((train_log, trlog), axis=0), np.concatenate((val_log, vlog), axis=0)
            tr_env = tr_env + [fi] * len(trlog)
            val_env = val_env + [fi] * len(vlog)
            
        
          
    if require_env:
        return train_data, val_data, train_log, val_log, np.array(tr_env), np.array(val_env)
    else:
        return train_data, val_data, train_log, val_log