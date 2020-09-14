#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 16:40:15 2020

@author: root
"""

import loader
import readDemoRecordBaseline_env as rdr
import numpy as np
import argparse
import getData as gd
import torch.nn.functional as F
from model_env_light import *
from torch import optim
import torch.utils.data
import matplotlib.pyplot as plt
from tqdm import tqdm
from triplet_loader import TripletLoader
from plot_confusion_matrix import plot_confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from radam import RAdam

#training parameters
parser = argparse.ArgumentParser(description='GKP training, fine_tune parameters')
parser.add_argument('--batch_size', type=int, default=32, metavar='N', help='batch size for training (128)')
parser.add_argument('--epochs', type=int, default=200, metavar='N', help='# of epochs for training (150)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (0.9)')
parser.add_argument('--lr', type=float, default=1e-3, metavar='N', help='learning rate (0.02)')
parser.add_argument('--lr2', type=float, default=1e-3, metavar='N', help='learning rate (0.02)')
parser.add_argument('--margin', type=float, default=1, metavar='N', help='margin of triplet loss (?)')
parser.add_argument('--alpha', type=float, default=2, metavar='N', help='weight for KLDLoss (?)')
parser.add_argument('--batchPerEpoch', type=int, default=3000, metavar='N', help='batch early stop in one epoch (?)')
parser.add_argument('--batch_size2', type=int, default=8, metavar='N', help='batch size for training (128)')
parser.add_argument('--epoch_change', type=int, default=200, metavar='N', help='epoch when changing learning strategy (50)')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if device == 'cuda':
  device = torch.device('cuda')

tune_mode_save = 'not tune'

def train(netFE, netTask, dataloader, optTask, optFE, epoch):
    netFE.train()
    netTask.train()
    
    CE = nn.CrossEntropyLoss()
    FE_loss, Task_loss = 0, 0
    APNL, ANPL = 0,0
    for batch_idx, ((P,A,N), targets, targetsEnv) in enumerate(dataloader): #~~
        if batch_idx > args.batchPerEpoch:
            print('exceed the limit of batch per epoch')
            break
        if torch.cuda.is_available():
          P, A, N, targets = P.to(device), A.to(device), N.to(device), targets.to(device)
        
        optTask.zero_grad()
        optFE.zero_grad()
        
        fP, fA, fN = netFE(P), netFE(A), netFE(N)
        
        if epoch <= args.epoch_change:
            tripLoss = F.triplet_margin_loss(fA, fP, fN, margin=0) + F.triplet_margin_loss(fA, fN, fP, margin=0)
            FE_loss += tripLoss.item()        
            APNL += F.triplet_margin_loss(fA, fP, fN, margin=0)
            ANPL += F.triplet_margin_loss(fA, fN, fP, margin=0)
            loss = tripLoss
        
        '''task phase'''
        tP, tA, tN = netTask(fP), netTask(fA), netTask(fN)         
              
        x = torch.transpose(torch.stack((tP,tA,tN),dim=1), 1, 2) #batch x 3(task) x 3(P,A,N)     
        y = targets.view(x.shape[0],1).repeat(1,3)
        
        CEloss = CE(x,y)
        loss = CEloss + tripLoss
        loss.backward()
        optTask.step()

        Task_loss += CEloss.item()
    if epoch == 0 or epoch == args.epochs//2 or epoch==args.epochs-1:
        print('APN:{}, ANP:{}'.format(APNL/batch_idx, ANPL/batch_idx))
    '''evaluate model on trainin set (not pairs)'''    
    netFE.eval()
    netTask.eval()
    (data, label, env) = dataloader.dataset.getTensors()
    with torch.no_grad():
        f = netFE(data.to(device))
        predTask = netTask(f).data.max(1)[1]
    
    c_tr_Task = 0
    for i in range(3):
        c_tr_Task += float(predTask[label==i].eq(i).cpu().sum().item())/sum(label==i).cpu().item()
    c_tr_Task = float(100*c_tr_Task/3)
    
    l_tr_Task = Task_loss / batch_idx
    l_tr_Env = FE_loss / batch_idx
    
    return c_tr_Task, l_tr_Task, l_tr_Env

'''this is actually validation, so the possible Envs are known for model'''
def test(netFE, netTask, dataloader):
    netFE.eval()
    netTask.eval()
    
    Loss = nn.CrossEntropyLoss()
    
    (data, label, env) = dataloader.dataset.getTensors()
    label = label.to(device)
    with torch.no_grad():
        f = netFE(data.to(device))
        outputTask = netTask(f)  #~~
        predTask = outputTask.data.max(1)[1]
    
    lossTask = Loss(outputTask, label).cpu().item()
    
    c_Task = 0
    for i in range(3):
        c_Task += float(predTask[label==i].eq(i).cpu().sum().item())/sum(label==i).cpu().item()
    c_Task = c_Task / 3
        
    '''return val Task accuracy, val Env accuracy, val Task loss, val Env loss (both are CrossEntropy loss)''' 
    return (float(100*c_Task)), lossTask

def train_model(netFE, netTask, epoch, tr_loader, te_loader, correct_ori, save, optTask, optFE): 
    c_tr_Task, l_tr_Task, l_tr_Env = train(netFE, netTask, tr_loader, optTask, optFE, epoch)
    
    correctTask, lossTask = test(netFE, netTask, te_loader)
    if correctTask > correct_ori:
        correct_ori = correctTask

        statFE = netFE.state_dict()
        statTask = netTask.state_dict()
        filenameFE = 'model/FE_'+str(epoch)+'_'+str(correct_ori)+'.tar'
        filenameTask = 'model/Task_'+str(epoch)+'_'+str(correct_ori)+'.tar'
        
        torch.save({
            'epoch': epoch,
            'state_dict': statFE,  #instead of saving whole model, only saving parameters
        }, filenameFE)
        
        torch.save({
            'epoch': epoch,
            'state_dict': statTask,  #instead of saving whole model, only saving parameters
        }, filenameTask)
    
        save = [filenameFE, filenameTask]
    
    return correctTask, correct_ori, save, c_tr_Task, l_tr_Task, l_tr_Env, lossTask

def get_best_model(netFE, netTask, f0, f1):
    tar = torch.load(f0)
    netFE.load_state_dict(tar['state_dict'])
    tar = torch.load(f1)
    netTask.load_state_dict(tar['state_dict'])

def adjust_lr_rate(optimizer, loss, times):
  if abs(sum(loss[-10:]) - sum(loss[-20:-10])) < 0.01 * sum(loss[-20:-10]) / times:
      times *= 10
      for param_group in optimizer.param_groups:
          param_group['lr'] *= 0.1
  return times
   

def show_TSNE(loader, val_loader, test_data, modelFE):
    data = loader.dataset.data
    feature = modelFE(torch.FloatTensor(data).to(device)).cpu().data.numpy()
    env = train_loader.dataset.env
    lentr = len(env)
    
    data = val_loader.dataset.data
    feature = np.concatenate((feature, modelFE(torch.FloatTensor(data).to(device)).cpu().data.numpy()), axis=0)
    venv = val_loader.dataset.env

    T = manifold.TSNE(n_components=2, init='pca', random_state=0)
    Yfeature = T.fit_transform(feature)
    
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    
    f = plt.figure(figsize=(8,8))
    ax = f.add_subplot(111)
    for i, s, c in zip([0, 1, 2, 3, 4, 5, 6, 7], [0,1,2,3,4,5,6,7], colors[:8]): #
        ax.scatter(Yfeature[:lentr][env == i, 0], Yfeature[:lentr][env == i, 1], color=c, alpha=.8, lw=0.1, label=str(s))
        ax.scatter(Yfeature[lentr:][venv == i, 0], Yfeature[lentr:][venv == i, 1], color=c, alpha=.1, lw=0.1)
    ax.legend() 
    ax.grid(True)
    plt.show()

    
def show_TSNE_task_env(loader, test_data, test_label, modelFE):
    data, label = loader.dataset.data, loader.dataset.label
    data = np.concatenate((data, test_data), axis=0)
    label = np.concatenate((label, test_label+3))
    
    feature = modelFE(torch.FloatTensor(data).to(device))
    
    T = manifold.TSNE(n_components=2, init='pca', random_state=0)
    Yfeature = T.fit_transform(feature.cpu().data.numpy())
    
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    
    f = plt.figure(figsize=(8,8))
    ax = f.add_subplot(111)
    for i, s, c in zip([0, 1, 2], ['left','right','rest'], colors[:3]):
        ax.scatter(Yfeature[label == i, 0], Yfeature[label == i, 1], alpha=.6, lw=0.1, label=s+'-tr', color=c)
        ax.scatter(Yfeature[label == i+2, 0], Yfeature[label == i+2, 1], alpha=.1, lw=0.1, label=s+'-te', color=c)
    ax.legend()
    ax.grid(True)
    plt.show()
    
    env_tr = loader.dataset.env
    env = np.concatenate((env_tr, np.array([9]*len(test_data))), axis=0)
    
    f = plt.figure(figsize=(8,8))
    ax = f.add_subplot(111)
    for i, s in zip([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], ['0','1','2','3','4','5','6','7','test']):
        ax.scatter(Yfeature[env == i, 0], Yfeature[env == i, 1], alpha=.8, lw=0.2, label=s)
    ax.legend()
    ax.grid(True)
    plt.show()    
    
def make_weights(label):
    num_classes = np.unique(label)[-1]+1
    
    nums = [0]*num_classes
    for i in range(num_classes):
        nums[i] = sum(label==i)
    
    N = float(len(label))
    weights = [0.]*len(label)
    
    for i in range(len(label)):
        weights[i] = N/nums[label[i]]
    
    return weights

def makeLoader(data, label, env, b_size=args.batch_size, w_sample=False): #make data loader from 2 dimensional data
    data = np.transpose(np.expand_dims(data, axis=1), (0, 1, 3, 2))
    dataset = TripletLoader(data, label, env, shuffle=False)

    if w_sample:
        weights = make_weights(label)
        weights = torch.FloatTensor(weights)
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
        loader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_size=b_size,
                                             sampler = sampler)

    else:
        loader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_size=b_size,
                                             shuffle=True,
                                             )
    return loader

def finalTest(netFE, netTask, test_data, test_label):
    text = ['left','right','rest']
    Loss = nn.CrossEntropyLoss()
    netFE.eval()
    netTask.eval()
    output = netTask(netFE(torch.FloatTensor(test_data).to(device)))
    pred = output.data.max(1)[1]
    #pred = voting(pred)
    loss = Loss(output, torch.tensor(test_label).to(device)).item()
    correct_num = sum(np.array(pred.cpu()) == test_label)  #.cpu()
    val_accuracy = correct_num / len(test_label)
    cmt = plot_confusion_matrix(test_label, pred.cpu(), np.array(text), title='accuracy: {0}%'.format(val_accuracy*100), normalize=True, show=False)
    
    return val_accuracy, loss, cmt
    

all_file = ['0320_2.txt','0320_3.txt','0529_2.txt','0529_3.txt','0225_2.txt','0225_3.txt','0424_3_fined.txt','0424_2_fined.txt','0425_2_fined.txt','0425_3_fined.txt']
bal_modes = ['no'] #,'old'
tune_modes = ['all'] #,'last_eval',,'last_eval', 
norm_modes = ['data'] #,'data','db','db_data','no'
correct_line = {}
accWithEnv = []
accWOEnv = []
Mrecalls = []
for norm_mode in norm_modes:
    for test_file in all_file:
        tune_mode_save = 'not tune'
        correct = []
        correct_ori = 0
        best = ''
        val_accuracy = 0
        cm = np.zeros((3,3))
        
        train_file = all_file.copy()
        for file in all_file:
            if test_file[0:5] in file:
                train_file.remove(file)
        
        print('train on '+ str(train_file))
        print('test on '+ str(test_file))
        
        test_file = [test_file]

        train_data, val_data, train_label, val_label, train_env, val_env = rdr.read_demo_record(train_file, require_bp = True, require_env = True, whiten=False, debase = ('db' in norm_mode))
        
        if 'data' in norm_mode:
            train_data, val_data = loader.standard(train_data), loader.standard(val_data)
        
        train_loader = makeLoader(train_data, train_label, train_env, w_sample=True)
        val_loader = makeLoader(val_data, val_label, val_env, w_sample=False)
        
        test_data, tdata, test_label, tlabel = rdr.read_demo_record(test_file, require_bp = True, whiten=False, debase = ('db' in norm_mode))
        test_data = np.concatenate((test_data, tdata),axis=0)
        test_label = np.concatenate((test_label, tlabel),axis=0)
        if 'data' in norm_mode:
            test_data = loader.standard(test_data)
        test_data = np.transpose(np.expand_dims(test_data, axis=1), (0, 1, 3, 2))
        
        '''here we are going to train'''
        for i in range(1):
            modelFE, modelTask = EEGNetFE2('LeakyReLU'), TaskNet2(num_classes = 3)
            modelFE.to(device) 
            modelTask.to(device) 
            
            optTask = optim.SGD(list(modelFE.parameters()) + list(modelTask.parameters()), lr = args.lr, momentum = args.momentum) #
            optEnv = optim.SGD(modelFE.parameters(), lr=args.lr2, momentum=args.momentum, weight_decay=0.001)
        
            correct_tr_Task, correct_val_Task, correct_te_Task = [], [], []
            loss_tr_Task, loss_tr_Env, loss_val_Task, loss_te_Task = [], [], [], []
            best = []
            decay_times = 1
            for epoch in tqdm(range(args.epochs)):
                c_val_Task, correct_ori, best, c_tr_Task, l_tr_Task, l_tr_Env, l_val_Task = train_model(modelFE, modelTask, epoch, train_loader, val_loader, correct_ori, best, optTask, optEnv)
                correct_val_Task.append(c_val_Task)
                correct_tr_Task.append(c_tr_Task)
                loss_tr_Task.append(l_tr_Task)
                loss_tr_Env.append(l_tr_Env)
                loss_val_Task.append(l_val_Task)
                
                c_te_Task, l_te_Task, _ = finalTest(modelFE, modelTask, test_data, test_label)
                correct_te_Task.append(c_te_Task*100)
                loss_te_Task.append(l_te_Task)
                
                
                if epoch == 0 or epoch == args.epoch_change//2 or epoch == args.epochs-1:
                    show_TSNE_task_env(train_loader, test_data, test_label, modelFE)

            x = np.arange(0,args.epochs)
                          
            ax = plt.subplot(111)
            ax.plot(x,correct_tr_Task,label='Task training acc')
            ax.plot(x,correct_val_Task,label='Task validation acc')
            ax.plot(x,correct_te_Task,label='Task test acc')
            ax.legend()
            ax.grid(True)
            plt.title('Accuracy (3 tasks) (' + str(len(np.unique(train_env))) + ' envs)')
            plt.show()
            
            ax = plt.subplot(111)
            ax.plot(x,loss_tr_Task,label='Task train loss (CE)')
            ax.plot(x,loss_tr_Env,label='FE train loss (tripletDisc)')
            ax.plot(x,loss_val_Task,label='Task val loss (CE)')
            ax.legend()
            ax.grid(True)
            plt.title('loss')
            plt.show()
        
        get_best_model(modelFE, modelTask, best[0], best[1])
         
        val_accuracy, val_loss, cmt = finalTest(modelFE, modelTask, test_data, test_label)
        
        print('-------final-------')
        print('testing accuracy {}, {} : {}%'.format(norm_mode, test_file[0], str(val_accuracy)))
        print(cmt)
        accWOEnv.append(val_accuracy)
        Mrecalls.append((cmt[0][0] + cmt[1][1] + cmt[2][2])/3)
        
