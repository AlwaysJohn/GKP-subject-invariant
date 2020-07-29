#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 21:38:59 2020

@author: root
"""

import loader
import readDemoRecordBaseline_env as rdr
import numpy as np
import argparse
import getData as gd
import torch.nn.functional as F
from model_env import *
from torch import optim
import torch.utils.data
import matplotlib.pyplot as plt
from tqdm import tqdm
from triplet_loader import TripletLoader
from plot_confusion_matrix import plot_confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn import manifold

#training parameters
parser = argparse.ArgumentParser(description='GKP training, fine_tune parameters')
parser.add_argument('--batch_size', type=int, default=32, metavar='N', help='batch size for training (128)')
parser.add_argument('--epochs', type=int, default=150, metavar='N', help='# of epochs for training (150)')
parser.add_argument('--epochs2', type=int, default=200, metavar='N', help='# of epochs for tuning (150)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (0.9)')
parser.add_argument('--lr', type=float, default=1e-4, metavar='N', help='learning rate (0.02)')
parser.add_argument('--lr2', type=float, default=1e-4, metavar='N', help='learning rate (0.02)')
parser.add_argument('--margin', type=float, default=1, metavar='N', help='margin of triplet loss (?)')
parser.add_argument('--alpha', type=float, default=0.5, metavar='N', help='weight for KLDLoss (?)')
parser.add_argument('--batchPerEpoch', type=int, default=3000, metavar='N', help='batch early stop in one epoch (?)')
#0527 - enlarge batch size in fine tune, let the bn layer learn the general mean, std of data of new subject 
parser.add_argument('--batch_size2', type=int, default=8, metavar='N', help='batch size for training (128)')
parser.add_argument('--epoch_change', type=int, default=50, metavar='N', help='epoch when changing learning strategy (50)')
args = parser.parse_args()


device = 'cuda' if torch.cuda.is_available() else 'cpu'

if device == 'cuda':
  device = torch.device('cuda')

tune_mode_save = 'not tune'

def train(netFE, netTask, netEnv, dataloader, optTask, optEnv, netEnvClass, optEnvClass, epoch):
    netFE.train()
    netTask.train()
    netEnv.train()
    netEnvClass.train()
    
    KLD = nn.KLDivLoss(reduction='batchmean')
    CE = nn.CrossEntropyLoss()
    Env_loss, Task_loss, Alphaloss = 0, 0, 0
    for batch_idx, ((P,A,N), targets, targetsEnv) in enumerate(dataloader): #~~
        if batch_idx > args.batchPerEpoch:
            print('exceed the limit of batch per epoch')
            break
        if torch.cuda.is_available():
          P, A, N, targets = P.to(device), A.to(device), N.to(device), targets.to(device)
        
        optTask.zero_grad()
        optEnv.zero_grad()
        optEnvClass.zero_grad()
        
        fP, fA, fN = netFE(P), netFE(A), netFE(N)
        
        '''environment phase'''
        eP, eA, eN = netEnv(fP.detach()), netEnv(fA.detach()), netEnv(fN.detach()) #.detach()
        tripLoss = torch.mean(F.relu(torch.norm(eA-eP, dim=1) - torch.norm(eA-eN, dim=1) + args.margin))
        
        #if epoch <= 20:
        #   tripLoss.backward()
        #   optEnv.step()
        Env_loss += tripLoss.item()        
        
        #eP, eA, eN = netEnv(fP.detach()), netEnv(fA.detach()), netEnv(fN.detach())    
        ecP, ecA, ecN = netEnvClass(eP), netEnvClass(eA), netEnvClass(eN) #.detach()
        x = torch.transpose(torch.stack((ecP,ecA,ecN),dim=1), 1, 2) #batch x env# x 3(P,A,N)     
        envcloss = CE(x,targetsEnv.to(device))
        
        loss = tripLoss #+ envcloss
        #loss.backward()
        optEnv.step()
        #optEnvClass.step()
        
        '''task phase'''
        tP, tA, tN = netTask(fP), netTask(fA), netTask(fN)
        eP, eA, eN = netEnv(fP), netEnv(fA), netEnv(fN)         
      
        '''stack shape : batch_size *2 '''
        stack_norm = torch.stack((torch.norm(eA-eP, dim=1), torch.norm(eA-eN, dim=1)), dim=1)
        tripletLoss = KLD(F.log_softmax(stack_norm, dim=1), (torch.ones(stack_norm.shape)*0.5).to(device))
        #tripletLoss = torch.mean(torch.norm(eP-eN, dim=1)).to(device)
        
        x = torch.transpose(torch.stack((tP,tA,tN),dim=1), 1, 2) #batch x 3(task) x 3(P,A,N)     
        y = targets.view(x.shape[0],1).repeat(1,3)
        
        CEloss, alphaloss = CE(x,y), args.alpha * tripletLoss
        loss = CEloss + alphaloss
        loss.backward()
        optTask.step()
        '''
        if epoch > args.epoch_change:
            loss = CEloss + alphaloss #+ tripLoss#- envcloss #
            
            loss.backward()
            optTask.step()
            #optEnv.step()
        else:
            loss = CEloss  + tripLoss + envcloss
        
            loss.backward()
            optEnv.step() #
            optEnvClass.step() #
            optTask.step()
         '''
        Task_loss += CEloss.item()
        Alphaloss += alphaloss.item()

    '''evaluate model on trainin set (not pairs)'''    
    netFE.eval()
    netTask.eval()
    netEnv.eval()
    netEnvClass.eval()
    (data, label, env) = dataloader.dataset.getTensors()
    with torch.no_grad():
        f = netFE(data.to(device))
        predTask = netTask(f).data.max(1)[1]
        predEnvC = netEnvClass(netEnv(f)).data.max(1)[1]
        
    #training accuracy
    correctTask_num = sum(np.array(predTask.cpu()) == np.array(label))
    correctEnv_num = sum(np.array(predEnvC.cpu()) == np.array(env))
    c_tr_Task = float(100*correctTask_num) / len(label)
    c_tr_Env = float(100*correctEnv_num) / len(label)
    l_tr_Task = Task_loss / batch_idx
    l_tr_Env = Env_loss / batch_idx
    l_tr_Alpha = Alphaloss / batch_idx
    
    if epoch % 40 ==0:
        show_TSNE(dataloader, netFE, netEnv)
    
    return c_tr_Task, c_tr_Env, l_tr_Task, l_tr_Env, l_tr_Alpha

'''this is actually validation, so the possible Envs are known for model'''
def test(netFE, netTask, netEnv, netEnvClass, dataloader):#~~
    netFE.eval()
    netTask.eval()
    netEnv.eval()
    netEnvClass.eval()
    
    Loss = nn.CrossEntropyLoss()
    
    (data, label, env) = dataloader.dataset.getTensors()
    label, env = label.to(device), env.to(device)
    with torch.no_grad():
        f = netFE(data.to(device))
        outputTask, outputEnv = netTask(f), netEnvClass(netEnv(f))  #~~
        predTask, predEnv = outputTask.data.max(1)[1], outputEnv.data.max(1)[1]
    
    lossTask = Loss(outputTask, label).cpu().item()
    lossEnv = Loss(outputEnv, env).cpu().item()
    
    c_Task = predTask.eq(label).cpu().sum().item()
    c_Env = predEnv.eq(env).cpu().sum().item()
    
    '''return val Task accuracy, val Env accuracy, val Task loss, val Env loss (both are CrossEntropy loss)''' 
    return (float(100 * c_Task)) / len(label), (float(100 * c_Env)) / len(label), lossTask, lossEnv

def train_model(netFE, netTask, netEnv, epoch, tr_loader, te_loader, correct_ori, save, optTask, optEnv, netEnvClass, optEnvClass): 
    c_tr_Task, c_tr_Env, l_tr_Task, l_tr_Env, l_tr_Alpha = train(netFE, netTask, netEnv, tr_loader, optTask, optEnv, netEnvClass, optEnvClass, epoch)
    
    correctTask, correctEnv, lossTask, lossEnv = test(netFE, netTask, netEnv, netEnvClass, te_loader)
    if correctTask > correct_ori:
        correct_ori = correctTask

        statFE = netFE.state_dict()
        statTask = netTask.state_dict()
        statEnv = netEnv.state_dict()
        #savefilename = 'model_demo_crosstime/result'+mode+'.tar'
        filenameFE = 'model/FE_'+str(epoch)+'_'+str(correct_ori)+'.tar'
        filenameTask = 'model/Task_'+str(epoch)+'_'+str(correct_ori)+'.tar'
        filenameEnv = 'model/Env_'+str(epoch)+'_'+str(correct_ori)+'.tar'
        
        torch.save({
            'epoch': epoch,
            'state_dict': statFE,  #instead of saving whole model, only saving parameters
        }, filenameFE)
        
        torch.save({
            'epoch': epoch,
            'state_dict': statTask,  #instead of saving whole model, only saving parameters
        }, filenameTask)
    
        torch.save({
            'epoch': epoch,
            'state_dict': statEnv,  #instead of saving whole model, only saving parameters
        }, filenameEnv)       

        save = [filenameFE, filenameTask, filenameEnv]
    
    return correctTask, correctEnv, correct_ori, save, c_tr_Task, c_tr_Env, l_tr_Task, l_tr_Env, lossTask, lossEnv, l_tr_Alpha

def get_best_model(netFE, netTask, netEnv, f0, f1, f2):
    tar = torch.load(f0)
    netFE.load_state_dict(tar['state_dict'])
    tar = torch.load(f1)
    netTask.load_state_dict(tar['state_dict'])
    tar = torch.load(f2)
    netEnv.load_state_dict(tar['state_dict']) 

def save_performance(p):
    import csv
    with open('pfm.csv', 'w+') as output:
        writer = csv.writer(output)
        writer.writerow(['file\\performance'] + list(p[list(p.keys())[0]].keys()))
        for k, v in p.items():
            writer.writerow([k]+list(v.values()))

def adjust_lr_rate(optimizer, loss, times):
  if abs(sum(loss[-10:]) - sum(loss[-20:-10])) < 0.01 * sum(loss[-20:-10]) / times:
      times *= 10
      for param_group in optimizer.param_groups:
          param_group['lr'] *= 0.1
  return times

def show_TSNE(loader, modelFE, modelEnv):
    data = loader.dataset.data
    feature = modelEnv(modelFE(torch.FloatTensor(data).to(device)))
    env = train_loader.dataset.env
    T = manifold.TSNE(n_components=2, init='pca', random_state=0)
    Yfeature = T.fit_transform(feature.cpu().data.numpy())
    
    f = plt.figure(figsize=(8,8))
    ax = f.add_subplot(111)
    for i, s in zip([0, 1, 2, 3, 4, 5, 6, 7], [0,1,2,3,4,5,6,7]):
        ax.scatter(Yfeature[env == i, 0], Yfeature[env == i, 1], alpha=.8, lw=0.2, label=str(s))
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
        print("hi")
    else:
        loader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_size=b_size,
                                             shuffle=True,
                                             )
    return loader


def voting(pred):
    pred = np.array(pred.cpu())
    result = [i for i in pred[0:3]]
    voters = [i for i in pred[0:3]]
    for p in pred[3:]:
        voters.append(p)
        result.append(max(set(voters[::-1]), key = voters.count))
        voters = voters[1:]
    
    return result


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
    

all_file = ['0320_2.txt','0320_3.txt','0225_2.txt','0225_3.txt','0529_2.txt','0529_3.txt','0424_3_fined.txt','0424_2_fined.txt','0425_2_fined.txt','0425_3_fined.txt']#,'0316_3.txt','0316_4.txt'
bal_modes = ['no'] #,'old'
tune_modes = ['all'] #,'last_eval',,'last_eval', 
norm_modes = ['no'] #,'data','db','db_data'
correct_line = {}
accWithEnv = []
accWOEnv = []

for norm_mode in norm_modes:
    for test_file in all_file: #all_file #,['0320_3.txt']
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

        train_data, val_data, train_label, val_label, train_env, val_env = rdr.read_demo_record(train_file, require_bp = True, require_env = True, whiten=True, debase = ('db' in norm_mode))
        
        if 'data' in norm_mode:
            train_data, val_data = loader.standard(train_data), loader.standard(val_data)
        
        train_loader = makeLoader(train_data, train_label, train_env, w_sample=True)
        val_loader = makeLoader(val_data, val_label, val_env, w_sample=False)
        
        test_data, tdata, test_label, tlabel = rdr.read_demo_record(test_file, require_bp = True, whiten=True, debase = ('db' in norm_mode))
        test_data = np.concatenate((test_data, tdata),axis=0)
        test_label = np.concatenate((test_label, tlabel),axis=0)
        if 'data' in norm_mode:
            test_data = loader.standard(test_data)
        test_data = np.transpose(np.expand_dims(test_data, axis=1), (0, 1, 3, 2))
        
        '''here we are going to train'''
        for i in range(1):
            mode = 'train'
            modelFE, modelTask, modelEnv = EEGNetFE('LeakyReLU'), TaskNet(num_classes = 3), EnvNet()
            modelFE.to(device) 
            modelTask.to(device) 
            modelEnv.to(device)
            
            modelEnvClass = EnvClassNet(num_classes = len(np.unique(train_env)))
            modelEnvClass.to(device)
            
            optTask = optim.SGD(list(modelFE.parameters()) + list(modelTask.parameters()), lr = args.lr, momentum = args.momentum) #
            optEnv = optim.SGD(modelEnv.parameters(), lr = args.lr2, momentum = args.momentum) #
            #optEnv = optim.SGD(list(modelFE.parameters()) + list(modelEnv.parameters()), lr = args.lr2, momentum = args.momentum)
            #optEnvClass = optim.SGD(modelEnvClass.parameters(), lr = args.lr)
            optEnvClass = optim.SGD(list(modelEnvClass.parameters())+list(modelEnv.parameters()), lr = args.lr)
            #optEnvClass = optim.SGD(list(modelEnvClass.parameters()) + list(modelEnv.parameters()) +list(modelFE.parameters()), lr = args.lr2*0.1)
            
            correct_tr_Task, correct_tr_Env, correct_val_Task, correct_val_Env, correct_te_Task = [], [], [], [], []
            loss_tr_Task, loss_tr_Env, loss_val_Task, loss_val_Env, loss_tr_Alpha, loss_te_Task = [], [], [], [], [], []
            best = []
            decay_times = 1
            for epoch in tqdm(range(args.epochs)):
                c_val_Task, c_val_Env, correct_ori, best, c_tr_Task, c_tr_Env, l_tr_Task, l_tr_Env, l_val_Task, l_val_Env, l_tr_Alpha = train_model(modelFE, modelTask, modelEnv, epoch, train_loader, val_loader, correct_ori, best, optTask, optEnv, modelEnvClass, optEnvClass)
                correct_val_Task.append(c_val_Task)
                correct_val_Env.append(c_val_Env)
                correct_tr_Task.append(c_tr_Task)
                correct_tr_Env.append(c_tr_Env)
                loss_tr_Task.append(l_tr_Task)
                loss_tr_Env.append(l_tr_Env)
                loss_val_Task.append(l_val_Task)
                loss_val_Env.append(l_val_Env)
                loss_tr_Alpha.append(l_tr_Alpha)
                
                c_te_Task, l_te_Task, _ = finalTest(modelFE, modelTask, test_data, test_label)
                correct_te_Task.append(c_te_Task*100)
                loss_te_Task.append(l_te_Task)
                
                '''add cross subject testing every epoch to check overfitting or not?'''
                
                '''
                if epoch > args.epoch_change:
                    decay_times = adjust_lr_rate(optTask, loss_tr_Task, decay_times)
                    print('lr_decay_times' + str(decay_times))
                '''
                if epoch == args.epoch_change:
                    get_best_model(modelFE, modelTask, modelEnv, best[0], best[1], best[2])
                
                    val_accuracy, val_loss, cmt = finalTest(modelFE, 
                                                            modelTask, test_data, test_label)
        
                    print('-------final-------')
                    print('testing accuracy before de-env {}, {} : {}%'.format(norm_mode, test_file[0], str(val_accuracy)))
                    print(cmt)
                    accWithEnv.append(val_accuracy)
                
            x = np.arange(0,args.epochs)
                          
            ax = plt.subplot(111)
            ax.plot(x,correct_tr_Task,label='Task training acc') # + str(train_file)
            ax.plot(x,correct_tr_Env,label='Env training acc')
            ax.plot(x,correct_val_Task,label='Task validation acc')
            ax.plot(x,correct_val_Env,label='Env validation acc')
            ax.plot(x,correct_te_Task,label='Task test acc')
            ax.legend()
            ax.grid(True)
            plt.title('Accuracy (3 tasks) (' + str(len(np.unique(train_env))) + ' envs)')
            plt.show()
            
            ax = plt.subplot(111)
            ax.plot(x,loss_tr_Task,label='Task train loss (CE)')
            ax.plot(x,loss_tr_Env,label='Env train loss (tripletDisc)')
            ax.plot(x,loss_tr_Alpha,label='Anti-Env train loss (tripletKLD)')
            ax.plot(x,loss_val_Task,label='Task val loss (CE)')
            ax.plot(x,loss_val_Env,label='Env val loss (CE)')
            ax.legend()
            ax.grid(True)
            plt.title('loss')
            plt.show()
        
        get_best_model(modelFE, modelTask, modelEnv, best[0], best[1], best[2])
         
        val_accuracy, val_loss, cmt = finalTest(modelFE, modelTask, test_data, test_label)
        
        print('-------final-------')
        print('testing accuracy {}, {} : {}%'.format(norm_mode, test_file[0], str(val_accuracy)))
        print(cmt)
        accWOEnv.append(val_accuracy)
        
