# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 14:21:26 2020

@author: shuyu
"""

import torch
import torch.nn as nn
import numpy as np
import pickle
import argparse
import torch.optim as optim
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from metric import *
from model import GanDTI

parser = argparse.ArgumentParser()
parser.add_argument("--lr", help = "learning rate", type = float, default = 0.001)
parser.add_argument("--ld", help = "learning rate decay", type = float, default = 0.5)
parser.add_argument("--epoch", help = "epoch", type = int, default = 30)
parser.add_argument("--features", help = "feature dimension", type = int, default = 40)
parser.add_argument("--GNN_depth", help = "gnn layer number", type = int, default = 3)
parser.add_argument("--MLP_depth", type = int, default = 2)
parser.add_argument('--weight_decay', type=float, default=1e-6, help='Weight decay (L2 loss on parameters).')
parser.add_argument("--mode", type = str, help = "regression or classification", default = 'classification')
parser.add_argument("--dataset", type = str, default = 'human')
args = parser.parse_args()

#load GPU or CPU
if torch.cuda.is_available():
    device = torch.device('cuda')
    print('program uses GPU. start now')
else:
    device = torch.device('cpu')
    print('program uses CPU')
    
def loadNpy(fileName, dtype):
    tensor = [dtype(data).to(device) for data in np.load(fileName + '.npy', allow_pickle= True)]
    return tensor

def loadPickle(fileName):
    with open(fileName, 'rb') as f:
        return pickle.load(f)

def load_data(dataset, mode):
    #load preprocessed data
    data_file = './'+ dataset +'/'
    compounds = loadNpy(data_file + 'compounds', torch.LongTensor)
    adjacencies = loadNpy(data_file +'adjacencies', torch.FloatTensor)
    proteins = loadNpy(data_file +'proteins', torch.LongTensor)
    fingerprintDict = loadPickle(data_file +'fingerprint.pickle')
    wordDict = loadPickle(data_file +'wordDict.pickle')
    compound_len = len(fingerprintDict)
    protein_len = len(wordDict)
    if mode == 'classification':
        interactions = loadNpy(data_file+ 'interactions', torch.LongTensor)
    elif mode == 'regression':
        interactions = loadNpy(data_file +'interactions', torch.FloatTensor)
    #zip to form the dataset and perform train test split
    dataset = list(zip(compounds, adjacencies, proteins, interactions))
    np.random.seed(seed=1234)
    np.random.shuffle(dataset)
    trainNumber = int(len(dataset)*0.8)
    trainData = dataset[:trainNumber]
    testData = dataset[trainNumber:]
    return trainData, testData, compound_len, protein_len #need revision

def train(dataset, mode, optimizer):
    np.random.shuffle(dataset)
    for data in dataset:
        optimizer.zero_grad()
        output = model(data)
        output.backward()
        optimizer.step()
    return output


def test_data_process(test_dataset):
    labels, predictions, scores = [], [], []
    for data in test_dataset:
        (label, predict, score) = model(data, train=False)
        labels.append(label)
        predictions.append(predict)
        scores.append(score)
    return labels, predictions, scores
        
def test_regression(test_dataset):
    labels, predictions = [], []
    for data in test_dataset:
        (label, predict, score) = model(data, train=False)
        labels.append(label)
        predictions.append(predict)
        #scores.append(score)
    RMSE = rmse(labels, predictions)
    Pearson = pearson(labels, predictions)
    return RMSE, Pearson

def test_classification(dataset):
    labels, predictions, scores = test_data_process(dataset) 
    AUC = roc_auc_score(labels, scores)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    predictions_array = np.array(predictions)
    labels_array = np.array(labels)
    roce1 = getROCE(predictions_array,labels_array, 0.5)
    roce2 = getROCE(predictions_array,labels_array, 1)
    roce3 = getROCE(predictions_array,labels_array, 2)
    roce4 = getROCE(predictions_array,labels_array, 5)
    return AUC, precision, recall, roce1, roce2, roce3, roce4

#load the train and test data
train_data, test_data, compound_len, protein_len = load_data(args.dataset, args.mode)
#load the model
torch.manual_seed(0)
model = GanDTI(compound_len, protein_len, args.features, args.GNN_depth, args.MLP_depth, args.mode
            ).to(device)
#optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=args.lr, 
            weight_decay=args.weight_decay, amsgrad=True)

def main(): 
    for epoch in range(args.epoch):
        model.train()    
        lossTrain = train(train_data, args.mode, optimizer)    
        model.eval()
        if args.mode == 'regression':
            rmse_test, pearson_test = test_regression(test_data)   
            print('Epoch:{:03d}'.format(epoch+1),
                  'train loss:{:.5f}'.format(lossTrain),
                  'rmse:{:5}'.format(str(rmse_test)),
                  'pearson:{:5}'.format(str(pearson_test))
                  )
        elif args.mode == 'classification':
            AUC_test, precision_test, recall_test, roce1, roce2, roce3, roce4 = test_classification(test_data)   
            print('Epoch:{:03d}'.format(epoch+1),
                  'train loss:{:.6f}'.format(lossTrain),
                  'AUC test:{:.6f}'.format(AUC_test),
                  'precision:{:4f}'.format(precision_test),
                  'recall:{:4f}'.format(recall_test),
                  'roce1', roce1,
                  'roce2', roce2,
                  'roce3', roce3,
                  'roce4', roce4
                  )

if __name__ == '__main__':
    main()

