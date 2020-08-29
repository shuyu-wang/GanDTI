# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 14:18:27 2020

@author: shuyu
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class GanDTI(nn.Module):
    def __init__(self, compound_len, protein_len, features, GNN_depth, MLP_depth, mode):
        super(GanDTI, self).__init__()
        
        self.mode = mode
        self.embed_compound = nn.Embedding(compound_len, features)
        self.embed_protein = nn.Embedding(protein_len, features)
        self.GNN_depth = GNN_depth
        self.GNN = nn.ModuleList(nn.Linear(features, features) for i in range(GNN_depth))
        self.W_att = nn.Linear(features, features)
        self.MLP_depth = MLP_depth
        self.MLP = nn.ModuleList(nn.Linear(features*2, features*2) for i in range(self.MLP_depth))
        self.classification_out = nn.Linear(2*features, 2)
        self.regression_out = nn.Linear(2*features, 1)
        self.dropout = nn.Dropout(0.5)
        
    def Attention(self, compound, protein):
        compound_h = torch.relu(self.W_att(compound))
        protein_h = torch.relu(self.W_att(protein))
        mult = compound @ protein_h.T
        weights = torch.tanh(mult)
        protein = weights.T * protein_h
        protein_vector = torch.unsqueeze(torch.mean(protein, 0), 0)
        return protein_vector
        
    def GraphNeuralNet(self, compound, A, GNN_depth):
        residual = compound
        for i in range(GNN_depth):
            compound_h = F.leaky_relu(self.GNN[i](compound))
            compound = compound + torch.matmul(A,compound_h)
        
        compound = compound + residual
        compound_vector = torch.unsqueeze(torch.mean(compound, 0), 0)
        return compound_vector
    
    def MLP_module(self, compound_protein, MLP_depth, mode):
        for i in range(MLP_depth):
            compound_protein = torch.relu(self.MLP[i](compound_protein))
        compound_protein = self.dropout(compound_protein)
        if mode == 'classification':
            out = self.classification_out(compound_protein)
        elif mode == 'regression':
            out = self.regression_out(compound_protein)
        
        return out
    
    def forward(self, data):
        compound, A, protein = data
        
        compound_embed = self.embed_compound(compound)
        compound_vector = self.GraphNeuralNet(compound_embed, A, self.GNN_depth)
        
        protein_embed = self.embed_protein(protein)
        protein_vector = self.Attention(compound_vector, protein_embed)
        
        compound_protein = torch.cat((compound_vector, protein_vector), 1)
        out = self.MLP_module(compound_protein, self.MLP_depth, self.mode)
        return out
    
    def __call__(self, data, train=True):

        feature_data, label_data = data[:-1], data[-1]
        predict_data = self.forward(feature_data)

        if train:
            if self.mode == 'classification':
                output = F.cross_entropy(predict_data, label_data)
            elif self.mode == 'regression':
                loss = nn.MSELoss()
                output = loss(predict_data[0].float(), label_data.float())
            return output
        else:
            labels = label_data.to('cpu').data.numpy()
            if self.mode == 'classification':
                predict_data = torch.sigmoid(predict_data).to('cpu').data.numpy()
                predict_result = list(map(lambda x: np.argmax(x), predict_data))
                predict_score = list(map(lambda x: x[1], predict_data))
            elif self.mode == 'regression':
                predict_result = predict_data[0].to('cpu').data.numpy()
                predict_score = predict_result
            return labels, predict_result, predict_score
    
    
        
        