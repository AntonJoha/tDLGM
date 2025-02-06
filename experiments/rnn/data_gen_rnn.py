from parameter_tuning import train_model
import torch
import torch.optim as optim
import torch.nn as nn
from parse_data import get_data, get_modified_values, get_binary_values, make_data_scalar
import numpy as np
from evaluation import evaluate_model
import random

class Datagen():
    
    def __init__(self, device):
        
        self.df = get_data()
        self.device = device
        self.cache_true = {}
        self.cache_generated = {}
        self.cache_test = {}
    
    def get_true_data(self, seq_len):
        if seq_len in self.cache_true:
            return self.cache_true[seq_len]
        x,y = self.make_data(self.df, self.device, seq_len)
        self.cache_true[seq_len] = self.true_data(x,y,self.device)
        
        self.get_generated_data(seq_len) # generate it as well, to save time later, don't care for result here

        return self.cache_true[seq_len]
        
    def get_generated_data(self, seq_len, variance=0.01, probability=0.1):
        name = str(seq_len) + str(variance) + str(probability)
        if name in self.cache_generated:
            return self.cache_generated[name]
        x,y = self.make_data(self.df, self.device, seq_len)
        self.cache_generated[name] = self.feature_engineering(x,y, self.device, variance, probability)
        self.get_true_data(seq_len) # generate it as well, to save time later, don't care for result here
        return self.cache_generated[name]
    
    def make_data(self, df, device, seq_len):

        x_train, y_train = [], []
        prev = []
        m = df.max()[0]
        #print(df)
        for row in df.values:
        
            if len(prev) < seq_len:
                before = [0]*(seq_len - len(prev))
                for a in prev:
                    before.append(a)
                x_train.append(torch.tensor(before,device=device))
            else:   
                x_train.append(torch.tensor(prev[-seq_len:],device=device))
            y_train.append(torch.tensor(row[0]/m,device=device))
            prev.append(row[0]/m)
        return torch.stack(x_train[:-500]).to(device),torch.stack(y_train[:-500]).to(device)

    def make_test_data(self, df, device, seq_len):

        x_train, y_train = [], []
        prev = []
        m = df.max()[0]
        #print(df)
        for row in df.values:
        
            if len(prev) < seq_len:
                before = [0]*(seq_len - len(prev))
                for a in prev:
                    before.append(a)
                x_train.append(torch.tensor(before,device=device))
            else:   
                x_train.append(torch.tensor(prev[-seq_len:],device=device))
            y_train.append(torch.tensor(row[0]/m,device=device))
            prev.append(row[0]/m)
        return torch.stack(x_train[-500:]).to(device),torch.stack(y_train[-500:]).to(device)
    
    
    
    def get_test_data(self, seq_len):
        if seq_len in self.cache_test:
            return self.cache_test[seq_len]
        x,y = self.make_test_data(self.df, self.device, seq_len)
        self.cache_test[seq_len] = self.true_data(x,y,self.device)
        
        self.get_generated_data(seq_len) # generate it as well, to save time later, don't care for result here

        return self.cache_test[seq_len]

    
    def true_data(self, X,Y, device):
        new_x, new_y = [], []
        for x, y in zip(X,Y):
            for i in range(1):
                curr_x = []
                for x_part in x:
                    to_add = x_part
                    if to_add > 1:
                        to_add = 1
                    if to_add < 0:
                        to_add = 0
                    curr_x.append([to_add])
                new_x.append(torch.tensor(curr_x,device=device))
                new_y.append(torch.tensor([y]).to(device))
        
        return torch.stack(new_x).float().to(device),torch.stack(new_y).float().to(device)


    def feature_engineering(self, X,Y, device, variance=0.01, probability=0.1):
        new_x, new_y = [], []
        count= 5
        for x, y in zip(X,Y):
            for i in range(count):
                curr_x = []
                for x_part in x:
                    if random.random() < probability:
                        to_add = x_part + np.random.normal(loc=0, scale=variance)
                    else:
                        to_add = x_part
                    if to_add > 1:
                        to_add = 1
                    if to_add < 0:
                        to_add = 0
                    curr_x.append([to_add])
                new_x.append(torch.tensor(curr_x,device=device))
                new_y.append(torch.tensor([y]).to(device))

        return torch.stack(new_x).float().to(device), torch.stack(new_y).float().to(device)

    
    