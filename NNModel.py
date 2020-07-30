#!/usr/bin/env python
# coding: utf-8

# In[3]:


import json
import pandas as pd
import sqlite3
import torch
from torch import nn
import numpy as np
import string
import os
from pathlib import Path
import torch.nn.functional as F


# In[ ]:


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1=nn.Linear(192, 100)
        self.fc2=nn.Linear(100, 60)
        self.fc3=nn.Linear(60, 40)        
        self.fc4=nn.Linear(40, 3)
    
    def forward(self, x):
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=F.relu(self.fc3(x))
        x=self.fc4(x)
        return F.log_softmax(x, dim=1)


# In[ ]:


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        
        self.hidden_dim = hidden_dim 
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :]) 
        return out

input_dim = 48
hidden_dim = 100
layer_dim = 2
output_dim = 3
seq_dim = 4
num_epochs=100


# In[ ]:

if __name__=='__main__':
	
	train=np.load('/content/drive/My Drive/DL/DATA.npy')
	test=np.load('/content/drive/My Drive/DL/TEST.npy')
	valid=np.load('/content/drive/My Drive/DL/VALID.npy')
	
	traindf=pd.read_json('snli_1.0/snli_1.0_train.jsonl',encoding='utf-8',lines=True)
	validdf=pd.read_json('snli_1.0/snli_1.0_dev.jsonl',encoding='utf-8',lines=True)
	testdf=pd.read_json('snli_1.0/snli_1.0_test.jsonl',encoding='utf-8',lines=True)
	traindf['annotator_labels']=traindf['annotator_labels'].apply(lambda text: " ".join(text))
	validdf['annotator_labels']=validdf['annotator_labels'].apply(lambda text: " ".join(text))
	testdf['annotator_labels']=testdf['annotator_labels'].apply(lambda text: " ".join(text))
	traindf.drop_duplicates()
	validdf.drop_duplicates()
	testdf.drop_duplicates()
	traindf.query('gold_label!="-"',inplace= True)
	validdf.query('gold_label!="-"',inplace= True)
	testdf.query('gold_label!="-"',inplace= True)
	from sklearn import preprocessing
	encoder = preprocessing.LabelEncoder()
	train_y = encoder.fit_transform(traindf['gold_label'])
	valid_y= encoder.fit_transform(validdf['gold_label'])
	test_y= encoder.fit_transform(testdf['gold_label'])
	list(encoder.classes_)
	
	TRAIN=[]
	TEST=[]
	VALID=[]
	for x,y in zip(train,train_y):
	  TRAIN.append([x,y])
	for x,y in zip(test,test_y):
	  TEST.append([x,y])
	for x,y in zip(valid,valid_y):
	  VALID.append([x,y])
	
	train_loader=torch.utils.data.DataLoader(TRAIN,batch_size=1000,shuffle=True)
	test_loader=torch.utils.data.DataLoader(TEST,batch_size=1000,shuffle=True)
	valid_loader=torch.utils.data.DataLoader(VALID,batch_size=1000,shuffle=True)
	
	if torch.cuda.is_available():
	    device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc. 
	    print("Running on the GPU")
	else:
	    device = torch.device("cpu")
	    print("Running on the CPU")
	        
	net=Net().to(device)
	print(net)
	
	optimizer = torch.optim.Adam(net.parameters(),lr=0.0001)
	for epoch in range(100):
	    temp=0
	    for data in train_loader:
	        X , y= data
	        X,y=X.to(device),y.to(device)
	        net.zero_grad()
	        output = net(X.view(-1,192))
	        loss = F.cross_entropy(output,y)
	        temp+=loss
	        loss.backward()
	        optimizer.step()
	    print('TrainLoss %f : epoch %d' %(loss.data,epoch))
	    net.eval()
	    correct = 0
	    total = 0
	    Vloss=0
	    with torch.no_grad():
	        for a in valid_loader:
	            text, labels = a
	            text, labels=text.to(device), labels.to(device)
	            outputs = net(text.view(-1,192))
	            _, predicted = torch.max(outputs.data, 1)
	            Vloss += F.nll_loss(outputs,labels)
	            total += labels.size(0)
	            correct += (predicted == labels).sum().item()
	        print('ValidationLoss %f : epoch %d' %(Vloss.data/len(valid_loader),epoch))  
	        print('Accuracy of the network on the 9K validation samples: %d %%' % (
	      100 * correct / total))
	
	net.eval()
	correct = 0
	total = 0
	Vloss=0
	with torch.no_grad():
	        for a in test_loader:
	            text, labels = a
	            text, labels=text.to(device), labels.to(device)
	            outputs = net(text.view(-1,192))
	            _, predicted = torch.max(outputs.data, 1)
	            Vloss += F.nll_loss(outputs,labels)
	            total += labels.size(0)
	            correct += (predicted == labels).sum().item()
	        print('TEstLoss %f ' %(Vloss.data/len(test_loader)))  
	        print('Accuracy of the network on the 9K validation samples: %d %%' % (
	      100 * correct / total))


	
	
	for epoch in range(num_epochs):
	    iter = 0  
	    for i, (images, labels) in enumerate(train_loader):
	        images = images.view(-1, seq_dim, input_dim).requires_grad_()
	        optimizer.zero_grad()
	        outputs = model1(images)
	        loss = criterion(outputs, labels)
	        loss.backward()
	        optimizer.step()
	
	        iter += 1
	
	        if iter % 500 == 0:
	            correct = 0
	            total = 0
	            for images, labels in valid_loader:
	                images = images.view(-1, seq_dim, input_dim)
	                outputs = model1(images)
	                _, predicted = torch.max(outputs.data, 1)
	                total += labels.size(0)
	                correct += (predicted == labels).sum()
	            accuracy = 100 * correct / total
	            print('Iteration: {}. Loss: {}. Accuracy: {}'.format(iter, loss.item(), accuracy))
	            
	
	




