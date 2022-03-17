#M-AI-AN Main Program
#Used to train M-AI-AN
#Used to classify images presented to M-AI-AN

#Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models # add models to the list
from torchvision.utils import make_grid
import os
import datetime

import numpy as np
import pandas as pd

#Model Definition
class ConvolutionalNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 3, 1)
        self.conv2 = nn.Conv2d(6, 16, 3, 1)
        self.fc1 = nn.Linear(47*47*16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 449)

    def forward(self, X):
        #Define dropout
        gendropout = nn.Dropout(p=0.1)
        detdropout = nn.Dropout(p=0.2)
        
        X = F.relu(self.conv1(X))
        X = F.avg_pool2d(X, 2, 2)
        X = gendropout(X)
        X = F.relu(self.conv2(X))
        X = F.avg_pool2d(X, 2, 2)
        X = detdropout(X)
        X = X.view(-1, 47*47*16)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)
        return F.log_softmax(X, dim=1)

MAiAn = ConvolutionalNetwork()
MAiAn

#Loss and optimization
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(MAiAn.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.9)


#Training Code
#----------------------------------------------
root = os.getcwd()
#Data Import
#Tensorize code
tensorize = transforms.Compose([
            transforms.ToTensor()
        ])

#import data
train_data = datasets.ImageFolder(root + '\\TestData\\train', transform = tensorize)
test_data = datasets.ImageFolder(root + '\\TestData\\test', transform = tensorize)

#Define classes
class_names = test_data.classes

#Define loaders
train_loader = DataLoader(train_data, batch_size=2, shuffle=True)
test_loader = DataLoader(test_data, batch_size=2, shuffle=False)

#Time stamp code
import time
start_time = time.time()

#Training settings and variables
epochs = 5
train_losses = []
test_losses = []
train_correct = []
test_correct = []

#Logging code
traininglog = open('TrainingLog.csv','w')
epoch_list = []
batch_list = []
loss_list = []
accuracy_list = []

for i in range(epochs):
    trn_corr = 0
    tst_corr = 0
    
    ##DEBUGGING CODE
    # Run the training batches
    for b, (X_train, y_train) in enumerate(train_loader):
        b+=1
        
        # Apply the model
        y_pred = MAiAn(X_train)  # we don't flatten X-train here
        loss = criterion(y_pred, y_train)
 
        # Tally the number of correct predictions
        predicted = torch.max(y_pred.data, 1)[1]
        batch_corr = (predicted == y_train).sum()
        trn_corr += batch_corr
        
        # Update parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        
        # Print interim results
        if b%200 == 0:
            curr_lr = optimizer.param_groups[0]['lr']
            print(f'Time: {datetime.datetime.now().strftime("%H:%M:%S")}  epoch: {i:2}  batch: {b:4} [{2*b:6}/{str(len(train_data))}]  loss: {loss.item():10.8f}  accuracy: {trn_corr.item()*100/(10*b):7.3f}% LRate:{curr_lr}')
            epoch_list.append(i)
            batch_list.append(b)
            loss_list.append(loss.item())
            accuracy_list.append(trn_corr.item()*100/(10*b))
            
        
    train_losses.append(loss)
    train_correct.append(trn_corr)
        
    # Run the testing batches
    with torch.no_grad():
        for b, (X_test, y_test) in enumerate(test_loader):

            # Apply the model
            y_val = MAiAn(X_test)

            # Tally the number of correct predictions
            predicted = torch.max(y_val.data, 1)[1] 
            tst_corr += (predicted == y_test).sum()
            
    loss = criterion(y_val, y_test)
    #Scheduler Code
    scheduler.step()
    test_losses.append(loss)
    test_correct.append(tst_corr)
        
print(f'\nDuration: {time.time() - start_time:.0f} seconds') # print the time elapsed   

#Log training results and export to file
traininglog.write("Training log - " + str(start_time) + "\n")
traininglog.write("Epoch" + "," + "Batch" + "," + "Loss" + "," + "Accuracy" + "\n")
for pos, val in enumerate(train_losses):
    traininglog.write(str(epoch_list[pos]) + "," + str(batch_list[pos]) + "," + str(loss_list[pos]) + "," + str(accuracy_list[pos]) + "\n")
traininglog.close()    
    

### EVALUATE Test Data ###
# Extract the data all at once, not in batches
test_load_all = DataLoader(test_data, batch_size=100, shuffle=False)

with torch.no_grad():
    correct = 0
    for X_test, y_test in test_load_all:
        y_val = MAiAn(X_test)  # we don't flatten the data this time
        predicted = torch.max(y_val,1)[1]
        correct += (predicted == y_test).sum()
print(f'Test accuracy: {correct.item()}/{len(test_data)} = {correct.item()*100/(len(test_data)):7.3f}%')

torch.save(MAiAN.state_dict(), 'M-AI-ANModel.pt')

#MAiAn.load_state_dict(torch.load('M-AI-ANModel.pt'))

#Image classifier
# x = 3167
# MAiAn.eval()
# with torch.no_grad():
    # new_pred = MAiAn(test_data[x][0].view(1,3,100,100)).argmax()
# print(f'Predicted value: {new_pred.item()} {class_names[new_pred.item()]}')