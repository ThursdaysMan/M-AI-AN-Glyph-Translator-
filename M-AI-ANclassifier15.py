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
from PIL import Image, ImageOps

import os
import datetime
import GlyLib as GL

import numpy as np
import pandas as pd

root = os.getcwd()

print("\n")
print("---------------------------------------")
print("@@@      M-AI-AN Classifier 15      @@@")
print("---------------------------------------")

set = 0
while set == 0:

    print("Please choose from the following options:")
    print("1 - Model Training")
    print("2 - Image Classification")
    print("3 - Model Details")
    print("X - Exit")
    choice = input()
    
    #Model Definition
    class ConvolutionalNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 6, 3, 1)
            self.conv2 = nn.Conv2d(6, 16, 2, 1)
            self.fc1 = nn.Linear(47*47*16, 360)
            self.fc2 = nn.Linear(360, 120)
            self.fc3 = nn.Linear(120, 1138)

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
    
    if choice == "1":
        print("-@-----------------------------------@-")
        print("Please name model export:")
        modelname = input()
        
        #Loss and optimization
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(MAiAn.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)


        #Training Code
        #----------------------------------------------
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

        torch.save(MAiAn.state_dict(), str(modelname) + ".pt")
        print("Training Complete")
        print("Model saved as: " +str(modelname) + ".pt" )
        print("\n")
        print("-@-----------------------------------@-")


    if choice == "2":
        #Image classifier
        
        print("Select Classification Option:")
        print("1 - Individual Test Data Classification")
        print("2 - Custom Image")
        print("3 - Batch Testing")
        print("X - Back to menu")
        classifychoice = input()
        
        #Tensorize code
        tensorize = transforms.Compose([
                        transforms.ToTensor()
                     ])
       
        
        #Load Model
        MAiAn.load_state_dict(torch.load('M-AI-ANfull.pt'))
        
        if classifychoice == "1":
            #Individual test classification
            
                     
            #import test data
            test_data = datasets.ImageFolder(root + '\\TestData\\test', transform = tensorize)
            
            #Define classes
            class_names = test_data.classes
            
            looper = 0
            while looper == 0:
                print("Please type the glyph type to use: ")
                glyphtypeinput = input()
                print("Please type the glyph image to use: ")
                individualglyphinput = input()
                x = -1
                f = 0
                n = 0
                
                for folder, subfolders, filenames in os.walk("TestData/test"):
                    x = x + 1
                    n = 0
                    #print("Folder :" + str(folder))
                    # print(subfolders)
                    # print("Filenames :" + str(filenames))
                    if folder.find(glyphtypeinput) > -1:
                        #print(folder[folder.find(glyphtypeinput):len(folder)])
                        f = x
                        #print(str(f))
                    for image in filenames:
                        n = n + 1
                        

                #Define Data Loader
                test_loader = DataLoader(test_data, batch_size=2, shuffle=False)

                #Begin Classification
                x = ((f - 1) * n) + int(individualglyphinput)
                MAiAn.eval
                with torch.no_grad():
                    new_pred = MAiAn(test_data[x][0].view(1,3,195,195)).argmax()
                print(f'Predicted value: {new_pred.item()} {class_names[new_pred.item()]} {GL.IdToWord.get("0" + str(class_names[new_pred.item()]))}')
                print("\n")

            
            print("\n")
            print("-@-----------------------------------@-")
        
        if classifychoice == "2":
            customchoice = input("Please give filename without extensions: ")
            
            #Sanitizes custome image for use in classifier
            workimage = Image.open(root + '\\CustomImage\\ClassifyArea\\' + customchoice + '.jpg')
            editedimage = ImageOps.fit(workimage, [195,195],centering=(0.5, 0.5))
            editedimage  = ImageOps.autocontrast(editedimage, cutoff=(0.05, 0.95), ignore = None, mask = None)
            print(editedimage.size)
            #editedimage  = ImageOps.grayscale(editedimage)
            
            CustomImage = tensorize(editedimage)
            
            #import test data
            test_data = datasets.ImageFolder(root + '\\TestData\\test', transform = tensorize)
            
            #Define classes
            class_names = test_data.classes
            
            MAiAn.eval
            with torch.no_grad():
                new_pred = MAiAn(CustomImage.view(1,3,195,195)).argmax()
            print(f'Predicted value: {new_pred.item()} {class_names[new_pred.item()]} {GL.IdToWord.get("0" + str(class_names[new_pred.item()]))}')
            print("\n")
            #Print List of probabilities
            with torch.no_grad():
                new_pred = torch.argsort(MAiAn(CustomImage.view(1,3,195,195)), dim=1,descending=True)
            print(f'Predicted values list:')
            #for i in range(len(new_pred[0])):
            for i in range(0,15):
                #print(len(new_pred[0]))
                #print(range(len(new_pred)))
                print("No: " + str(new_pred[0][i].item()) + " ID: " + str(class_names[new_pred[0][i].item()]))
                #print(GL.IdToWord.get("0" + str(class_names[new_pred[0][i].item()])))
            #print(str(new_pred))
            print("\n")
            print("-@-----------------------------------@-")
            
            
        if classifychoice == "3":    
            #Batch Testing
            
            #import data
            test_data = datasets.ImageFolder(root + '\\TestData\\test', transform = tensorize)
            #Tests against a large amount of data
            test_load_all = DataLoader(test_data, batch_size=100, shuffle=False)        
            #Define classes
            class_names = test_data.classes
            
            MAiAn.eval
            with torch.no_grad():
                correct = 0
                for X_test, y_test in test_load_all:
                    batch = 0
                    end = 0
                    for pos, val in enumerate(X_test):
                        end = end + 1
                        prediction = MAiAn(val.view(1,3,195,195)).argmax()
                        if prediction.item() == y_test[pos]:
                            batch = batch + 1
                    print("Batch Accuracy: " + str(batch) + "/" + str(end))
                    
                    correct = correct + batch
            print(f'Test accuracy: {correct}/{len(test_data)}')
            print("\n")
            print("-@-----------------------------------@-")
        
        if classifychoice == "X":
            print("Returning to menu")
            print("\n")
            print("-@-----------------------------------@-")
        
    if choice == "3":
        print(MAiAn)
        print("\n")
        print("-@-----------------------------------@-")
        
    if choice == "X":
        set = 1