#M-AI-AN Glyph Data Analyzer
#Analyzes glyph import data to be used for M-AI-AN training

#Imports
import os
import numpy as np
import pandas as pd
import DataSegments as DS
from torchvision import datasets, transforms
from PIL import Image, ImageOps

#Function area
#Size Calculation code
def imagesizemetacalc(imagelist):
    imagesizemeta = []
    for image in imagelist:
        with Image.open(image) as image:
                try:
                    imagesizemeta.append(image.size)
                except:
                    print("Error with image: " + image)
    return imagesizemeta

#Segment printing code
def segmentlist():
    print("\n")
    print("List of current Data Segments")
    print("---------------------------------------")
    print("\n")
    for i in DS.MED.keys():
        print(i + " : " + str(DS.MED.get(i)))
    print("\n")
    print("\n")
    print("-@-----------------------------------@-")
    

#Start
print("---------------------------------------")
print("@@@ M-AI-AN Glyph Data Analyzer 1.0 @@@")
print("---------------------------------------")

set = 0
while set == 0:
    print("Please choose from the following options:")
    print("1 - Image statistics")
    print("2 - Image padding")
    print("3 - Image homogenisation transform")
    print("4 - Image segmentation")
    print("5 - WIP - Edit image segments")
    print("6 - Training Data creation")
    print("X - Exit")
    choice = input()

    #Image import 
    imagelist = []
    for folder, subfolders, filenames in os.walk("TestData/"):
        for image in filenames:
            imagelist.append(folder + image)

    #Image size statistics
    if choice == "1":
        #Creates list of size metadata
        imagesizemeta = imagesizemetacalc(imagelist)

        #Create pandas dataframe for analysis
        df = pd.DataFrame(imagesizemeta)
        
        #Run summary statistics on image widths
        print(str(df))
        print("\n")
        print("Width Metadata: " + "\n" + str(df[0].describe()))
        print("\n")
        print("Height Metadata: " + "\n" + str(df[1].describe()))
        print("\n")
        print("-@-----------------------------------@-")

    #Image manipulation
    if choice == "2":
        for image in imagelist:
            workimage = Image.open(image)
            workimage = workimage.convert('L')
            pixels = workimage.load()
            dimension = workimage.size[0] - workimage.size[1]
            
            #Adds additional padding depending on deficit on height/width
            if dimension > 0:
                nheight = workimage.size[0] 
                nwidth = workimage.size[1] + dimension
                nleft = 0
                ntop = round(dimension/2)
            elif dimension < 0:
                nheight = workimage.size[0] - dimension
                nwidth = workimage.size[1] 
                ntop = 0
                nleft = round(0 - dimension/2)
            
            #Adds additional whitespace around the glyphs
            if dimension != 0:
                #Use below if RGB implemented
                #paddedimage = Image.new(workimage.mode, (nwidth, nheight), (255,255,255))
                
                paddedimage = Image.new(workimage.mode, (nwidth, nheight), 255)
                paddedimage.paste(workimage,(nleft,ntop))
                #paddedimage = paddedimage.convert('RGB')
                paddedimage.save(image,quality=100)
             
        print("\n")
        print("Padding complete - please check image files")
        print("\n")
        print("-@-----------------------------------@-")
    
    if choice == "3":
        #Checks if you are ready to scale up images by comparing image sizes
        imagesizemeta = imagesizemetacalc(imagelist)
        df = pd.DataFrame(imagesizemeta)
        resizedim = int(np.percentile(df[1],50))
        if np.percentile(df[1],50) != np.percentile(df[0],50):
            print("Your image files are not correctly scaled - please try option 2")
            print("\n")
            print("---------------------------------------")
        else:
            for image in imagelist:
                workimage = Image.open(image)
                workimage = workimage.convert('L')
                resizedimage = ImageOps.scale(workimage, resizedim/workimage.size[0], resample=3)
                resizedimage.save(image,quality=100)

            print("\n")
            print("Image homogenisation scaling complete - please check image files")
            print("\n")
            print("-@-----------------------------------@-") 
            
            
           
    if choice == "4":
        segmentlist()

    
    if choice == "5":
        print("Work in progress")
    
    
    if choice == "6":
        #Lists segments to allow users to choose
        segmentlist()
        templist = []
        for i in DS.MED.keys():
            templist.append(i)
        print("Please type which segement/s you would like to create training data for :")
        segmentchoice = input()
        print("Please type how many samples for each glyph you want to create :")
        samplechoice = input()
        
        #Defines transforms to perform on the dataset
        transform = transforms.Compose([
            transforms.RandomRotation(degrees=(-20,20),fill=[255]),
            transforms.RandomPerspective(distortion_scale=0.3, p=0.3,fill=[255]),
            transforms.ToTensor()
        ])
        
        try:
            os.mkdir("TestData/train")
            os.mkdir("TestData/test")
        except:
            print("\n")
            print("Test and train folders already present")
            print("\n")
        
        #Starts data creation process
        for i in DS.MED.keys():
            if i == segmentchoice:
                Ulim = DS.MED.get(i)[1]
                Llim = DS.MED.get(i)[0]
                for pos, image in enumerate(imagelist):
                    print(image[9:15])
                    
                    if int(image[9:15]) >= int(Llim) and int(image[9:15]) <= int(Ulim):
                        
                        print("\n")
                        print("---------------------------------------")
                        print(image)
                        
                        try:
                            os.mkdir("TestData/test/" + str(image[9:15]))
                            os.mkdir("TestData/train/" + str(image[9:15]))

                        except:
                            print("Directory already exists: " + str(image[10:15]))
                            
                        workimage = Image.open(image)
                        workimage = workimage.convert('L')
                        for sampleno in range(int(samplechoice)):
                            #Alters image within random parameters
                            alteredimage = transform(workimage)
                            
                            #Converts into pillow and saves to file
                            finalimage = transforms.ToPILImage()(alteredimage.to('cpu'))
                            
                            #Saves to test folder
                            if sampleno <= int(samplechoice) * 0.3:
                                print(str(image[0:15]) + "/test/" + str(sampleno) + ".jpg")
                                finalimage.save("TestData/test/" + str(image[9:15]) + "/" + str(sampleno) + ".jpg",quality=100)
                            
                            #Saves to train folder
                            elif sampleno > int(samplechoice) * 0.3:
                                print(str(image[0:15]) + "/train/" + str(sampleno) + ".jpg")
                                finalimage.save("TestData/train/" + str(image[9:15]) + "/" + str(sampleno) + ".jpg",quality=100)

        print("\n")
        print("Image creation complete - please check image files")
        print("\n")
        print("-@-----------------------------------@-") 

    
    if choice == "X":
        print("---------------------------------------")
        print("           Exiting program")    
        print("-@-----------------------------------@-")
        set = 1
    