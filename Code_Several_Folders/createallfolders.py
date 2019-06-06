# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 20:58:16 2019

@author: eduar
"""

#import functionspop
import glob
import os
import pylab as py
import numpy as np

## Some key parameters:
fs=32000 #Sampling Frequency
window_size=100 #For envelope

##Gets all the folders
current_dir = os.getcwd()
listsubdirs=[]
for item in os.listdir(current_dir):
    if os.path.isdir(os.path.join(current_dir, item)):
        if item == "__pycache__":
            continue
        else:
            listsubdirs+=[item]
    
for item in range(len(listsubdirs)):
    print("Working on folder:" + listsubdirs[item])
    os.chdir(listsubdirs[item])
    l=os.getcwd()
    if glob.glob("*.smr") != []:
        file = glob.glob("*.smr")[0]
    else:
        continue
    
    #Check if file with units and windows is in the folder
    if not os.path.isfile("unitswindow.txt"):
        info_list = [] 
        unit=""
        windowbeg=""
        windowend=""
        print("Enter the useful units and windows (press * to terminate) ")
        while unit != "*":
            unit = input("Unit?")
            windowbeg= input("Beginning of Window [in seconds]?")
            windowend= input("End of Window [in seconds]?")
            if(unit!="*"):
                info_list+=[[unit,windowbeg,windowend]]
        np.savetxt("unitswindow.txt", info_list, fmt="%s")
    
    #Check if file with silent period and npys is in the folder
    if not os.path.isfile("info.txt"):
        info = []
        songfile=input("Name of file with the song ")
        raw=input("Name of file with Raw LFP ")
        rawfiltered=input("Name of the file with Filtered LFP ")
        silbeg= input("Beginning of silent period [in seconds]? ")
        silend= input("End of silent period [in seconds]? ")
        info+=[[songfile],
               [raw],
               [rawfiltered],
               [silbeg],
               [silend]]
        np.savetxt("info.txt", info, fmt="%s")    
        
    ## Starts creating the files and folders
    functionspop.createsave(file)
    with open("..\\unitswindow.txt", "r") as datafile:
        s=datafile.read().split()[0::3]
    for i in range(len(s)):
        os.mkdir("Unit_"+s[i])
        os.chdir("Unit_"+s[i])
        os.mkdir("Results")
        os.mkdir("Figures")
        os.chdir("..")
    
    #Gives you the option to plot the analogs/spiketrains
    if input("Want to run plotplots?").lower() == "y":
        smr= l + "\\" + file
        functionspop.plotplots(smr)
        py.waitforbuttonpress(60)
    else:
        pass
    
    
    infos= open("..\\info.txt", "r")
    infosr= infos.read().splitlines()
    songfile=infosr[0]
    raw=infosr[1]
    rawfiltered=infosr[2]
    basebeg=int(infosr[3])
    basend=int(infosr[4])
    motifile="..\\" +"labels.txt"
    smr="..\\" + file
    ### Get the tones/sybcuts for further analysis
    if not os.path.isfile("CheckSylsFreq.txt"):
        fich= open("CheckSylsFreq.txt", mode="w+")
        fich.close()
    
    ### Get LFP downsampled
    print("Obtaining the Downsampled LFP..")
    functionspop.lfpdown(raw, fs)
    
    ## Get Spikeshapes
    print("Obtaining the SpikeShapes..")
    functionspop.spikeshapes(smr, raw, rawfiltered)
    
    ### Get the tones/sybcuts for further analysis
    if not os.path.isfile("..\\MeanA.txt"):
        functionspop.gettones(songfile,motifile,fs, window_size)
        
    os.chdir(current_dir)
    

   
