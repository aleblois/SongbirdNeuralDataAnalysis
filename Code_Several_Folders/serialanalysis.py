# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 20:58:16 2019

@author: eduar
"""
#Necessary modules
#import functionspop
import os
import pylab as py

## Some key parameters:
fs=32000 #Sampling Frequency
n_iterations=1000 #For bootstrapping
window_size=100 #For envelope
lags=100 #For Autocorrelation
alpha=0.05 #For P value
premot=0.05 # Premotor window
binwidth=0.02 #for PSTH

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
    os.chdir(listsubdirs[item]+"\\Files")    
    
    infos= open("..\\info.txt", "r")
    infosr= infos.read().splitlines()
    songfile=infosr[0]
    raw=infosr[1]
    rawfiltered=infosr[2]
    basebeg=int(infosr[3])
    basend=int(infosr[4])
    motifile="..\\labels.txt"
    
    #### Analysis
    with open("..\\unitswindow.txt", "r") as datafile:
            s=datafile.read().split()[0::3]
    for i in range(len(s)):
        spikefile= s[i]
        
        #PSTH
        if input("PTSH?").lower() == "y" or "":
            functionspop.psth(spikefile+".txt", motifile, fs, basebeg, basend, binwidth)
            os.chdir("Unit_"+spikefile)
            py.savefig("PSTH.tif")
            py.close()
            os.chdir("..")
        else:
            pass
        
        #ISI
        if input("ISI?").lower() == "y" or "":
            functionspop.ISI(spikefile+".txt")
            os.chdir("Unit_"+spikefile)
            py.savefig("ISI.tif")
            py.close()
            os.chdir("..")
        else:
            pass
        
        #Correlation Duration
        if input("Correlation Duration?").lower() == "y" or "":
            os.chdir("Unit_"+spikefile)
            functionspop.corrduration("..\\"+spikefile+".txt", "..\\"+motifile, n_iterations, fs, alpha)
            os.chdir("..")
        else:
            pass
        
        
        #Correlation Spectral Entropy
        if input("Correlation Spectral Entropy?").lower() == "y" or "":
            os.chdir("Unit_"+spikefile)
            functionspop.corrspectral("..\\"+ songfile, "..\\"+motifile, fs, "..\\"+spikefile+".txt", window_size, n_iterations, alpha, premot)
            os.chdir("..")
        else:
            pass
        
        #Correlation Pitch
        if input("Correlation Pitch?").lower() == "y" or "":
            os.chdir("Unit_"+spikefile)
            functionspop.corrpitch("..\\"+ songfile, "..\\"+motifile, lags, window_size, fs, "..\\"+spikefile+".txt", n_iterations, alpha, premot)
            os.chdir("..")
        else:
            pass
        
        #Correlation Amplitude
        if input("Correlation Amplitude?").lower() == "y" or "":
            os.chdir("Unit_"+spikefile)
            functionspop.corramplitude("..\\" + songfile, "..\\"+motifile, fs, "..\\"+spikefile+".txt", window_size, n_iterations, alpha, premot)
            os.chdir("..")
        else:
            pass

    os.chdir(current_dir)