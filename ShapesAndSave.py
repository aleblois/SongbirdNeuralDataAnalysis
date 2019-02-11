# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 14:20:57 2019

@author: eduar

-->> This is the third script of the series to obtain data from Spike2 files. 
This script will generate .txt files with the spikeshapes (Initial Time, Final Time, Analog points inside Window).

Run this script considering the series (file to be read has to be defined in ReadAndPlot.py)
"""
# Necessary Packages and Files
from CreateAndSave import *

import os

# Create and save the spikeshapes
windowsize=150

for i in range(n_spike_trains):
    Chprov = data.list_units[i].annotations["id"]
    Label = Chprov.split("#")[1]
    channel = np.loadtxt(Chprov+".txt")
    print(Chprov)
    print("Starting to get the spikeshapes... Grab a book or something, this might take a while!")
    a=np.array([],int) 
    x=np.empty([1,windowsize+2],int)
    for j in range(len(channel)):
        a= np.where(time >= channel[j])[0][0]
        analogtxt=analog[1][a:a+windowsize].reshape(1,windowsize)
        y = np.array([[a],[a+windowsize]], np.int32).reshape(1,2)
        res = append(y,analogtxt).reshape(1,-1)
        x=np.append(x,res, axis=0)
    b=x[1:]    
    print("\n" + "Voilà!")   
    np.savetxt("SpikeShape#"+Label+".txt", b, header="First column = Initial Time; Second column = Final Time; Third Column = First Spike Shape value, etc")
    #Plot an example
    window1=int(b[0][0])
    window2=int(b[0][1])
    figure()
    plot(time[window1:window2],analog[1][window1:window2])
    tight_layout()
    show()