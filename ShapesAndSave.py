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
windowsize=150 #Define here the number of points that suit your window (to get accurately the shape of the spike)

for i in range(n_spike_trains):
    Chprov = data.list_units[i].annotations["id"]
    Label = Chprov.split("#")[1]
    answer = input("You can get the spikeshapes of each channel or all channels. Use Each or All.") 
    if answer.lower()[0] == "e":
        for k in range(n_spike_trains):
            Chprov = data.list_units[k].annotations["id"]
            Label = Chprov.split("#")[1]
            answer2 = input("Nice! The next channel is:" + str(Chprov) + ". Should I create a file for it? [Y/n]")                     
            if answer2.lower() == "y":
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
                answer3= input("Do you want to see the plots? [Y/n]?")
                if answer3.lower() == "y":
                    print("Ok! Let's go!")
                    #Plot an example
                    window1=int(b[0][0])
                    window2=int(b[0][1])
                    figure()
                    plot(time[window1:window2],analog[1][window1:window2])
                    tight_layout()
                    show()
                else:
                    print("Ok, maybe next time..")
            else:
                print("Ok...")
                continue
    if answer.lower()[0] == "a":
        answer4 = input("Would you like to see an example of spike from each file? [Y/n]?")
        for h in range(n_spike_trains):
            Chprov1 = data.list_units[h].annotations["id"]
            Label1 = Chprov1.split("#")[1]
            channel1 = np.loadtxt(Chprov1+".txt")
            print(Chprov1)
            print("Starting to get the spikeshapes... Grab a book or something, this might take a while!")
            a1=np.array([],int) 
            x1=np.empty([1,windowsize+2],int)
            for j in range(len(channel1)):
                a1= np.where(time >= channel1[j])[0][0]
                analogtxt1=analog[1][a1:a1+windowsize].reshape(1,windowsize)
                y1 = np.array([[a1],[a1+windowsize]], np.int32).reshape(1,2)
                res1 = append(y1,analogtxt1).reshape(1,-1)
                x1=np.append(x1,res1, axis=0)
            b1=x1[1:]    
            print("\n" + "Voilà!")   
            np.savetxt("SpikeShape#"+Label1+".txt", b1, header="First column = Initial Time; Second column = Final Time; Third Column = First Spike Shape value, etc")
            if answer4.lower() == "y":
                window1=int(b[0][0])
                window2=int(b[0][1])
                figure()
                plot(time[window1:window2],analog[1][window1:window2])
                tight_layout()
                show()
        break
    else:
        break
