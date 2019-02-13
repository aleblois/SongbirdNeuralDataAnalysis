# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 15:06:12 2019

@author: eduar

-->> This is the first script of the series to obtain data from Spike2 files. 
This script will read .smr files, extract the content, and give you the option to plot it.
"""

import matplotlib
import neo
import numpy as np
from pylab import *

# Create a reader
reader = neo.io.Spike2IO(filename="CSC1_light.smr")
# Read the block
data = reader.read()[0]

# Extract data segments
data_seg=data.segments[0]

# Start time (s)
t_start=float(data_seg.t_start)

# Stop time (s)
t_stop=float(data_seg.t_stop)

# Information regarding the content of data segments
print(data_seg.size)

# Count analog signals number
n_analog_signals=len(data_seg.analogsignals)

# Count spike trains number
n_spike_trains=len(data_seg.spiketrains)

# Extract analog_signals and puts each array inside a list
analog=[]
for i in range(n_analog_signals):
    analog += [i]
    analog[i]= data_seg.analogsignals[i].as_array()
    
print("analog: This list contains " + str(n_analog_signals) + " analog signals!")
    
# Analog signals number of steps
as_steps=len(analog[0])

# Compute time series for analog signals
time=linspace(t_start,t_stop,as_steps)

# Extract spike trains and put each array inside a list
sp=[]
for i in range(n_spike_trains):
    sp += [i]
    sp[i]= data_seg.spiketrains[i].as_array()

print("sp: This list contains " + str(n_spike_trains) + " spiketrains!")

answer= input("Do you want to see the plots? (Y/n)")
if answer=="" or answer.lower()[0] == "y":
    print("Ok! Let's go!")
    #Plot of Analog Signals
    figure()
    for i in range(data_seg.size.get("analogsignals")):
        subplot(n_analog_signals,1,i+1)
        plot(time,analog[i])
        xlabel("time (s)")
        title("Analog signal of: " + data_seg.analogsignals[i].name.split(" ")[2])
    tight_layout()
    
    #Plot of Spike Trains
    Labels=[]
    for i in range(len(data.list_units)):
        Chprov = data.list_units[i].annotations["id"]
        Labels += [Chprov]
    
    figure()
    yticks(np.arange(0, 11, step=1), )
    xlabel("time (s)")
    title("Spike trains")
    res=-1
    
    for j in sp:
        res=res+1
        scatter(j,res+zeros(len(j)),marker="|")
        legend((Labels), bbox_to_anchor=(1, 1))        
    tight_layout()
    show()
   
else:
    print("Ok, maybe next time..")
    pass
           

    

