"""
Created on Tu Feb 5 2019

@author: eduar

-->> This is the second script of the series to obtain data from Spike2 files. 
This script will generate spiketimes files (.txt), analogical signal files (.npy), a dataframe with Channels/Labels/LFPnumbers and a summary file.

Run this script considering the series (file to be read has to be defined in ReadAndPlot.py)
"""

#Import necessary packages/variables 
from ReadAndPlot import *
import datetime
import os
import pandas
import numpy as np

#Create new folder and change directory
today= datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
os.mkdir(today)
os.chdir(os.path.expanduser(today))

#Create DataFrame (LFP should be indicated by the subject) and SpikeTime files
res=[]
LFP = input("Enter LFP number:")
for i in range(n_spike_trains):
    Chprov = data.list_units[i].annotations["id"]
    Chprov2 = Chprov.split("#")[0]
    Ch = Chprov2.split("ch")[1]                     
    Label = Chprov.split("#")[1]
    res += [[int(Ch), int(Label), int(LFP)]]
    df = pandas.DataFrame(data=res, columns= ["Channel", "Label", "LFP number"])
    np.savetxt(Chprov+".txt", data_seg.spiketrains[i].as_array(), header="Spiketimes of: "+str(Chprov)) #Creates files with the Spiketimes.

print(df)

#Save dataframe with Channels/Labels/LFPnumbers
file = open("Channels_Label_LFP.txt", "w+")
file.write(str(df))
file.close()

#Create and Save Binary/.NPY files of Analog signals
for i in range(n_analog_signals):
    temp=data_seg.analogsignals[i].name.split(" ")[2][1:-1]
    np.save(temp, data_seg.analogsignals[i].as_array())
        
#Create and Save Summary about the File
an=["File of origin: " + data.file_origin, "Number of AnalogSignals: " + str(n_analog_signals)]
for i in range(n_analog_signals):
    anlenght= str(data.children_recur[i].size)
    anunit=str(data.children_recur[i].units).split(" ")[1]
    anname=str(data.children_recur[i].name)
    ansampling_rate=str(data.children_recur[i].sampling_rate)
    antime = str(str(data.children_recur[i].t_start) + " to " + str(data.children_recur[i].t_stop))
    an+=[["Channel Name: " + anname, "Lenght: "+ anlenght, "Unit: " + anunit, "Sampling Rate: " + ansampling_rate, "Duration: " + antime]]
    
spk=["Number of SpikeTrains: " + str(n_spike_trains)]    
for i in range(n_analog_signals, n_spike_trains + n_analog_signals):
    spkid = str(data.children_recur[i].annotations["id"])
    spkcreated = str(data.children_recur[i].annotations["comment"])
    spkname= str(data.children_recur[i].name)
    spksize = str(data.children_recur[i].size)
    spkunit = str(data.children_recur[i].units).split(" ")[1]
    spk+=[["Channel Id: " + spkid, "Created on: " + spkcreated, "Name: " + spkname, "Size: "+ spksize, "Unit: " + spkunit]]

final = an + spk

with open("summary.txt", "w+") as f:
    for item in final:
        f.write("%s\n" % "".join(item))
f.close()        
    
print("\n"+"All files were created!")
