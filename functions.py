## @author: eduar
#  Documentation for this module.
#
#  Created on Wed Feb  6 15:06:12 2019; -*- coding: utf-8 -*-; 

import matplotlib
import neo
import numpy as np
from pylab import *
import os
import datetime
import pandas
import scipy.io
import h5py
import scipy.signal

file="CSC1_light_LFPin.smr"
songnumber=1
motifile="motif_times_2018_05_06_11_12_43.mat"

## Documentation for a function.
#
# Read .smr file:
def read(file):
    reader = neo.io.Spike2IO(filename=file)
    # Read the block
    data = reader.read()[0]
    data_seg=data.segments[0]
    return data, data_seg

## Documentation for a function.
#
# Get information inside .smr file
def getinfo(file):
    data, data_seg= read(file)
     # Get the informations of the file
    t_start=float(data_seg.t_start)
    t_stop=float(data_seg.t_stop)
    as_steps=len(data_seg.analogsignals[0])
    time=linspace(t_start,t_stop,as_steps)
    n_analog_signals=len(data_seg.analogsignals)
    n_spike_trains=len(data_seg.spiketrains)
    ansampling_rate=int(data.children_recur[0].sampling_rate)
    return n_analog_signals, n_spike_trains, time, ansampling_rate

## Documentation for a function.
#
# Get the arrays inside the .smr file
def getarrays(file):#Transforms analog signals into arrays inside list
    data, data_seg= read(file)
    n_analog_signals, n_spike_trains, time, ansampling_rate = getinfo(file)
    # Extract analogs and put each array inside a list
    analog=[]
    for i in range(n_analog_signals):
        analog += [data_seg.analogsignals[i].as_array()]
    print("analog: This list contains " + str(n_analog_signals) + " analog signals!")
    # Extract spike trains and put each array inside a list
    sp=[]
    for k in range(n_spike_trains):
        sp += [data_seg.spiketrains[k].as_array()]
    print("sp: This list contains " + str(n_spike_trains) + " spiketrains!")
    return analog, sp

## Documentation for a function.
#
# Plot the analog signals and spiketrains inside .smr file
def plotplots(file):
    data, data_seg= read(file)
    n_analog_signals, n_spike_trains, time, ansampling_rate = getinfo(file)
    analog, sp = getarrays(file)
    #Plot of Analogs
    figure()
    for i in range(n_analog_signals):
        subplot(len(analog),1,i+1)
        plot(time,analog[i])
        xlabel("time (s)")
        title("Analog signal of: " + data_seg.analogsignals[i].name.split(" ")[2])
    tight_layout()
    #Plot of Spike Trains
    Labels=[]
    for i in range(n_spike_trains):
        Chprov = data.list_units[i].annotations["id"]
        Labels += [Chprov]
    figure()
    yticks(np.arange(0, 11, step=1), )
    xlabel("time (s)")
    title("Spike trains")
    res=-1
    count=0
    for j in sp:
        colors=["black","blue", "red", "pink", "purple", "grey", "limegreen", "aqua", "magenta", "darkviolet", "orange"]
        res=res+1
        scatter(j,res+zeros(len(j)),marker="|", color=colors[count])
        legend((Labels), bbox_to_anchor=(1, 1))
        count+=1        
    tight_layout()
    show()

## Documentation for a function.
#
# Create files summary.txt, the spiketimes(.txt) and the analog(.npy) inside a new folder    
def createsave(file):
    data, data_seg= read(file)
    n_analog_signals, n_spike_trains, time, ansampling_rate = getinfo(file)
    analog, sp = getarrays(file)
    
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
    file = open("Channels_Label_LFP.txt", "w+")
    file.write(str(df))
    file.close()
    
    #Create and Save Binary/.NPY files of Analog signals
    for j in range(n_analog_signals):
        temp=data_seg.analogsignals[j].name.split(" ")[2][1:-1]
        np.save(temp, data_seg.analogsignals[j].as_array())
    
    #Create and Save Summary about the File
    an=["File of origin: " + data.file_origin, "Number of AnalogSignals: " + str(n_analog_signals)]
    for k in range(n_analog_signals):
        anlenght= str(data.children_recur[k].size)
        anunit=str(data.children_recur[k].units).split(" ")[1]
        anname=str(data.children_recur[k].name)
        ansampling=str(data.children_recur[i].sampling_rate)
        antime = str(str(data.children_recur[k].t_start) + " to " + str(data.children_recur[k].t_stop))
        an+=[["Analog index:" + str(k) + " Channel Name: " + anname, "Lenght: "+ anlenght, " Unit: " + anunit, " Sampling Rate: " + ansampling + " Duration: " + antime]]    
    
    spk=["Number of SpikeTrains: " + str(n_spike_trains)]    
    for l in range(n_analog_signals, n_spike_trains + n_analog_signals):
        spkid = str(data.children_recur[l].annotations["id"])
        spkcreated = str(data.children_recur[i].annotations["comment"])
        spkname= str(data.children_recur[l].name)
        spksize = str(data.children_recur[l].size)
        spkunit = str(data.children_recur[l].units).split(" ")[1]
        spk+=[["SpikeTrain index:" + str(l-n_analog_signals) + " Channel Id: " + spkid, " " + spkcreated, " Name: " + spkname, " Size: "+ spksize, " Unit: " + spkunit]]
    final = an + spk
    with open("summary.txt", "w+") as f:
        for item in final:
            f.write("%s\n" % "".join(item))
    f.close()        
    print("\n"+"All files were created!")
    
## Documentation for a function.
#
# Get and save the spikeshapes (.txt)    
def spikeshapes(file):
    data, data_seg= read(file)
    n_analog_signals, n_spike_trains, time, ansampling_rate = getinfo(file)
    analog, sp = getarrays(file)
    createsave(file)
    windowsize=int(ansampling_rate*2/1000) #Define here the number of points that suit your window (set to 2ms)
    numberLFP = int(input("Please type the number of the LFP channel"))
    numberanalogspikes = int(input("Please type the number of the SpikeAnalog"))
    # Create and save the spikeshapes
    answer4 = input("Would you like to see an example of spike from each file? [Y/n]?")
    for m in range(n_spike_trains):
        Chprov1 = data.list_units[m].annotations["id"]
        Label1 = Chprov1.split("#")[1]
        channel1 = np.loadtxt(Chprov1+".txt")
        print(Chprov1)
        print("Starting to get the spikeshapes... Grab a book or something, this might take a while!")
        x1=np.empty([1,windowsize+2],int)
        for n in range(len(channel1)):
            a1= int(channel1[n]*ansampling_rate)-57
            analogtxt1=analog[numberLFP][a1:a1+windowsize].reshape(1,windowsize)
            y1 = np.array([[a1],[a1+windowsize]], np.int32).reshape(1,2)
            res1 = np.append(y1,analogtxt1).reshape(1,-1)
            x1=np.append(x1,res1, axis=0)
        b1=x1[1:]    
        print("\n" + "Voilà!")   
        np.savetxt("SpikeShape#"+Label1+".txt", b1, header="First column = Initial Time; Second column = Final Time; Third Column = First Spike Shape value, etc")
        if answer4 == "" or answer4.lower()[0] == "y":
            window1=int(b1[0][0])
            window2=int(b1[0][1])
            figure()
            subplot(2,1,1)
            plot(analog[numberLFP][window1:window2])
            subplot(2,1,2)
            plot(analog[numberanalogspikes][window1+57:window2+57])
            tight_layout()
            show()
## Documentation for a function.
#
# Downsample 1000Hz the LPF signal and save it as .npy file
def lfpdown(file):
    data, data_seg= read(file)
    n_analog_signals, n_spike_trains, time, ansampling_rate = getinfo(file)
    analog, sp = getarrays(file)
    numberLFP = int(input("Please type the number of the LFP channel"))
    ana=analog[numberLFP] #LFP raw analog signal
    def mma(series,window):
        return convolve(series,repeat(1,window)/window,"same")
    
    s_ana=ana[0:][:,0] #window of the array
    s_ana_1=mma(s_ana,100) #convolved version
    figure()
    plot(s_ana)
    plot(s_ana_1)
    show()
           
    c=[]
    for i in range(len(s_ana_1)):
        if i%32==0:
            c+=[s_ana_1[i]]
            
    d=np.array(c)
    np.save("LFPDownsampled", d)
    figure()
    plot(c)  
    show()
    return d

## Documentation for a function.
#
# Gets the motif times   
def cutwaves(file, songnumber, motifile):
    analog, sp = getarrays(file)
    n_analog_signals, n_spike_trains, time, ansampling_rate = getinfo(file)
    ana=analog[songnumber]
    mt = scipy.io.loadmat(motifile)
    mtall = transpose(mt.get("all_motif_times"))
    mtsyb = mt.get("syllable_times")
    res=array([1,2]).reshape(-1,2)
    for i in range(len(mtall)):
        x= (mtall[i][0] + mean(mtsyb[:,0]))* ansampling_rate #Will compute the beginning of the window
        x2= (mtall[i][0] + mtsyb[i][4]) * ansampling_rate #Will compute the end of the window
        res=append(res, (x,x2)).reshape(-1,2)
    return res[1:]

## Documentation for a function.
#
# Generates spectrogram of the motifs in the song raw signal     
def spectrogram_custom(file, songnumber, motifile, resnumber):
    res=cutwaves(file, songnumber, motifile)
    n_analog_signals, n_spike_trains, time, ansampling_rate = getinfo(file)
    analog, sp = getarrays(file)
    b=int(res[resnumber][0])
    b2=int(res[resnumber][1]*1.01) #Gives some freedom to the end of the window to be sure that will get the whole last syllable
    rawsong1=analog[songnumber][b:b2].reshape(1,-1)
    rawsong=rawsong1[0][0:int(ansampling_rate*1.05)] #Allows to standardize the window for all motifs (default= ~1.05s)
    window =('hamming')
    overlap = 64
    nperseg = 1024
    noverlap = nperseg-overlap
    fs=ansampling_rate # set here the sampling frequency
    #Compute and plot spectrogram
    (f,t,sp)=scipy.signal.spectrogram(rawsong, fs, window, nperseg, noverlap, mode='complex')
    figure()
    subplot(2,1,1)
    plot(rawsong)
    subplot(2,1,2)
    imshow(10*np.log10(np.square(abs(sp))), origin="lower", aspect="auto", interpolation="none", cmap="inferno")
    colorbar()
    tight_layout()

## Documentation for a function.
#
# Generates PSTH for 1 syllable (default= first syb)   
def psth(spnumber, motifile):
    analog, sp = getarrays(file)
    mt = scipy.io.loadmat(motifile)
    mtall = transpose(mt.get("all_motif_times"))
    mtsyb = mt.get("syllable_times")
    shoulder= 0.05 #50 ms
    #spused=sp[1]
    meandurall=mean(mtsyb[:,1]-mtsyb[:,0])
    binwidth=0.02
    fig, (a,a1,a2) = plt.subplots(1,3)
    spikes1=[]
    res=-1
    count=0
    n0,n1=0,2
    for i in range(len(mtall)):
        step1=[]
        step2=[]
        step3=[]
        step4=[]
        beg= (mtall[i][0] + mtsyb[i][1]) #Will compute the beginning of the window
        end= (mtall[i][0] + mtsyb[i][2]) #Will compute the end of the window
        step1=spused[where(np.logical_and(spused >= beg-shoulder, spused <= end+shoulder) == True)]-beg
        step2=step1[where(np.logical_and(step1 >= 0, step1 <= end-beg) == True)]*(meandurall/(end-beg))
        step3=step1[where(np.logical_and(step1 >= end-beg, step1 <= (end-beg)+shoulder) == True)]+(meandurall-(end-beg))
        step4=step1[where(step1>=(meandurall+shoulder))]
        spikes1+=[step2,step3]
        res=res+1
        spikes2=spikes1
        spikes2=concatenate(spikes2[n0:n1])
        #d.scatter(spikes2,res+zeros(len(spikes2)),marker="|")
        count+=1
        n0+=2
        n1+=2
        d.set_xlim(-shoulder,(shoulder+meandurall)+binwidth)
        d.tick_params(
            axis='y',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            left=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelleft=False)
    c.set_ylabel("Motif number")
    a.set_xlabel("Time [s]")
    spikes=sort(concatenate(spikes1))
    normfactor=len(mtall)*binwidth
    a.set_ylabel('Spikes/s')
    a1.hist(spikes, bins=arange(-shoulder,(shoulder+meandurall)+binwidth, 0.02), color='b', edgecolor='black', linewidth=1, weights=ones(len(spikes))/normfactor)
    a.set_xlim(-shoulder,(shoulder+meandurall)+binwidth)
    a1.tick_params(
            axis='y',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            left=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelleft=False)
    a.tick_params(
            axis='y',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0)
    fig.show()