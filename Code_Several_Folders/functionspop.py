# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 10:28:22 2019

@author: eduar
"""

## @author: Eduarda Centeno
#  Documentation for this module.
#
#  Created on Wed Feb  6 15:06:12 2019; -*- coding: utf-8 -*-; 


#################################################################################################################################
#################################################################################################################################
# This code was built with the aim of allowing the user to work with Spike2 .smr files and further perfom correlation analyses ##
# between specific acoustic features and neuronal activity.                                                                    ##
# In our group we work with Zebra finches, recording their neuronal activity while they sing, so the parameters here might     ##
# have to be ajusted to your specific data.                                                                                    ##                                                                                                                              ##
#################################################################################################################################
#################################################################################################################################

### Necessary packages
import neo
import nolds
import numpy as np
import pylab as py
import os
import matplotlib.lines as mlines
import pandas
import scipy.io
import scipy.signal
import scipy.stats
import scipy.fftpack
import scipy.interpolate
import random
from statsmodels.tsa.stattools import acf
from matplotlib.ticker import FormatStrFormatter

#############################################################################################################################
# This block includes some functions that will be used several times in the code, but will not be individually documented:  #
                                                                                                                            #
def sortsyls(motifile):
    #Read and import files that will be needed
    f=open(motifile, "r")
    imported = f.read().splitlines()
    
    #Excludes everything that is not a real syllable
    a=[] ; b=[] ; c=[] ; d=[]; e=[]
    arra=np.empty((1,2)); arrb=np.empty((1,2)); arrc=np.empty((1,2))
    arrd=np.empty((1,2)); arre=np.empty((1,2))
    for i in range(len(imported)):
        if imported[i][-1] == "a":
            a=[imported[i].split(",")]
            arra=np.append(arra, np.array([int(a[0][0]), int(a[0][1])], float).reshape(1,2), axis=0)
        if imported[i][-1] == "b": 
            b=[imported[i].split(",")]
            arrb=np.append(arrb, np.array([int(b[0][0]), int(b[0][1])], float).reshape(1,2), axis=0)
        if imported[i][-1] == "c": 
            c=[imported[i].split(",")]  
            arrc=np.append(arrc, np.array([int(c[0][0]), int(c[0][1])], float).reshape(1,2), axis=0)
        if imported[i][-1] == "d": 
            d=[imported[i].split(",")] 
            arrd=np.append(arrd, np.array([int(d[0][0]), int(d[0][1])], float).reshape(1,2), axis=0)
        if imported[i][-1] == "e": 
            e=[imported[i].split(",")]   
            arre=np.append(arre, np.array([int(e[0][0]), int(e[0][1])], float).reshape(1,2), axis=0)
            
    arra=arra[1:]; arrb=arrb[1:]; arrc=arrc[1:]; arrd=arrd[1:] ; arre=arre[1:]
    k=[arra,arrb,arrc,arrd,arre]
    finallist=[]
    for i in k:
        #print(i.size)
        if i.size != 0:
            finallist+=[i]
        else:
            continue
        
    return finallist

def tellme(s):
    print(s)
    py.title(s, fontsize=10)
    py.draw()


def smoothed(inputSignal,fs, smooth_win=10):
        squared_song = np.power(inputSignal, 2)
        len = np.round(fs * smooth_win / 1000).astype(int)
        h = np.ones((len,)) / len
        smooth = np.convolve(squared_song, h)
        offset = round((smooth.shape[-1] - inputSignal.shape[-1]) / 2)
        smooth = smooth[offset:inputSignal.shape[-1] + offset]
        smooth = np.sqrt(smooth)
        return smooth
    
    
#Fast loop to check visually if the syllables are ok. I've been finding problems in A syllables, so I recommend checking always before analysis.
def checksyls(songfile,motifile, beg, end ):
    arra, arrb, arrc, arrd, arre= sortsyls(motifile)
    song=np.load(songfile)
    #Will filter which arra will be used
    answer=input("Which syllable?")
    if answer.lower() == "a":
        used=arra
    elif answer.lower() == "b":
        used=arrb
    elif answer.lower() == "c":
        used=arrc    
    elif answer.lower() == "d":
        used=arrd
    
    print("This syb has "+ str(len(used)) + " renditions.")
    
    for i in range(beg,end):
        py.figure()
        py.plot(song[int(used[i][0]):int(used[i][1])])

""" The two following functions were obtained from 
http://ceciliajarne.web.unq.edu.ar/investigacion/envelope_code/ """
def window_rms(inputSignal, window_size):
        a2 = np.power(inputSignal,2)
        window = np.ones(window_size)/float(window_size)
        return np.sqrt(np.convolve(a2, window, "valid"))
    
def getEnvelope(inputSignal, window_size):
# Taking the absolute value

    absoluteSignal = []
    for sample in inputSignal:
        absoluteSignal.append (abs (sample))

    # Peak detection

    intervalLength = window_size # change this number depending on your signal frequency content and time scale
    outputSignal = []

    for baseIndex in range (0, len (absoluteSignal)):
        maximum = 0
        for lookbackIndex in range (intervalLength):
            maximum = max (absoluteSignal [baseIndex - lookbackIndex], maximum)
        outputSignal.append (maximum)

    return outputSignal

def jumpsyl(spikefile2):
    with open("..\\CheckSylsFreq"+spikefile2+".txt", "r") as datafile:
        fich=datafile.read().split()[1::4]    
    return fich
    
###############################################################################################################################





##############################################################################################################################
# From now on there will be the core functions of this code, which will be individually documented:                          #
                                                                                                                             #

##
##
# 
# This  function will allow you to read the .smr files from Spike2.
def read(file):
    reader = neo.io.Spike2IO(filename=file) #This command will read the file defined above
    data = reader.read()[0] #This will get the block of data of interest inside the file
    data_seg=data.segments[0] #This will get all the segments
    return data, data_seg


## 
#
# This  function will allow you to get information inside the .smr file.
# It will return the number of analog signals inside it, the number of spike trains, 
# a numpy array with the time (suitable for further plotting), and the sampling rate of the recording.
def getinfo(file):
    data, data_seg= read(file)
     # Get the informations of the file
    t_start=float(data_seg.t_start) #This gets the time of start of your recording
    t_stop=float(data_seg.t_stop) #This gets the time of stop of your recording
    as_steps=len(data_seg.analogsignals[0]) 
    time=np.linspace(t_start,t_stop,as_steps)
    n_analog_signals=len(data_seg.analogsignals) #This gets the number of analogical signals of your recording
    n_spike_trains=len(data_seg.spiketrains) #This gets the number of spiketrains signals of your recording
    ansampling_rate=int(data.children_recur[0].sampling_rate) #This gets the sampling rate of your recording
    return n_analog_signals, n_spike_trains, time, ansampling_rate


## 
#
# This  function will get the analogical signals and the spiketrains from the .smr file and return them in the end as arrays.
def getarrays(file): #Transforms analog signals into arrays inside list
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


## 
#
# This  function will allow you to plot the analog signals and spiketrains inside the .smr file.
def plotplots(file):
    data, data_seg= read(file)
    n_analog_signals, n_spike_trains, time, ansampling_rate = getinfo(file)
    analog, sp = getarrays(file)
    #Plot of Analogs
    py.figure()
    for i in range(n_analog_signals):
        py.subplot(len(analog),1,i+1)
        py.plot(time,analog[i])
        py.xlabel("time (s)")
        py.ylabel("Amplitude")
        py.title("Analog signal of: " + data_seg.analogsignals[i].name.split(" ")[2])
    py.tight_layout()
    #Plot of Spike Trains
    Labels=[]
    for i in range(n_spike_trains):
        Chprov = data.list_units[i].annotations["id"]
        Labels += [Chprov]
    py.figure()
    py.yticks(np.arange(0, 11, step=1), )
    py.xlabel("time (s)")
    py.title("Spike trains")
    py.ylabel("Number of spike trains")
    res=-1
    count=0
    for j in sp:
        colors=["black","blue", "red", "pink", "purple", "grey", "limegreen", "aqua", "magenta", "darkviolet", "orange"] #This was decided from observing which order SPIKE2 defines the colors for the spiketrains
        res=res+1
        py.scatter(j,res+np.zeros(len(j)),marker="|", color=colors[count])
        py.legend((Labels), bbox_to_anchor=(1, 1))
        count+=1        
    py.tight_layout()
    py.show()


## 
#
# This function will create a few files inside a folder which will be named according to the date and time.
# The files are:
#
#  1 - summary.txt : this file will contain a summary of the contents of the .smr file.
#
#  2- the spiketimes as .txt: these files will contain the spiketimes of each spiketrain.
#
#  3- the analog signals as .npy: these files will contain the raw data of the analog signals.   
def createsave(file):
    data, data_seg= read(file)
    n_analog_signals, n_spike_trains, time, ansampling_rate = getinfo(file)
    analog, sp = getarrays(file)
    
    #Create new folder and change directory
    today= "Files"
    os.mkdir(today)
    os.chdir(os.path.expanduser(today))
    
    #Create DataFrame (LFP should be indicated by the subject) and SpikeTime files
    res=[]
    LFP = input("Enter LFP number:")
    if os.path.isfile("..//unitswindow.txt"):
        for i in range(n_spike_trains):
            Chprov = data.list_units[i].annotations["id"]
            Chprov2 = Chprov.split("#")[0]
            Ch = Chprov2.split("ch")[1]                     
            Label = Chprov.split("#")[1]
            res += [[int(Ch), int(Label), int(LFP)]]
            df = pandas.DataFrame(data=res, columns= ["Channel", "Label", "LFP number"])
            with open("..\\unitswindow.txt", "r") as datafile:
                s=datafile.read().split()
            d=s[0::3]
            x=np.array(s).reshape((-1,3))
            if Chprov in d and x.size >=3:
                arr= data_seg.spiketrains[i].as_array()
                where=d.index(Chprov)
                windowbeg=int(x[where][1])
                windowend=int(x[where][2])
                if windowend==-1:
                    windowend=arr[-1]
                tosave= arr[np.where(np.logical_and(arr >= windowbeg , arr <= windowend) == True)]
                np.savetxt(Chprov+".txt", tosave) #Creates files with the Spiketimes.
            else:
                np.savetxt(Chprov+".txt", data_seg.spiketrains[i].as_array())
    else:
        for i in range(n_spike_trains):
            Chprov = data.list_units[i].annotations["id"]
            Chprov2 = Chprov.split("#")[0]
            Ch = Chprov2.split("ch")[1]                     
            Label = Chprov.split("#")[1]
            res += [[int(Ch), int(Label), int(LFP)]]
            df = pandas.DataFrame(data=res, columns= ["Channel", "Label", "LFP number"])
            np.savetxt(Chprov+".txt", data_seg.spiketrains[i].as_array()) #Creates files with the Spiketimes.
        
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
        antime = str(str(data.children_recur[k].t_start) + " to " + str(data.children_recur[k].t_stop))
        an+=[["Analog index:" + str(k) + " Channel Name: " + anname, "Lenght: "+ anlenght, " Unit: " + anunit, " Sampling Rate: " + str(ansampling_rate) + " Duration: " + antime]]    
    
    spk=["Number of SpikeTrains: " + str(n_spike_trains)]    
    for l in range(n_analog_signals, n_spike_trains + n_analog_signals):
        spkid = str(data.children_recur[l].annotations["id"])          
        spkcreated = str(data.children_recur[l].annotations["comment"])
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

    
## 
#
# This function will get and save the spikeshapes (.txt) from the Raw unfiltered neuronal signal.
#
# Arguments:
#
# file is the .smr file
#
# raw is the .npy file containing the Raw unfiltered neuronal signal
#
# rawfiltered is the .npy containing the spike2 filtered neuronal signal    
def spikeshapes(file, raw, rawfiltered):
    data, _= read(file)
    _, n_spike_trains, _, ansampling_rate = getinfo(file)
    LFP=np.load(raw)
    notLFP=np.load(rawfiltered)
    windowsize=int(ansampling_rate*2/1000) #Define here the number of points that suit your window (set to 2ms)
    # Create and save the spikeshapes
    # This part will iterate through all the .txt files containing the spiketimes inside the folder.
    answer4 = input("Would you like to see an example of spike from each file? [Y/n]?")
    for m in range(n_spike_trains):
        Chprov1 = data.list_units[m].annotations["id"]
        #Label1 = Chprov1.split("#")[1]
        channel1 = np.loadtxt(Chprov1+".txt")    
        print(Chprov1)
        print("Starting to get the spikeshapes... Grab a book or something, this might take a while!")
        x1=np.empty([1,windowsize+2],int)
        for n in range(len(channel1)):
            a1= int(channel1[n]*ansampling_rate)-57
            if a1 == -57:
                continue
            analogtxt1=LFP[a1:a1+windowsize].reshape(1,windowsize)
            y1 = np.array([[a1],[a1+windowsize]], np.int32).reshape(1,2)
            res1 = np.append(y1,analogtxt1).reshape(1,-1)
            x1=np.append(x1,res1, axis=0)
        b1=x1[1:]    
        print("\n" + "Voilà!")   
        np.savetxt("SpikeShape#"+Chprov1+".txt", b1, header="First column = Initial Time; Second column = Final Time; Third Column = First Spike Shape value, etc")
        if answer4 == "" or answer4.lower()[0] == "y":
            window1=int(b1[0][0])
            window2=int(b1[0][1])
            py.fig,(s,s1) = py.subplots(2,1)
            s.plot(LFP[window1:window2])
            s.set_title("SpikeShape from Raw Unfiltered")
            s1.plot(notLFP[window1+57:window2+57])
            s.set_ylabel("Amplitude")
            s1.set_ylabel("Amplitude")
            s1.set_xlabel("Sample points")
            s1.set_title("SpikeShape from Raw Filtered Spike2") # Just like you would see in Spike2
            py.tight_layout()
            py.show()
 
           
## 
#
# This function will downsample your LFP signal to 1000Hz and save it as .npy file
def lfpdown(LFPfile, fs): #LFPfile is the .npy one inside the new folder generated by the function createsave (for example, CSC1.npy)
    fs1=int(fs/1000)
    rawsignal=np.load(LFPfile)
    def mma(series,window):
        return np.convolve(series,np.repeat(1,window)/window,"same")
    
    rawsignal=rawsignal[0:][:,0] #window of the array, in case you want to select a specific part
    conv=mma(rawsignal,100) #convolved version
    c=[]
    for i in range(len(conv)):
        if i%fs1==0:
            c+=[conv[i]]
            
    downsamp=np.array(c)
    np.save("LFPDownsampled", downsamp)
    answer=input("Want to see the plots? Might be a bit heavy. [Y/n]")
    if answer == "" or answer.lower()[0] == "y":
        py.fig,(s,s1) = py.subplots(2,1) 
        s.plot(rawsignal)
        s.plot(conv)
        s.set_title("Plot of RawSignal X Convolved Version")
        s1.plot(downsamp)
        s1.set_title("LFP Downsampled")
        s.set_ylabel("Amplitude")
        s1.set_ylabel("Amplitude")
        s1.set_xlabel("Sample points")
        py.show()
        py.tight_layout()


## 
#
# This function generates spectrogram of the motifs in the song raw signal. 
# To be used with the new matfiles.
#    
# Arguments:
#    
# songfile is the .npy file containing the song signal.
#
# beg, end : are the index that would correspond to the beginning and the end of the motif/syllable (check syllables annotations file for that)
#
# fs = sampling frequency    
def spectrogram(songfile, beg, end, fs):
    analog= np.load(songfile)
    rawsong1=analog[beg:end].reshape(1,-1)
    rawsong=rawsong1[0]
    #Compute and plot spectrogram
    #(f,t,sp)=scipy.signal.spectrogram(rawsong, fs, window, nperseg, noverlap, scaling="density", mode="complex")
    py.fig, ax = py.subplots(2,1)
    ax[0].plot(rawsong)
    ax[0].set_ylabel("Amplitude")
    ax[0].set_xlabel("Sample points")
    _,_,_,im = ax[1].specgram(rawsong,Fs=fs, NFFT=980, noverlap=930, scale_by_freq=False, mode="default", pad_to=915, cmap="inferno")
    #py.imshow(10*np.log10(np.square(abs(sp))), origin="lower", aspect="auto", interpolation="none", cmap="inferno")
    ax[1].tick_params(
                        axis="x",          # changes apply to the x-axis
                        which="both",      # both major and minor ticks are affected
                        bottom=False,      # ticks along the bottom edge are off
                        top=False,         # ticks along the top edge are off
                        labelbottom=False)
    ax[1].set_ylabel("Frequency")
    cbar=py.colorbar(im, ax=ax[1])
    cbar.ax.invert_yaxis()
    cbar.set_ticks(np.linspace(cbar.vmin, cbar.vmax, 5, dtype=float))
    cbar.ax.set_yticklabels(np.floor(np.linspace(np.floor(cbar.vmin), cbar.vmax, 5)).astype(int))
    py.tight_layout() 


## 
#
# This function generates a PSTH for motifs. 
# To be used with the new matfiles.
#
# Arguments:    
#
# spikefile is the .txt file with the spiketimes.
#
# motifile is the .txt file containing the annotations of the beggining and end of each syllable/motif.
#
# fs is the sampling frequency
#
# basebeg is the start time for baseline computation
#
# basend is the end time for baseline computation    
def psth(spikefile, motifile, fs, basebeg, basend, binwidth):     
    sybs=["A","B","C","D","E"]
    finallist=sortsyls(motifile)
    #Starts to plot the PSTH
    spused=np.loadtxt(spikefile)
    shoulder= 0.05 #50 ms
    meandurall=0
    f = open("CheckSylsFreq"+spikefile[:-4]+".txt", "w+")
    # This part will result in an iteration through all the syllables, and then through all the motifs inside each syllable.
    py.fig, ax = py.subplots(2,len(finallist), figsize=(18,15), sharey=False)
    for i in range(len(finallist)):
        if len(finallist) == 1:
            shapes = (1,)
            shapes2 = (0,)
        else:
            shapes=(1,i)
            shapes2=(0,i)
        used=finallist[i]/fs # sets which array from finallist will be used.
        meandurall=np.mean(used[:,1]-used[:,0])
        spikes1=[]
        res=-1
        spikes=[]
        basespk=[]
        n0,n1=0,3
        for j in range(len(used)):
            step1=[]
            step2=[]
            step3=[]
            beg= used[j][0] #Will compute the beginning of the window
            end= used[j][1] #Will compute the end of the window
            step1=spused[np.where(np.logical_and(spused >= beg-shoulder, spused <= end+shoulder) == True)]-beg
            stepsholneg=step1[step1<0]
            step2=step1[np.where(np.logical_and(step1 >= 0, step1 <= end-beg) == True)]*(meandurall/(end-beg))
            step3=step1[np.where(np.logical_and(step1 >= end-beg, step1 <= (end-beg)+shoulder) == True)]+(meandurall-(end-beg))
            spikes1+=[stepsholneg,step2,step3]
            res=res+1
            spikes2=spikes1
            spikes3=np.concatenate(spikes2[n0:n1]) # Gets the step2 and step3 arrays for scatter
            ax[shapes].scatter(spikes3,res+np.zeros(len(spikes3)),marker="|", color="black")
            n0+=3
            n1+=3
            bins=np.arange(-shoulder,meandurall+shoulder, step=binwidth)
            ax[shapes].set_xlim(min(bins), max(bins))
            ax[shapes].set_xticks([min(bins),0,meandurall,max(bins)])
            normfactor=len(used)*binwidth
            ax[shapes2].set_xlim(min(bins), max(bins))
            ax[shapes2].set_title("Syllable " + sybs[i])
            ax[shapes2].tick_params(
                    axis="x",          # changes apply to the x-axis
                    which="both",      # both major and minor ticks are affected
                    bottom=False,      # ticks along the bottom edge are off
                    top=False,         # ticks along the top edge are off
                    labelbottom=False)
            basecuts=np.random.choice(np.arange(basebeg,basend))
            test2=spused[np.where(np.logical_and(spused >= basecuts, spused <= basecuts+meandurall) == True)]-basecuts
            basespk+=[test2]
        # Computation of baseline
        b=np.sort(np.concatenate(basespk))
        u,_= py.histogram(b, bins=np.arange(0,meandurall+binwidth,binwidth), weights=np.ones(len(b))/normfactor)
        basemean=np.mean(u)
        stdbase=np.std(u)
        axis=np.arange(meandurall/3,meandurall*2/3,binwidth)
        ax[shapes2].plot(axis,np.ones((len(axis),))*basemean, color = "g")
        ax[shapes2].plot(axis,np.ones((len(axis),))*(basemean+stdbase), color = "black")
        ax[shapes2].plot(axis,np.ones((len(axis),))*(basemean-stdbase), color = "black", ls="dashed")
        # Computation of spikes
        spikes=np.sort(np.concatenate(spikes2))
        y1,x1= py.histogram(spikes, bins=bins, weights=np.ones(len(spikes))/normfactor)
        if np.mean(y1) < 5:
            f.writelines("Syllable " + str(sybs[i]) +" : " + str(np.mean(y1)) + "\n")
        ax[shapes].axvline(x=0, color="grey", linestyle="--")
        ax[shapes].axvline(x=meandurall, color="grey", linestyle="--")
        #ax[shapes2].hist(spikes, bins=bins, color="b", edgecolor="black", weights=np.ones(len(spikes))/normfactor)
        #ax[0].plot(x1[:-1]+binwidth/2,y1, color="red")
        x2=np.delete(x1,-2)
        x2[1:-1]=x2[1:-1]+binwidth/2
        inter = scipy.interpolate.interp1d(x2, y1, kind="linear")
        xnew=np.linspace(min(x2),max(x2), num=100)
        ax[shapes2].plot(xnew,inter(xnew), color="red")
        py.fig.subplots_adjust(hspace=0)
        black_line = mlines.Line2D([], [], color="black", label="+STD")
        black_dashed  = mlines.Line2D([], [], color="black", label="+STD", linestyle="--")
        green_line  = mlines.Line2D([], [], color="green", label="Mean")
        ax[shapes2].legend(handles=[black_line,black_dashed,green_line], loc="upper left", prop={'size': 6})
        ax[shapes].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    if len(finallist) == 1:
        ax[0].set_ylabel("Spikes/Sec")
        ax[1].set_ylabel("Motif number")
    else:
        ax[0,0].set_ylabel("Spikes/Sec")
        ax[1,0].set_ylabel("Motif number")
        values = np.array([])
        values2 = np.array([])
        top = np.array([])
        top2 = np.array([])
        for lim in range(len(finallist)):
            values = np.array(ax[0,lim].get_ylim())
            values2 = np.array(ax[1,lim].get_ylim())
            top = np.sort(np.append(top, values))
            top2 = np.sort(np.append(top2, values2))
        for limreal in range(len(finallist)):
            ax[0,limreal].set_ylim(0,max(top))
            ax[1,limreal].set_ylim(min(top2),max(top2))        
    wind=py.get_current_fig_manager()
    wind.window.showMaximized()
    py.fig.subplots_adjust(top=0.957, bottom=0.072, left=0.032, right=0.984, hspace=0.0, wspace=0.109)
    #py.fig.tight_layout()
    py.fig.text(0.5, 0.02, "Time(seconds)", va="center", ha="center")
    f.close()
    

    
## 
#
# Generates correlations for each syllable. 
# To be used it with new matfiles.
#
# Arguments:
#
# spikefile is the .txt file with the spiketimes.
#
# motifile is the .txt file containing the annotations of the beggining and end of each syllable/motif.
#
# n_iterations is the number of iterations for the bootstrapping
#
# fs is the sampling frequency
def corrduration(spikefile, motifile, n_iterations,fs, alpha):
    print(os.getcwd())      
    #Read and import mat file (new version)
    sybs=["A","B","C","D","E"]
    finallist=sortsyls(motifile)    
    #Starts to compute correlations and save the data into txt file (in case the user wants to use it in another software)
    spused=np.loadtxt(spikefile)
    check=jumpsyl(spikefile[3:-4])
    final=[]
    f = open("SummaryDuration.txt", "w+")
    for i in range(len(finallist)):
        if sybs[i] in check:
            continue
        else:        
            used=finallist[i]/fs
            dur=used[:,1]-used[:,0]
            array=np.empty((1,2))
            statistics=[]
            for j in range(len(used)):
                step1=[]
                beg= used[j][0] #Will compute the beginning of the window
                end= used[j][1] #Will compute the end of the window
                step1=spused[np.where(np.logical_and(spused >= beg, spused <= end) == True)]
                array=np.append(array, np.array([[dur[j]],[np.size(step1)/dur[j]]]).reshape(-1,2), axis=0)
            array=array[1:]
            os.chdir("Results")
            np.savetxt("Data_Raw_Corr_Duration_Result_Syb"+str(sybs[i])+".txt", array, header="First column is the duration value, second is the number of spikes.")
            os.chdir("..")
            threshold = 3 #Standard Deviation threshold for Z score identification of outliers
            z = np.abs(scipy.stats.zscore(array))
            array=array[(z < threshold).all(axis=1)]
            if len(array)<3:
                continue
            else:               
                s1=scipy.stats.shapiro(array[:,0])[1]
                s2=scipy.stats.shapiro(array[:,1])[1]
                s3=np.array([s1,s2])
                s3=s3>alpha
                homo=scipy.stats.levene(array[:,0],array[:,1])[1]
                if  s3.all() == True and homo > alpha: #test for normality
                    final=scipy.stats.pearsonr(array[:,0],array[:,1]) #if this is used, outcome will have no clear name on it
                    statistics+=[[final[0],final[1]]]
                    # Bootstrapping
                    for q in range(n_iterations):
                        resample=np.random.choice(array[:,0], len(array[:,0]), replace=True)
                        res=scipy.stats.spearmanr(array[:,1],resample)
                        statistics+=[[res[0],res[1]]]
                else: 
                    final=scipy.stats.spearmanr(array[:,0],array[:,1]) #if this is used, outcome will have the name spearman on it
                    statistics+=[[final[0],final[1]]]
                    # Bootstrapping
                    for x in range(n_iterations):
                        resample=np.random.choice(array[:,0], len(array[:,0]), replace=True)
                        res=scipy.stats.spearmanr(array[:,1],resample)
                        statistics+=[[res[0],res[1]]]
                os.chdir("Results")
                np.savetxt("Data_Boot_Corr_Duration_Result_Syb"+str(sybs[i])+".txt", np.array(statistics), header="First column is the correlation value, second is the p value. First line is the original correlation, all below are the bootstrapped correlations.") #First column is the correlation value, second is the p value.
                os.chdir("..")
                print("Syllable " + str(sybs[i]) +": " + str(final))
                f.writelines("Syllable " + str(sybs[i]) +": " + str(final) + "\n")
          
## 
#
# This function allows you to see the envelope for song signal.
#
# Arguments:
#
# songfile is the .npy file containing the song signal.
#
# beg, end : are the index that would correspond to the beginning and the end of the motif/syllable (check syllables annotations file for that)
#
# window_size is the size of the window for the convolve function. 
def plotEnvelopes(songfile, beg, end, window_size): 
    inputSignal=np.load(songfile)
    inputSignal=np.ravel(inputSignal[beg:end])
    
    outputSignal=getEnvelope(inputSignal, window_size)
    rms=window_rms(inputSignal, window_size)
    
    # Plots of the envelopes
    py.fig, (a,b,c) =py.subplots(3,1, sharey=True)
    py.xlabel("Sample Points")
    a.plot(abs(inputSignal))
    a.set_ylabel("Amplitude")
    a.set_title("Raw Signal")
    b.plot(abs(inputSignal))
    b.plot(outputSignal)
    b.set_ylabel("Amplitude")
    b.set_title("Squared Windows")
    c.plot(abs(inputSignal))
    c.plot(rms)           
    c.set_ylabel("Amplitude")
    c.set_title("RMS")
    py.tight_layout()
    py.show()

## 
#
# This function will perform the Fast Fourier Transform to obtain the power spectrum of the syllables.
#
# Arguments:
#
# songfile is the .npy file containing the song signal.
#
# beg, end : are the index that would correspond to the beginning and the end of the motif/syllable (check syllables annotations file for that)
#
# fs is the sampling rate
def powerspectrum(songfile, beg, end, fs):
    signal=np.load(songfile) #The song channel raw data
    signal=signal[beg:end] #I selected just one syllable A to test
    print ("Frequency sampling", fs)
    l_audio = len(signal.shape)
    if l_audio == 2:
        signal = signal.sum(axis=1) / 2
    N = signal.shape[0]
    print ("Complete Samplings N", N)
    secs = N / float(fs)
    print ("secs", secs)
    Ts = 1.0/fs # sampling interval in time
    print ("Timestep between samples Ts", Ts)
    t = scipy.arange(0, secs, Ts) # time vector as scipy arange field / numpy.ndarray
    FFT = abs(scipy.fft(signal))**2 # if **2 is power spectrum, without is amplitude spectrum
    FFT_side = FFT[range(int(N/2))] # one side FFT range
    freqs = scipy.fftpack.fftfreq(signal.size, t[1]-t[0])
    freqs_side = freqs[range(int(N/2))]
    py.subplot(311)
    py.plot(t, signal, "g") # plotting the signal
    py.xlabel("Time")
    py.ylabel("Amplitude")
    py.subplot(312)
    py.plot(freqs, FFT, "r") # plotting the complete fft spectrum
    py.xlabel("Frequency (Hz)")
    py.title("Double-sided")
    py.ylabel("Power")
    py.subplot(313)
    py.plot(freqs_side, abs(FFT_side), "b") # plotting the positive fft spectrum
    py.xlabel("Frequency (Hz)")
    py.title("Single sided")
    py.ylabel("Power")
    py.tight_layout()
    py.show()

## 
#
# This function can be used to obtain the pitch of specific tones inside a syllable.
# It will execute an autocorrelation for the identification of the pitchs.
#
# Arguments:
#
# songfile is the .npy file containing the song signal.
#
# motifile is the .txt file containing the annotations of the beggining and end of each syllable/motif.
#
# lags is the number of lags for the autocorrelation
#
# window_size is the size of the window for the convolve function (RMS of signal)
#
# fs is the sampling rate   
def corrpitch(songfile, motifile, lags, window_size,fs,spikefile, n_iterations, alpha, premot):
    
   #Read and import files that will be needed
    spused=np.loadtxt(spikefile)
    song=np.load(songfile)
    finallist=sortsyls(motifile)  
    fichier = open("SummaryCorrPitch.txt", "w+")
    y=["MeanA.txt","MeanB.txt","MeanC.txt","MeanD.txt"]
    Syls=["A","B","C","D"]
    check=jumpsyl(spikefile[3:-4])
    for obj in range(len(finallist)):
        if Syls[obj] in check:
            continue
        else:                
            used=finallist[obj]
            means = np.loadtxt("..\\..\\"+y[obj]).astype(int)
            syb=song[int(used[0][0]):int(used[0][1])]
        
            # Autocorrelation and Distribution 
            for m in range(1,len(means)):
                spikespremot=[]
                spikesdur=[]
                freq2=[]
                coords5=[]
                fig=py.figure(figsize=(18,15))
                gs=py.GridSpec(2,2)
                a2=fig.add_subplot(gs[0,0]) # First row, first column
                a3=fig.add_subplot(gs[0,1]) # First row, second column
                a1=fig.add_subplot(gs[1,:]) 
                fig.suptitle("Syllable " + Syls[obj] + " Tone " + str(m))
                for n in range(len(used)):
                    syb=song[int(used[n][0]):int(used[n][1])] #Will get the syllables for each rendition
                    sybcut=syb[means[m-1]:means[m]] #Will apply the cuts for the syllable
                    x2=np.arange(0,len(acf(sybcut,nlags=int(lags))),1)
                    f=scipy.interpolate.interp1d(x2,acf(sybcut, nlags=int(lags)), kind="quadratic")
                    xnew=np.linspace(min(x2),max(x2), num=1000)
                    a1.plot(xnew,f(xnew))
                    a1.set_xlabel("Number of Lags")
                    a1.set_ylabel("Autocorrelation score")
                a1.set_label(tellme("Want to keep it? Key click (x2) for yes, mouse click for no"))
                if not py.waitforbuttonpress(30):
                    py.close()
                    continue            
                else:
                    py.waitforbuttonpress(30)
                    while True:           
                        coord=[]
                        while len(coord) < 2:
                            tellme("Select the points for the peak.") #You should choose in the graph the range that representes the peak
                            coord = np.asarray(py.ginput(2, timeout=-1, show_clicks=True))
                        scat=a1.scatter(coord[:,0],coord[:,1], s=50, marker="X", zorder=10, c="b")
                        tellme("Happy? Key click for yes, mouse click for no")
                        if py.waitforbuttonpress(30):
                            break
                        else:
                            scat.remove()
                    coords5=coord[:,0]*10 # times ten is because of the linspace being 1000
                    a1.clear()
                    
                #From now it will use the coordinates of the peak to plot the distribution and the interpolated version of the peak    
                for x in range(len(used)):
                    syb=song[int(used[x][0]):int(used[x][1])]
                    sybcut=syb[means[m-1]:means[m]]
                    x2=np.arange(0,len(acf(sybcut,nlags=int(lags))),1)
                    f=scipy.interpolate.interp1d(x2,acf(sybcut, nlags=int(lags)), kind="quadratic")
                    xnew=np.linspace(min(x2),max(x2), num=1000)
                    a1.plot(xnew,f(xnew))
                    x3=xnew[int(coords5[0]):int(coords5[1])]
                    g=scipy.interpolate.interp1d(x3,f(xnew)[int(coords5[0]):int(coords5[1])], kind="cubic")
                    xnew2=np.linspace(min(x3),max(x3), num=1000)
                    a2.plot(xnew2,g(xnew2))
                    peak=np.argmax(g(xnew2))
                    freq2+=[xnew2[peak]]
                    beg=(used[x][0] + means[m-1])/fs
                    end=(used[x][0] + means[m])/fs
                    step1=spused[np.where(np.logical_and(spused >= beg-premot, spused <= beg) == True)]
                    step2=spused[np.where(np.logical_and(spused >= beg, spused <= end) == True)]
                    spikespremot+=[[np.size(step1)/(beg-(beg-premot))]]
                    spikesdur+=[[np.size(step2)/(end-beg)]]
                statistics=[]
                statistics2=[]
                spikesdur=np.array(spikesdur)[:,0]
                spikespremot=np.array(spikespremot)[:,0]
                freq2=np.array(freq2)
                freq2=np.reciprocal(freq2/fs)
                total = np.column_stack((freq2,spikespremot,spikesdur))
                os.chdir("Results")
                np.savetxt("Data_Raw_Corr_Pitch_Result_Syb" + Syls[obj] + "_tone_" + str(m) + ".txt", total, header="First column is the pitch value, second is the number of spikes inside premotor window, third is the number of spikes inside 'during' window.")
                os.chdir("..")
                #Here it will give you the possibility of computing the correlations and Bootstrapping
                threshold = 3 #Standard Deviation threshold for Z score identification of outliers
                total1=np.column_stack((freq2,spikespremot))
                total2=np.column_stack((freq2,spikesdur))
                z1 = np.abs(scipy.stats.zscore(total1))
                z2 = np.abs(scipy.stats.zscore(total2))
                total1=total1[(z1 < threshold).all(axis=1)]
                total2=total2[(z2 < threshold).all(axis=1)]
                a = total1[:,1] == 0
                b = total2[:,1] == 0
                #This will get the data for Pitch vs Premotor
                if len(total1) < 3 or all(a) == True:
                    pass
                else:
                    s1=scipy.stats.shapiro(total1[:,0])[1] #Pitch column
                    s2=scipy.stats.shapiro(total1[:,1])[1] #Premot Column
                    homo=scipy.stats.levene(total1[:,0],total[:,1])[1]
                    comb1=np.array([s1,s2,homo])
                    comb1=comb1>alpha
                    if  comb1.all() == True: #test for normality
                        final=scipy.stats.pearsonr(total1[:,0],total1[:,1]) #if this is used, outcome will have no clear name on it
                        statistics+=[[final[0],final[1]]]
                        # Bootstrapping
                        for q in range(n_iterations):
                            resample=np.random.choice(total1[:,0], len(total1[:,0]), replace=True)
                            res=scipy.stats.spearmanr(total1[:,1],resample)
                            statistics+=[[res[0],res[1]]]
                    else: 
                        final=scipy.stats.spearmanr(total1[:,0],total1[:,1]) #if this is used, outcome will have the name spearman on it
                        statistics+=[[final[0],final[1]]]
                        # Bootstrapping
                        for q in range(n_iterations):
                            resample=np.random.choice(total1[:,0], len(total1[:,0]), replace=True)
                            res=scipy.stats.spearmanr(total1[:,1],resample)
                            statistics+=[[res[0],res[1]]]
                    os.chdir("Results")
                    np.savetxt("Data_Boot_Corr_Pitch_Result_Syb" + Syls[obj] + "_tone_" + str(m)+ "_Premotor.txt", statistics, header="First column is the correlation value, second is the p value. First line is the original correlation, all below are the bootstrapped correlations.")
                    fichier.writelines("Syllable " + Syls[obj] + "_tone_" + str(m)+ "_Premotor:" + str(final) + "\n")
                    os.chdir("..")
                    print(final)
                #This will get the data for Pitch vs During     
                if len(total2) < 3 or all(b) == True:
                    pass
                else:
                    s1=scipy.stats.shapiro(total2[:,0])[1] #Pitch column
                    s2=scipy.stats.shapiro(total2[:,1])[1] #During Column
                    homo=scipy.stats.levene(total2[:,0],total2[:,1])[1]
                    comb1=np.array([s1,s2,homo])
                    comb1=comb1>alpha
                    if  comb1.all() == True: #test for normality
                        final=scipy.stats.pearsonr(total2[:,0],total2[:,1]) #if this is used, outcome will have no clear name on it
                        statistics2+=[[final[0],final[1]]]
                        # Bootstrapping
                        for q in range(n_iterations):
                            resample=np.random.choice(total2[:,0], len(total2[:,0]), replace=True)
                            res=scipy.stats.spearmanr(total2[:,1],resample)
                            statistics2+=[[res[0],res[1]]]
                    else: 
                        final=scipy.stats.spearmanr(total2[:,0],total2[:,1]) #if this is used, outcome will have the name spearman on it
                        statistics2+=[[final[0],final[1]]]
                        # Bootstrapping
                        for q in range(n_iterations):
                            resample=np.random.choice(total2[:,0], len(total2[:,0]), replace=True)
                            res=scipy.stats.spearmanr(total2[:,1],resample)
                            statistics2+=[[res[0],res[1]]]
                    os.chdir("Results")
                    np.savetxt("Data_Boot_Corr_Pitch_Result_Syb" + Syls[obj] + "_tone_" + str(m)+ "_During.txt", statistics2, header="First column is the correlation value, second is the p value. First line is the original correlation, all below are the bootstrapped correlations.")
                    fichier.writelines("Syllable " + Syls[obj] + "_tone_" + str(m)+ "_During:" + str(final) + "\n")
                    os.chdir("..")
                    print(final)                  
                a2.set_xlabel("Number of Lags")
                a2.set_ylabel("Autocorrelation score")
                a3.hist(freq2, bins=int(np.mean(freq2)*0.01))
                a3.set_xlabel("Frequency (Hz)")
                a1.set_xlabel("Number of Lags")
                a1.set_ylabel("Autocorrelation score")
                a1.set_label(tellme("Now let's select the frequency. Key click (x2) for yes, mouse click for no")) #Here you will be asked to select a point in the peak that could represent the frequency (just to get an estimation)
                if not py.waitforbuttonpress(30):
                    os.chdir("Figures")
                    py.savefig("Corr_Pitch_syb"+ Syls[obj] +"_tone"+ str(m)+".jpg")
                    py.close()
                    os.chdir("..")
                    continue            
                else:
                    py.waitforbuttonpress(30)
                    while True:
                        freq = []
                        while len(freq) < 1:
                            tellme("Select the point for the frequency.")
                            freq = np.asarray(py.ginput(1, timeout=-1, show_clicks=True))
                        scat= a1.scatter(freq[:,0],freq[:,1], s=50, marker="X", zorder=10, c="b") 
                        ann=a1.annotate(str(int(np.reciprocal(freq[:,0]/fs))) +" Hz", xy=(freq[:,0],freq[:,1]), xytext=(freq[:,0]*1.2,freq[:,1]*1.2),
                                    arrowprops=dict(facecolor="black", shrink=0.05))
                                    
                        tellme("Happy? Key click for yes, mouse click for no")
                        if py.waitforbuttonpress(30):
                            os.chdir("Figures")
                            py.savefig("Corr_Pitch_syb"+ Syls[obj] +"_tone"+ str(m)+".jpg")
                            py.close()
                            os.chdir("..")
                            break
                        else:
                            ann.remove()
                            scat.remove()
			        
## 
#
# This function can be used to obtain the amplitude and its correlations of specific tones inside a syllable.
# It will allow you to work with the means or the area under the curve (integration)
#
# Arguments:
#
# songfile is the .npy file containing the song signal.
#
# motifile is the .txt file containing the annotations of the beggining and end of each syllable/motif.
#
# fs is the sampling rate.
#
# means is the .txt that contains the cutting points for the tones. If None, it will allow you to create this list of means by visual inspection of plots. 
def corramplitude(songfile, motifile, fs, spikefile, window_size, n_iterations, alpha, premot):
    
    #Read and import files that will be needed
    spused=np.loadtxt(spikefile)
    song=np.load(songfile)
    finallist=sortsyls(motifile)  
    f = open("SummaryCorrAmp.txt", "w+")
    y=["MeanA.txt","MeanB.txt","MeanC.txt","MeanD.txt"]
    Syls=["A","B","C","D"]
    check=jumpsyl(spikefile[3:-4])
    for g in range(len(finallist)):
        if Syls[g] in check:
            continue
        else:
            used=finallist[g]
            means = np.loadtxt("..\\..\\"+y[g]).astype(int)
            syb=song[int(used[0][0]):int(used[0][1])]
            
            # Autocorrelation and Distribution 
            for m in range(1,len(means)):
                spikespremot=[]
                spikesdur=[]
                amps=[]
                fig=py.figure(figsize=(18,15))
                gs=py.GridSpec(2,3)
                a1=fig.add_subplot(gs[0,:]) # First row, first column
                a2=fig.add_subplot(gs[1,0]) # First row, second column
                a3=fig.add_subplot(gs[1,1])
                a4=fig.add_subplot(gs[1,2])
                statistics=[]
                statistics2=[]
                for n in range(len(used)):
                    syb=song[int(used[n][0]):int(used[n][1])] #Will get the syllables for each rendition
                    sybcut=syb[means[m-1]:means[m]] #Will apply the cuts for the syllable
                    smooth=smoothed(np.ravel(sybcut),fs)
                    beg=(used[n][0] + means[m-1])/fs
                    end=(used[n][0] + means[m])/fs
                    step1=spused[np.where(np.logical_and(spused >= beg-premot, spused <= beg) == True)]
                    step2=spused[np.where(np.logical_and(spused >= beg, spused <= end) == True)]
                    spikespremot+=[[np.size(step1)/(beg-(beg-premot))]]
                    spikesdur+=[[np.size(step2)/(end-beg)]]
                    amps+=[np.mean(smooth)]
                a1.plot(abs(sybcut))
                a1.plot(smooth)
                a1.set_title("Syllable " + Syls[g] + " Tone " + str(m))
                a1.set_ylabel("Amplitude")
                a1.set_xlabel("Sample points")
                spikesdur=np.array(spikesdur)[:,0]
                spikespremot=np.array(spikespremot)[:,0]
                amps=np.array(amps)
                total = np.column_stack((amps,spikespremot,spikesdur))
                os.chdir("Results")
                np.savetxt("Data_Raw_Corr_Amplitude_Result_Syb" + Syls[g] + "_tone_" + str(m) + "_Mean.txt", total, header="First column is the amplitude value, second is the number of spikes inside premotor window, third is the number of spikes inside 'during' window.")
                os.chdir("..")
                total1=np.column_stack((amps,spikespremot))
                total2=np.column_stack((amps,spikesdur))
                a2.hist(amps)
                a2.set_title("Distribution of the Raw Means")
                a2.set_ylabel("Frequency")
                a2.set_xlabel("Mean Values")
                #Start of Correlations
                threshold = 3 #Standard Deviation threshold for Z score identification of outliers
                z1 = np.abs(scipy.stats.zscore(total1))
                z2 = np.abs(scipy.stats.zscore(total2))
                total1=total1[(z1 < threshold).all(axis=1)]
                total2=total2[(z2 < threshold).all(axis=1)]
                a = total1[:,1] == 0
                b = total2[:,1] == 0
                if len(total1) < 3 or all(a) == True:
                    pass
                else:
                    s1=scipy.stats.shapiro(total1[:,0])[1] #Amplitude column
                    s2=scipy.stats.shapiro(total1[:,1])[1] #Premot Column
                    homo=scipy.stats.levene(total1[:,0],total[:,1])[1]
                    comb1=np.array([s1,s2,homo])
                    comb1=comb1>alpha
                    #This will get the data for Amplitude vs Premotor
                    if  comb1.all() == True: #test for normality
                        final=scipy.stats.pearsonr(total1[:,0],total1[:,1]) #if this is used, outcome will have no clear name on it
                        statistics+=[[final[0],final[1]]]
                        # Bootstrapping
                        for q in range(n_iterations):
                            resample=np.random.choice(total1[:,0], len(total1[:,0]), replace=True)
                            res=scipy.stats.spearmanr(total1[:,1],resample)
                            statistics+=[[res[0],res[1]]]
                    else: 
                        final=scipy.stats.spearmanr(total1[:,0],total1[:,1]) #if this is used, outcome will have the name spearman on it
                        statistics+=[[final[0],final[1]]]
                        # Bootstrapping
                        for q in range(n_iterations):
                            resample=np.random.choice(total1[:,0], len(total1[:,0]), replace=True)
                            res=scipy.stats.spearmanr(total1[:,1],resample)
                            statistics+=[[res[0],res[1]]]
                    os.chdir("Results")
                    np.savetxt("Data_Boot_Corr_Amplitude_Result_Syb" + Syls[g] + "_tone_" + str(m)+ "_Premotor_Mean.txt", statistics, header="First column is the correlation value, second is the p value. First line is the original correlation, all below are the bootstrapped correlations.")
                    os.chdir("..")
                    f.writelines("Syllable " + Syls[g] + "_tone_" + str(m)+ "_Premotor:" + str(final) + "\n")
                    print(final)
                    a3.hist(np.array(statistics)[:,0])
                    a3.set_title("Bootstrap Premotor")
                    a3.set_xlabel("Correlation Values")
                #This will get the data for Amplitude vs During     
                if len(total2) < 3 or all(b) == True:
                    pass
                else:
                    s1=scipy.stats.shapiro(total2[:,0])[1] #Amplitude column
                    s2=scipy.stats.shapiro(total2[:,1])[1] #During Column
                    homo=scipy.stats.levene(total2[:,0],total2[:,1])[1]
                    comb1=np.array([s1,s2,homo])
                    comb1=comb1>alpha
                    if  comb1.all() == True: #test for normality
                        final=scipy.stats.pearsonr(total2[:,0],total2[:,1]) #if this is used, outcome will have no clear name on it
                        statistics2+=[[final[0],final[1]]]
                        # Bootstrapping
                        for q in range(n_iterations):
                            resample=np.random.choice(total2[:,0], len(total2[:,0]), replace=True)
                            res=scipy.stats.spearmanr(total2[:,1],resample)
                            statistics2+=[[res[0],res[1]]]
                    else: 
                        final=scipy.stats.spearmanr(total2[:,0],total2[:,1]) #if this is used, outcome will have the name spearman on it
                        statistics2+=[[final[0],final[1]]]
                        # Bootstrapping
                        for q in range(n_iterations):
                            resample=np.random.choice(total2[:,0], len(total2[:,0]), replace=True)
                            res=scipy.stats.spearmanr(total2[:,1],resample)
                            statistics2+=[[res[0],res[1]]]
                    os.chdir("Results")
                    np.savetxt("Data_Boot_Corr_Amplitude_Result_Syb" + Syls[g] + "_tone_" + str(m)+ "_During_Mean.txt", statistics2, header="First column is the correlation value, second is the p value. First line is the original correlation, all below are the bootstrapped correlations.")
                    os.chdir("..")
                    f.writelines("Syllable " + Syls[g] + "_tone_" + str(m)+ "_During:" + str(final) + "\n")
                    a4.hist(np.array(statistics2)[:,0])
                    a4.set_title("Bootstrap During")
                    a4.set_xlabel("Correlation Values")
                    print(final)
                os.chdir("Figures")
                py.savefig(fname="Corr_Amplitude_syb"+ Syls[g] +"_tone"+ str(m) +".jpg")
                py.close()
                os.chdir("..")
##
# This function computes the Spectral Entropy of a signal. 
#The power spectrum is computed through fft. Then, it is normalised and assimilated to a probability density function.
#
# Arguments:
#    ----------
#    signal : list or array
#        List or array of values.
#    sampling_rate : int
#        Sampling rate (samples/second).
#    bands : list or array
#        A list of numbers delimiting the bins of the frequency bands. If None the entropy is computed over the whole range of the DFT (from 0 to `f_s/2`).
#
#    Returns
#    ----------
#    spectral_entropy : float
#        The spectral entropy as float value.
def complexity_entropy_spectral(signal, sampling_rate, bands=None):
    """
    Based on the `pyrem <https://github.com/gilestrolab/pyrem>`_ repo by Quentin Geissmann.
    
    Example
    ----------
    >>> import neurokit as nk
    >>>
    >>> signal = np.sin(np.log(np.random.sample(666)))
    >>> spectral_entropy = nk.complexity_entropy_spectral(signal, 1000)

    Notes
    ----------
    *Details*

    - **Spectral Entropy**: Entropy for different frequency bands.


    *Authors*

    - Quentin Geissmann (https://github.com/qgeissmann)

    *Dependencies*

    - numpy

    *See Also*

    - pyrem package: https://github.com/gilestrolab/pyrem
    """

    psd = np.abs(np.fft.rfft(signal))**2
    psd /= np.sum(psd) # psd as a pdf (normalised to one)

    if bands is None:
        power_per_band= psd[psd>0]
    else:
        freqs = np.fft.rfftfreq(signal.size, 1/float(sampling_rate))
        bands = np.asarray(bands)

        freq_limits_low = np.concatenate([[0.0],bands])
        freq_limits_up = np.concatenate([bands, [np.Inf]])

        power_per_band = [np.sum(psd[np.bitwise_and(freqs >= low, freqs<up)])
                for low,up in zip(freq_limits_low, freq_limits_up)]

        power_per_band= np.array(power_per_band)[np.array(power_per_band) > 0]

    spectral = - np.sum(power_per_band * np.log2(power_per_band))
    return(spectral)

## 
#
# This function can be used to obtain the spectral entropy and its correlations of specific tones inside a syllable.
#
# Arguments:
#
# songfile is the .npy file containing the song signal.
#
# motifile is the .txt file containing the annotations of the beggining and end of each syllable/motif.
#
# fs is the sampling rate
#
# means is the a .txt that contains the cutting points for the tones. If None, it will allow you to create this list of means by visual inspection of plots. 
def corrspectral(songfile, motifile, fs, spikefile, window_size, n_iterations, alpha, premot):
    spused=np.loadtxt(spikefile)
    song=np.load(songfile)
    finallist=sortsyls(motifile)  
    f = open("SummaryCorrSpecEnt.txt", "w+")
    y=["MeanA.txt","MeanB.txt","MeanC.txt","MeanD.txt"]
    Syls=["A","B","C","D"]
    check=jumpsyl(spikefile[3:-4])  
    for g in range(len(finallist)):
        if Syls[g] in check:
            continue
        else:
            used=finallist[g]
            means = np.loadtxt("..\\..\\"+y[g]).astype(int)
            syb=song[int(used[0][0]):int(used[0][1])]
            # Autocorrelation and Distribution 
            for m in range(1,len(means)):
                spikespremot=[]
                spikesdur=[]
                specent=[]
                fig=py.figure(figsize=(18,15))
                gs=py.GridSpec(1,3)
                a2=fig.add_subplot(gs[0,0]) # First row, second column
                a3=fig.add_subplot(gs[0,1])
                a4=fig.add_subplot(gs[0,2])
                statistics=[]
                statistics2=[]
                for n in range(len(used)):
                    syb=song[int(used[n][0]):int(used[n][1])] #Will get the syllables for each rendition
                    sybcut=syb[means[m-1]:means[m]] #Will apply the cuts for the syllable
                    SE=complexity_entropy_spectral(sybcut[:,0],fs)
                    beg=(used[n][0] + means[m-1])/fs
                    end=(used[n][0] + means[m])/fs
                    step1=spused[np.where(np.logical_and(spused >= beg-premot, spused <= beg) == True)]
                    step2=spused[np.where(np.logical_and(spused >= beg, spused <= end) == True)]
                    spikespremot+=[[np.size(step1)/(beg-(beg-premot))]]
                    spikesdur+=[[np.size(step2)/(end-beg)]]
                    specent+=[[SE]]
                fig.suptitle("Syllable " + Syls[g] + " Tone " + str(m))
                spikesdur=np.array(spikesdur)[:,0]
                spikespremot=np.array(spikespremot)[:,0]
                specent=np.array(specent)
                total = np.column_stack((specent,spikespremot,spikesdur))
                os.chdir("Results")
                np.savetxt("Data_Raw_Corr_SpecEnt_Result_Syb" + Syls[g] + "_tone_" + str(m) + ".txt", total, header="First column is the spectral value, second is the number of spikes inside premotor window, third is the number of spikes inside 'during' window.")
                os.chdir("..")
                #Here it will give you the possibility of computing the correlations and Bootstrapping
                threshold = 3 #Standard Deviation threshold for Z score identification of outliers
                total1=np.column_stack((specent,spikespremot))
                total2=np.column_stack((specent,spikesdur))
                z1 = np.abs(scipy.stats.zscore(total1))
                z2 = np.abs(scipy.stats.zscore(total2))
                total1=total1[(z1 < threshold).all(axis=1)]
                total2=total2[(z2 < threshold).all(axis=1)]
                a = total1[:,1] == 0
                b = total2[:,1] == 0
                a2.hist(specent)
                a2.set_title("Distribution of the Raw Spectral Entropy")
                a2.set_ylabel("Frequency")
                a2.set_xlabel("Spectral Values")
                #This will get the data for Spectral Entropy vs Premotor
                if len(total1) < 3 or all(a) == True:
                    pass
                else:
                    s1=scipy.stats.shapiro(total1[:,0])[1] #Spectral Entropy column
                    s2=scipy.stats.shapiro(total1[:,1])[1] #Premot Column
                    homo=scipy.stats.levene(total1[:,0],total[:,1])[1]
                    comb1=np.array([s1,s2,homo])
                    comb1=comb1>alpha
                    if  comb1.all() == True: #test for normality
                        final=scipy.stats.pearsonr(total1[:,0],total1[:,1]) #if this is used, outcome will have no clear name on it
                        statistics+=[[final[0],final[1]]]
                        # Bootstrapping
                        for q in range(n_iterations):
                            resample=np.random.choice(total1[:,0], len(total1[:,0]), replace=True)
                            res=scipy.stats.spearmanr(total1[:,1],resample)
                            statistics+=[[res[0],res[1]]]
                    else: 
                        final=scipy.stats.spearmanr(total1[:,0],total1[:,1]) #if this is used, outcome will have the name spearman on it
                        statistics+=[[final[0],final[1]]]
                        # Bootstrapping
                        for q in range(n_iterations):
                            resample=np.random.choice(total1[:,0], len(total1[:,0]), replace=True)
                            res=scipy.stats.spearmanr(total1[:,1],resample)
                            statistics+=[[res[0],res[1]]]
                    os.chdir("Results")
                    np.savetxt("Data_Boot_Corr_SpecEnt_Result_Syb" + Syls[g] + "_tone_" + str(m)+ "_Premotor.txt", statistics, header="First column is the correlation value, second is the p value. First line is the original correlation, all below are the bootstrapped correlations.")
                    os.chdir("..")
                    f.writelines("Syllable " + Syls[g] + "_tone_" + str(m)+ "_Premotor:" + str(final) + "\n")
                    print(final)
                    a3.hist(np.array(statistics)[:,0])
                    a3.set_title("Bootstrap Premotor")
                    a3.set_xlabel("Correlation Values")
                #This will get the data for Spectral Entropy vs During     
                if len(total2) < 3 or all(b) == True:
                    pass
                else:
                    s1=scipy.stats.shapiro(total2[:,0])[1] #Spectral Entropy column
                    s2=scipy.stats.shapiro(total2[:,1])[1] #During Column
                    homo=scipy.stats.levene(total2[:,0],total2[:,1])[1]
                    comb1=np.array([s1,s2,homo])
                    comb1=comb1>alpha
                    if  comb1.all() == True: #test for normality
                        final=scipy.stats.pearsonr(total2[:,0],total2[:,1]) #if this is used, outcome will have no clear name on it
                        statistics2+=[[final[0],final[1]]]
                        # Bootstrapping
                        for q in range(n_iterations):
                            resample=np.random.choice(total2[:,0], len(total2[:,0]), replace=True)
                            res=scipy.stats.spearmanr(total2[:,1],resample)
                            statistics2+=[[res[0],res[1]]]
                    else: 
                        final=scipy.stats.spearmanr(total2[:,0],total2[:,1]) #if this is used, outcome will have the name spearman on it
                        statistics2+=[[final[0],final[1]]]
                        # Bootstrapping
                        for q in range(n_iterations):
                            resample=np.random.choice(total2[:,0], len(total2[:,0]), replace=True)
                            res=scipy.stats.spearmanr(total2[:,1],resample)
                            statistics2+=[[res[0],res[1]]]
                    os.chdir("Results")
                    np.savetxt("Data_Boot_Corr_SpecEnt_Result_Syb" + Syls[g] + "_tone_" + str(m)+ "_During.txt", statistics2, header="First column is the correlation value, second is the p value. First line is the original correlation, all below are the bootstrapped correlations.")   
                    os.chdir("..")
                    f.writelines("Syllable " + Syls[g] + "_tone_" + str(m)+ "_During:" + str(final) + "\n")
                    print(final)
                    a4.hist(np.array(statistics2)[:,0])
                    a4.set_title("Bootstrap During")
                    a4.set_xlabel("Correlation Values")
                os.chdir("Figures")
                py.savefig("Corr_SpecEnt_syb"+ Syls[g] +"_tone"+ str(m)+".jpg")
                py.close()
                os.chdir("..")
    
    
def gettones(songfile,motifile,fs, window_size):
    finallist=sortsyls(motifile)  
    song=np.load(songfile)
    Syls=["A","B","C","D"]
    
    #Will plot an exmaple of the syllable for you to get an idea of the number of chunks
    for x in range(len(finallist)):        
        fig, az = py.subplots()
        used=finallist[x]
        example=song[int(used[0][0]):int(used[0][1])]
        abso=abs(example)
        az.plot(example)
        az.plot(abso)
        rms=window_rms(np.ravel(example),window_size)
        az.plot(rms)
        az.set_title("This is syb "+ Syls[x] + ".  Click on graph to move on.")
        py.waitforbuttonpress(10)
        if input("Want to keep?").lower() == "n":
            continue
        else:            
            numcuts=int(input("Number of chunks?"))
            py.close()
        
        # Will provide you 4 random exmaples of syllables to stablish the cutting points
        coords2=[]
        for j in range(4):           
           j=random.randint(0,len(used)-1)
           fig, ax = py.subplots()
           syb=song[int(used[j][0]):int(used[j][1])]
           abso=abs(syb)
           ax.plot(abso)
           rms=window_rms(np.ravel(syb),window_size)
           ax.plot(rms)
           py.waitforbuttonpress(10)
           while True:
               coords = []
               while len(coords) < numcuts+1:
                   tellme("Select the points to cut with mouse")
                   coords = np.asarray(py.ginput(numcuts+1, timeout=-1, show_clicks=True))
               scat = py.scatter(coords[:,0],coords[:,1], s=50, marker="X", zorder=10, c="r")    
               tellme("Happy? Key click for yes, mouse click for no")
               if py.waitforbuttonpress():
                   break
               else:
                   scat.remove()
           py.close()
           coords2=np.append(coords2,coords[:,0])
        
        #Will keep the mean coordinates for the cuts
        coords2.sort()
        coords2=np.split(coords2,numcuts+1)
        means=[]
        for k in range(len(coords2)):
            means+=[int(np.mean(coords2[k]))]
        np.savetxt("../Mean"+Syls[x]+".txt", means) 
        # Will plot how the syllables will be cut according to the avarage of the coordinates clicked before by the user    
        py.figure()
        py.plot(syb)
        for l in range(1,len(means)):
            py.plot(np.arange(means[l-1],means[l-1]+len(syb[means[l-1]:means[l]])),syb[means[l-1]:means[l]])
        py.savefig("../Cut"+ Syls[x]+".jpg")

def ISI(spikefile):
    spikes=np.loadtxt(spikefile)
    times=np.sort(np.diff(spikes))*1000
    py.hist(times, bins= np.arange(np.min(times), np.max(times), 1))
    py.xscale('log')
    py.xlabel("Millisecond (ms)")
    py.ylabel("Counts/bin")