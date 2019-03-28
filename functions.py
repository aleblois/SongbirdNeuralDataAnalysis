## @author: eduar
#  Documentation for this module.
#
#  Created on Wed Feb  6 15:06:12 2019; -*- coding: utf-8 -*-; 

# Necessary packages
import neo
import numpy as np
import pylab as py
import os
import datetime
import pandas
import scipy.io
import scipy.signal
import scipy.stats
import scipy.fftpack
import scipy.interpolate
import random
from statsmodels.tsa.stattools import acf

#file="CSC1_light_LFPin.smr" #Here you define the .smr file that will be analysed
#songfile="CSC10.npy" #Here you define which is the file with the raw signal of the song
#motifile="rawsong_06_05_2018_annot.txt" #Here you define what is the name of the file with the motif stamps/times

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

    
## 
#
# This function will get and save the spikeshapes (.txt) from the LFP signal.
# Arguments:
#
# file is the .smr file
#
# LFPfile is the .npy file containing the LFP signal
#
# notLFPfile is the .npy containing the neuron data, but which of course is not the LFP nor the Song.    
def spikeshapes(file, LFPfile, notLFPfile):
    data, _= read(file)
    _, n_spike_trains, _, ansampling_rate = getinfo(file)
    LFP=np.load(LFPfile)
    notLFP=np.load(notLFPfile)
    windowsize=int(ansampling_rate*2/1000) #Define here the number of points that suit your window (set to 2ms)
    # Create and save the spikeshapes
    # This part will iterate through all the .txt files containing the spiketimes inside the folder.
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
            analogtxt1=LFP[a1:a1+windowsize].reshape(1,windowsize)
            y1 = np.array([[a1],[a1+windowsize]], np.int32).reshape(1,2)
            res1 = np.append(y1,analogtxt1).reshape(1,-1)
            x1=np.append(x1,res1, axis=0)
        b1=x1[1:]    
        print("\n" + "Voilà!")   
        np.savetxt("SpikeShape#"+Label1+".txt", b1, header="First column = Initial Time; Second column = Final Time; Third Column = First Spike Shape value, etc")
        if answer4 == "" or answer4.lower()[0] == "y":
            window1=int(b1[0][0])
            window2=int(b1[0][1])
            py.figure()
            py.subplot(2,1,1)
            py.plot(LFP[window1:window2])
            py.subplot(2,1,2)
            py.plot(notLFP[window1+57:window2+57])
            py.tight_layout()
            py.show()
 
           
## 
#
# This function will downsample your LFP signal to 1000Hz and save it as .npy file
def lfpdown(LFPfile): #LFPfile is the .npy one inside the new folder generated by the function createsave (for example, CSC1.npy)
    ana=np.load(LFPfile)
    def mma(series,window):
        return np.convolve(series,np.repeat(1,window)/window,"same")
    
    s_ana=ana[0:][:,0] #window of the array
    s_ana_1=mma(s_ana,100) #convolved version
    c=[]
    for i in range(len(s_ana_1)):
        if i%32==0:
            c+=[s_ana_1[i]]
            
    d=np.array(c)
    np.save("LFPDownsampled", d)
    answer=input("Want to see the plots? Might be a bit heavy. [Y/n]")
    if answer == "" or answer.lower()[0] == "y":
        py.fig,(s,s1) = py.subplots(2,1) 
        s.plot(s_ana)
        s.plot(s_ana_1)
        s.set_title("Plot of RawSignal X Convolved Version")
        s1.plot(c)
        s1.set_title("LFP Downsampled")
    py.show()
    py.tight_layout()
    return d


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
    window =("hamming")
    overlap = 64
    nperseg = 1024
    noverlap = nperseg-overlap
    #Compute and plot spectrogram
    (f,t,sp)=scipy.signal.spectrogram(rawsong, fs, window, nperseg, noverlap, mode="complex")
    py.figure()
    py.subplot(2,1,1)
    py.plot(rawsong)
    py.subplot(2,1,2)
    py.imshow(10*np.log10(np.square(abs(sp))), origin="lower", aspect="auto", interpolation="none", cmap="inferno")
    py.colorbar()
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
def psth(spikefile, motifile):        
    #Read and import mat file (new version)
    f=open(motifile, "r")
    imported = f.read().splitlines()
    samplingrate=32000
    #Excludes everything that is not a real syllable
    a=[] ; b=[] ; c=[] ; d=[]; e=[]
    arra=np.empty((1,2)); arrb=np.empty((1,2)); arrc=np.empty((1,2))
    arrd=np.empty((1,2)); arre=np.empty((1,2))
    for i in range(len(imported)):
        if imported[i][-1] == "a":
            a=[imported[i].split(",")]
            arra=np.append(arra, np.array([int(a[0][0])/samplingrate, int(a[0][1])/samplingrate], float).reshape(1,2), axis=0)
        if imported[i][-1] == "b": 
            b=[imported[i].split(",")]
            arrb=np.append(arrb, np.array([int(b[0][0])/samplingrate, int(b[0][1])/samplingrate], float).reshape(1,2), axis=0)
        if imported[i][-1] == "c": 
            c=[imported[i].split(",")]  
            arrc=np.append(arrc, np.array([int(c[0][0])/samplingrate, int(c[0][1])/samplingrate], float).reshape(1,2), axis=0)
        if imported[i][-1] == "d": 
            d=[imported[i].split(",")] 
            arrd=np.append(arrd, np.array([int(d[0][0])/samplingrate, int(d[0][1])/samplingrate], float).reshape(1,2), axis=0)
        if imported[i][-1] == "e": 
            e=[imported[i].split(",")]   
            arre=np.append(arre, np.array([int(e[0][0])/samplingrate, int(e[0][1])/samplingrate], float).reshape(1,2), axis=0)
            
    arra=arra[1:]; arrb=arrb[1:]; arrc=arrc[1:]; arrd=arrd[1:] ; arre=arre[1:]  
    #Starts to plot the PSTH
    spused=np.loadtxt(spikefile)
    shoulder= 0.05 #50 ms
    binwidth=0.02
    tes=0
    sep=0
    adjust=0
    meandurall=0
    py.fig, ax = py.subplots(2,1)
    k=[arra,arrb,arrc,arrd]
    x2=[]
    y2=[]
    # This part will result in an iteration through all the syllables, and then through all the motifs inside each syllable. 
    for i in range(len(k)):
            used=k[i] # sets which array from k will be used.
            adjust+=meandurall+sep
            meandurall=np.mean(used[:,1]-used[:,0])
            spikes1=[]
            res=-1
            spikes=[]
            n0,n1=0,2
            for j in range(len(used)):
                step1=[]
                step2=[]
                step3=[]
                beg= used[j][0] #Will compute the beginning of the window
                end= used[j][1] #Will compute the end of the window
                step1=spused[np.where(np.logical_and(spused >= beg-shoulder, spused <= end+shoulder) == True)]-beg
                step2=step1[np.where(np.logical_and(step1 >= 0, step1 <= end-beg) == True)]*(meandurall/(end-beg))
                step3=step1[np.where(np.logical_and(step1 >= end-beg, step1 <= (end-beg)+shoulder) == True)]+(meandurall-(end-beg))
                spikes1+=[step2+adjust,step3+adjust]
                res=res+1
                spikes2=spikes1
                spikes3=spikes1
                spikes2=np.concatenate(spikes2[n0:n1])
                ax[1].scatter(spikes2,res+np.zeros(len(spikes2)),marker="|")
                n0+=2
                n1+=2
                sep=0.08
                ax[1].set_xlim(-shoulder,(shoulder+meandurall)+binwidth+adjust)
                ax[1].set_ylabel("Motif number")
                ax[1].set_xlabel("Time [s]")
                normfactor=len(used)*binwidth
                ax[0].set_ylabel("Spikes/s")
                bins=np.arange(-shoulder,(shoulder+meandurall)+binwidth, step=binwidth)
                ax[0].set_xlim(-shoulder,(shoulder+meandurall)+binwidth+adjust)
                ax[0].tick_params(
                        axis="x",          # changes apply to the x-axis
                        which="both",      # both major and minor ticks are affected
                        bottom=False,      # ticks along the bottom edge are off
                        top=False,         # ticks along the top edge are off
                        labelbottom=False)
            spikes=np.sort(np.concatenate(spikes3))
            y1,x1= py.histogram(spikes, bins=bins+adjust, weights=np.ones(len(spikes))/normfactor)
            ax[0].hist(spikes, bins=bins+adjust, color="b", edgecolor="black", linewidth=1, weights=np.ones(len(spikes))/normfactor, align="left")
            x2+=[x1[tes:]]
            y=np.append(y1,0)
            y2+=[y[tes:]]
            tes=2 #This is necessary for the proper execution of the interpolation
    x2=np.concatenate(x2)
    x2=np.sort(x2)
    y2=np.concatenate(y2)
    f = scipy.interpolate.interp1d(x2, y2, kind='cubic')
    xnew=np.linspace(min(x2),max(x2), num=1000)
    ax[0].plot(xnew,f(xnew), color="r")
    py.fig.subplots_adjust(hspace=0)

    
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
def correlation(spikefile, motifile, n_iterations,fs):      
    #Read and import mat file (new version)
    f=open(motifile, "r")
    imported = f.read().splitlines()
    samplingrate=fs
    #Excludes everything that is not a real syllable
    a=[] ; b=[] ; c=[] ; d=[]; e=[]
    sybs=["A","B","C","D","E"]
    arra=np.empty((1,2)); arrb=np.empty((1,2)); arrc=np.empty((1,2))
    arrd=np.empty((1,2)); arre=np.empty((1,2))
    for i in range(len(imported)):
        if imported[i][-1] == "a":
            a=[imported[i].split(",")]
            arra=np.append(arra, np.array([int(a[0][0])/samplingrate, int(a[0][1])/samplingrate], float).reshape(1,2), axis=0)
        if imported[i][-1] == "b": 
            b=[imported[i].split(",")]
            arrb=np.append(arrb, np.array([int(b[0][0])/samplingrate, int(b[0][1])/samplingrate], float).reshape(1,2), axis=0)
        if imported[i][-1] == "c": 
            c=[imported[i].split(",")]  
            arrc=np.append(arrc, np.array([int(c[0][0])/samplingrate, int(c[0][1])/samplingrate], float).reshape(1,2), axis=0)
        if imported[i][-1] == "d": 
            d=[imported[i].split(",")] 
            arrd=np.append(arrd, np.array([int(d[0][0])/samplingrate, int(d[0][1])/samplingrate], float).reshape(1,2), axis=0)
        if imported[i][-1] == "e": 
            e=[imported[i].split(",")]   
            arre=np.append(arre, np.array([int(e[0][0])/samplingrate, int(e[0][1])/samplingrate], float).reshape(1,2), axis=0)
            
    arra=arra[1:]; arrb=arrb[1:]; arrc=arrc[1:]; arrd=arrd[1:] ; arre=arre[1:]
    dura=arra[:,1]-arra[:,0]; durb=arrb[:,1]-arrb[:,0];durc=arrc[:,1]-arrc[:,0]; durd=arrd[:,1]-arrd[:,0]; dure=arre[:,1]-arre[:,0]
    #Starts to compute correlations and save the data into txt file (in case the user wants to use it in another software)
    spused=np.loadtxt(spikefile)
    k=[arra,arrb,arrc,arrd]
    g=[dura,durb,durc,durd,dure]
    final=[]
    for i in range(len(k)):
            used=k[i]
            dur=g[i]
            array=np.empty((1,2))
            statistics=[]
            for j in range(len(used)):
                step1=[]
                beg= used[j][0] #Will compute the beginning of the window
                end= used[j][1] #Will compute the end of the window
                step1=spused[np.where(np.logical_and(spused >= beg, spused <= end) == True)]
                array=np.append(array, np.array([[dur[j]],[np.size(step1)/dur[j]]]).reshape(-1,2), axis=0)
            array=array[1:]
            #np.savetxt("Data_Corr_Dur_syb"+str(sybs[i])+".txt", array)
            threshold = 3 #Standard Deviation threshold for Z score identification of outliers
            z = np.abs(scipy.stats.zscore(array))
            array=array[(z < threshold).all(axis=1)]
            alpha=0.05
            s1=scipy.stats.shapiro(array[:,0])[1]
            s2=scipy.stats.shapiro(array[:,1])[1]
            s3=np.array([s1,s2])
            homo=scipy.stats.levene(array[:,0],array[:,1])[1]
            if  s3.all() > alpha and homo > alpha: #test for normality
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
            np.savetxt("Data_Boot_Corr_Result_Syb"+str(sybs[i])+".txt", np.array(statistics)) #First column is the correlation value, second is the p value.
            print("Syllable " + str(sybs[i]) +": " + str(final))                

          
## 
#
# This function allows you to get the envelope for song signal.
#
# Arguments:
#
# songfile is the .npy file containing the song signal.
#
# beg, end : are the index that would correspond to the beginning and the end of the motif/syllable (check syllables annotations file for that)
#
# window_size is the size of the window for the convolve function. 
def getEnvelope(songfile, beg, end, window_size): 
    inputSignal=np.load(songfile)
    inputSignal=np.ravel(inputSignal[beg:end])
    def window_rms(inputSignal, window_size):
        a2 = np.power(inputSignal,2)
        window = np.ones(window_size)/float(window_size)
        return np.sqrt(np.convolve(a2, window, 'valid'))
    
    def getEnvelope(inputSignal):
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
    
    outputSignal=getEnvelope(inputSignal)
    rms=window_rms(inputSignal, window_size)
    
    # Plots of the envelopes
    py.fig, (a,b,c) =py.subplots(3,1)
    a.plot(abs(inputSignal))
    
    b.plot(abs(inputSignal))
    b.plot(outputSignal)
    
    c.plot(abs(inputSignal))
    c.plot(rms)           
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
def fft(songfile, beg, end, fs):
    signal=np.load(songfile) #The song channel raw data
    signal=signal[beg:end] #I selected just one syllable A to test
    fs_rate = fs
    print ("Frequency sampling", fs_rate)
    l_audio = len(signal.shape)
    if l_audio == 2:
        signal = signal.sum(axis=1) / 2
    N = signal.shape[0]
    print ("Complete Samplings N", N)
    secs = N / float(fs_rate)
    print ("secs", secs)
    Ts = 1.0/fs_rate # sampling interval in time
    print ("Timestep between samples Ts", Ts)
    t = scipy.arange(0, secs, Ts) # time vector as scipy arange field / numpy.ndarray
    FFT = abs(scipy.fft(signal))**2 # if **2 is power spectrum, without is amplitude spectrum
    FFT_side = FFT[range(int(N/2))] # one side FFT range
    freqs = scipy.fftpack.fftfreq(signal.size, t[1]-t[0])
    freqs_side = freqs[range(int(N/2))]
    py.subplot(311)
    p1 = py.plot(t, signal, "g") # plotting the signal
    py.xlabel('Time')
    py.ylabel('Amplitude')
    py.subplot(312)
    p2 = py.plot(freqs, FFT, "r") # plotting the complete fft spectrum
    py.xlabel('Frequency (Hz)')
    py.ylabel('Count dbl-sided')
    py.subplot(313)
    p3 = py.plot(freqs_side, abs(FFT_side), "b") # plotting the positive fft spectrum
    py.xlabel('Frequency (Hz)')
    py.ylabel('Count single-sided')
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
# window_size is the size of the window for the convolve function (Envelopes)
def pitch(songfile, motifile, lags, window_size): #spikefile is the txt file with the spiketimes       
    #Read and import mat file (new version)
    song=np.load(songfile)
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
    
    def tellme(s):
        print(s)
        py.title(s, fontsize=16)
        py.draw()
    
    answer=input("Which syllable?")
    if answer.lower() == 'a':
        used=arra
    elif answer.lower() == 'b':
        used=arrb
    elif answer.lower() == 'c':
        used=arrc    
    elif answer.lower() == 'd':
        used=arrd
    
    def window_rms(inputSignal, window_size):
        a2 = np.power(inputSignal,2)
        window = np.ones(window_size)/float(window_size)
        return np.sqrt(np.convolve(a2, window, 'valid'))
    fig, az = py.subplots()
    example=song[int(used[0][0]):int(used[0][1])]
    abso=abs(example)
    az.plot(example)
    az.plot(abso)
    rms=window_rms(np.ravel(example),window_size)
    az.plot(rms)
    az.set_title("Click on graph to move on.")
    py.waitforbuttonpress()
    numcuts=int(input("Number of chunks?"))
    py.close()
    coords2=[]
    for i in range(4):           
       i=random.randint(0,len(used)-1)
       fig, ax = py.subplots()
       syb=song[int(used[i][0]):int(used[i][1])]
       abso=abs(syb)
       ax.plot(abso)
       rms=window_rms(np.ravel(syb),window_size)
       ax.plot(rms)
       py.waitforbuttonpress()
       while True:
           coords = []
           while len(coords) < numcuts+1:
               tellme('Select the points to cut with mouse')
               coords = np.asarray(py.ginput(numcuts+1, timeout=-1, show_clicks=True))
           py.scatter(coords[:,0],coords[:,1], s=50, marker='X', zorder=10, c='r')    
           tellme('Happy? Key click for yes, mouse click for no')
           if py.waitforbuttonpress():
               break
       py.close()
       coords2=np.append(coords2,coords[:,0])
    
    coords2.sort()
    coords2=np.split(coords2,numcuts+1)
    means=[]
    for i in range(len(coords2)):
        means+=[int(np.mean(coords2[i]))]
        
    py.plot(syb)
    for j in range(1,len(means)):
        py.plot(np.arange(means[j-1],means[j-1]+len(syb[means[j-1]:means[j]])),syb[means[j-1]:means[j]])   

    # Autocorrelation
    for j in range(1,len(means)):
        py.figure()
        for i in range(len(used)):
            syb=song[int(used[i][0]):int(used[i][1])]
            sybcut=syb[means[j-1]:means[j]]
            x2=np.arange(0,len(acf(sybcut,nlags=int(lags))),1)
            f=scipy.interpolate.interp1d(x2,acf(sybcut, nlags=int(lags)), kind='quadratic')
            xnew=np.linspace(min(x2),max(x2), num=1000)
            py.plot(xnew,f(xnew))    
