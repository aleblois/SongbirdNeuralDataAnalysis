import numpy as np
import scipy as sp
import neo
from scipy.io import loadmat
import scipy.signal
from scipy.io import wavfile
import matplotlib as mplt
import matplotlib.pyplot as plt
import os
import glob
import math
from threading import Thread

###################################################################################
# Block 0: Define variables and functions
###################################################################################
#Spectrogram parameters
#Default windowing function in spectrogram function
#window =('tukey', 0.25) 
window =('hamming')
overlap = 64
nperseg = 1024
noverlap = nperseg-overlap
colormap = "jet"

###########################################################################
#              Set here threshold for segmentation                        #
###########################################################################
#threshold=1.6e-4 # for files recorded with RTXI from recording box (above the chamber)
threshold=2e-9 # for files recorded with Neuralynx
min_syl_dur=0.02
min_silent_dur= 0.005
smooth_win=10
#Contains the labels of the syllables from a single .wav file
labels = []
syl_counter=0
Nb_syls=0
keep='' #Save chunk of audio data or discard it

# Initial default for smooth_win = 2
def smooth_data(rawsong, samp_freq, freq_cutoffs=None, smooth_win=10):

    if freq_cutoffs is None:
        # then don't do bandpass_filtfilt
        filtsong = rawsong
    else:
        filtsong = bandpass_filtfilt(rawsong, samp_freq, freq_cutoffs)

    squared_song = np.power(filtsong, 2)
	
    len = np.round(samp_freq * smooth_win / 1000).astype(int)
    h = np.ones((len,)) / len
    smooth = np.convolve(squared_song, h)
    offset = round((smooth.shape[-1] - filtsong.shape[-1]) / 2)
    smooth = smooth[offset:filtsong.shape[-1] + offset]
    return smooth

	
def bandpass_filtfilt(rawsong, samp_freq, freq_cutoffs=(500, 10000)):
    """filter song audio with band pass filter, run through filtfilt
    (zero-phase filter)

    Parameters
    ----------
    rawsong : ndarray
        audio
    samp_freq : int
        sampling frequency
    freq_cutoffs : list
        2 elements long, cutoff frequencies for bandpass filter.
        If None, no cutoffs; filtering is done with cutoffs set
        to range from 0 to the Nyquist rate.
        Default is [500, 10000].

    Returns
    -------
    filtsong : ndarray
    """

    if freq_cutoffs[0] <= 0:
        raise ValueError('Low frequency cutoff {} is invalid, '
                         'must be greater than zero.'
                         .format(freq_cutoffs[0]))

    Nyquist_rate = samp_freq / 2
    if freq_cutoffs[1] >= Nyquist_rate:
        raise ValueError('High frequency cutoff {} is invalid, '
                         'must be less than Nyquist rate, {}.'
                         .format(freq_cutoffs[1], Nyquist_rate))

    if rawsong.shape[-1] < 387:
        numtaps = 64
    elif rawsong.shape[-1] < 771:
        numtaps = 128
    elif rawsong.shape[-1] < 1539:
        numtaps = 256
    else:
        numtaps = 512

    cutoffs = np.asarray([freq_cutoffs[0] / Nyquist_rate,
                          freq_cutoffs[1] / Nyquist_rate])
    # code on which this is based, bandpass_filtfilt.m, says it uses Hann(ing)
    # window to design filter, but default for matlab's fir1
    # is actually Hamming
    # note that first parameter for scipy.signal.firwin is filter *length*
    # whereas argument to matlab's fir1 is filter *order*
    # for linear FIR, filter length is filter order + 1
    b = scipy.signal.firwin(numtaps + 1, cutoffs, pass_zero=False)
    a = np.zeros((numtaps+1,))
    a[0] = 1  # make an "all-zero filter"
    padlen = np.max((b.shape[-1] - 1, a.shape[-1] - 1))
    filtsong = scipy.signal.filtfilt(b, a, rawsong, padlen=padlen)
    #filtsong = filter_song(b, a, rawsong)
    return (filtsong)
	
	
def segment_song(amp,
                 segment_params={'threshold': 5000, 'min_syl_dur': 0.2, 'min_silent_dur': 0.02},
                 time_bins=None,
                 samp_freq=None):
    """Divides songs into segments based on threshold crossings of amplitude.
    Returns onsets and offsets of segments, corresponding (hopefully) to syllables in a song.
    Parameters
    ----------
    amp : 1-d numpy array
        Either amplitude of power spectral density, returned by compute_amp,
        or smoothed amplitude of filtered audio, returned by evfuncs.smooth_data
    segment_params : dict
        with the following keys
            threshold : int
                value above which amplitude is considered part of a segment. default is 5000.
            min_syl_dur : float
                minimum duration of a segment. default is 0.02, i.e. 20 ms.
            min_silent_dur : float
                minimum duration of silent gap between segment. default is 0.002, i.e. 2 ms.
    time_bins : 1-d numpy array
        time in s, must be same length as log amp. Returned by Spectrogram.make.
    samp_freq : int
        sampling frequency

    Returns
    -------
    onsets : 1-d numpy array
    offsets : 1-d numpy array
        arrays of onsets and offsets of segments.

    So for syllable 1 of a song, its onset is onsets[0] and its offset is offsets[0].
    To get that segment of the spectrogram, you'd take spect[:,onsets[0]:offsets[0]]
    """

    if time_bins is None and samp_freq is None:
        raise ValueError('Values needed for either time_bins or samp_freq parameters '
                         'needed to segment song.')
    if time_bins is not None and samp_freq is not None:
        raise ValueError('Can only use one of time_bins or samp_freq to segment song, '
                         'but values were passed for both parameters')

    if time_bins is not None:
        if amp.shape[-1] != time_bins.shape[-1]:
            raise ValueError('if using time_bins, '
                             'amp and time_bins must have same length')

    above_th = amp > segment_params['threshold']
    h = [1, -1]
    # convolving with h causes:
    # +1 whenever above_th changes from 0 to 1
    # and -1 whenever above_th changes from 1 to 0
    above_th_convoluted = np.convolve(h, above_th)

    if time_bins is not None:
        # if amp was taken from time_bins using compute_amp
        # note that np.where calls np.nonzero which returns a tuple
        # but numpy "knows" to use this tuple to index into time_bins
        onsets = time_bins[np.where(above_th_convoluted > 0)]
        offsets = time_bins[np.where(above_th_convoluted < 0)]
    elif samp_freq is not None:
        # if amp was taken from smoothed audio using smooth_data
        # here, need to get the array out of the tuple returned by np.where
        # **also note we avoid converting from samples to s
        # until *after* we find segments** 
        onsets = np.where(above_th_convoluted > 0)[0]
        offsets = np.where(above_th_convoluted < 0)[0]

    if onsets.shape[0] < 1 or offsets.shape[0] < 1:
        return None, None  # because no onsets or offsets in this file

    # get rid of silent intervals that are shorter than min_silent_dur
    silent_gap_durs = onsets[1:] - offsets[:-1]  # duration of silent gaps
    if samp_freq is not None:
        # need to convert to s
        silent_gap_durs = silent_gap_durs / samp_freq
    keep_these = np.nonzero(silent_gap_durs > segment_params['min_silent_dur'])
    onsets = np.concatenate(
        (onsets[0, np.newaxis], onsets[1:][keep_these]))
    offsets = np.concatenate(
        (offsets[:-1][keep_these], offsets[-1, np.newaxis]))

    # eliminate syllables with duration shorter than min_syl_dur
    syl_durs = offsets - onsets
    if samp_freq is not None:
        syl_durs = syl_durs / samp_freq
    keep_these = np.nonzero(syl_durs > segment_params['min_syl_dur'])
    onsets = onsets[keep_these]
    offsets = offsets[keep_these]

#    if samp_freq is not None:
#        onsets = onsets / samp_freq
#        offsets = offsets / samp_freq

    return onsets, offsets

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
		
def read(file):
    reader = neo.io.Spike2IO(filename=file) #This command will read the file defined above
    #print(reader)
    data = reader.read()[0] #This will get the block of data of interest inside the file
    #print(data)
    data_seg=data.segments[0] #This will get all the segments
    #print(data_seg)
    return data, data_seg
	
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
    return analog, sp, ansampling_rate
	


class Ask_Labels(Thread):

    def __init__(self, nb_segmented_syls):
        Thread.__init__(self)
        self.nb_segmented_syls = nb_segmented_syls

    def run(self):
        global labels 
        global syl_counter
        global Nb_syls
        global keep
        syl_counter=0		
        labels = []
        Nb_syls=self.nb_segmented_syls
        keep = input("Keep chunk ? (y or nothing)")
        if keep=='y':
           print("Total of %d segmented syllables" % Nb_syls)
           for i in range (0, Nb_syls):
               label = input("Label ?")
               labels.append(label)
		

			
#######################################################################################
## Block 1: Go to right folder, select the song file to be processed, filter and segment the song
#######################################################################################		
##Go to the right location
pwd = os.getcwd()
#os.chdir("..\data roman labelling")
#os.chdir(".\data roman labelling")
#Enter name of the file to be processed
songfile = 'CSC10_08_06_2018.smr'
[analog,sp,fs]=getarrays(songfile)
rawsong=analog[0]

#Convert (M,1) to (M,) i.e. array to array_like, for np.convolve
rawsong=np.transpose(rawsong)
rawsong = rawsong[0,:]

#Bandpass filter, square and lowpass filter
#cutoffs : 1000, 8000
amp = smooth_data(rawsong,fs,freq_cutoffs=(1000, 8000),smooth_win=smooth_win)
#No band_pass filtering during smoothing!!
#amp = smooth_data(rawsong,fs,freq_cutoffs=None)

#Segment song
(onsets, offsets) = segment_song(amp,segment_params={'threshold': threshold, 'min_syl_dur': min_syl_dur, 'min_silent_dur': min_silent_dur},samp_freq=fs)
shpe = len(onsets)
#Compute the duration of silence gaps (difference between next onset and previous offset))
onsets_diff=onsets[1::]
offsets_diff=offsets[0:-1]
diff = onsets_diff-offsets_diff;
#Compute the number of 'pauses' in the acoustic file: i.e. the number of silent periods lasting more than 0.5 sec
pauses = np.nonzero(diff > (np.round(0.5*fs).astype(int))) #indexes of onset[i+1]-offset[i] (i.e. pauses) that are longer than 5 sec.
pauses = pauses[0][:]
pauses = np.append(pauses,shpe-1) #Add the index of the last element in the offsets array
nb_pauses = len(pauses)
print("Nb pauses in file: %d" % nb_pauses)
i_chunk = 0 # i as initial: beginning of the chunk
f_chunk = 0 # f as final: end of the chunk

counter = 0 #Keeps track of the onsets of syllables
counter_pauses = 0
counter_chunk = 0


while counter<shpe :
    i_chunk = max(onsets[counter]-np.round(0.1*fs).astype(int), 0) #For the beginning os the chunk, take a bit earlier that the onset of the syllable
    f_chunk = offsets[pauses[counter_pauses]] #Previous value of the f_chunk
	#Next: increase the size of the chunck until it is at least 5 s long, unless we are at the end of file
    while (((f_chunk-i_chunk) < (np.round(5*fs).astype(int))) and (counter_pauses <  (nb_pauses-1))):
        counter_pauses +=1
        f_chunk = offsets[pauses[counter_pauses]]
    
    #if counter_pauses == (nb_pauses-1):
    f_chunk = f_chunk + np.round(0.1*fs).astype(int) #For the end of the chunk, take a bit more than the offset of the syllable
    counter = pauses[counter_pauses]+1 # For next syllable onset
    chunk = rawsong[i_chunk:f_chunk]
    len_chunk=len(chunk)
    
    #Bandpass filter, square and lowpass filter the chunk
    #cutoffs : 1000, 8000
    amp_chunk = smooth_data(chunk,fs,freq_cutoffs=(1000, 8000),smooth_win=smooth_win)
    #print(amp_chunk.shape)
    #If no band_pass filtering during smoothing is desired, comment above and uncomment bellow
    #amp = smooth_data(rawsong,fs,freq_cutoffs=None)
    
    #Segment song
    (onsets_chunk, offsets_chunk) = segment_song(amp_chunk,segment_params={'threshold': threshold, 'min_syl_dur': min_syl_dur, 'min_silent_dur': min_silent_dur},samp_freq=fs)
    shpe_chunk = len(onsets_chunk)

    
    ########################################################################################
    # Block 3: Plot spectrogram and onset/offsets of song segments
    ########################################################################################
    #Compute and plot spectrogram
    (f,t,sp)=scipy.signal.spectrogram(chunk, fs, window, nperseg, noverlap, mode='complex')
    plt.figure()
    plt.imshow(10*np.log10(np.square(abs(sp))), origin="lower", aspect="auto", interpolation="none")
    plt.colorbar()
    
    #Plot smoothed amplitude with onset and offset of song segments
    plt.figure() 
    x=np.arange(len(amp_chunk))
    plt.plot(x,amp_chunk)
    #Plot onsets and offsets
    for i in range(0,shpe_chunk):
        plt.axvline(x=onsets_chunk[i])
        plt.axvline(x=offsets_chunk[i],color='r')	
    
    
    #########################################################################################
    ## Block 2: Launch thread for labeling of segments
    #########################################################################################
    # Create thread for labels input
    thread_1 = Ask_Labels(shpe_chunk)
    # Start thread
    thread_1.start()
	
    plt.show()
    #########################################################################################
    # Block 4: Write onsets/offsets/labels to a file on disk
    #########################################################################################
    #Join threads
    thread_1.join()
    
    #Write chunk of raw data and the (onsets, offsets, labels) into separate folders
    if keep=='y':
       os.chdir(".\Annotations")
       current_dir = os.getcwd()
       chunk_number=''
       if((counter_chunk+32)<10):
          chunk_number='00'+str(counter_chunk+32)
       elif((counter_chunk+32)<100): 
          chunk_number='0'+str(counter_chunk+32)
       else:
          chunk_number=str(counter_chunk+32)
       file_path = current_dir+'\\'+songfile[0:18]+'_raw_chunk_'+chunk_number+'_annot.txt'
       file_to_write= open(file_path,"w+") 
       for j in range(0, shpe_chunk):
           file_to_write.write("%d,%d,%s\n" % (onsets_chunk[j],offsets_chunk[j],labels[j]))
       
       #Write to file from buffer, i.e. flush the buffer
       file_to_write.flush()
       file_to_write.close
	   
       os.chdir("..\Raw_songs")
       current_dir = os.getcwd()
       file_name = songfile[0:18]+'_raw_chunk_'+chunk_number+'.npy'
       np.save(file_name,chunk)
#       file_to_write= open(file_path,"w+") 
#       for j in range(0, len_chunk):
#           file_to_write.write("%13.11f\n" % (chunk[j]))     
#       #Write to file from buffer, i.e. flush the buffer
#       file_to_write.flush()
#       file_to_write.close
       
       print('Chunk %d labeled' % counter_chunk)
       counter_chunk +=1
       
	   #Go back to the right folder for the next loop
       os.chdir("..")
    

##Auxiliary code to plot amplitude of signal and smoothed amplitude
##Plot song signal amplitude
#plt.figure() 
#x=np.arange(len(rawsong))
#plt.plot(x,rawsong)
#
#
##Plot smoothed amplitude of the song
#plt.figure();
#x=np.arange(len(amp))
#plt.plot(x,amp)




