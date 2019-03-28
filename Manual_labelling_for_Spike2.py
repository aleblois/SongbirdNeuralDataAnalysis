import numpy as np
import scipy as sp
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

#Threshold for segmentation
#threshold=1.6e-4 # for files recorded with RTXI from recording box (above the chamber)
threshold=2e-9 # for files recorded with Neuralynx
min_syl_dur=0.02
min_silent_dur= 0.005
smooth_win=10
#Contains the labels of the syllables from a single .wav file
labels = []
syl_counter=0
Nb_syls=0
write=0

#Durations in seconds
min_chunk_duration = 5
max_chunk_duration = 25

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

	
class Ask_Labels(Thread):

    def __init__(self, nb_segmented_syls):
        Thread.__init__(self)
        self.nb_segmented_syls = nb_segmented_syls

    def run(self):
        global labels 
        global syl_counter
        global Nb_syls
        global write
        syl_counter=0		
        labels = []
        write = 0
        Nb_syls=self.nb_segmented_syls
		
        keep = input("Keep chunk ? (y or nothing)")
        if keep=='y':
           write = 1
           print("Total of %d segmented syllables" % Nb_syls)
           for i in range (0, Nb_syls):
               label = input("Label ?")
               labels.append(label)
        else:
           write = 0
		



			
#######################################################################################
## Block 1: Go to right folder, select the song file to be processed, filter and segment the song
#######################################################################################		
##Go to the right location
pwd = os.getcwd()
#os.chdir("..\SyllablesClassification\Koumura Data Set\Song_Data\\raw_song")
#os.chdir("..\SyllablesClassification\Koumura Data Set\Song_Data\I_09_04_2018\\raw_song\\Test_Spike2")
#Enter name of the file to be processed
#songfile = 'file_1550590539.txt'
#songfile = 'rawsong_06_05_2018_short_'


#Read song file	
rawsong = []
#fs=30303.0
fs=32000.0
#if txt
#rawsong=np.loadtxt(songfile)
#if mat
rawsong=np.load("CSC10.npy")
#rawsong = rawsong['rawsong_short']
rawsong=np.transpose(rawsong)
#print(rawsong.shape)
rawsong = rawsong[0,:] # convert to (M,) array-like for np.convolve

#Bandpass filter, square and lowpass filter
#cutoffs : 1000, 8000
amp = smooth_data(rawsong,fs,freq_cutoffs=(1000, 8000),smooth_win=smooth_win)
#No band_pass filtering during smoothing!!
#amp = smooth_data(rawsong,fs,freq_cutoffs=None)

#Segment song
(onsets, offsets) = segment_song(amp,segment_params={'threshold': threshold, 'min_syl_dur': min_syl_dur, 'min_silent_dur': min_silent_dur},samp_freq=fs)
shpe = len(onsets)
#onsets = np.round(onsets * fs).astype(int)
#offsets = np.round(offsets * fs).astype(int)


onsets_diff=onsets[1::]
offsets_diff=offsets[0:-1]
diff = onsets_diff-offsets_diff;
pauses = np.nonzero(diff > (np.round(0.5*fs).astype(int))) #indexes of onset[i+1]-offset[i] (i.e. pauses) that are longer than 5 sec.
pauses = pauses[0][:]
pauses = np.append(pauses,shpe-1) #Add the index of the last element in the offsets array
nb_pauses = len(pauses)
print("Nb pauses in file: %d" % nb_pauses)
#print(pauses[0][0])
#print(onsets)
#print(onsets_diff)
#
#print(offsets)
#print(offsets_diff)

i_chunk = 0
f_chunk = 0

counter = 0 #Keeps track of the onsets of syllables
counter_pauses = 0
counter_chunk = 0



while counter<shpe :
    i_chunk = max(onsets[counter]-np.round(0.1*fs).astype(int), 0) # Take a bit earlier that the onset of the syllable
    f_chunk = offsets[pauses[counter_pauses]]
    while (((f_chunk-i_chunk) < (np.round(5*fs).astype(int))) and (counter_pauses <  (nb_pauses-1))):
        counter_pauses +=1
        #print(counter_pauses)
        #print(pauses[counter_pauses])
        f_chunk = offsets[pauses[counter_pauses]]
    
    #if counter_pauses == (nb_pauses-1):
    f_chunk = f_chunk + np.round(0.1*fs).astype(int) # Take a bit more than the offset of the syllable
    counter = pauses[counter_pauses]+1 # For next syllable onset
    chunk = rawsong[i_chunk:f_chunk]
    len_chunk=len(chunk)
    
    #Bandpass filter, square and lowpass filter
    #cutoffs : 1000, 8000
    amp_chunk = smooth_data(chunk,fs,freq_cutoffs=(1000, 8000),smooth_win=smooth_win)
    #print(amp_chunk.shape)
    #No band_pass filtering during smoothing!!
    #amp = smooth_data(rawsong,fs,freq_cutoffs=None)
    
    #Segment song
    (onsets_chunk, offsets_chunk) = segment_song(amp_chunk,segment_params={'threshold': threshold, 'min_syl_dur': min_syl_dur, 'min_silent_dur': min_silent_dur},samp_freq=fs)
    shpe_chunk = len(onsets_chunk)
    #onsets = np.round(onsets * fs).astype(int)
    #offsets = np.round(offsets * fs).astype(int)

    
    ########################################################################################
    # Block 3: Plot spectrogram and onset/offsets of song segments
    ########################################################################################
    #Compute and plot spectrogram
    (f,t,sp)=scipy.signal.spectrogram(chunk, fs, window, nperseg, noverlap, mode='complex')
    plt.figure()
    plt.imshow(10*np.log10(np.square(abs(sp))), origin="lower", aspect="auto", interpolation="none")
    plt.colorbar()
    
    #plt.figure()
    #plt.imshow(10*np.log10(np.square(abs(sp))), origin="lower", aspect="auto", cmap=colormap, interpolation="none")
    #plt.colorbar()
    
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
    
    #Write file with onsets, offsets, labels
    if write:
       os.chdir(".\Annotations")
       current_dir = os.getcwd()
       file_path = current_dir+'\\'+'CSC10_raw_annot_'+str(counter_chunk)+'.txt'
       file_to_write= open(file_path,"w+") 
       for j in range(0, shpe_chunk):
           file_to_write.write("%d,%d,%s\n" % (onsets_chunk[j],offsets_chunk[j],labels[j]))
       
       #Write to file from buffer, i.e. flush the buffer
       file_to_write.flush()
       file_to_write.close
	   
       os.chdir("..\Raw_songs")
       current_dir = os.getcwd()
       file_path = current_dir+'\\'+'CSC10_raw_chunk_'+str(counter_chunk)+'.txt'
       file_to_write= open(file_path,"w+") 
       for j in range(0, len_chunk):
           file_to_write.write("%13.11f\n" % (chunk[j]))
       
       #Write to file from buffer, i.e. flush the buffer
       file_to_write.flush()
       file_to_write.close
       
       print('Chunk %d labeled' % counter_chunk)
       counter_chunk +=1
       
       os.chdir("..")
    

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




