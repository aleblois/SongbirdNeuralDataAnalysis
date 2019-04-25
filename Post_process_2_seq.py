import numpy as np
import scipy as sp
from scipy.io import loadmat
import scipy.signal
from scipy.io import wavfile
import matplotlib as mplt
import matplotlib.pyplot as plt
import os
import glob
from threading import Thread
import math

##Check location of libraries
#npdir = os.path.dirname(np.__file__)
#print("NumPy is installed in %s" % npdir)
#spdir = os.path.dirname(sp.__file__)
#print("SciPy is installed in %s" % spdir)
#print(np.__version__)
#print(sp.__version__)
#print(mplt.__version__)
##End check location

#l_t = ['a','b','c','d','b']
#
##print(l_t[np.where(l_t='b')])
#idx=np.where((np.char.find(l_t,'b'))>=0)
#print(idx)
#print(idx[0])# The array of indeces
#print(idx[0][:])# The first index
#pauses = np.where(l_t='b') #indexes of onset[i+1]-offset[i] (i.e. pauses) that are longer than 5 sec.
#print(pauses)

os.chdir(".\Post_process")
current_dir = os.getcwd()
file_name = 'Mean_Std_dev.txt'	
mean=[]
std_dev=[]
labels_uniq = [] #The list of labels used to label the noise and the syllables. Typically: ['n','a','b','c','d']
label=''
with open(file_name) as f:
     lines = f.readlines()
     for line in lines:
         #line = str(lines)
         #print("line: %s" % line)
         splt = line.split(",")
         mean.append(int(splt[0]))
         std_dev.append(int(splt[1]))

f.close	
Nb_inter_syl=len(mean)
print(mean)
print(std_dev)
print(Nb_inter_syl)


file_name = 'Labels.txt'	
with open(file_name) as f:
     lines = f.readlines()
     for line in lines:
         #line = str(lines)
         #print("line: %s" % line)
         labels_uniq.append(line[0])

f.close	
Nb_labels=len(labels_uniq)
print(Nb_labels)
print(labels_uniq)
	

	

##Go to the location of the annotations of the predicted songs
pwd = os.getcwd()
os.chdir("..\Test_Songs_predict")
#Enter name of the file to be processed
#songfile = 'CSC10_raw_chunk_40.txt'

annot_files_list = glob.glob('*.txt')

#Read song file	
rawsong = []
#fs=30303.0
fs=32000.0
slack_var = 3


#Enter the syllable labels for the syllables of the stereotyped part of the motif
##############################################
#        Enter labels                        #
##############################################



#file_num is the index of the file in the songfiles_list
for file_num, annot_file in enumerate(annot_files_list):
    #Read song_annot file	
    print('File name %s' % annot_file)

    onsets = []
    offsets = []
    labels = [] 

    with open(annot_file) as f:
         lines = f.readlines()
         for line in lines:
             #line = str(lines)
             #print("line: %s" % line)
             splt = line.split(",")
             onsets.append(int(splt[0]))
             offsets.append(int(splt[1]))
			 
			 #If just: labels.append(str(splt[2])) then, labels will be ['1\n','2\n','3\n','1\n',...]. Need to get rid of \n
			 #This is done bellow
			 
             splt_nwline = (splt[2]).split("\n")
             labels.append(str(splt_nwline[0]))
		 
             #print("type of onsets %s %s %s " % (type(onsets[0]),type(offsets[0]),type(labels[0])))
             #print("%f %f %f" % (float(splt[0]),float(splt[1]),float(splt[2])))

    f.close	
    Nl=len(labels)
    #Check consistency of sequence of syllables
    #Consistency of syllable 'a' (first syllable in the motif)
    #idx_a=np.where((np.char.find(labels,labels_uniq[0]))>=0)
    #print(idx)
    #print(idx[0][0])
    #idx_a = idx_a[0]
    for i in range(0,Nl):
       found=False
       j=1 #label of noise is labels_uniq[0]
       while(not(found) and (j<Nb_labels)):
        found = (labels[i]==labels_uniq[j])
        j+=1
		
       if found:
         j=j-1
         if(j==1): #i.e. if syllable 'a' was found
           if(i<(Nl-1)): # this is not the last syllable in the annotation file
              if(labels[i+1]!=labels_uniq[j+1]): #if next syllable is not 'b' or 'b' comes too late or too early
                labels[i]=labels_uniq[0] #set label to be that of noise
         elif(j==(Nb_labels-1)): #i.e. if last syllable was found
           if(i>0): # this is not the first syllable in the annotation file
              if(labels[i-1]!=labels_uniq[j-1]): #if previous syllable is not the expected one or if the expected syllable is too late or too early
                labels[i]=labels_uniq[0] #set label to be that of noise				 
         elif(j==(Nb_labels-2)): #i.e. if previous to last syllable was found       
           if(i>0): # this is not the first syllable in the annotation file
              if(labels[i-1]!=labels_uniq[j-1]): #if previous syllable is not the expected one
                labels[i]=labels_uniq[0] #set label to be that of noise
         else: #i.e. if another syllable was found
           if(i<(Nl-1)): # this is not the last syllable in the annotation file
              if(labels[i+1]!=labels_uniq[j+1]): #if next syllable is not the expected one
                labels[i]=labels_uniq[0] #set label to be that of noise           
           if(i>0): # this is not the first syllable in the annotation file
              if(labels[i-1]!=labels_uniq[j-1]): #if previous syllable is not the expected one
                labels[i]=labels_uniq[0] #set label to be that of noise
		 

    os.remove(annot_file)
    current_dir = os.getcwd()
    file_path = current_dir+'\\'+annot_file
    file_to_write= open(file_path,"w+") 
    for j in range(0, Nl):
        file_to_write.write("%d,%d,%s\n" % (onsets[j],offsets[j],labels[j]))
    
    #Write to file from buffer, i.e. flush the buffer
    file_to_write.flush()
    file_to_write.close
    
    