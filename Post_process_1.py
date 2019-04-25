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


####################################################################################################
#                  PURPOSE OF THE SCRIPT
#Go to the annotations of the files used to train the machine learning algorithm
#Ask for the characters used to label the motif and the noise. 
#Compute the mean inter-syllable interval and std dev
#Store mean, std_dev and labels used for motif and noise into files
#####################################################################################################

#Go to the location of the annotations of the predicted songs
pwd = os.getcwd()
os.chdir(".\Training_Songs_annot")
annot_files_list = glob.glob('*.txt')

#Enter the syllable labels for the syllables of the stereotyped part of the motif
#############################################################
#               Enter labels                                #
#############################################################
labels_uniq = [] #The list of labels used to label the noise and the syllables. Typically: ['n','a','b','c','d']
label=''
print("Enter the characters used to label the noise and then the stereotyped part of the motif in the right order (press * to terminate) ")
while label!='*':
    label = input("Label ?")
    if(label!='*'):
      labels_uniq.append(label)

Nb_labels=len(labels_uniq)
print(Nb_labels)
print(labels_uniq)
arr_labels_uniq_Nb = np.zeros(Nb_labels-2) # contains the number of occurrences of each inter-syllable interval

arr_std_dev = np.zeros(Nb_labels-2)
arr_mean = np.zeros(Nb_labels-2)
inter_syl_dur = [ [] for i in range(Nb_labels-2) ]

#################################################################
#        Extract onsets, offsets, labels                        #
#################################################################
#file_num is the index of the file in the songfiles_list
for file_num, annot_file in enumerate(annot_files_list):
    #Read song_annot file	
    #print('File name %s' % annot_file)

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
    #Store inter-syllable interval durations
    for i in range(0,Nl):
       found=False
       j=1 #label of noise is labels_uniq[0]
       while(not(found) and (j<Nb_labels-1)):
        found = (labels[i]==labels_uniq[j])
        j+=1

       if found:
         j=j-1
         if(i<(Nl-1)): # this is not the last syllable in the annotation file
            if(labels[i+1]==labels_uniq[j+1]): #if next syllable is the expected syllable
              inter_syl_dur[j-1].append((onsets[i+1]-offsets[i])) # add inter-syllable interval

###################################################################
#                Compute std_dev and mean                         #
###################################################################
for i in range(Nb_labels-2):
   arr_aux = np.asarray(inter_syl_dur[i])
   arr_mean[i] = np.mean(arr_aux)

for i in range(Nb_labels-2):
   arr_aux = np.asarray(inter_syl_dur[i])
   arr_std_dev[i] = np.std(arr_aux)
   
print(arr_mean)
print(arr_std_dev)


###################################################################
#                Store mean and std_dev to files                  #
###################################################################
os.chdir("..\Post_process")
current_dir = os.getcwd()
file_path = current_dir+'\\'+'Mean_Std_dev.txt'
file_to_write= open(file_path,"w+") 
for j in range(0, Nb_labels-2):
    file_to_write.write("%d,%d\n" % (arr_mean[j],arr_std_dev[j]))

#Write to file from buffer, i.e. flush the buffer
file_to_write.flush()
file_to_write.close


file_path = current_dir+'\\'+'Labels.txt'
file_to_write= open(file_path,"w+") 
for j in range(0, Nb_labels):
    file_to_write.write("%s\n" % labels_uniq[j])
   
#Write to file from buffer, i.e. flush the buffer
file_to_write.flush()
file_to_write.close   