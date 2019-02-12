# SongbirdNeuralDataAnalsysis
Tool development for analysing neural and vocalization data collected from songbirds

This module contains a series of scripts to obtain/analyse data from Spike2 files (.smr):

1- ReadAndPlot.py : This is the first script of the series.  This script will read .smr files, extract the content, and give you the option to plot it.

2 - CreateAndSave.py : This is the second script of the series. This script will create a new folder (name: date/time) inside your working directory and put inside it the spiketimes files (.txt), analogical signal files (.npy), a dataframe with Channels/Labels/LFPnumbers and a summary file containing key information about the .smr file.

3 - ShapeAndSave.py : This is the third script of the series. This script will generate .txt files with the spikeshapes (Initial Time, Final Time, Analog points inside Window), and plot an example of spike from each file. Please keep in mind that this step might take a long time (depending on the size of the spiketrains).

How to use this series?
  The idea behind this series is to be as simple as possible, as automated as possible. So, in order to obtain the contents of a .smr file, the user should define in the first script (ReadAndPlot.py) which .smr file will be read, and then resave the script. Now, to get all the information at once, the user has to run the last script. By doing that, all the previous scripts will be called in the order of the series, generating all the files inside the final folder.
  
Other comments:
  When building this series, we wanted it to also interact with the user and give him/her some options that can influence in how fast the script will run. For example, when the first script (ReadAndPlot.py) is ran, the user will see in the console a Yes/No question about plotting or not the data (this step can take some time, so the user might decide that it's better not to do it). Other than that, the script two (CreateAndSave.py) will ask the user to type the LFP number of each channel, since this information can not be obtained from the .smr file.
