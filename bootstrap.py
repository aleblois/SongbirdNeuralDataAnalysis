# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 13:10:07 2019

@author: eduar
"""

import os
import numpy as np
import glob
import pylab as py
from scipy import stats 

which='Data_Boot_Corr_Pitch'
which2='During'
n_iterations=1000

current_dir = os.getcwd()
listsubdirs=[]

total = np.empty(shape=(n_iterations,1))
for item in os.listdir(current_dir):
    if os.path.isdir(os.path.join(current_dir, item)):
        if item == "__pycache__":
            continue
        else:
            listsubdirs+=[item]
    
for item in range(len(listsubdirs)):
    print("Working on folder:" + listsubdirs[item])
    os.chdir(listsubdirs[item])
    
    with open("unitswindow.txt", "r") as datafile:
        s=datafile.read().split()[0::3]
    for i in range(len(s)):
        os.chdir(".\\Files\\Unit_"+s[i]+"\\Results")
        annot_files_list = glob.glob('*.txt')
        for name in annot_files_list:
            if which in name and which2 in name:
                print(name)
                read = np.loadtxt(name)[:,1][1:].reshape(-1,1)
                print(read[:5])
                total = np.column_stack((total,read))
        os.chdir("..\\..\\..")
    os.chdir(current_dir)
bol=total[:,1:]<0.05
print("Total of <0.05: " + str(np.sum(np.size(np.where(bol==True)))) + " out of " + str(bol.size))
print("In first row: " + str(np.sum(np.size(np.where(bol[0,:]==True)))) + " out of " + str(bol[0,:].size))
np.savetxt("boot"+which+which2+".txt", total[:,1:], header="Total of <0.05: " + str(np.sum(np.size(np.where(bol==True)))) + " out of " + str(bol.size))

cases=np.array([])
for row in range(len(bol)):
    cases=np.append(cases,(np.sum(np.size(np.where(bol[row,:]==True)))/len(bol[row,:]))*100)
    
py.hist(cases, density=True)
xt = py.xticks()[0]  
xmin, xmax = min(xt), max(xt)  
lnspc = np.linspace(xmin, xmax, len(cases))
m, s = stats.norm.fit(cases) # get mean and standard deviation  
pdf_g = stats.norm.pdf(lnspc, m, s) # now get theoretical values in our interval  
py.plot(lnspc, pdf_g, label="Norm") # plot it
py.xlabel("Significant values per case in %")
        