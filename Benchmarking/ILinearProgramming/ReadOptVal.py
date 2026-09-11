# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 15:31:33 2025

@author: MarcelSeelbach(MSE)
"""

import pickle
import numpy as np

from os import listdir
from os.path import isfile, join

import matplotlib.pyplot as plt 


#Download files from https://github.com/henriquebecker91/phd/tree/master/instances. 
#and change the mypath variable appropriately. 
mypath= 'C:/Users/MarcelSeelbach(MSE)/Documents/Gitlab/magicapp/NewIdeaEmphasizeCost/phd-BMC-1/instances/'
onlyfiles = [f for f in sorted(listdir(mypath)) if isfile(join(mypath, f))]


allExperiments=[ 'PlotsLinprog' ] #[ 'PlotsLinprog','PlotsLinprogNoRot','PlotsLinprogNoHeur' ]

for k in range (len(onlyfiles)):
    if  onlyfiles[k].startswith("CW")  or onlyfiles[k].startswith("CU") :

        enToPrint=''
        for s in range( len( allExperiments ) ):
            
            try:
                favorite_color = pickle.load( open( allExperiments[s] +"/save"+onlyfiles[k]+".p", "rb" ) )
               #https://wiki.python.org/moin/UsingPickle.html
                enToPrint+= str(-favorite_color[1][0])+', '
            except:
                pass
        print(onlyfiles[k]+': '+enToPrint)