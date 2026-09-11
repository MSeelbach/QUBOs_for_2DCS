# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 15:31:33 2025

@author: MarcelSeelbach(MSE)
"""

import pickle
import numpy as np

from os import listdir
from os.path import isfile, join

from os.path import getmtime

import matplotlib.pyplot as plt 


#Download files from https://github.com/henriquebecker91/phd/tree/master/instances. 
#and change the mypath variable appropriately. 
mypath= 'C:/Users/MarcelSeelbach(MSE)/Documents/Gitlab/magicapp/NewIdeaEmphasizeCost/phd-BMC-1/instances/'
onlyfiles = [f for f in sorted(listdir(mypath)) if isfile(join(mypath, f))]


allExperiments=[ 'PlotsLinprog','PlotsLinprogNoRot','PlotsLinprogNoHeur' ]
for s in range( len( allExperiments ) ):
    print(allExperiments[s])
    modifAlt=0

    for k in range (len(onlyfiles)):
        if  onlyfiles[k].startswith("CW")  or onlyfiles[k].startswith("CU") :
              
              enToPrint=''
            
              try:
                    #favorite_color = pickle.load( open( allExperiments[s] +"/save"+onlyfiles[k]+".p", "rb" ) )
                    modif= getmtime(allExperiments[s] +"/save"+onlyfiles[k]+".p")
                    
                   
                    enToPrint+= str((modif-modifAlt))+', ' #/60 for minutes
                    modifAlt= modif
              except:
                    pass
              print(onlyfiles[k]+': '+enToPrint)