# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 10:07:50 2025

@author: MarcelSeelbach(MSE)
"""
import numpy as np
import csv

from itertools import combinations, groupby

import timeit
import pickle
import numpy as np

from scipy.sparse import csr_matrix, vstack
import scipy.sparse 

import matplotlib.path as mpath 
import matplotlib.patches as mpatches 
import matplotlib.pyplot as plt 
import matplotlib.colors as mcolors

import NealSolver

import Highssolver
import random


from os import listdir
from os.path import isfile, join


#Download files from https://github.com/henriquebecker91/phd/tree/master/instances. 
#and change the mypath variable appropriately. 
mypath= 'C:/Users/MarcelSeelbach(MSE)/Documents/Gitlab/magicapp/NewIdeaEmphasizeCost/phd-BMC-1/instances/' #where are the problem instances stored?
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]


rotationDis= True

StoreProblemInformation=[]
NrProblems=0
LWreadoutSuccess=False

for m in range(16,len(onlyfiles)):
    if  onlyfiles[m].startswith("CW")  or onlyfiles[m].startswith("CU") :
        asList=[]
        with open(mypath + onlyfiles[m], newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|',skipinitialspace=True)
            for row in spamreader:
                asList.append(row)
        try:
            W,L = int(asList[0][0]),int(asList[0][1])
            LWreadoutSuccess=True
        except:
            pass
    
        if len(asList[4])==1:
            asList=[]
            with open(mypath + onlyfiles[m], newline='') as csvfile:
                spamreader = csv.reader(csvfile, delimiter='\t', quotechar='|',skipinitialspace=True)
                for row in spamreader:
                    asList.append(row)
        
        if   LWreadoutSuccess==False:
            try:
                W,L = int(asList[0][0]),int(asList[0][1])
                LWreadoutSuccess=True
            except:
                print('L W information has neither tab nor white space as delimiter')
        
        
        l=[]
        w=[]
        value= []
        be=[]
        for k in range(2,len(asList)):
            if len(asList[k])>1:
                l.append(int(asList[k][0]))
                w.append(int(asList[k][1]))
                value.append(int(asList[k][2]))
                be.append(int(asList[k][3]))
        if all(valEl == 0 for valEl in value):
             value= [l[i]*w[i] for i in range(len(w))]
        #print('Instance '+str(m))
        #print(l)
        #print(w)
        print(asList[1])
        print(np.sum(be))
        StoreProblemInformation.append([onlyfiles[m],L,W,l,w,value,be])
    
    


def EnergyLin(c,vec):
    #Compute Energy if the objective is linear
    return c.T @ vec 
def numpyArrayToNealDict(initSol):
    '''
    Parameters
    ----------
    initSol :numpy array
        solution to be put in dictionary form.

    Returns
    -------
    dictToReturn : Dict
        Convert to a dictionary in the form required by the Neal solver.

    '''
    dictToReturn={}
    for k in range(initSol.shape[0]):
        dictToReturn.update({k: int(initSol[k])}) 
    return dictToReturn

for problemIter, problem in enumerate(StoreProblemInformation):

    start= timeit.timeit()
    
    [descrStr,L,W ,l,w,value,be]= problem
    print((L,W))
    #Demand
    requiredPieces  = [(l[i], w[i]) for i in range (len(l))]
    print('Demand:')
    print(be)
    print('(L,W):')
    print((L,W))

    m=len(be)
    
    
  
    
    l=[]
    w=[]
    beTilde=[]
    valueTilde=[]
    SameParts=[]
    
    for i in range(m):
        sameList=[]
        for s in range(be[i]):
            l.append( requiredPieces[i][1])
            w.append( requiredPieces[i][0]) 
            beTilde.append(1)
            valueTilde.append(value[i])
            SameParts.append(i)
        
    

    
    l.append(0)
    w.append(W)
    l.append(L)
    w.append(0)
    beTilde.append(1)
    beTilde.append(1)
    if rotationDis==False:
        
        valueTilde=valueTilde+valueTilde
    valueTilde.append(0)
    valueTilde.append(0)
    
    be=beTilde
 
    if rotationDis==True:
        N=  np.sum( be )
    else:
        N=  2*np.sum( be )-2
        
    startturnedAround= np.sum( be )-2
    
    NrVariables=  2* N**2
    
    
    
    lListsimple= [  l[ i ] for i in range(len(l)) for z in range(be[i])  ]#+ [0] +[L]
    wListsimple= [  w[ i ] for i in range(len(w)) for z in range(be[i])  ]#+[W]+ [0]
    
    if rotationDis==False:
        lList= lListsimple[0:-2]+wListsimple[0:-2]+lListsimple[-2:]
        wList= wListsimple[0:-2]+lListsimple[0:-2]+wListsimple[-2:]  #Either rotate the piece or not
    else:
        lList= lListsimple
        wList= wListsimple
    
    
    
    def RecursiveCuttingPlan(solution,currentEntry, doneEntries, xPos ,yPos , cuttingStage, positionInfo=[] , rev=0 ) :
                '''
                Parameters
                ----------
                solution : numpy array
                     x and y variables from the publication
                currentEntry : int
                    what entry of the vector are we looking at
                doneEntries : List
                    what enntries were already visited?
                xPos : float
                    current x position on plate
                yPos :float
                    current y position on plate
                cuttingStage : int
                    What is the current cutting stage
                positionInfo : list
                    Where are we on the plate
                rev : Bool, optsional
                    reverse cutting direction. The default is 0.
        
                Returns
                -------
                Given a solution vector it should be determined where the required pieces are on the plate.
                The function will help out the actual plotting.
        
                '''  
                #Find Out neighbouring Pieces
                xCuts= (cuttingStage+rev)%2 
                yCuts= 1- xCuts
                startingIndex = 0
                if xCuts == 0 or ( xCuts==1 and rev==1  ):#DOUBLE CHECK
                    startingIndex = int(NrVariables/2)
                doneEntries[currentEntry]=1
                childCounter=0
    
                (piece1, piece2) = ((currentEntry  %  int(NrVariables/2)) // N ,  (currentEntry  % int(NrVariables/2)) % N)
                currentPiece= piece2# xCuts* piece1+ yCuts* piece2   #DOUBLE CHECK
                parentPiece=  piece1 # (1-xCuts)* piece1+ (1-yCuts)* piece2 
                print(piece1)
                print(piece2)
                posInfo=[]
                currentxPos=xPos 
                currentyPos=yPos
                if currentPiece!=N-1 and currentPiece!=N-2:
    
                    currentxPos=xPos + xCuts* ( lList[currentPiece ]   )
                    currentyPos=yPos + yCuts* ( wList[currentPiece  ]  )
                    
                    if cuttingStage==1 or cuttingStage==2:
            
                        posInfo=[  [(xPos, yPos), cuttingStage , ( lList[currentPiece ],wList[currentPiece  ] , len(lList)-1 ,  currentPiece, currentEntry   )  ] ]
                    else: # yCuts*lList[parentPiece] + xCuts* wList[parentPiece]
                        posInfo=[  [(xPos, yPos), cuttingStage , ( lList[currentPiece ],wList[currentPiece  ] ,parentPiece ,  currentPiece , currentEntry  )  ] ]
    
                
    
    
                for t in range( startingIndex+ N* currentPiece, startingIndex+ N* (currentPiece+1) ):
                    
                    
                        newlength = lList[(t -startingIndex) % N ]
                        newwidth  = wList[(t -startingIndex) % N ]
                        
                        #if lList[currentPiece]>= newlength and xCuts==1: 
                        #    print(t)
                        #if wList[currentPiece]>= newwidth and yCuts==1: 
                         #   print(t)
                        
                        if solution[t]==1 and doneEntries[t]==0:
                            #check if there are more rectangles in subsequent cutting stages                    
                            posInfo.extend(RecursiveCuttingPlan(solution,t, doneEntries, currentxPos  ,  currentyPos    , cuttingStage+1 ,positionInfo=[], rev= rev ))
                            
                            currentxPos+=  xCuts* ( newlength )
                            
                            currentyPos+=  yCuts* ( newwidth )
                            
                            childCounter+=1
                        
               
                return posInfo
    
    
   
    
    def setUpLinProg(additFactor=1):
        '''
        Set up a linear programm to solve the problem 
        
        Output: objective vector c and and in. constraints [A,b]
        '''
        
        
        c= -csr_matrix(np.array( valueTilde*(2*N)  ), dtype=int)
        Asize= csr_matrix( (2*N ,NrVariables), dtype=int )
        bsize= np.zeros( (2*N ), dtype=int )
        
        Ademand= csr_matrix( (N ,NrVariables), dtype=int )
        bdemand= csr_matrix( np.ones(N ), dtype=int )
        
        
        lengthConstraints=[]
        widthConstraints=[]

    
        ADat=[]
        ARow=[]
        ACol=[]               
        for k in range(N):
            
            #all parts should be placed inequality 
            
       
            
            AllOccPlaces=[]
            
            for s in range(2*N):
  

                AllOccPlaces.append(s*N+(k))
                if rotationDis==False:
                    if k < N-2 :
                        if k >=startturnedAround :
    
                            AllOccPlaces.append(s*N+k-startturnedAround)
                        else: 
    
                            AllOccPlaces.append(s*N+k+startturnedAround)
                        
                    
            for occPlace in AllOccPlaces:
                ADat.append(1
                            )
                ARow.append(k)
                ACol.append(occPlace)
                    
                    
                    
        Ademand= csr_matrix((ADat, (ARow, ACol)), shape= (N, NrVariables))

            
        ADat=[]
        ARow=[]
        ACol=[]        
        
        
        for t in range(N**2):
            
            #if t//N!= N-2:
            ADat.append(lList[t%N])
            ACol.append(t)
            ARow.append( (t)//N)
        
            ADat.append(-lList[t//N])
            ACol.append(t+N**2)
            ARow.append( (t)%N)
            
            #ADat.append(-lList[t%N])
            #ACol.append(t)
            #ARow.append( (t)%N)
            ADat.append(lList[t%N])
            ACol.append(t+N**2)
            ARow.append( (t)%N)
            
            if t//N !=N-1:
            
                ADat.append(wList[t%N])
                ACol.append(t+N**2)
                ARow.append( (t)//N+N)
            
                ADat.append(-wList[t//N])
                ACol.append(t)
                ARow.append( (t)%N+N)
                
                #ADat.append(-wList[t%N])
                #ACol.append(t+N**2)
                #ARow.append( (t)%N+N) 
                ADat.append(wList[t%N])
                ACol.append(t)
                ARow.append( (t)%N+N)
                
                
            #add aditional terms to avoid infeasibility
            
            
        
        Asize= csr_matrix((ADat, (ARow, ACol)), shape= (2*N, NrVariables))       
        VariableDir= NrVariables//2

        fixVariableToZero = np.zeros(NrVariables, dtype=int)
        fixVariableToOne  = np.zeros(NrVariables, dtype=int)
    
        for k in range(N):
            fixVariableToZero[ VariableDir + k*N +N-1 ]=1
            fixVariableToZero[ VariableDir + k*N +N-2 ]=1
            fixVariableToZero[   k*N +N-1 ]=1
            fixVariableToZero[   k*N +N-2 ]=1
            
            fixVariableToZero[ VariableDir + N*(N-1) +k ]=1
            
            fixVariableToZero[   N*(N-1) +k ]=1
            fixVariableToZero[   N*(N-2) +k+ VariableDir ]=1
            fixVariableToZero[k*N+k]=1
            fixVariableToZero[VariableDir +k*N+k]=1
            
            
            for s in range(N):
                
                if wList[k]<wList[s]:
                
                    fixVariableToZero[k * N + s]=1
                if lList[k]<lList[s]:
    
                    fixVariableToZero[k * N + s + VariableDir ]=1
    
        
        fixVariableToZero[VariableDir + N*(N-1) + N-2]=0
    
        fixVariableToOne [VariableDir + N*(N-1) + N-2]=1
        
        #fixVariableToOne[  N*(N-1) + N-2]=1
    
        fixVariableNotToZero= csr_matrix(1-fixVariableToZero)
    
    
        return c, [(Asize, Ademand), (bsize, bdemand)], fixVariableToZero, fixVariableToOne  
        #all parts should be placed 
    
    
    
    

   
    sol=0
    multipleTrys=1
    Energy=0
    for s in range(multipleTrys):
        c, [(Asize, Ademand), (bsize, bdemand)] , fixVariableToZero, fixVariableToOne  = setUpLinProg()#Still additFactor
        A= vstack([Asize,Ademand])
        b= vstack([bsize.reshape((2*N,1)), bdemand.T])
        restVariables= np.where(fixVariableToZero+fixVariableToOne==0)[0]
        
        
        
        
        bounds = [  np.zeros(NrVariables, dtype=int)+fixVariableToOne,np.ones(NrVariables, dtype=int)-fixVariableToZero]
        sol = Highssolver.linprog(c,A, b ,None,None, bounds = bounds ,integrality=[1]*(NrVariables),Inputoptions= { 'disp':True,'presolve':True, 'mip_heuristic_effort':0.9,'time_limit':180 },SaveStr=descrStr+'.txt')  #NealSolver.runSimulatedFullyConnectedQUBOSparse(Q[np.ix_(restVariables[:], restVariables[:])])#initial_states =[numpyArrayToNealDict(np.ones(Q.shape[0]))]) #initial_states =[numpyArrayToNealDict(solBestSim)])
        
       
        
     
        Energy=EnergyLin(c.toarray().T, sol)
      
    
   
    
    cuttingInfo=RecursiveCuttingPlan(sol,  N*(N-1)+N-2 ,  np.zeros(NrVariables)  , 0 ,0 , 1, positionInfo=[] , rev=0  ) 
    fig = plt.figure(figsize= (10,(W*10)//L) ) 
    
    ax = fig.subplots(1, 1)
    ax.set_xlim([0, L*1.1])
    ax.set_ylim([0, W*1.1])
    
    OuterBound = plt.Rectangle((0,0),      L ,W,  edgecolor= 'black', linestyle= '-', linewidth= 2 , facecolor='none' ) 
    ax.add_patch(OuterBound)
    first=True
    for  rectangle in cuttingInfo:
          coord= rectangle[0]
          pp1 = plt.Rectangle(coord , rectangle[-1][0] ,rectangle[-1][1], linestyle= '-' ,  edgecolor= 'green', facecolor='none' ) 
          
          cutDir =  (rectangle[1])%2
          if first==True:
              cutDir=0
              first=False
        
          ppSchnitt = plt.Rectangle(coord ,  ( 1-cutDir ) *rectangle[-1][0] + cutDir*lList[ rectangle[-1][2]  ]  , cutDir* rectangle[-1][1]+ ( 1-cutDir ) * wList[ rectangle[-1][2]] , linestyle= '--' ,  edgecolor= 'black', facecolor='none' ) 
    
          labelNr=rectangle[-1][3]
          labelStr=''
          if labelNr<startturnedAround:
              labelStr= str(SameParts[labelNr])
          else:
              labelStr= str(SameParts[labelNr-startturnedAround])+ ' T'

          
          
          ax.text( coord[0]  + rectangle[-1][0]/2 ,  coord[1] + rectangle[-1][1]/2, labelStr )
          ax.add_patch(pp1)
          
          
    plt.title('Problem '+descrStr)
    plt.savefig('PlotsLinprogNoRot/' + descrStr + '.png')
    end= timeit.timeit()

    pickle.dump([sol, Energy  ,end-start ], open( "PlotsLinprogNoRot/save"+descrStr+".p", "wb" ) )    
    plt.show()
    
   
    
     