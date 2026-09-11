# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 10:07:50 2025

Main class to carry out experiments on the QUBO formulation for the 2D cutting stock problem 
QUBO problems for small problems are saved for later numeric simulations
@author: MarcelSeelbach(MSE)

"""
import numpy as np
import csv

from itertools import combinations, groupby

import timeit
import time
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
import QUBOSA

from os import listdir
from os.path import isfile, join


mypath= 'C:/Users/MarcelSeelbach(MSE)/Documents/Gitlab/magicapp/NewIdeaEmphasizeCost/phd-BMC-1/instances/'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

ColorList=list(mcolors.XKCD_COLORS.values())
random.shuffle(ColorList)

rotationDis= True # Consider Problem without allowing for rotations.
StoreProblemInformation=[]
NrProblems=0
LWreadoutSuccess=False

'''
Defining the 3 cutting stock problems of interest.
1. Problem with 3 pieces. Everything fits on the plate. SmallProblem
2. Problem with 2 pieces. Everything fits on the plate. SmallestProblem
3. Problem with 2 pieces. Only one piece fits. SmallestProblemNoSolPos

'''
W,L=24,10

w=[10,9,8]
l=[8 ,9,4]

value= [0,0,0]
be=[1,1,1]

if all(valEl == 0 for valEl in value):
     value= [l[i]*w[i] for i in range(len(w))]
#print('Instance '+str(m))

StoreProblemInformation.append(['SmallProblem',L,W,l,w,value,be])

W,L=24,10

w=[10,9]
l=[8 ,9]

value= [0,0]
be=[1,1]

if all(valEl == 0 for valEl in value):
     value= [l[i]*w[i] for i in range(len(w))]
#print('Instance '+str(m))

StoreProblemInformation.append(['SmallestProblem',L,W,l,w,value,be])

W,L=15,10

w=[10,7]
l=[8 ,9]

value= [0,0]
be=[1,1]

if all(valEl == 0 for valEl in value):
     value= [l[i]*w[i] for i in range(len(w))]
#print('Instance '+str(m))

StoreProblemInformation.append(['SmallestProblemNoSolPos',L,W,l,w,value,be])
def EnergyLin(c,vec):
    ''' Short function to calculate linear objective'''
    return c.T @ vec 


timeSave=[] #Store information about runtime

for problemIter, problem in enumerate(StoreProblemInformation):
    
    start= time.time()
    [descrStr,L,W ,l,w,value,be]= problem
    print('Problem: ' +descrStr )
    #Demands:
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
    
    for i in range(m):#[3,4,8,9]:#range(m):
        sameList=[]
        for s in range(be[i]): # Every piece is demanded once in the model. be is initial demand, while beTilde has only once as entry
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
    
    NrPlates=1
 
    if rotationDis==True:
        N=  np.sum( be )
    else:
        N=  2*np.sum( be )-2
        
    startturnedAround= np.sum( be )-2
    
    NrVariables=  2* N**2 # Nr Variables
    
    
    lListsimple= [  l[ i ] for i in range(len(l)) for z in range(be[i])  ]#+ [0] +[L]
    wListsimple= [  w[ i ] for i in range(len(w)) for z in range(be[i])  ]#+[W]+ [0]
    
    if rotationDis==False:
        lList= lListsimple[0:-2]+wListsimple[0:-2]+lListsimple[-2:]
        wList= wListsimple[0:-2]+lListsimple[0:-2]+wListsimple[-2:]  #Either rotate the piece or not
    else:
        lList= lListsimple
        wList= wListsimple
    
    
    
    def RecursiveCuttingPlan(XYsolution,currentEntry, doneEntries, xPos ,yPos , cuttingStage, positionInfo=[] , rev=0 ) :
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
                
                relevantL=[lList[currentPiece ]]
                relevantW=[wList[currentPiece ]]
                
                if currentPiece not in range(N-NrPlates-1,N):
                
                    currentxPos=xPos + xCuts* ( np.max(relevantL)   )
                    currentyPos=yPos + yCuts* (np.max(relevantW)  )
            
                    
                    if cuttingStage==1 or cuttingStage==2:
                        parentParameter= len(lList)-1
                    else: # yCuts*lList[parentPiece] + xCuts* wList[parentPiece]
                        parentParameter= parentPiece
                    CoordList=[(xPos,yPos)] 
                    posInfo=[  [CoordList, cuttingStage , ( relevantL,relevantW   , parentParameter,  [currentPiece] , currentEntry  )  ] ]

    
    
                for t in range( startingIndex+ N* currentPiece, startingIndex+ N* (currentPiece+1) ):
                    
                        nextPieceIndex=(t -startingIndex) % N 
                        newlength = lList[nextPieceIndex]
                        newwidth  = wList[nextPieceIndex]
                        
                        #if lList[currentPiece]>= newlength and xCuts==1: 
                        #    print(t)
                        #if wList[currentPiece]>= newwidth and yCuts==1: 
                        #    print(t)
                        
                        if XYsolution[t]==1 and doneEntries[t]==0:
                            
                      
                            
                            #check if there are more rectangles in subsequent cutting stages                    
                            posInfo.extend(RecursiveCuttingPlan(XYsolution,t  ,doneEntries, currentxPos  ,  currentyPos    , cuttingStage+1 ,positionInfo=[], rev= rev ))
                            
                            currentxPos+=  xCuts* ( newlength )
                            
                            currentyPos+=  yCuts* ( newwidth )
                            
                            childCounter+=1
                        
               
                return posInfo
    
    
    

    
    def setUpLinProgUnr(additFactor=1):
        '''
        Set up a linear programm to solve the problem 
        
        Output: objective vector c and and in. constraints [A,b]
        '''
        c= -csr_matrix(np.array( valueTilde*(2*N) + [0]*(N**2) ), dtype=int)
        Qcost= csr_matrix( (NrVariables ,NrVariables), dtype=int )

        Asize= csr_matrix( (2*N ,NrVariables), dtype=int )
        
        bsize= np.zeros( (2*N ), dtype=int )
        
        Qdemand= csr_matrix( (NrVariables ,NrVariables), dtype=int )
        
        Qzcorr= csr_matrix( (NrVariables ,NrVariables), dtype=int )
        
        Ademand= csr_matrix( (N ,NrVariables), dtype=int )

        bdemand= csr_matrix( np.ones(N), dtype=int )
        
        lengthConstraints=[]
        
        widthConstraints=[]

    
        ADat=[]
        ARow=[]
        ACol=[]   
        ADat2=[]
        ARow2=[]
        ACol2=[]                    
        for k in range(N):
            
            #all parts should not be placed double inequality 
            
       
            
            AllOccPlaces=[]
            
            for s in range(2*N):
  

                AllOccPlaces.append(s*N+(k))
                if rotationDis==False:
                    if k < N-2 :
                        if k >=startturnedAround :
    
                            AllOccPlaces.append(s*N+k-startturnedAround)
                        else: 
    
                            AllOccPlaces.append(s*N+k+startturnedAround)
                        
                    
            for occPlace1 in AllOccPlaces:
                for occPlace2 in AllOccPlaces:
                    if occPlace1!= occPlace2:
                        ADat.append(1)
                        ARow.append(occPlace1)
                        ACol.append(occPlace2)
                             
            for occPlace in AllOccPlaces:
                ADat2.append(1)
                ARow2.append(k)
                ACol2.append(occPlace)
                    
                    
                    
        Ademand= csr_matrix((ADat2, (ARow2, ACol2)), shape= (N, NrVariables))
 
        Qdemand= csr_matrix((ADat, (ARow, ACol)), shape= (NrVariables, NrVariables))




        ListQuadraticTerms = [[],[]] # First width, then length
        
           
        
        ADat=[]
        ARow=[]
        ACol=[]  
        
        for t in range(N**2):
            
            #if t//N!= N-2:
            ADat.append( lList[t%N])
            ACol.append(t)
            ARow.append( (t)//N)
        
            ADat.append(-lList[t//N])
            ACol.append(t+N**2)
            ARow.append( (t)%N)
            
            #ADat.append(-lList[t%N])
            #ACol.append(t)
            #ARow.append( (t)%N)
            ADat.append( lList[t%N])
            ACol.append(t+N**2)
            ARow.append( (t)%N)
            
            if t//N !=N-1:
            
                ADat.append( wList[t%N])
                ACol.append(t+N**2)
                ARow.append( (t)//N+N)
            
                ADat.append(-wList[t//N])
                ACol.append(t)
                ARow.append( (t)%N+N)
                
                #ADat.append(-wList[t%N])
                #ACol.append(t+N**2)
                #ARow.append( (t)%N+N) 
                ADat.append( wList[t%N])
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
    
    
        return c, Qcost , [(Asize, Ademand), (bsize, bdemand)],  Qdemand,Qzcorr, ListQuadraticTerms ,fixVariableToZero, fixVariableToOne  
        #all parts should be placed 
    
    
    
    

   
    sol=0
    multipleTrys=1
    Energy=0
    EnergyLinear=0
    EnergyBest=0
    for s in range(multipleTrys):
        
        c, Qcost , [(Asize, Ademand) , (bsize, bdemand)]  ,  Qdemand,Qzcorr, ListQuadraticTerms, fixVariableToZero, fixVariableToOne  = setUpLinProgUnr()#Still additFactor
              
        restVariables= np.where(fixVariableToZero+fixVariableToOne==0)[0]
        
        bounds = [  np.zeros(NrVariables, dtype=int)+fixVariableToOne,np.ones(NrVariables, dtype=int)-fixVariableToZero]
        bounds = [  np.zeros(NrVariables, dtype=int)+fixVariableToOne[:NrVariables],np.ones(NrVariables, dtype=int)-fixVariableToZero[:NrVariables]]

        A= vstack([Asize,Ademand])
        b= vstack([bsize.reshape((2*N,1)), bdemand.T])

        startLin=time.time()
        solLin = Highssolver.linprog(c[:NrVariables] , A.toarray()[:,:NrVariables] , b ,None,None, bounds = bounds ,integrality=[1]*(NrVariables),Inputoptions= { 'disp':True,'presolve':True, 'mip_heuristic_effort':0.9,'time_limit':180 },SaveStr=descrStr+'.txt')  #NealSolver.runSimulatedFullyConnectedQUBOSparse(Q[np.ix_(restVariables[:], restVariables[:])])#initial_states =[numpyArrayToNealDict(np.ones(Q.shape[0]))]) #initial_states =[numpyArrayToNealDict(solBestSim)])
        endLin= time.time()
        
        whereToSave='Plots/' +descrStr
        solList, bestSol,EnLin, EnQuad, bestIter = QUBOSA.AugmentedLagrangianQuad( solLin , scipy.sparse.diags(c.toarray(), [0]) ,Asize, bsize, Qdemand,ListQuadraticTerms ,fixVariableToZero, fixVariableToOne,lList,wList, whereToSave , maxiter=10)
        EnQuadVal= [EnQuadel[0,0] for EnQuadel in EnQuad ]

        plt.plot(EnLin, label='Lin')   
        plt.plot(EnQuadVal, label= 'Quad')       
        plt.legend()
        plt.savefig('Plots/' +descrStr + 'CompPlot.png', bbox_inches='tight')
       
        plt.show()
        print('hallo')
        EnergyLinear=EnergyLin((c.toarray())[0,:solLin.shape[0]].T, solLin)
        EnergyBest=EnergyLin((c.toarray())[0,:solLin.shape[0]].T, bestSol[:solLin.shape[0]])
    end= time.time()
    timeSave.append((end-start-endLin+startLin, endLin-startLin,bestIter, EnergyLinear.copy(),EnergyBest.copy() ))

    def Plot(solXY, furtherDesc=None,iterDescr='') :   
          
         fac= 7
         fig = plt.figure(figsize= (fac,NrPlates*(W*fac)//L) ) 
         
         
         
         axs = fig.subplots(NrPlates, 1)
         if NrPlates>1:
             axs = axs.ravel()
         else:
             axs= [axs]
         
         for PlateNr in range(NrPlates):
             
             cuttingInfo=RecursiveCuttingPlan(solXY , N*(N-1)+N-2-PlateNr , [], np.zeros(NrVariables)  , 0 ,0 , 1, positionInfo=[] , rev=0  )         
             ax=axs[PlateNr]
             ax.set_xlim([0, L*1.1])
             ax.set_ylim([0, W*1.1])
         
             OuterBound = plt.Rectangle((0,0),      L ,W,  edgecolor= 'black', linestyle= '-', linewidth= 2 , facecolor='none' ) 
             ax.add_patch(OuterBound)
             first=True
             for  rectangle in cuttingInfo:
                   
                   
                   pp1List=[]
                   
                   for joinedIter in range(len( rectangle[-1][0])):
                       coord= rectangle[0][joinedIter]
    
                       
                       
                       labelNr=rectangle[-1][3][joinedIter]
                       labelStr=''
                       if labelNr<startturnedAround:
                           labelStr= str(SameParts[labelNr])
                       else:
                           labelStr= str(SameParts[labelNr-startturnedAround])+ ' T'
                           
                       pp1 = plt.Rectangle(coord , rectangle[-1][0][joinedIter] ,rectangle[-1][1][joinedIter], linestyle= '-' ,  edgecolor= 'black', facecolor=ColorList[SameParts[labelNr]] ) #TABLEAU_COLORS

                       
                       ax.text( coord[0]  + rectangle[-1][0][joinedIter]/2 ,  coord[1] + rectangle[-1][1][joinedIter]/2, labelStr )
    
                       
                       pp1List.append(pp1)
                   
                   
                   
                   cutDir =  (rectangle[1])%2
                   if first==True:
                       cutDir=0
                       first=False
                 
                   coord= rectangle[0][-1]
                   width= rectangle[-1][0][-1]
                   height= rectangle[-1][1][-1]
                   #ppSchnitt = plt.Rectangle(coord ,  ( 1-cutDir ) *np.sum(rectangle[-1][0]) + cutDir*lList[ rectangle[-1][2]  ]  , cutDir* np.sum( rectangle[-1][1])+ ( 1-cutDir ) * wList[ rectangle[-1][2]] , linestyle= '--' ,  edgecolor= 'black', facecolor='none' ) 
                   #Cutline= plt.Line2D([coord[0],coord[0] +( 1-cutDir ) *np.sum(rectangle[-1][0]) + cutDir*lList[ rectangle[-1][2]  ] ] ,[coord[1],coord[1]+  cutDir* np.sum( rectangle[-1][1])+ ( 1-cutDir ) * wList[ rectangle[-1][2]] ], linestyle= '--',  edgecolor= 'black')
                   #ax.add_line(Cutline)
                   #ax.add_patch(ppSchnitt)
                   if len(pp1List)>1:
                       if cutDir%2==0:
                           joinedRect = plt.Rectangle(rectangle[0][0] , np.sum(rectangle[-1][0]),np.max(rectangle[-1][1]), linestyle= '--' ,  edgecolor= 'black', facecolor='none' ) 
                       else:
                           joinedRect = plt.Rectangle(rectangle[0][0] , np.max(rectangle[-1][0]),np.sum(rectangle[-1][1]), linestyle= '--' ,  edgecolor= 'black', facecolor='none' ) 
    
                       ax.add_patch(joinedRect)
    
                   for ppRectangle in pp1List:
                       ax.add_patch(ppRectangle)
               
               
         plt.suptitle('Problem '+descrStr+ ' '+furtherDesc, fontsize=20, y=0.93)
         #plt.subPlotsNotRot_adjust(top=0.85)
    
         plt.savefig('Plots/' + descrStr+iterDescr  + '.png', bbox_inches='tight')
        
         plt.show()
         
    def PlotMultiple(solList , furtherDesc=None,iterDescr=''):
        '''
        Parameters
        ----------
        solList : List of numpy arrays
            solList which should be plotted
        solZList : List of numpy arrays
            z variables if applicable
        furtherDesc : String, optional
            Further description. The default is None.
        iterDescr : String, optional
            Used to describe the iteration. The default is ''.

        Returns
        -------
        Saves the plots.

        '''
        for k in range(len(solList)):
                Plot(solList[k] ,  furtherDesc=furtherDesc,iterDescr=str(k))
     
     
    PlotMultiple(solList,[solElem[2*N**2: ] for solElem in solList] , 'Augmented Lagrangian '+str(len(restVariables))+'Qubits')
    plt.show()
    
    Plot(bestSol, bestSol [2*N**2: ] , 'Augmented Lagrangian',  iterDescr='BestSol' )#Plot(bestSol,np.zeros(int(bestSol.shape[0]//2)) , 'Augmented Lagrangian',  iterDescr='BestSol' )
    Plot(solLin,np.zeros(int(solLin.shape[0]//2)) , 'HiGHS Linear Programming '+str(len(restVariables))+' Nr Qubits',iterDescr='Lin')

    #pickle.dump([sol, Energy  ,end-start ], open( "Plots/save"+descrStr+".p", "wb" ) )    
    
    plt.show()
    
pickle.dump(timeSave, open( "Plots/saveAll"+descrStr+".p", "wb" ) )    
    

   
     