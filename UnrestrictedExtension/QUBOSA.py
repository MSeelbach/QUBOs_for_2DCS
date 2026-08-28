# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 21:47:22 2019

@author: Marcel
"""

import matplotlib.pyplot as plt
import numpy as np

import NealSolver

from scipy.sparse import csr_matrix, vstack

import pickle 
import scipy



def FixVarinQUBO(Q, fixVariableToZero, fixVariableToOne ):
    restVariables = np.where(fixVariableToZero+fixVariableToOne==0)[0]
    linSetToOne = np.diag(Q@fixVariableToOne) + np.diag(fixVariableToOne.T @ Q)
    Qreturn = np.zeros((len(restVariables), len(restVariables)))
    Qfull = Q+linSetToOne

    try:
        Qreturn = Qfull[np.ix_(restVariables[:], restVariables[:])]
    except:
        Qreturn = Qfull.toarray()[np.ix_(restVariables[:], restVariables[:])]

    
    return csr_matrix( Qreturn )

def FullSol(sol,fixVariableToZero, fixVariableToOne):
    solreturn= np.zeros(len(fixVariableToOne), dtype=int )
    solreturn+= fixVariableToOne 
    restVariables= np.where(fixVariableToZero+fixVariableToOne==0)[0]
    solreturn[np.ix_(restVariables[:])]= sol[:]
    return solreturn 


def ChangeToIsing(Q):
    
    couplings= Q/4 
    bias=( Q@np.ones(Q.shape[0])+ Q.T@np.ones(Q.shape[0]) )/4
    return couplings, bias

def ChangesolToQubo(sol):
    solQUBO=  np.zeros(len(sol), dtype=int)
    solQUBO= (sol+ np.ones(len(sol), dtype = int))/2
    return solQUBO

def Energy(vec,mat):
    return vec.T@mat@vec


def CheckFeasibility(sol,Aineq, bineq):
    
    
    feasibility = ( Aineq@sol<= bineq ).all()
    
    
    return feasibility

def CheckUnique(sol,Qdemand):
    
    return (Energy(sol,Qdemand)==0).all()


def GetQuadratFunction(i,N , xOry,Aineq, ListQuadraticTerms):
    
    Qi = ListQuadraticTerms[xOry][i].copy()
    
    
    Qi += scipy.sparse.diags( Aineq[ N* xOry+ i ].toarray().ravel() )
    
    
    return Qi



def AugmentedLagrangianQuad(sol, Obj,Qcost,Aineq, bineq, Qdemand,Qzcorr,ListQuadraticTerms,fixVariableToZero, fixVariableToOne,lList ,wList,whereToSave  ,maxiter=15):
    
    #Implemented According to https://arxiv.org/pdf/2507.12159
    N= Aineq.shape[0]//2       
    mu = np.array(wList+lList)**2 #np.max (np.abs(Obj.data))/10000  #Think about this more    
    lambdas =   np.ones(Aineq.shape[0])
    returnFull= np.zeros(Obj.shape[0])    
    bestSol= returnFull.copy()    
    bestEnergy= Energy(bestSol, Obj+ Qcost)    
    bestLambdas= lambdas.copy()
    
    returnList=[]
    damping=0#10**-1
    EnQuad = []
    EnLin=[]
    bestIter=0
    for k in range(maxiter):
        
        print(k)
       
        xTerms= [ GetQuadratFunction(i,N , 0 ,Aineq, ListQuadraticTerms)  for i in range(N) ]
        
        yTerms= [ GetQuadratFunction(i,N , 1 ,Aineq, ListQuadraticTerms)  for i in range(N) ]

        TermList= xTerms+yTerms        

        #QObjinp, Qlambdinp, const= getQMatrixfrom(Obj,Aineq, bineq, lambdas, mu /2)
        if True:
            QUBOPart, QUBOPartQuadr= getQMatrixfromList( FixVariableSquare(TermList,N ,returnFull),  Aineq,  bineq, lambdas, QuadratBool=False,mu=mu )
        else:
            
            lambdas = bestLambdas
            QUBOPart = getQMatrixfromList( TermList, Aineq,   bineq, lambdas, QuadratBool=True,mu=mu )
        
        QObjinp=Obj+ Qcost
        
       
        #TODO multiply with lambda and square if z is fixed 
        
        
        objFact=20#=10 from previous Experiments
        lagrangeFact = 1        
        
        outNumber=100
        avgNumber=1
        #damping=  0
        demandFact = 500#25
        zCorrFact= demandFact
        fewzFact= 3*10**5

        if True: 
            QObjinpWithoutZPart,QdemandWithoutZPart,QzcorrWithoutZPart=FixVariableSquare([QObjinp,Qdemand,Qzcorr],N,returnFull)
            fullQUBO=objFact*QObjinpWithoutZPart + lagrangeFact*(QUBOPart+ QUBOPartQuadr)+demandFact*QdemandWithoutZPart+ zCorrFact*QzcorrWithoutZPart+damping*scipy.sparse.diags((0.5*np.ones(returnFull.shape)-returnFull)[:QUBOPart.shape[0]])
            fullQUBO= np.array(fullQUBO)
        
        else: 
            
            AllZ= np.zeros((1,3*N**2))
            AllZ[0,2*N**2:]=1
            ReduceZQubo= getEqualityConstraint(AllZ, np.array([0])) 
            fullQUBO=objFact*QObjinp + lagrangeFact*QUBOPart+demandFact*Qdemand+fewzFact*ReduceZQubo+ zCorrFact*Qzcorr+damping*scipy.sparse.diags((0.5*np.ones(returnFull.shape)-returnFull)[:])
            fullQUBO= np.array(fullQUBO)
            
       
        smallerQUBO=  FixVarinQUBO(fullQUBO, fixVariableToZero[:fullQUBO.shape[0]], fixVariableToOne[:fullQUBO.shape[0]] )
        plt.imshow( smallerQUBO.toarray())#np.maximum( np.minimum( fullQUBO,10*np.ones(fullQUBO.shape)),-10*np.ones(fullQUBO.shape)))
        
        pickle.dump(smallerQUBO, open( whereToSave+'QUBOLambdIt'+str(k)+'.p', "wb" ) )
        
        plt.colorbar()
        plt.savefig(whereToSave+'ImgPlotsLambdIt'+str(k)+'.png', format='png')
        plt.show()
        #smallerQUBO.dtype=int
        returnSmall=NealSolver.runSimulatedQUBOSparseMult(smallerQUBO,outNumber=outNumber)#initial_states =[numpyArrayToNealDict(np.ones(Q.shape[0]))])
        
    
        if  True: 
            
            previousz= returnFull[2*N**2:].copy()
            
            returnFull=np.hstack([FullSol(returnSmall[0],fixVariableToZero[:fullQUBO.shape[0]], fixVariableToOne[:fullQUBO.shape[0]] ) ,previousz ])      
            returnFullList=[ np.hstack([FullSol(returnSmall[t],fixVariableToZero[:fullQUBO.shape[0]], fixVariableToOne[:fullQUBO.shape[0]]),previousz ]) for t in range (outNumber)]
            EnLin.append(Energy(returnFull[:2*N**2], QUBOPart))
            EnQuad.append(Energy(returnFull[:2*N**2], QUBOPartQuadr))
        else:
                       
            returnFull=FullSol(returnSmall[0],fixVariableToZero, fixVariableToOne )        
            returnFullList=[ FullSol(returnSmall[t],fixVariableToZero, fixVariableToOne) for t in range (outNumber)]
            
        lastLambdas= lambdas.copy()
        lambdas  += np.average([ mu  *  np.maximum( np.zeros(Aineq.shape[0]) , (Aineq*returnFullList[returnFullElementNr] - bineq) )  for returnFullElementNr in range(avgNumber)],axis=0)
        #lambdas  +=  mu * ( Aineq*returnFull - bineq)  
        
        
        #lambdas= np.ones(lambdas.shape)* np.max(lambdas)
        
        
        #mu=0.9*mu
        
        returnList.append(returnFull)
        for returnFullElement in returnFullList:
            newEnergy=Energy(returnFullElement,Obj+ Qcost)
            if newEnergy<bestEnergy and CheckFeasibility(returnFullElement,Aineq, bineq) and CheckUnique(returnFullElement,Qdemand):
                    bestEnergy= newEnergy
                    bestSol=  returnFullElement.copy()
                    bestLambdas= lastLambdas.copy()
                    bestIter=k
    
    return   returnList, bestSol , EnLin, EnQuad, bestIter
    

def Solve(sol, Obj,Aineq, bineq, Qdemand,fixVariableToZero, fixVariableToOne):
        
    eps=3
    lambdas= np.linalg.solve(   Aineq @ Aineq.T , Aineq@ ( np.diag(Obj) - eps* (sol- np.ones(sol.shape) )) )
            
    QObjinp, Qlambdinp, const= getQMatrixfrom(Obj,Aineq, bineq, lambdas)
     

    
    # np.diag(QObjinp)/ np.diag(Qlambdinp)+ eps*(sol- np.ones(sol.shape)/2)   

    lagrangeFact=1
    
    demandFact=500#25
    fullQUBO=QObjinp-  lagrangeFact*Qlambdinp+demandFact*Qdemand
    
    smallerQUBO=  FixVarinQUBO(fullQUBO, fixVariableToZero, fixVariableToOne )
    
    N= smallerQUBO.shape[0]
    energies= []
    for t in range(2**N):
        energies.append(Energy( np.array(IntegerToList(t,N)),smallerQUBO ))
    #plt.plot(energies)
    
    returnSmall=NealSolver.runSimulatedFullyConnectedQUBO(smallerQUBO)#initial_states =[numpyArrayToNealDict(np.ones(Q.shape[0]))])

    returnFull=FullSol(returnSmall,fixVariableToZero, fixVariableToOne)


    return returnFull


def IntegerToList(n,Length): 
    binary = [0]*Length
    num=n
    for i in range(1,Length+1):
        bit = num % 2
        binary[Length-i]= int( bit)
        num = num / 2
    return binary




def getEqualityConstraint( A, b ):
            
            
            return A.T@ A -scipy.sparse.diags(( A.T@b + (b.T @  A).T )[:])




def OptimizeDirectly( coefficients ,A, b ):
            
            
    return  scipy.sparse.diags((coefficients*A).toarray().ravel())



def getQMatrixfrom(QObj, Aineq, bineq, lambdas, mu =0):
    '''
    Parameters
    ----------
    QObj : objective term
    Qineqlist : list of matrix
    bineq : bound list

    Returns
    -------
    Q : numpy array
    '''
    
    Qlambd = OptimizeDirectly(lambdas, Aineq, bineq)
    Qlambd += mu*  getEqualityConstraint( Aineq , np.zeros (Aineq.shape[0]) )

    const= bineq
    
    return QObj, Qlambd, const

def FixVariableSquare(QuboList, N ,currSol):
    
    QlambList= []    
    fixVariableToZero= np.zeros(currSol.shape)
    fixVariableToOne= np.zeros(currSol.shape)

    fixVariableToZero[np.where (currSol==0)[0]]=1 
    fixVariableToOne[ np.where (currSol==1)[0]]=1
    fixVariableToZero[:2*N**2]=0
    fixVariableToOne[:2*N**2]=0
    for k in range(len(QuboList)):
        


        
        QlambList.append(  FixVarinQUBO( QuboList[k].toarray() , fixVariableToZero, fixVariableToOne )  )
        
        
        
    return QlambList
    
    
def getQMatrixfromList( TermsList,  Aineq,  bineq, lambdas, QuadratBool=False,mu =0):
    '''
    Parameters
    ----------
    QObj : objective term
    Qineqlist : list of matrix
    bineq : bound list

    Returns
    -------
    Q : numpy array
    '''
    
    NrTotalVar= TermsList[0].shape[0]
    
    Qlambd=  csr_matrix( (NrTotalVar, NrTotalVar), dtype= float)
    QlambdQuad=  csr_matrix( (NrTotalVar, NrTotalVar), dtype= float)

    
    if QuadratBool==False:
    
         for s in range(len(TermsList)):
            
                
                Qlambd +=lambdas[s]* TermsList[s]# OptimizeDirectly(lambdas[s], TermsList[s], bineq[s])
                
         QlambdQuad +=   getEqualityConstraint(   (Aineq.toarray().T*np.sqrt(mu)).T[:,:NrTotalVar] , np.zeros (Aineq.shape[0]) )
            
    else: 
        
        
         for s in range(len(TermsList)):

                    Qlambd +=lambdas[s]* TermsList[s] #OptimizeDirectly(lambdas[s], TermsList[s], bineq[s])
           
    
    return  Qlambd, QlambdQuad


