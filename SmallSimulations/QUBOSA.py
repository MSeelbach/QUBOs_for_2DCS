# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 21:47:22 2019

@author: Marcel

This class implements the Augmented Lagrangian method described in the publication "Formulating 2D Cutting Stock Problems as QUBO".

"""

import matplotlib.pyplot as plt
import numpy as np

from QUBOs_for_2DCS.utilityclasses import NealSolver

from scipy.sparse import csr_matrix, vstack

import pickle 
import scipy



def FixVarinQUBO(Q, fixVariableToZero, fixVariableToOne ):
    '''
    Parameters
    ----------
    Q : numpy array or csr matrix
        QUBO Coupling matrix
    fixVariableToZero : numpy array
        The variables that we want to set to zero by force.
    fixVariableToOne : numpy array
        The variables that we want to set to one by force.

    Returns 
    -------
    smaller QUBO coupling matrix in a sparse format. 

    '''
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
    '''
    Parameters
    ----------
    sol : numpy array
        solution for the entries that were not fixed before
    fixVariableToZero : numpy array
      The variables that we want to set to zero by force.
    fixVariableToOne : numpy array
      The variables that we want to set to one by force.

    Returns
    -------
    solreturn : numpy array
        solution binary vector that also includes entries that were forced to be zero or one.

    '''
    solreturn= np.zeros(len(fixVariableToOne), dtype=int )
    solreturn+= fixVariableToOne 
    restVariables= np.where(fixVariableToZero+fixVariableToOne==0)[0]
    solreturn[np.ix_(restVariables[:])]= sol[:]
    return solreturn 


def ChangeToIsing(Q):
    '''
    Parameters
    ----------
    Q : np array
        QUBO couplings matrix

    Returns
    -------
    couplings : numpy arra
        Couplings of the Ising problem 
    bias : numpy array
        Bias vector of the Ising problem

    '''
    couplings= Q/4 
    bias=( Q@np.ones(Q.shape[0])+ Q.T@np.ones(Q.shape[0]) )/4
    return couplings, bias


def Energy(vec,mat):
    '''
    Parameters
    ----------
    vec : numpy array
        vector for energy calculation
    mat : numpy array
        matrix for energy calculation

    Returns 
    -------
    Energy

    '''
    return vec.T@mat@vec


def CheckFeasibility(sol,Aineq, bineq):
    '''
    Parameters
    ----------
    sol : numpy array
        Binary solution vector
    Aineq : numpy array
        Matrix for linear inequality
    bineq : numpy array
        vector to describe linear inequality

    Returns
    -------
    feasibility : Boolean
    Is the solution feasible for the inequality constraints?

    '''
    
    feasibility = ( Aineq@sol<= bineq ).all()
    
    
    return feasibility

def CheckUnique(sol,Qdemand):
    #Are there many vectors with energy of zero?

    return (Energy(sol,Qdemand)==0).all()


def GetQuadratFunction(i,N , xOry,Aineq, ListQuadraticTerms):
    '''
    Parameters
    ----------
    i : int
        For what inequality do we want to find the higher order terms?
    N : int
        Total Nr. variables s. paper
    xOry : int
        are cuts in x or y direction considered
    Aineq : numpy array
        Inequality constraints
    ListQuadraticTerms : list
        What are the quadratic terms for the setting specified above

    Returns
    -------
    Qi : scipy.sparse
        Helper function to tackle the unconstrained case as described in the Appendix

    '''
    Qi = ListQuadraticTerms[xOry][i].copy()
    
    
    Qi += scipy.sparse.diags( Aineq[ N* xOry+ i ].toarray().ravel() )
    
    
    return Qi



def AugmentedLagrangianQuad(sol, Obj,Aineq, bineq, Qdemand,ListQuadraticTerms,fixVariableToZero, fixVariableToOne,lList ,wList,whereToSave  ,maxiter=15):
    '''
    Parameters
    ----------
    Obj : numpy array
        Objective in the optimization
    Qcost : numpy array
        Important for the unrestricted case if there are terms with z-variables in the objective
    Aineq : numpy array
        Size inequalities matrix
    bineq : numpy array
        Size inequaliteis vector
    Qdemand : numpy array
        Addition to the QUBO modell to describe the demand
    Qzcorr : numpy array
        Interesing for the unconstrained case
   ListQuadraticTerms : list
      What are the quadratic terms for the setting specified above
    fixVariableToZero : numpy array
        The variables that we want to set to zero by force.
    fixVariableToOne : numpy array
        The variables that we want to set to one by force.
    lList : list
        List of lengths.
    wList : list
        List of widths.
    whereToSave : String
        save possible plots.
    maxiter : int, Optional
        For how many iterations is the method carried out? The default is 15.

    Returns
    -------
    returnList:
        List of iterates       
    bestSol:
        Best solution that occured
    EnLin:
        Energy due to objective original problem 
    EnQuad
        Energy due to objective quadratic terms
    
    bestIter
        At what iteration did the best energy occured. 
    '''    
    #Implemented According to https://arxiv.org/pdf/2507.12159
    N= Aineq.shape[0]//2       
    mu = np.array(wList+lList)**2 #np.max (np.abs(Obj.data))/10000  #Possible Initializations    
    lambdas =   np.ones(Aineq.shape[0])
    returnFull= np.zeros(Obj.shape[0])    
    bestSol= returnFull.copy()    
    bestEnergy= Energy(bestSol, Obj)    
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

        QUBOPart, QUBOPartQuadr= getQMatrixfromList( TermList,  Aineq,  bineq, lambdas, QuadratBool=False,mu=mu )
        
        #TODO multiply with lambda and square if z is fixed 
               
        objFact=20#=10 from previous Experiments
        lagrangeFact = 1        
        
        outNumber=100
        avgNumber=1
        #damping=  0
        demandFact = 500#25
        zCorrFact= demandFact
        fewzFact= 3*10**5

        QObjinpWithoutZPart,QdemandWithoutZPart,QzcorrWithoutZPart=[Obj,Qdemand]
        fullQUBO=objFact*QObjinpWithoutZPart + lagrangeFact*(QUBOPart+ QUBOPartQuadr)+demandFact*QdemandWithoutZPart+ zCorrFact*QzcorrWithoutZPart+damping*scipy.sparse.diags((0.5*np.ones(returnFull.shape)-returnFull)[:QUBOPart.shape[0]])
        fullQUBO= np.array(fullQUBO)
             
        smallerQUBO=  FixVarinQUBO(fullQUBO, fixVariableToZero[:fullQUBO.shape[0]], fixVariableToOne[:fullQUBO.shape[0]] )
        plt.imshow( smallerQUBO.toarray())#np.maximum( np.minimum( fullQUBO,10*np.ones(fullQUBO.shape)),-10*np.ones(fullQUBO.shape)))
        
        pickle.dump(smallerQUBO, open( whereToSave+'QUBOLambdIt'+str(k)+'.p', "wb" ) )
        
        plt.colorbar()
        plt.savefig(whereToSave+'ImgPlotsLambdIt'+str(k)+'.png', format='png')
        plt.show()

        #smallerQUBO.dtype=int        
        returnSmall=NealSolver.runSimulatedQUBOSparseMult(smallerQUBO,outNumber=outNumber)#initial_states =[numpyArrayToNealDict(np.ones(Q.shape[0]))])
        
        
        
        previousz= returnFull[2*N**2:].copy()
        
        returnFull=np.hstack([FullSol(returnSmall[0],fixVariableToZero[:fullQUBO.shape[0]], fixVariableToOne[:fullQUBO.shape[0]] ) ,previousz ])      
        returnFullList=[ np.hstack([FullSol(returnSmall[t],fixVariableToZero[:fullQUBO.shape[0]], fixVariableToOne[:fullQUBO.shape[0]]),previousz ]) for t in range (outNumber)]
        EnLin.append( Energy(returnFull[:2*N**2], QUBOPart ))
        EnQuad.append( Energy(returnFull[:2*N**2], QUBOPartQuadr ))
   
        
        lastLambdas= lambdas.copy()
        lambdas  += np.average([ mu  *  np.maximum( np.zeros(Aineq.shape[0]) , (Aineq*returnFullList[returnFullElementNr] - bineq) )  for returnFullElementNr in range(avgNumber)],axis=0)
        #lambdas  +=  mu * ( Aineq*returnFull - bineq)  
        
        
        #lambdas= np.ones(lambdas.shape)* np.max(lambdas)
        
        
        #mu=0.9*mu
        
        returnList.append(returnFull)
        for returnFullElement in returnFullList:
            newEnergy=Energy(returnFullElement,Obj)
            if newEnergy<bestEnergy and CheckFeasibility(returnFullElement,Aineq, bineq) and CheckUnique(returnFullElement,Qdemand):
                    bestEnergy= newEnergy
                    bestSol=  returnFullElement.copy()
                    bestLambdas= lastLambdas.copy()
                    bestIter=k
    
    return   returnList, bestSol , EnLin, EnQuad, bestIter
    




def IntegerToList(n,Length): 
    '''
    Parameters
    ----------
    n : int
        Integer to be written in binary representation
    Length : int
        How long should the output binary representation be?

    Returns
    -------
    binary : List
        binary representation of input integer
    '''
    binary = [0]*Length
    num=n
    for i in range(1,Length+1):
        bit = num % 2
        binary[Length-i]= int( bit)
        num = num / 2
    return binary




def getEqualityConstraint( A, b ):
            '''
            Parameters
            ----------
            A : numpy array
                Matrix for equality constraint
            b : numpy array
                Vector for equality constraint
        
            Returns
            -------
            Penalty term that enforces the equality constraint.
        
            '''
            
            return A.T@ A -scipy.sparse.diags(( A.T@b + (b.T @  A).T )[:])




def OptimizeDirectly( coefficients ,A, b ):
    '''
    Parameters
    ----------
    coefficients : numpy array
        lambda parameters
    A : numpy array
        Matrix for equality constraint
    b : numpy array
        Vector for equality constraint

    Returns
    -------
    Instead of having a penalty term the equality is here linearly added to the QUBO with penalty parameters
    '''
            
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


