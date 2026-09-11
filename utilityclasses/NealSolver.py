# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 17:11:26 2025

@author: MarcelSeelbach(MSE)

Multiple functions to solve QUBOs via neal
"""

import numpy as np

import neal



def runSimulatedFullyConnectedQUBO(W,numOfReads=1000, numOfSweeps=100 ,beta_schedule_type="geometric",initial_states=None):
    '''
    Parameters
    ----------
    W : numpy array
        QUBO matrix.
    numOfReads : int, optional
        How many samples? The default is 1000.
    numOfSweeps : int, optional
        How many sweeps? The default is 100.
    beta_schedule_type : String, optional
        The default is "geometric". Other option would be linear.
    initial_states : dict like output of neal, optional. The default is None.

    Returns
    -------
    binary numpy vector
        Returns the best solution from the state histogram.
    occ : List
        How often did the states occur. (commented out)
    Result : List
        Full state histogram. (commented out)

    '''
    
    
    m = W.shape[0]
    dic={}
    for t1 in range(m):
        for t2 in range(m):

            if W[  t1  ,  t2  ]!=0 and t1!=t2: 
                dic[(  t1  ,t2 )]= W[  t1  ,  t2  ]    

    for t1 in range(m):
                dic[(  t1  ,t1 )]= W[  t1  ,  t1  ]    
        


    sampler  =  neal.SimulatedAnnealingSampler()
    sampleset = sampler.sample_qubo(dic ,num_reads=numOfReads, num_sweeps=numOfSweeps,beta_schedule_type=beta_schedule_type,  initial_states=initial_states)


    Result=[]
    best=1e+5 #should work regardles of initialization due to diter==0 in line 40
    occ=0
    for diter,datum in enumerate(sampleset.data(['sample', 'energy', 'num_occurrences'])):   
             #      print(datum.sample, "Energy: ", datum.energy, "Occurrences: ", datum.num_occurrences)
                     Result.append([datum.sample,  datum.energy,  datum.num_occurrences])
                     if diter==0 or datum.energy== best:# samples are in increasing order
                         best =  datum.energy
                         occ+= datum.num_occurrences

    return np.array(  list( sampleset.first[0].values()),dtype = float )#, occ, Result


def runSimulatedQUBOSparse(W,numOfReads=1000, numOfSweeps=100 ,beta_schedule_type="geometric",initial_states=None):
    
    '''
    Parameters
    ----------
    W : numpy array
        QUBO matrix.
    numOfReads : int, optional
        How many samples? The default is 1000.
    numOfSweeps : int, optional
        How many sweeps? The default is 100.
    beta_schedule_type : String, optional
        The default is "geometric". Other option would be linear.
    initial_states : dict like output of neal, optional. The default is None.

    Returns
    -------
    binary numpy vector
        Returns the best solution from the state histogram.
        Same as previous method but also work for sparse input.
    occ : List
        How often did the states occur. (commented out)
    Result : List
        Full state histogram. (commented out)

    '''
    
    d = W.todok() #"Convert this matrix to Dictionary Of Keys format." from https://docs.scipy.org/doc//scipy-1.9.2/reference/generated/scipy.sparse.coo_matrix.todok.html
    sampler  =  neal.SimulatedAnnealingSampler()
    sampleset = sampler.sample_qubo(dict(d) ,num_reads=numOfReads, num_sweeps=numOfSweeps,beta_schedule_type=beta_schedule_type,  initial_states=initial_states)


    #Result=[]
    #best=1e+5
    #occ=0
    #for diter,datum in enumerate(sampleset.data(['sample', 'energy', 'num_occurrences'])):   
             #      print(datum.sample, "Energy: ", datum.energy, "Occurrences: ", datum.num_occurrences)
   #                  Result.append([datum.sample,  datum.energy,  datum.num_occurrences])
   #                  if diter==0 or datum.energy== best:# samples are in increasing order
   #                      best =  datum.energy
   
#                      occ+= datum.num_occurrences

    return np.array(  list( sampleset.first[0].values()),dtype = float )#, occ, Result



def runSimulatedQUBOSparseMult(W,numOfReads=1000, numOfSweeps=100 ,beta_schedule_type="geometric",outNumber=10,initial_states=None):

      '''
      Parameters
      ----------
      W : numpy array
          QUBO matrix.
      numOfReads : int, optional
          How many samples? The default is 1000.
      numOfSweeps : int, optional
          How many sweeps? The default is 100.
      beta_schedule_type : String, optional
          The default is "geometric". Other option would be linear.
      outnumber: int, optional
          Returns that many states of the output histogram. The default is 10.
      initial_states : dict like output of neal, optional. The default is None.

      Returns
      -------
 
      Result : List
      Same method as above but returns 'outnumber' entries of the state histogram.    
      Full state histogram. (commented out)

      '''
      d = W.todok()


      sampler  =  neal.SimulatedAnnealingSampler()
      sampleset = sampler.sample_qubo(dict(d) ,num_reads=numOfReads, num_sweeps=numOfSweeps,beta_schedule_type=beta_schedule_type,  initial_states=initial_states)


      Result=[]
      #best=1e+5
      #occ=0
      for diter,datum in enumerate(sampleset.data(['sample', 'energy', 'num_occurrences'])):   
               #      print(datum.sample, "Energy: ", datum.energy, "Occurrences: ", datum.num_occurrences)
                   if diter>= outNumber:
                       break    
                   Result.append(np.array(  list( datum.sample.values()),dtype = float ))
                      
     #                  if diter==0 or datum.energy== best:# samples are in increasing order
     #                      best =  datum.energy
     
  #                      occ+= datum.num_occurrences

      return Result

  
