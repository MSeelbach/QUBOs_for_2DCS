# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 17:11:26 2025

@author: MarcelSeelbach(MSE)
"""

import numpy as np

import neal

#Simulated annealing Solver
def runSimulatedFullyConnectedQUBO(W,numOfReads=1000, numOfSweeps=100 ,beta_schedule_type="geometric",initial_states=None):
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
    best=1e+5
    occ=0
    for diter,datum in enumerate(sampleset.data(['sample', 'energy', 'num_occurrences'])):   
             #      print(datum.sample, "Energy: ", datum.energy, "Occurrences: ", datum.num_occurrences)
                     Result.append([datum.sample,  datum.energy,  datum.num_occurrences])
                     if diter==0 or datum.energy== best:# samples are in increasing order
                         best =  datum.energy
                         occ+= datum.num_occurrences

    return np.array(  list( sampleset.first[0].values()),dtype = float )#, occ, Result


def runSimulatedQUBOSparse(W,numOfReads=1000, numOfSweeps=100 ,beta_schedule_type="geometric",initial_states=None):
    d = W.todok()


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

  
