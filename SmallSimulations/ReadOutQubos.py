# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 15:10:36 2026

@author: MarcelSeelbach(MSE)

Read out the QUBO problems.
Investigate if the problem had to be solved in multiple iterations.
"""
import numpy as np 
import pickle 
import matplotlib.pyplot as plt

whereToSave= 'Plots/SmallestProblemNoSolPos'

k=0
QUBO=  pickle.load( open( whereToSave+'QUBOLambdIt'+str(k)+'.p', "rb" ) )
print(QUBO.toarray())
plt.imshow(QUBO.toarray())
plt.colorbar()

oldQUBO=QUBO.toarray()
for k in range(10):
    QUBO=  pickle.load( open( whereToSave+'QUBOLambdIt'+str(k)+'.p', "rb" ) )
    print(oldQUBO -QUBO.toarray())
    oldQUBO=QUBO.toarray()
