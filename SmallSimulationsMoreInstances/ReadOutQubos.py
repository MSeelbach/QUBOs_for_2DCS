# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 15:10:36 2026

@author: MarcelSeelbach(MSE)
"""
import numpy as np 
import pickle 
import matplotlib.pyplot as plt

Pieces= 2 # 2 or 3

W=9   # range(10,24) for 3 pieces #  range(9,16) for 2 pieces


whereToSave= 'Plots/'+str(Pieces)+'PieceProblemW'+str(W) 

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
