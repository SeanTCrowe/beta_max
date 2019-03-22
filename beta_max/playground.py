#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 17:55:07 2019

@author: sean

This is a test file

"""
import numpy as np
import pandas as pd
from  matplotlib import pyplot as plt
from  pandas_datareader import data as web
import FinMod


goog=web.DataReader('GOOG',data_source='yahoo',start="1/1/2018",end='1/1/2019')

sp500=web.DataReader('spy',data_source='yahoo',start="1/1/2018",end='1/1/2019')


#print(FinMod.Sharpe(goog))


print(FinMod.Fitter(goog,sp500))


plt.plot((goog['Close']-goog['Close'].mean())*(sp500['Close']-sp500['Close'].mean()))
plt.show()
print(((goog['Close']-goog['Close'].mean())*(sp500['Close']-sp500['Close'].mean())).mean())

#print(FinMod.Vola(goog))

#a=FinMod.VolMin(1)

#print(a)

#print(a[:,-1])


#Testing volmin and sharpemax

C=np.array([[0.2**2,0.011],[0.011,0.1**2]])
import numpy as np
import numpy.linalg as linalg

    #Find Eigenvalues and eigenvecors of C
eigenValues, eigenVectors = linalg.eig(C)

    #Order eigenvectors according to their eigenvalues
delta=FinMod.VolMin(C)

var=np.dot(delta.T,np.dot(C,delta))


print(delta)
print(FinMod.l1norm(delta))
print(np.sqrt(var))
print(np.sqrt(eigenValues))

print(FinMod.RWBMax(C,np.array([1.2,1.5])[:,None]))

b=np.array([1.2,1.5])[:,None]

print(np.dot(b.T,FinMod.RWBMax(C,np.array([1.2,1.5])[:,None])))




A=np.zeros((2,2))

A[1][1]=1
print(A)



