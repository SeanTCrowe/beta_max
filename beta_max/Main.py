#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 21:49:18 2019

@author: sean crowe

In this script we are going to implement the strategy in documentation using the
functinons in FinMod

"""
import numpy as np
import pandas as pd
import numpy.linalg as linalg
from  matplotlib import pyplot as plt
from  pandas_datareader import data as web
import FinMod

#Stocks that we care about
stocks=['MMM','ALB','BHGE','CBOE','DVA','EL','FISV','GD','HP','IDXX','KMI'\
        ,'LMT','MCD','NKE','PCAR','O','STX','TJX','UNH','VNO','WMB']

N=len(stocks)#number of stocks


#get the data for the above stocks and the covariance matrix
C,TheData=FinMod.CovMat(stocks)

#use this as you're benchmark
sp500=web.DataReader('spy',data_source='yahoo',start="1/1/2018",end='1/1/2019')


#initialize a vector to hold the betas
beta=np.zeros((N,1))

#initialize a return vector
ret=np.zeros((N,1))


for n in range(0,N):
    
    p,stnderror_Slope,stnderror_Int=FinMod.Fitter(TheData[n],sp500)
    beta[n]=p[0]
    
for m in range(0,N):
    
    stoc=TheData[m]
    
    ret[m]=stoc['DR'].mean()
    
    



delta=FinMod.RWBMax(C,beta)




    
    








