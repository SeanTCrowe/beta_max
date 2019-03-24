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

t1="1/1/2017"
t2="5/1/2017"



#get the data for the above stocks and the covariance matrix
C,TheData=FinMod.CovMat(stocks,t1,t2)

#use this as you're benchmark
sp500=web.DataReader('spy',data_source='yahoo',start=t1,end=t2)


#New cell, don't want to import new data constantly
#%%



#initialize a vector to hold the betas
beta=np.zeros((N,1))

#initizlize a vector to hold the errors of the betas
dbeta=np.zeros((N,1))

#initialize a return vector
ret=np.zeros((N,1))


for n in range(0,N):
    
    p,stnderror_Slope,stnderror_Int=FinMod.Fitter(TheData[n],sp500)
    beta[n]=p[0]
    dbeta[n]=stnderror_Slope
    
for m in range(0,N):
    
    stoc=TheData[m]
    
    ret[m]=(stoc['DR']).mean()
    
    



delta=FinMod.SharpeMax(C,ret)
#ddelta=FinMod.RWBMax(C,dbeta)


#for m in range(0,N):
#    
#    if (abs(ddelta[m])/abs(delta[m]))>0.3:
#        
#        delta[m]=0
##        
#delta=delta/FinMod.l1norm(delta)
print(delta)    

#%%
#now we use the delta from the last year on the next year.


t1p='5/1/2017'
t2p='9/1/2017'
Cp,TheDatap=FinMod.CovMat(stocks,t1p,t2p)

sp500p=web.DataReader('spy',data_source='yahoo',start=t1p,end=t2p)



#initialize a new return vector
retp=np.zeros((N,1))

betap=np.zeros((N,1))
dbetap=np.zeros((N,1))


for n in range(0,N):
    
    p,stnderror_Slope,stnderror_Int=FinMod.Fitter(TheDatap[n],sp500p)
    betap[n]=p[0]
    dbetap[n]=stnderror_Slope


for m in range(0,N):
    
    stoc=TheDatap[m]
    
    print(m)
    
    retp[m]=(stoc['DR']).mean()
    
print(np.dot(retp.T,delta)*252)
    




    
    








