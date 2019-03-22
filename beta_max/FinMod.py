#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 22:06:09 2019

@author: sean crowe

This is the toolkit script.

Some basic functions we need,

-Calculate alpha and beta for a given stock
-calculate the sharpe ratio for a given stock
-calulate the volatility of a stock, over a given time period.
-Minimize volitility of a port
-Maximize sharpe of a port
-calculate moving average crossing points

Needful things: 
    
    Make sure to generalize things so that we can use minute level 
    data as well as daily data.
    
    Make a function that can make a covariance matrix


"""

def Fitter(stock,bench):
    
    #This is a function that takes a stock and benchmark and computes alpha and 
    #beta. The inputs are pandas data frames. probably one should use the sp500
    # as the bench mark. Just pass whole data frames to this function
    
  
    
    import numpy as np
    
    
    bench['DR']=bench['Close'].pct_change(1)
    stock['DR']=stock['Close'].pct_change(1)
    
    p=np.polyfit(bench['DR'].values[1:],stock['DR'].values[1:],1)
    
    sigb=bench['DR'].std()
    sigStock=stock['DR'].std()
    
    xbar=bench['DR'].mean()
    
    n=len(bench['DR'])
    
    s=sigStock*np.sqrt((1-(p[0]**2)*((sigb/sigStock)**2))*(n/(n-2)))
    
    stnderror_Slope=(1/np.sqrt(n-2))*np.sqrt((sigStock/sigb)**2-p[0]**2)
    stnderror_Int=(1/np.sqrt(n))*s*np.sqrt(1+(xbar/sigb)**2)
    
    return p,stnderror_Slope,stnderror_Int


def Sharpe(stock,days=252):
    
    #This function computes the sharpe ratio of a given stock. days is the time
    # period we consider the sharpe ratio over. It's wasteful, but import the 
    # whole datafram. to make sure everything is consistent.
    
    import numpy as np
    
    
    stock['Daily Return']=stock['Close'].pct_change(1)
    
    sharpe=np.sqrt(days)*stock['Daily Return'].rolling(window=days).mean()/\
    (stock['Daily Return'].rolling(window=days).std())
    return sharpe.mean()

def Vola(stock,days=252):
    
    #This is a script that computes the volatility of the returns of of the a given stock over a 
    #period of time. By default we choose one year as the time period.
    import numpy as np
    
    stock['Daily Return']=stock['Close'].pct_change(1)
    sig=np.sqrt(days)*stock['Daily Return'].std() #annualized std of the dr.
    
    return sig

def l1norm(x):
    
    norm=sum(abs(x))
    
    return norm

##############################################################


def VolMin(C):
    
    # This is a function that minimizes the volatility of a portfolio, given a
    # covariance matrix.
    
    import numpy as np
    import numpy.linalg as linalg

    #Find Eigenvalues and eigenvecors of C
    eigenValues, eigenVectors = linalg.eig(C)

    #Order eigenvectors according to their eigenvalues
    idx = eigenValues.argsort()[::-1]   
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]
    
    tol=10**(-4)
    
    #Designate the face we look at. Its a standing vector
    face=np.sign(eigenVectors[:,-1])[:,None]
    facel2norm=face/np.sqrt(len(face))
    
    #make the inital delta,  THis is a FINE guess
    delta=(eigenVectors[:,-1]/l1norm(eigenVectors[:,-1]))[:,None]
    
   

    def graddescent_Vol(delta_n):
        
        #This function minimizes the volatility respecting the portfolio constraint.
        #using gradient descent
        
        #learning rate
        alpha=0.01
        
        delta_np1=delta_n-2*alpha*(np.dot(C,delta_n)-np.dot(np.dot(C,delta_n).T,facel2norm)*facel2norm)
        
        return delta_np1
    
    def resid(delta_n):
        
        var=np.dot(delta_n.T,np.dot(C,delta_n))
        r=l1norm((np.dot(C,delta_n)-var*face))
        
        return r
    
    r=1
    counter=0
    deltan=delta
    while r>=tol:
        
        deltan=graddescent_Vol(deltan)
        #deltan=deltan/l1norm(deltan)
        r=resid(deltan)
        counter+=1
        if counter>=10000:
            print('gradient descent failed to converge')
            break
    
    

    return deltan




##############################################################
def SharpeMax(C,ret):
    
    import numpy as np
    import numpy.linalg as linalg
    
    Cinv=linalg.inv(C)
    delta=np.dot(Cinv,ret)
    delta=delta/l1norm(delta)
    
    return delta

##############################################################
def RWBMax(C,beta):
    
    #This function maximizes the risk weighted beta. If you aim to short you need
    # to multipy by-1.
    
    import numpy as np
    import numpy.linalg as linalg
    
    Cinv=linalg.inv(C)
    delta=np.dot(Cinv,beta)
    delta=delta/l1norm(delta)
    
    return delta


def CovMat(stocks,dt=252):
    
    import numpy as np
    import pandas as pd
    from  matplotlib import pyplot as plt
    from  pandas_datareader import data as web
    
    #This is a function that finds the covariance matrix with a windowing
    #period that can be varied. We should only have to compute this when
    # moving averages are crossed. This function also returns all other stock data
    # back to main
    
    #needful things, fix the timing.
    
    #make an empty data frame to store all the data
    TheData = []
    
    #count the number of stocks
    n=len(stocks)
    
    for things in stocks:
        TheData.append(web.DataReader(things,data_source='yahoo',start="1/1/2018",end='1/1/2019'))
    
    for m in TheData:
        
        m['DR']=m['Close'].pct_change(1)
        
    
    
    C= np.zeros((n,n))
    
    for k in range(0,n):
        
        for m in range(0,n):
            
            stockk=TheData[k]
            stockm=TheData[m]
            
            C[k,m]=((stockk['DR']-stockk['DR'].mean())*(stockm['DR']-stockm['DR'].mean())).mean()
    

    return C,TheData
    
    
        









