# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 15:18:19 2016

@author: alana
"""

import numpy as np
#import NeuralNet
    
def Parse(fileName, params):
    #get data from the file
    X = np.genfromtxt(fileName, delimiter=',', usecols = (range(params)))
    y = np.genfromtxt(fileName, delimiter=',', usecols = (params), dtype=str)
    
    #change strings into indexed outputs
    for i in range(len(y)):
        if (y[i]=='Iris-setosa'):
            y[i]=int(0)
        elif (y[i]=='Iris-versicolor'):
            y[i]=int(1)
        elif (y[i]=='Iris-virginica'):
            y[i]=int(2)

    y=y.astype(float) #otherwise stays as string
    
    return X, y
    
#remove one at a time... very time intensive
def CrossValidate(numIterations, X,y):
    runningError =0    
    
    #leave one out cross-validation... good, but inefficient
    for i in range(len(y)):
        print(str(i))
        XTrain = X
        yTrain = y
        x_out = X[i,:]
        y_out = y[i]
        np.delete(XTrain,(i), axis=0)
        np.delete(yTrain,(i), axis=0)
        
        ann = NeuralNet2.ANN()
        ann.StochasticGD(numIterations,XTrain,yTrain)
        error=ann.CalculateError(x_out,y_out)
        runningError+=error
    
    totalError=runningError/len(y)
    return totalError