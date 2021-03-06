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