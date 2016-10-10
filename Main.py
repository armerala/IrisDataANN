# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 18:10:21 2016

@author: alana
"""

import DataParser
import NeuralNet2
 
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

X,y = DataParser.Parse('IrisData.txt', 4)
ann = NeuralNet2.ANN()
ann.StochasticGD(100, X, y)