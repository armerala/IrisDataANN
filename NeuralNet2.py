# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 14:57:13 2016

@author: alana
"""

#############################################################################################
#An Artificial Neural Network(ANN) framework, made from scratch.                            #
#Has one hidden layers of variable size, as well as input and output layers of variable size#
#The activiation function for hidden layer 1 is tanh, while it is softmax for the output    #                  
#layer.                                                                                     #
#                                                                                           #
#Made by Alan Armero                                                                        #
#############################################################################################

import numpy as np
import random as rand


def SoftMax(x):
    xt = np.exp(x - np.max(x))
    return xt / np.sum(xt)

class ANN:
    def __init__(self, inputDim=4, hidDim=4, outputDim=3, alpha = 0.01, regLambda = .01):
        #dimensionality of each layer
        self.inputDim = inputDim
        self.hidDim = hidDim
        self.outputDim = outputDim
        
        #learning parameters
        self.alpha = alpha #learning rate
        self.regLambda = regLambda #lambda regularization parameter
        
        #create weight matrix for layer 1 of size parameter dim x hidden layer 1 dim
        self.W1 = np.random.uniform(-np.sqrt(1/inputDim), np.sqrt(1/inputDim), (inputDim, hidDim))
        self.b1 = np.zeros((1, hidDim))
        
        #create weight matrix for layer 2 of size hidden layer 1 dim x hidden layer 2 dim
        self.W2 = np.random.uniform(-np.sqrt(1/hidDim), np.sqrt(1/hidDim), (hidDim,outputDim))
        self.b2 = np.zeros((1,outputDim))

        
    #forward propogation algorithm takes a matrix "x" of any size x inputDim
    def ForProp(self, X):            
        #signal vector for hidden layer
        #tanh activation function
        S1 = X.dot(self.W1) + self.b1
        Z1 = np.array(np.tanh(S1))
        
        #vector for the final output layer
        S2 = Z1.dot(self.W2)+ self.b2
        #softmax for output layer activation
        expScores = np.array(np.exp(S2))
        output = np.array(expScores/(np.sum(expScores, axis=1, keepdims=True)))    
        return output,Z1

    #cross-entropy error of an input set and a vector y of size N
    #cross-entropy error
    def CalculateError(self, output, y):
        #calculate total error against the vector y for the neurons where output = 1 (the rest are 0)    
        totalError = 0
        for i in range(0,len(y)):
            totalError += -np.log(np.array(output)[i, int(y[i])])

        #now account for regularizer
        totalError+=(self.regLambda/self.inputDim) * (np.sum(np.square(self.W1))+np.sum(np.square(self.W2)))     
    
        error=totalError/len(y) #divide ny N
        return error
                
    def StochasticGD(self, numIterations, X, y):
        
        
        #storage for the best performing weights in-sample
        bestError = 1000000 #very poor practice. Note to self: fix this       
        bestW1 = np.zeros((self.inputDim, self.hidDim))
        bestb1 = np.zeros((1, self.hidDim))
        bestW2 = np.zeros((self.hidDim, self.outputDim))
        bestb2 = np.zeros((1,self.outputDim))
                       
        for i in range (0,numIterations):
            #select random x_n to perform SGD
            n = np.random.randint(len(y))            
            x_n= X[n,:]
            #for whatever reason, the above line retrurns shape of (,4) instead of (1,4)
            #thus we have to reshape, it seems
            x_n = x_n.reshape((1,4))
            #forward propogation and get probabilities of each
            output,z1 = self.ForProp(x_n)
            
            #backprop
            #alert: some fuckaging with the dimensions here... pay attgention to sizes
            #calculate deltas for each layers
            delta3 = output
            delta3-=1
            delta2 = (delta3.dot(self.W2.T))*(1-(z1**2))
            
            #calculate individual gradients from the deltas
            dW2 = (z1.T).dot(delta3)
            db2 = np.sum(delta3, axis=0, keepdims = True)
            dW1 = (x_n.T).dot(delta2)
            db1 = np.sum(delta2, axis = 0, keepdims=True)
            
            #add regularizer slowdown to gradients
            dW2 += self.regLambda * self.W2
            dW1 += self.regLambda * self.W1
            
            #adjust weights
            self.W2-= self.alpha * dW2
            self.b2-= self.alpha * db2
            self.W1-= self.alpha * dW1
            self.b1-= self.alpha * db1
            
            #reclaculate error
            newOutput,___ = self.ForProp(X)
            error = self.CalculateError(newOutput, y)
            #if error went down, log these weights
            if(error<bestError):
                bestError = error
                bestW1 = self.W1
                bestb1 = self.b1
                bestW2 = self.W2
                bestb2 = self.b2
            if (i%1000==0):
                print(bestError)
            
        #after loop take on the best values
        self.W1 = bestW1
        self.b1 = bestb1
        self.W2 = bestW2
        self.b2 = bestb2
        
        print("In Sample Error: " + str(bestError))