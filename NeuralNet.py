# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 14:57:13 2016

@author: alana
"""

#############################################################################################
#An Artificial Neural Network(ANN) framework, made from scratch.                            #
#Has 2 hidden layers of variable size, as well as input and output layers of variable size. #
#The activiation function for hidden layer 1 & 2 is tanh, while it is softmax for the output#                  
#layer.                                                                                     #
#                                                                                           #
#Made by Alan Armero                                                                        #
#############################################################################################

#from IrisData import*
import numpy as np
import random as rand


def SoftMax(x):
    xt = np.exp(x - np.max(x))
    return xt / np.sum(xt)

class ANN:
    def __init__(self, lay0Dim=4, lay1Dim=3, lay2Dim=3, outputDim=3, alpha = 0.01, regLambda = .01):
        #dimensionality of each layer
        self.lay0Dim = lay0Dim
        self.lay1Dim = lay1Dim
        self.lay2Dim = lay2Dim
        self.outputDim = outputDim
        
        #learning parameters
        self.alpha = alpha #learning rate
        self.regLambda = regLambda #lambda regularization parameter
        
        #create weight matrix for layer 1 of size parameter dim x hidden layer 1 dim
        self.W1 = np.random.uniform(-np.sqrt(1/lay0Dim), np.sqrt(1/lay0Dim), (lay0Dim, lay1Dim))
        self.b1 = np.zeros((1, lay1Dim))
        
        #create weight matrix for layer 2 of size hidden layer 1 dim x hidden layer 2 dim
        self.W2 = np.random.uniform(-np.sqrt(1/lay0Dim), np.sqrt(1/lay0Dim), (lay1Dim,lay2Dim))
        self.b2 = np.zeros((1,lay2Dim))
        
        #create weight matrix for layer 3 of size layer 2 dim x output dim
        self.W3 = np.random.uniform(-np.sqrt(1/lay0Dim), np.sqrt(1/lay0Dim), (lay2Dim, outputDim))
        self.b3 = np.zeros((1,outputDim))

        
    #forward propogation algorithm takes a matrix "x" of any size x inputDim
    def ForProp(self, X):            
            #signal vector for hidden layer 1
            #tanh activation function
            S1 = X.dot(self.W1) + self.b1
            Z1 = np.tanh(S1)
            
            #signal vector for hidden layer 2
            #tanh activation function
            S2 = Z1.dot(self.W2)+self.b2
            Z2 = np.tanh(S2)
            
            #vector for the final output layer
            S3 = Z2.dot(self.W3)+ self.b3
            #softmax for output layer activation
            expScores = np.exp(S3)
            output = expScores/(np.sum(expScores, axis=1, keepdims=True))
            return output,Z1,Z2

    #cross-entropy error of an input set and a vector y of size N
    #slightly fucked up right now
    def CalculateError(self, X, y):
        #matrix of raw outputs from the tanh
        #ignore the underscords, the function returns more than we need        
        output,_,__ = self.ForProp(X)   
        
        #cross-entropy error
        #calculate total error against the vector y for the neurons where output = 1 (the rest are 0)
        totalError = 0
        for i in range(0,len(y)):
            totalError += -np.log(output[i, int(y[i])])

        #now account for regularizer
        #does bias get included in this?
        totalError+=self.regLambda/self.lay0Dim * (np.sum(np.square(self.W1))+np.sum(np.square(self.W2)) + np.sum(np.square(self.W3)))     
    
        error=totalError/len(y) #divide by 1/N
        return error
                
    def StochasticGD(self, numIterations, X, y):
        
        
        #storage for the best performing weights in-sample
        bestError = 1000000 #very poor practice. Note to self: fix this       
        bestW1 = np.zeros((self.lay0Dim, self.lay1Dim))
        bestb1 = np.zeros((1, self.lay1Dim))
        bestW2 = np.zeros((self.lay1Dim, self.lay2Dim))
        bestb2 = np.zeros((1,self.lay2Dim))
        bestW3 = np.zeros((self.lay2Dim, self.outputDim))
        bestb3 = np.zeros((1,self.outputDim))
                       
        for i in range (0,numIterations):
            #select random x_n to perform SGD
            n = np.random.randint(0,len(y))            
            x_n= X[n,:]
            #for whatever reason, the above line retrurns shape of (,4) instead of (1,4)
            #thus we have to reshape, it seems
            x_n = x_n.reshape((1,4))
            
            #forward propogation and get probabilities of each
            output,z1,z2 = self.ForProp(x_n)
            
            #TODO:
            #need to calculate error for every forward propogation to see if we gain in-sample error for every iteration

            #backprop
            #alert: some fuckaging with the dimensions here... pay attgention to sizes

            delta4 = output #size of 1 x outputDim
            delta4-=1 #softmax derivate is simply output-1            
            
            #tanh deriv is slightly is the familiar (1-(signal^2))
            dW3 = (z2.T).dot(delta4) #gradient of W2
            db3 = np.sum(delta4,axis=0,keepdims=True) #gradient of bias is simply sum
            delta3 = delta4.dot(self.W3.T)*(1-(z2**2)) #size of 1 x hidden layer 2 Dim
            
            dW2 = (z1.T).dot(delta3)
            db2 = np.sum(delta3, axis=0, keepdims=True)
            delta2 = delta3.dot(self.W2.T)*(1-(z1**2)) #size of 1 x hidden layer 1 DIm
            
            dW1 = (x_n.T).dot(delta2)
            db1 = np.sum(delta2, axis=0, keepdims=True)

            #add regularizer to gradients
            dW3 += self.regLambda*self.W3
            dW2 += self.regLambda*self.W2
            dW1 += self.regLambda*self.W1
            
            #adjust weights
            self.W3-= self.alpha * dW3
            self.W3-= self.alpha * db3
            self.W2-= self.alpha * dW2
            self.b2-= self.alpha * db2
            self.W1-= self.alpha * dW1
            self.b1-= self.alpha * db1
            
            
            #if error went down, log those weights as the best so far
            error = self.CalculateError(X, y)
            if(error<bestError):
                bestError = error
                bestW1 = self.W1
                bestb1 = self.b1
                bestW2 = self.W2
                bestb2 = self.b2
                bestW3 = self.W3
                bestb3 = self.b3
            
        #after loop take on the best values
        self.W1 = bestW1
        self.b1 = bestb1
        self.W2 = bestW2
        self.b2 = bestb2
        self.W3 = bestW3
        self.b3 = bestb3
        
        finalError = self.CalculateError(X, y)
        print("In Sample Error: " + str(finalError))