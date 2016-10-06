# IrisDataANN
A very much WIP artificial neural network for Iris Data. It doesn't work quite yet, but it should be good to go soon, there's just some debugging that needs to happen.

##Synopsys:
This code is a smaller project I'm working on as I continue to explore the world of machine learning. It uses the famous Iris data (the flower) from the UCI Machine Learning Database. It expands heavily on the logistic regression model that is in another repo on this account as well as some other side projects I've created along the way. It's takes advantage of vector-wise programming and calculations are done with matrix-wise operations to allow the ability to easily expand and contract the Artificial Neural Network as well as to make computation efficient.

It employs a 3-layer architecture, where the input layer is of variable size, as it's hidden layer, and its output layer. Though, the default parameters I have set are to accomodate the Iris Data Set. The parameters are as follows. The input layer is of dimension 4. The hidden layer is of dimension three. The ouput layer is also of dimension three.

The network uses stochastic gradient descent to train and a weight-decay regularizer. Error is measured using the cross-entropy function.
