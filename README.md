# Introduction to Deep Learning
This repository holds my solutions for the [Introduction to Deep Learning](https://dvl.in.tum.de/teaching/i2dl-ss19/) course of the summer semester 2019 held by [Prof. Dr. Laura Leal-Taixé](https://dvl.in.tum.de/team/lealtaixe/) and [Prof. Dr. Matthias Nießner](http://www.niessnerlab.org/members/matthias_niessner/profile.html).

The course was offered by the [Dynamic Vision and Learning Group](https://dvl.in.tum.de/) at [Technische Universität München (TUM)](https://www.tum.de/).

***

## Exercise 0
### Introduction
- Introduction to IPython and Jupyter notebooks
- Interaction with external python code
- Some random hands on examples

### Data Preparation
- Some tasks on data handling and preparation
- Data loading and visualization on the CIFAR-10 dataset
- Splitting into training, validation and test sets
- Mean image subtraction

***

## Exercise 1
### Softmax
- Implementation of a fully-vectorized loss function for the Softmax classifier
- Implementation of the fully-vectorized expression for its analytic gradient
- Check of the implementation with numerical gradient
- Usage of a validation set to tune the learning rate and regularization strength
- Optimization of the loss function with SGD
- Visualization of the final learned weights

Highest achieved score: 35.98% correct classified classes (Rank 322 / 428)

### Two layer net
- Implementation of a fully-connected two layer neural network to perform classification on the CIFAR-10 dataset
- Implementation of the forward and backward pass
- Training of the NN and hyperparameter training
- Visialization of the learned weights

Highest achieved score: 51.13% correct classified classes (Rank 216 / 393)

### Features
- Improvement of the Two layer net by using extracted image features instead of raw image data
- Feature extraction: Histogram of Oriented Gradients (HOG) and color histogram using the hue channel ins HSV color space
- Training of the NN on the extracted features

***

## Exercise 2
### Fully Connected Nets
- Implementation of a modular fully connected neural network
- Affine layer: forward and backward
- ReLU layer: forward and backward
- Sandwich layers: affine-relu
- Loss layers: Softmax
- Implementation of a solver class to run the training process decoupled from the network model
- Implementation of different update rules: SGD, SGD+Momentum, Adam
- Hyperparameter tuning and model training

Highest achieved score: 50.34% correct classified classes

### Batch Normalization
- Implementation of a batch normalization layer
- Training of a network with batch normalization
- Comparison of different weight initializations and the interaction with batchnorm

Highest achieved score: 53.89% correct classified classes (Rank 31 / 391)

### Dropout
- Implementaion of a dropout layer

### House Prices
- Implementation of a network to predict house prices
- Exploration of the House Price Data
- Dealing with missing data and non-numerical values
- Data normalization
- Training of a NN and regression to predict house prices

### House Prices - Data Analysis
- House Price Data exploration and visualization
- Comparison of different data axes 

***

## PyTorch Intro and CNN
### CNN Layers
- Implementation of a convolutional layer
- Implementation of a max pooling layer

### PyTorch Introduction
- Introduction to PyTorch

### Classification CNN
- Implementation of a convolutional neural network using PyTorch
- Network architecture: `conv - relu - max pool - fc - dropout - relu - fc`
- Implementation of a solver class to run the update steps on the model
- Training of the network
- Visualization of the learned filters and the loss and accuracy history

Highest achieved score: 68.00% correct classified classes
