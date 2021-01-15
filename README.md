# Deep_Learning_Exercises
My projects for the course of Neural Networks and Deep Learning by prof. Alberto Testolin attended in 2019/2020 under Department of Information Engineering at University of Padua.

## FFNN_From_Scratch: Implementing a Neural Network in numpy
A Neural Network (NN) with two hidden layers and a framework (grid search) to search systematically for the best architecture and hyper-parameters (HPs) between some possible options defined with some a priori is implemented from scratch. Different activation functions, an early stopping condition, gradient clipping, L1 and L2 regularization and a k fold cross-validation are implemented. The model is trained in order to tackle a regression problem, i.e. the approximation of an unknown function. 

The pourpose of the project is to understand the complexity of building a deep learning framework, so that once one will start to use frameworks like TensorFlow or PyTorch will have more insights on their functioning.


## FFNN_Pytorch: Predicting MNIST characters with Feed-Forward Neural Network
A Neural Network (NN) in Pytorch is implemented to tackle classification problem on MNIST dataset. 
I then show how to use two consecutive random searches to find the best model and discuss its performance. Finally I present some techniques that can help in visualizing what each neuron is learning.

The notable aspects of this project are:
* The use of a random search with prior distributions based on some heuristics, e.g. for the number of neurons in the hidden layers the compression rate (the ratio between neurons in one layer and the previous one) that we want to hold in general.
* The visualization of the features encoded by neurons by receptive feld technique and gradient
ascent over the image.

## NLP_RNNs: Text generation with LSTM
A Pytorch a Recurrent Neural Network with Long Short Term Memory (LSTM) cells working at word-level is implemented. Then the network is trained on some books by Charles Dickens and finally generated some sample text providing different sequences as context seed.

## Autoencoders: Compressing MNIST dataset with autoencoders
A Pytorch Autoencoder, tuning its parameters through a random search and training it on the 
MNIST dataset, is implemented. Then the compression-performance trade-off is explored, varying the dimension of the encoded space of the 
autoencoder, and testing for different dimensions the performance on the test set in three different setups: using the test set
uncorrupted, adding to it gaussian noise or occluding part of the images. Then the same testing procedure is repeated on 
denoising autoencoders. Finally a method for generating images with the Autoencoder sampling from the encoded 
space is implemented analizing the smoothness and homogeneity properties of a bi-dimensional encoded space.

## Reinforcement_Learning: Reinforcement learning with tabular methods
Two different reinforcement learning (RL) algorithms from the tabular methods class, SARSA and Q-learning, are implemented for solving maze.
Actually this was very basic RL assignment that required less work than the previous ones present in my portfolio, so if you're interested I suggest you check it out at https://github.com/matteoguida/Quantum-Information-and-Computing .
