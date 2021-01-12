"""
NEURAL NETWORKS AND DEEP LEARNING
PHYSICS OF DATA - Department of Physics and Astronomy
A.A. 2019/20 (6 CFU)
Dr. Alberto Testolin, Dr. Federico Chiariotti


Author: Matteo Guida
Date of creation : 25/10/2020


Description:

Recurrent neural network in PyTorch for natural language processing on Charles Dickens books.

- The program loads and runs the trained model in order to generate lenght number of words starting from initial seed.

Usage example: python trained_model.py --seed "Procrastination is the thief of time. Collar him." --lenght 40

"""


import torch 
import numpy as np 
import network 

import argparse

########################
## INPUTS ##########
########################

parser = argparse.ArgumentParser(prog = '\nGenerate lenght number of words starting from initial seed.\n',
                            description = 'The program loads and runs the trained model in order to carry out the task.')

# The Chimes quote. 
parser.add_argument('--seed', type=str, nargs='?', default='So may the New Year be a happy one to you, happy to many more whose happiness depends on you.', help='Initial seed.')
parser.add_argument('--lenght', type=int, nargs='?', default=50, help='Number of words generated.')


args = parser.parse_args()
loaded_net = torch.load('./model/best_network.dat')
np.random.seed(1)
print("\n")
loaded_net.generate_words(input_seed = args.seed, n_words=args.lenght)
print("\n")