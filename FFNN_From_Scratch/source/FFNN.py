"""
NEURAL NETWORKS AND DEEP LEARNING
PHYSICS OF DATA - Department of Physics and Astronomy
A.A. 2019/20 (6 CFU)
Dr. Alberto Testolin, Dr. Federico Chiariotti


Author: Matteo Guida
Date of creation : 30/09/2020


Description:

A 2D regression problem is addressed with a feed-forward neural network implemented from scratch. 
The following feature are added in order to get good results:
    1) Cross-validation to make the training more robust.
    2) Different activation functions to compute the hidden layer values, (ReLU, Leaky ReLU, ELU, Swish).
    3) Regularization L1 and L2 of network weights to tackle overfitting.
    4) Validation-based early stopping to increase generalization performance.

"""
#%%
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import pickle
import itertools
from copy import deepcopy
from random import shuffle

class Network():
    
    def __init__(self, params_dict):
        
        # The Class is initialized to the first configuration from the ones possible with the dict combinations. 
        self.neurons_architecture = np.asarray(params_dict["neurons_architecture"][0,:]).flatten()

        ### WEIGHT INITIALIZATION (Xavier)
        # Initialize hidden weights and biases (layer 1)
        self.Ni, self.Nh1, self.Nh2, self.No = self.neurons_architecture
        
        self.Wh1 = (np.random.rand(self.Nh1, self.Ni) - 0.5) * np.sqrt(12 / (self.Nh1 + self.Ni))
        self.Bh1 = np.zeros([self.Nh1, 1])
        self.WBh1 = np.concatenate([self.Wh1, self.Bh1], 1) # Weight matrix including biases
        self.reset_WBh1 = deepcopy(self.WBh1)

        # Initialize hidden weights and biases (layer 2)
        self.Wh2 = (np.random.rand(self.Nh2, self.Nh1) - 0.5) * np.sqrt(12 / (self.Nh2 + self.Nh1))
        self.Bh2 = np.zeros([self.Nh2, 1])
        self.WBh2 = np.concatenate([self.Wh2, self.Bh2], 1) # Weight matrix including biases
        self.reset_WBh2 = deepcopy(self.WBh2)

        # Initialize output weights and biases
        self.Wo = (np.random.rand(self.No, self.Nh2) - 0.5) * np.sqrt(12 / (self.No + self.Nh2))

        self.Bo = np.zeros([self.No, 1])
        self.WBo = np.concatenate([self.Wo, self.Bo], 1) # Weight matrix including biases
        self.reset_WBo = deepcopy(self.WBo)
        

        self.lr       = params_dict["lr"][0]
        self.en_decay = params_dict["en_decay"][0]
        self.lr_final = params_dict["lr_final"][0]
        self.lr_decay = params_dict["lr_decay"][0]
        self.num_epochs = int(params_dict["n_epochs"][0])
        self.regularization = params_dict["regularization"][0]
        self.penalty = params_dict["penalty"][0]
        self.grad_clipping = params_dict["grad_clipping"][0]
        self.act = eval("self."+params_dict["activation"][0])
        self.act_der = eval("self.grad_"+ params_dict["activation"][0])
        self.clip_norm = params_dict["clip_norm"][0]  


    def set_network(self):
        """
        Description:
        
        The function reset the weights to random values in case the neurons_architecture is the same of the one when a Network 
        object is instantiated, otherwise do the same operation according with the new neurons_architecture. 
        
        """
        np.random.seed(4)
        self.Ni,self.Nh1, self.Nh2, self.No = np.asarray(self.neurons_architecture).flatten()
        self.Wh1 = (np.random.rand(self.Nh1, self.Ni) - 0.5) * np.sqrt(12 / (self.Nh1 + self.Ni))

        self.Bh1 = np.zeros([self.Nh1, 1])
        self.WBh1 = np.concatenate([self.Wh1, self.Bh1], 1) 
        self.Wh2 = (np.random.rand(self.Nh2, self.Nh1) - 0.5) * np.sqrt(12 / (self.Nh2 + self.Nh1))
        
        self.Bh2 = np.zeros([self.Nh2, 1])
        self.WBh2 = np.concatenate([self.Wh2, self.Bh2], 1) 

        
        self.Wo = (np.random.rand(self.No, self.Nh2) - 0.5) * np.sqrt(12 / (self.No +self.Nh2))
        
        self.Bo = np.zeros([self.No, 1])
        self.WBo = np.concatenate([self.Wo, self.Bo], 1)
        return None

    def forward(self, x, additional_out=False):
        """
        Description:
        
        Given the input point x the function return the result obtained from the forward propagation in the feed-forward neural network. 
        
        """
        # Convert to numpy array
        x = np.array(x)
        
        ### Hidden layer 1
        # Add bias term
        X = np.append(x, 1)

        # Forward pass (linear)
        H1 = np.matmul(self.WBh1, X)
        # Activation function
        Z1 = self.act(H1)
        
        ### Hidden layer 2
        # Add bias term
        Z1 = np.append(Z1, 1)
        # Forward pass (linear)
        H2 = np.matmul(self.WBh2, Z1)
        
        # Activation function
        Z2 = self.act(H2)
        
        ### Output layer
        # Add bias term
        Z2 = np.append(Z2, 1)
        # Forward pass (linear)
        Y = np.matmul(self.WBo, Z2)
        # NO activation function
        
        if additional_out:
            return Y.squeeze(), Z2
        
        # squeeze remove single-dimensional entries from the shape of an array.
        return Y.squeeze()
        

    def update(self, x, label):
        """
        Description:
        
        The function implements the update rule based on backpropagation algorithm procedure in the vanilla version or 
        adding a L1 or L2 regularization term based on the params_dict prescriptions. 
        Moreover a gradient clipping is added in order to avoid very large updates for the weights, which lead them to diverge. 
        
        """
        # Convert to numpy array
        X = np.array(x)
        
        ### Hidden layer 1
        # Add bias term
        X = np.append(X, 1)
        # Forward pass (linear)
        H1 = np.matmul(self.WBh1, X)
        # Activation function
        Z1 = self.act(H1)

        ### Hidden layer 2
        # Add bias term
        Z1 = np.append(Z1, 1)
        # Forward pass (linear)
        H2 = np.matmul(self.WBh2, Z1)
        # Activation function
        Z2 = self.act(H2)
        
        ### Output layer
        # Add bias term
        Z2 = np.append(Z2, 1)
        # Forward pass (linear)
        Y = np.matmul(self.WBo, Z2)
        # NO activation function

        ### BACKPROPAGATION

        # Evaluate the derivative terms
        D1 = Y - label    
        D2 = Z2

        D3 = self.WBo[:,:-1]
        
        D4 = self.act_der(H2)
        D5 = Z1

        D6 = self.WBh2[:,:-1]
        D7 = self.act_der(H1)
        D8 = X
        
        # Layer Error
        Eo = D1
        Eh2 = np.matmul(Eo, D3) * D4
        Eh1 = np.matmul(Eh2, D6) * D7
        
        
        # Derivative for weight matrices
                
        dWBo = np.matmul(Eo.reshape(-1,1), D2.reshape(1,-1))
        
        dWBh2 = np.matmul(Eh2.reshape(-1,1), D5.reshape(1,-1))
        
        dWBh1 = np.matmul(Eh1.reshape(-1,1), D8.reshape(1,-1))

        ##########################
        ## GRADIENT CLIPPING  
        ##########################
        
        if self.grad_clipping is True:
            # The norm of the gradient w.r.t. each layer is computed. 
            grad_norm_dWBo  = np.linalg.norm(dWBo.flatten())
            grad_norm_dWBh2 = np.linalg.norm(dWBh2.flatten())
            grad_norm_dWBh1 = np.linalg.norm(dWBh1.flatten())
            # if the norm is grater than the threshold clip_norm the gradient is rescaled so that its value does not 
            # fall outside [-clip_norm,clip_norm]
            if grad_norm_dWBo > self.clip_norm :
                dWBo = dWBo/grad_norm_dWBo*self.clip_norm 
            if grad_norm_dWBh2 > self.clip_norm :
                dWBh2 = dWBh2/grad_norm_dWBh2*self.clip_norm 
            if grad_norm_dWBh1 > self.clip_norm :
                dWBh1 = dWBh1/grad_norm_dWBh1*self.clip_norm 

        ##########################
        ## L1 REGULARIZATION 
        ##########################
        if self.regularization == 'L1':
            
            # Compute penalty 
            WBh1_sgn = np.ones(self.WBh1.shape)
            WBh1_sgn[self.WBh1<0] = -1

            WBh2_sgn = np.ones(self.WBh2.shape)
            WBh2_sgn[self.WBh2<0] = -1

            WBo_sgn = np.ones(self.WBo.shape)
            WBo_sgn[self.WBo<0] = -1
        
            # Update the weights values 
            self.WBh1 -= self.lr * (dWBh1 + self.penalty * WBh1_sgn)
            self.WBh2 -= self.lr * (dWBh2 + self.penalty * WBh2_sgn)
            self.WBo -= self.lr * (dWBo + self.penalty * WBo_sgn)
            
            # Evaluate L1 regularized loss function
            loss = (Y - label)**2/2 + self.penalty * (np.abs(self.WBh1).sum() + np.abs(self.WBh2).sum() + np.abs(self.WBo).sum())
            
        ##########################
        ## L2 REGULARIZATION 
        ##########################
        elif self.regularization == 'L2':
        
            # Update the weights
            self.WBh1 -= self.lr * (dWBh1 + self.penalty * self.WBh1)
            self.WBh2 -= self.lr * (dWBh2 + self.penalty * self.WBh2)
            self.WBo -= self.lr * (dWBo + self.penalty * self.WBo)
            
            # Evaluate L2 regularized loss function
            loss = (Y - label)**2/2 + self.penalty *( (self.WBh1**2).sum() + (self.WBh2**2).sum() + (self.WBo**2).sum() )
        
        else: # vanilla version without regularization. 
            
            # Update the weights
            self.WBh1 -= self.lr * dWBh1
            self.WBh2 -= self.lr * dWBh2
            self.WBo -= self.lr * dWBo
            
            # Evaluate loss function
            loss = (Y - label)**2/2     
        
        return loss

    ##########################
    ## ACTIVATION FUNCTIONS 
    ##########################

    def relu(self,x):
        """ Evaluate rectified linear unit activation function on the elements
        of input x."""
        return np.maximum(0,x)

    def grad_relu(self,x):
        """
        Evaluate the first derivative of  ReLu function on the elements
        of input x"""
        return np.greater(x, 0).astype(int)

    def elu(self,x,alpha=0.9):
        """ Evaluate exponential linear unit activation function on the elements
        of input x."""
        return np.where(x > 0, x, alpha * (np.exp(x) - 1))

    def grad_elu(self,x,alpha=0.9):
        """
        Evaluate the first derivative of exponential linear unit  function on the elements
        of input x"""
        return np.where(x > 0, np.ones_like(x), alpha * np.exp(x))

    def leaky_relu(self,x, alpha=0.01):
        """ Evaluate leaky rectified linear unit activation function on the elements
        of input x."""
        return np.where(x > 0, x, x * alpha)  

    def grad_leaky_relu(self,x, alpha=0.01):
        """
        Evaluate the first derivative of leaky rectified linear unit function on the elements
        of input x"""
        dx = np.ones_like(x)
        dx[x < 0] = alpha
        return dx  

    def swish(self,x):
        """ Evaluate swish activation function on the elements
        of input x."""
        return x/(1-np.exp(-x))

    def grad_swish(self,x):
        """
        Evaluate the first derivative of swish function on the elements
        of input x"""
        return self.swish(x) + 1/(1+np.exp(-x))*(1-self.swish(x))


    def save_model(self):
        np.save("WBh1", self.WBh1)
        np.save("WBh2", self.WBh2)
        np.save("WBo", self.WBo)

    def load_model(self, WBh1, WBh2, WBo):
        self.WBh1 = WBh1
        self.WBh2 = WBh2
        self.WBo = WBo

    def train(self, x_train, y_train, x_val, y_val, params_dict, early_stopping=True ,return_log=False, out_log=False,save=False):
        """
        Description:
        
        The function implements the training of the FFNN given a num_epochs with the possibility to add a learning rate
        decay and an early stopping. We call evaluation because this function will be used iteratively in K-fold cross validation, but
        can be also used to evaluate the loss on test set. 
        
        """
        # Lists to store train/validation loss at each epoch 
        if out_log or return_log or early_stopping: 
            train_loss_log = []
            val_loss_log = []
        
        self.set_network()

        if self.en_decay:
            self.lr_decay = (self.lr_final / self.lr)**(1 / self.num_epochs)
        
        
        for num_ep in range(self.num_epochs):
            
            # Learning rate decay during each epoch
            if self.en_decay:
                self.lr *= self.lr_decay
                
            # Train single epoch (sample by sample, no batch for now)
            train_loss_vec = [self.update(x, y) for x, y in zip(x_train, y_train)]
            avg_train_loss = np.mean(train_loss_vec)
            
            # Validation of network
            y_val_est = np.array([self.forward(x) for x in x_val])

            avg_val_loss = np.mean((y_val_est - y_val)**2/2)
            
            # Store results in the defined lists
            if return_log or out_log or early_stopping:
                train_loss_log.append(avg_train_loss)
                val_loss_log.append(avg_val_loss)
                if out_log:
                    print('Epoch %d - Train loss: %.5f - Val loss: %.5f' % (num_ep + 1, avg_train_loss, avg_val_loss))
            
            ##########################
            ## EARLY STOPPING  
            ##########################

            # The training cannot stop before the first 400 epochs. After that if in the last 200 epochs an improvement of the validation loss
            # of 5% does not occur w.r.t. the 200 epochs before the training is stopped.  
            if early_stopping and num_ep > 400:
                if (np.mean(val_loss_log[-200:]) - np.mean(val_loss_log[-400:-200]))/np.mean(val_loss_log[-400:-200])*100> -5:
                    # print("Early stopped at epoch:",num_ep )
                    break 
                else:
                    continue
            
        # The function can return the last train/validation loss or it can return also their values for each epoch. 
        if save is True: 
                self.save_model()
        if return_log is True:
            return train_loss_log, val_loss_log, avg_train_loss, avg_val_loss
        else:
            return avg_train_loss, avg_val_loss
        
    def Kfold_cross_val(self, x_data, label, params_dict,k_fold=6):
        """
        Description:
        
        The function implements K-fold cross validation to make the training procedure more robust with a view to it on unseen data. 
        Not strict procedures are used to fix k, but a reasonable default value is selected. 
        
        """
        block = int(len(x_data)/k_fold)
        training_error = []
        val_error = []
        randomize = np.arange(len(x_data))
        shuffle(randomize)
        x_data = x_data[randomize]
        label = label[randomize]
        for i in range(k_fold):
            # Re-initialize network to reset its weights, or add new ones according with new architecture. 
            x_validation,y_validation = x_data[i*block:(i+1)*block],label[i*block:(i+1)*block]
            x_train_kfold,y_train_kfold = np.delete(x_data,range(i*block,(i+1)*block),axis=0),np.delete(label,range(i*block,(i+1)*block))
            # Train the model with the union of k-1 groups and get the validation error with the validation set.
            final_train_loss, final_val_loss = self.train(x_train=x_train_kfold,y_train=y_train_kfold, 
            x_val=x_validation,y_val= y_validation,params_dict=params_dict)
            # Saved validation error for each iteration of the cycle. 
            val_error.append(final_val_loss) 
            training_error.append(final_train_loss) 

        avg_val_error = np.asarray(val_error).mean() 
        std_val_error = np.asarray(val_error).std() 

        print("______________________________________________________")
        print("AVERAGE VALIDATION ERROR",round(avg_val_error,3), "±", round(std_val_error,3))
        print("______________________________________________________")
    
        return np.asarray(training_error),avg_val_error,std_val_error, 



    def grid_search(self, params_dict,x_data, label):
        """
        Description:
        
        The function implements a grid-search for all the combination of the hyperparameters present in the dictionary.
        For each possible combination the k-fold cross validation procedure is performed. 
        
        """
        # Compute all combinations of dict keys and values
        combinations = [dict(zip(params_dict.keys(), a)) for a in itertools.product(*params_dict.values())]
        list_avg_val_losses = []
        list_std_val_losses = []
        # Perform Cross Validation for each combination of parameters
        for combination in trange(len(combinations)):
            print("\nPerforming training", combination, "out of", len(combinations))
            print(combinations[combination]["neurons_architecture"])
            self.neurons_architecture = combinations[combination]["neurons_architecture"]
            self.set_network()
            self.act = eval("self."+combinations[combination]["activation"])
            self.act_der = eval("self.grad_"+ combinations[combination]["activation"])
            self.num_epochs = int(combinations[combination]["n_epochs"])
            self.lr = combinations[combination]["lr"]
            self.en_decay = combinations[combination]["en_decay"]
            self.lr_final = combinations[combination]["lr_final"]
            self.lr_decay = combinations[combination]["lr_decay"]
            self.regularization = combinations[combination]["regularization"]
            self.penalty = combinations[combination]["penalty"]
            self.grad_clipping = combinations[combination]["grad_clipping"]
            self.clip_norm = combinations[combination]["clip_norm"]

            print("\nConfiguration:",combinations[combination])

            _,avg_valid_loss,std_valid_loss = self.Kfold_cross_val(x_data=x_data, label=label,params_dict=params_dict)
            list_avg_val_losses.append(avg_valid_loss)
            list_std_val_losses.append(std_valid_loss)
        return combinations, list_avg_val_losses, list_std_val_losses     
        

# %%
