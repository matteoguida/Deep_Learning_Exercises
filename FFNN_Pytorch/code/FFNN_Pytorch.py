"""
NEURAL NETWORKS AND DEEP LEARNING
PHYSICS OF DATA - Department of Physics and Astronomy
A.A. 2019/20 (6 CFU)
Dr. Alberto Testolin, Dr. Federico Chiariotti


Author: Matteo Guida
Date of creation : 30/09/2020


Description:

The multiclassification problem of the MNIST dataset is addressed with a feed-forward neural network implemented from scratch. 
The following feature are added in order to get good results:
    1) Cross-validation to make the training more robust.
    2) Random search for hyperparameter optimization. 
    3) Regularization L2 of network weights and dropout to reduce the overfitting
    4) Validation-based early stopping to increase generalization performance.


"""
#%%
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import itertools
from copy import deepcopy
from random import shuffle
import torch
import torch.nn as nn
import torch.optim as optim
class NN_model(nn.Module):
    
    def __init__(self, params_dict):
        super().__init__()
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.neurons_architecture = np.asarray(params_dict["neurons_architecture"][0,:]).flatten()
        self.Ni, self.Nh1, self.Nh2, self.No = self.neurons_architecture
        self.fc1 = nn.Linear(in_features=self.Ni, out_features=self.Nh1)
        self.fc2 = nn.Linear(self.Nh1, self.Nh2)
        self.fc3 = nn.Linear(self.Nh2, self.No)
        self.drop = nn.Dropout(params_dict["dropout"][0])
        self.act  = params_dict["activation"][0]
    
    def set_network(self,params_dict):
        """
        Description:

        The function reset the weights to random values in case the neurons_architecture is the same of the one instantiated with the Network 
        object, otherwise do the same operation according with the new neurons_architecture. 

        """
        np.random.seed(4)
        # Linear implies the standard operation: y = Wx + b
        self.fc1 = nn.Linear(in_features=self.Ni, out_features=self.Nh1).to(self.device)
        self.fc2 = nn.Linear(self.Nh1, self.Nh2).to(self.device)
        self.fc3 = nn.Linear(self.Nh2, self.No).to(self.device)
        self.drop = nn.Dropout(params_dict["dropout"][0]).to(self.device)
        self.act  = params_dict["activation"][0].to(self.device)
        
    
    def forward(self, x, additional_out=False):
        
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.act(self.fc2(x)))
        out = self.fc3(x)    
        if additional_out:
            return out, x
        
        return out
    
    def accuracy(self, x_test,y_test):
        with torch.no_grad():
            self.eval() 
            outputs = self.forward(x_test.to(self.device)).to(self.device)
            _, predicted = torch.max(outputs.data, 1)
            predicted = predicted.cpu().numpy()
            y_test = y_test.cpu().numpy()
            total = len(y_test)
            correct = (predicted == y_test).sum()
        accuracy = 100 * correct / total
        return accuracy
    
    def trainin(self,data_train, label_train, data_val, label_val, params_dict,early_stopping=True , return_log=False,verbose=False,save_model=False):
        
        # Re-initialize network to reset its weights, or add new ones according with new architecture. 
        self.set_network(params_dict=params_dict)
        
        self.num_epochs  = int(params_dict["n_epochs"][0])
        loss_fn = nn.CrossEntropyLoss()
        ### Define an optimizer
        lr = params_dict["lr"][0]
        optimizer = eval(params_dict["optimizer"][0])(self.parameters(), lr=lr, weight_decay=params_dict["penalty"][0])
        train_loss_log = []
        val_loss_log = []
        val_accuracy_log = []

        for num_epoch in range(self.num_epochs):

            # Training
            self.train() 
            # Eventually clear previous recorded gradients.
            optimizer.zero_grad()
            net_output = self(data_train.to(self.device)).to(self.device)
            # Evaluate loss
            loss = loss_fn(net_output.to(self.device), label_train.to(self.device))
            # Backward pass
            loss.backward()
            # Update
            optimizer.step()
            
            # Validation
            self.eval() # Evaluation mode.
            with torch.no_grad(): # No need to track the gradients
                net_output = self(data_val.to(self.device))
                # Evaluate global loss
                val_loss = loss_fn(net_output.to(self.device), label_val.to(self.device))
                val_accuracy=self.accuracy(x_test=data_train,y_test=label_train)
            
            train_loss = float(loss.data)
            validat_loss = float(val_loss.data)

            val_accuracy_log.append(val_accuracy)
            train_loss_log.append(train_loss)
            val_loss_log.append(validat_loss)
            if verbose == True:
                print('\n\nEpoch', num_epoch + 1)
                print("Training loss :", round(train_loss,3))
                print("Validation loss :", round(validat_loss,3))
                print("Validation accuracy", round(val_accuracy,3))

            ##########################
            ## EARLY STOPPING  
            ##########################

            # The training cannot stop before the first 200 epochs. After that if 
            # 1) in the last 200 epochs the mean of loss function does not
            # present an improvement of the validation loss of 5% w.r.t. mean of the values of the loss function in the 200 epochs before the training is stopped.  
            # OR 
            # 2) if the minimum accuracy in the last two epochs is worse of 1% w.r.t. the mean of the accuracy in the 10 epochs before
            # THEN
            # stop the training. 
            if early_stopping and num_epoch > 200:

                if (np.mean(val_loss_log[-200:]) - np.mean(val_loss_log[-400:-200]))/np.mean(val_loss_log[-400:-200])*100> -5 or \
                  np.min(val_accuracy_log[-2:]) - np.mean(val_accuracy_log[-11:-2]) < - 1:
                    break 
                else:
                    continue

        if save_model is True:
            torch.save(self, 'best_network.dat')

        if return_log is True:
            return train_loss_log,val_loss_log,val_accuracy_log,train_loss,val_loss,train_loss, validat_loss
        else:
            return train_loss, validat_loss,val_accuracy

    def Kfold_cross_val(self, params_dict, x_data, label,k_fold=5):
        """
        Description:
        
        The function implements K-fold cross validation to make the training procedure more robust with a view to it on unseen data. 
        Not strict procedures are used to fix k, but a reasonable default value is selected. 
        
        """
        block = int(len(x_data)/k_fold)
        training_error_list = []
        val_error_list = []
        val_accuracy_list = []
        randomize = np.arange(len(x_data))
        shuffle(randomize)
        x_data = x_data[randomize]
        label = label[randomize]
        for i in range(k_fold):
            x_validation,y_validation = x_data[i*block:(i+1)*block],np.asarray(label[i*block:(i+1)*block],dtype=int)
            x_train_kfold,y_train_kfold = np.delete(x_data,range(i*block,(i+1)*block),axis=0),np.asarray(np.delete(label,range(i*block,(i+1)*block),axis=0),dtype=int)
            # Train the model with the union of k-1 groups and get the validation error with the validation set.
            final_train_loss, final_val_loss,val_accuracy = self.trainin(data_train=torch.Tensor(x_train_kfold),label_train=torch.LongTensor(y_train_kfold).squeeze(),
             data_val=torch.Tensor(x_validation),label_val= torch.LongTensor(y_validation).squeeze(),params_dict=params_dict, return_log=False,verbose=False)
            # Saved validation error,validation accuracy and the training error for each iteration of the cycle. 
            val_error_list.append(final_val_loss) 
            val_accuracy_list.append(val_accuracy) 
            training_error_list.append(final_train_loss) 
        
        avg_val_error = np.asarray(val_error_list).mean() 
        std_val_error = np.asarray(val_error_list).std() 
        avg_val_accuracy = np.asarray(val_accuracy_list).mean() 
        std_val_accuracy = np.asarray(val_accuracy_list).std() 
        
        print("______________________________________________________")
        print("AVERAGE VALIDATION ERROR",round(avg_val_error,3), "±", round(std_val_error,3))
        print("AVERAGE VALIDATION ACCURACY",round(avg_val_accuracy,3), "±", round(std_val_accuracy,3))
        print("______________________________________________________")
    
        return np.asarray(training_error_list),avg_val_error,std_val_error,avg_val_accuracy,std_val_accuracy
    
def get_list_rand_search(params_dict):
    """
    Description:
    
    The function allows to implement the extraction of the hyperparameters according with the random search.
    
    """
    input_layer = 784
    output_layer = 10
    activation =  params_dict["activation"]
    inf_lim_exp_lr = params_dict["lr"][0]
    sup_lim_exp_lr = params_dict["lr"][1]
    inf_lim_drop = params_dict["dropout"][0]
    sup_lim_drop = params_dict["dropout"][1]
    inf_lim_exp_penalty = params_dict["penalty"][0]
    sup_lim_exp_penalty = params_dict["penalty"][1]
    optimizer = params_dict["optimizer"]
    ncombinations = params_dict["ncombinations"]
    n_epochs = params_dict["n_epochs"]
    
    check = False
    while check is False:
        n_neurons_h1 = np.rint(np.random.normal(loc=(output_layer/input_layer)**(1/3)*input_layer, 
                                            scale=np.sqrt(output_layer/input_layer)*input_layer, size=ncombinations))
        n_neurons_h2 = np.rint(np.random.normal(loc=(output_layer/input_layer)**(2/3)*input_layer, 
                                            scale=output_layer/input_layer*input_layer, size=ncombinations))
        if (np.all(n_neurons_h1>0) and np.all(n_neurons_h2>0)):
            check=True 
    activation = np.random.choice(activation, ncombinations)
    optimizer = np.random.choice(optimizer, ncombinations)
    dropout = np.random.uniform(inf_lim_drop, sup_lim_drop, ncombinations)
    lr_vals = 10**(np.random.uniform(inf_lim_exp_lr, sup_lim_exp_lr, ncombinations))
    L2_vals = 10**(np.random.uniform(inf_lim_exp_penalty, sup_lim_exp_penalty, ncombinations))   
    n_epochs = [n_epochs[0] for _ in range(ncombinations[0])]

    combinations = [ [n_neurons_h1[i], n_neurons_h2[i], activation[i], optimizer[i], 
                    dropout[i], lr_vals[i], L2_vals[i],n_epochs[i]] for i in range(ncombinations[0])]
    
    return combinations

# if __name__ == "__main__":

#     import numpy as np
#     import scipy.io as sio
#     import torch.optim as optim
#     import torch
#     import torch.nn as nn
#     import torch

#     data = sio.loadmat('MNIST.mat')
#     numbers = data['output_labels']
#     images = data['input_images']

#     randomize = np.arange(numbers.shape[0])
#     np.random.shuffle(randomize)
#     images = images[randomize]
#     numbers = numbers[randomize]

#     params_dict = {"neurons_architecture":np.matrix([249, 200, 100, 10]),
#                 "activation":  [nn.LeakyReLU()],
#                 "n_epochs":  [int(2e3)],
#                     "penalty" : [1e-3],
#                 "optimizer" : ['torch.optim.Adam'],
#                 "lr" : [3e-3],
#                 "dropout" : [0.3]
#                     }
#     print("params_dict",params_dict)
#     # The division bewween test-train is 1:9.
#     TRAIN_PERCENTAGE = 0.90

#     X_train = torch.Tensor(images[ : int(TRAIN_PERCENTAGE*len(X_train))])
#     X_test  = torch.Tensor(images[int(TRAIN_PERCENTAGE*len(images)) : ])

#     y_train = torch.LongTensor(numbers[ : int(TRAIN_PERCENTAGE*len(y_train))]).squeeze()
#     y_test  = torch.LongTensor(numbers[int(TRAIN_PERCENTAGE*len(numbers)) : ]).squeeze()

#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     network1 = NN_model(params_dict).to(device)

#     train_loss_log,test_loss_log,val_accuracy_log = network1.trainin(data_train=X_train, label_train=y_train,
#                                         data_val=X_test,label_val=y_test,params_dict=params_dict)

#     fig, ax = plt.subplots(1,2, figsize=(20,6))
#     ax[0].plot(train_loss_log, label='Train Loss')
#     ax[0].plot(test_loss_log, label='Test Loss')
#     ax[0].set_ylabel("Loss",fontsize=20)
#     ax[0].set_xlabel("Epoch",fontsize=20)
#     ax[0].set_title('Learning Curve',fontsize=20)
#     ax[0].tick_params(labelsize=14)
#     ax[0].legend(fontsize=14)
#     ax[0].grid(alpha=0.2)

#     ax[1].plot(train_loss_log, label='Train Loss')
#     ax[1].plot(test_loss_log, label='Test Loss')
#     ax[1].set_ylabel("Loss",fontsize=20)
#     ax[1].set_xlabel("Epoch",fontsize=20)
#     ax[1].set_title('Learning Curve',fontsize=20)
#     ax[1].tick_params(labelsize=14)
#     ax[1].legend(fontsize=14)
#     ax[1].set_yscale('log')
#     ax[1].grid(alpha=0.2)

