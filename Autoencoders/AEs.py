"""
NEURAL NETWORKS AND DEEP LEARNING

ICT FOR LIFE AND HEALTH - Department of Information Engineering
PHYSICS OF DATA - Department of Physics and Astronomy
COGNITIVE NEUROSCIENCE AND CLINICAL NEUROPSYCHOLOGY - Department of Psychology

A.A. 2019/20 (6 CFU)
Dr. Alberto Testolin, Dr. Federico Chiariotti

Author: Matteo Guida

Assigment 4: Images reconstruction with an autoencoder (AE), denoising autoencoder (DAE) and generative capabilities exploration sampling from latent space.

    - The class Autoencoder is implemented where the model is initialized and trained by the functions train_epoch, test_epoch and trainin. The functions for the DAE are implemented in a very similar way but the model
      is feed with corrupted data and tested with uncorrupted ones. 
    - The function get_list_rand_search allows to implement the extraction of the hyperparameters according with the random search.
    - With the function encode_rep, generate_from_encoded_sample and images_smooth_sampling the sampling from latent space is performed.

 
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import random
from copy import deepcopy

from torch import nn
from tqdm import trange, tqdm


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Autoencoder(nn.Module):
    
    def __init__(self,encoded_space_dim,params_dict):
        super().__init__()
        
        self.dropout = params_dict["dropout"][0]
        ### Encoder
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Dropout(self.dropout),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Dropout(self.dropout),
            nn.Conv2d(16, 32, 3, stride=2, padding=0),
            nn.ReLU(True),
            nn.Dropout(self.dropout)
        )
        self.encoder_lin = nn.Sequential(
            nn.Linear(3 * 3 * 32, 64),
            nn.ReLU(True),
            nn.Dropout(self.dropout),
            nn.Linear(64, encoded_space_dim)
        )
        
        ### Decoder
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 64),
            nn.ReLU(True),
            nn.Dropout(self.dropout),
            nn.Linear(64, 3 * 3 * 32),
            nn.ReLU(True),
            nn.Dropout(self.dropout)
        )
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, output_padding=0),
            nn.ReLU(True),
            nn.Dropout(self.dropout),
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.Dropout(self.dropout),
            nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1)
        )
        self.loss_fn = torch.nn.MSELoss()

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

    def encode(self, x):
        # Apply convolutions
        x = self.encoder_cnn(x)
        # Flatten
        x = x.view([x.size(0), -1])
        # Apply linear layers
        x = self.encoder_lin(x)
        return x
    
    def decode(self, x):
        # Apply linear layers
        x = self.decoder_lin(x)
        # Reshape
        x = x.view([-1, 32, 3, 3])
        # Apply transposed convolutions
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x

    # Network training

    ### Training function
    def train_epoch(self, dataloader, optimizer):
        # Training
        self.train()
        loss_list = []
        for sample_batch in dataloader:
            # Extract data and move tensors to the selected device
            image_batch = sample_batch[0].to(device)
            # Forward pass
            output = self(image_batch)
            loss = self.loss_fn(output, image_batch)
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.data.cpu().numpy())
        loss_list = np.asarray(loss_list)
        return np.mean(loss_list)
    
    def dae_train_epoch(self, dataloader_orig,dataloader_noised, optimizer):
        # Training
        self.train()
        loss_list = []
        for sample_batch_orig,sample_batch_noised in zip(dataloader_orig,dataloader_noised):

            # Extract data and move tensors to the selected device
            image_batch_orig = sample_batch_orig[0].to(device)
            image_batch_noised = sample_batch_noised[0].to(device)

            # Forward pass
            output = self(image_batch_noised)
            loss = self.loss_fn(output, image_batch_orig)
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.data.cpu().numpy())

        loss_list = np.asarray(loss_list)
        return np.mean(loss_list)

    # Network validation/testing

    def test_epoch(self, test_dataloader):
        self.eval() 
        loss_list = []
        with torch.no_grad(): 
            conc_out = torch.Tensor().float()
            conc_label = torch.Tensor().float()
            for sample_batch in test_dataloader:
                image_batch = sample_batch[0].to(device)
                out = self(image_batch)
                conc_out = torch.cat([conc_out, out.cpu()])
                conc_label = torch.cat([conc_label, image_batch.cpu()]) 
            val_loss = self.loss_fn(conc_out, conc_label)

        return torch.mean(val_loss.data)

    
    def dae_test_epoch(self,  dataloader_orig,dataloader_noised):
        
        self.eval() 
        loss_list = []
        with torch.no_grad(): 
            conc_out = torch.Tensor().float()
            conc_label = torch.Tensor().float()
            for sample_batch_orig,sample_batch_noised in zip(dataloader_orig,dataloader_noised):
                image_batch_orig = sample_batch_orig[0].to(device)
                image_batch_noised = sample_batch_noised[0].to(device)
                out = self(image_batch_noised)
                conc_out = torch.cat([conc_out, out.cpu()])
                conc_label = torch.cat([conc_label, image_batch_orig.cpu()]) 
            val_loss = self.loss_fn(conc_out, conc_label)

        return torch.mean(val_loss.data)
    
    
    def trainin(self,train_dataloader,params_dict,test_dataloader,verbose = False,return_log=True):
        train_loss_log = []
        val_loss_log = []
        optimizer = eval(params_dict["optimizer"][0])(self.parameters(), lr=params_dict["lr"][0], weight_decay=params_dict["penalty"][0])
        for epoch in range(params_dict["n_epochs"][0]):
            ### Training
            train_loss = self.train_epoch(dataloader=train_dataloader, optimizer=optimizer) 
            train_loss_log.append(train_loss)
            ### Validation
            val_loss = self.test_epoch(test_dataloader=test_dataloader) 
            val_loss_log.append(val_loss)
            if verbose is True:
                print('\n\nEPOCH', epoch + 1)
                print("TRAINING LOSS",round(float(train_loss),3))
                print("VALIDATION LOSS",round(float(val_loss.numpy()),3))
        if return_log is False:
            return train_loss,val_loss
        else:
            return train_loss_log,val_loss_log,train_loss,val_loss
    
    def dae_trainin(self,train_orig_dataloader,train_noise_dataloader,params_dict,test_orig_dataloader,test_noise_dataloader,verbose = False,return_log=True):
        train_loss_log = []
        val_loss_log = []
        optimizer = eval(params_dict["optimizer"][0])(self.parameters(), lr=params_dict["lr"][0], weight_decay=params_dict["penalty"][0])
        for epoch in range(params_dict["n_epochs"][0]):
            ### Training
            train_loss = self.dae_train_epoch(dataloader_orig=train_orig_dataloader,dataloader_noised=train_noise_dataloader,
                                              optimizer=optimizer) 
            train_loss_log.append(train_loss)
            ### Validation
            val_loss = self.dae_test_epoch(dataloader_orig=train_orig_dataloader,dataloader_noised=train_noise_dataloader) 
            val_loss_log.append(val_loss)
            if verbose is True:
                print('\n\nEPOCH', epoch + 1)
                print("TRAINING LOSS",round(float(train_loss),3))
                print("VALIDATION LOSS",round(float(val_loss.numpy()),3))
        if return_log is False:
            return train_loss,val_loss
        else:
            return train_loss_log,val_loss_log,train_loss,val_loss
 
    def save_comparative_plot(self, sample_orig, sample_noised, filename):

        img_noised = next(iter(sample_noised))[0][0].squeeze().to(device)
        img_noised_data = next(iter(sample_noised))[0].to(device)

        fig, axs = plt.subplots(1, 2, figsize=(10,4))
        axs[0].imshow(img_noised.cpu().numpy(), cmap='Greys') 
        axs[0].set_title('Original Noised Image',fontsize=18)
        axs[0].tick_params(labelsize=18)
        self.eval()
        with torch.no_grad():
            rec_img  = self(img_noised_data)
        axs[1].imshow(rec_img.data[0][0].cpu().squeeze().numpy(), cmap='Greys')
        axs[1].set_title('Reconstructed Image DAE',fontsize=18)
        axs[1].tick_params(labelsize=18)
        plt.savefig("./plots/"+str(filename)+'.png',bbox_inches="tight")
        return None
    
    def encode_rep(self,dataset):
        # Given a dataset a list with the encoded representation of the input tensors is returned. 
        encoded_list = []
        for sample in dataset:
            img = sample[0].unsqueeze(0).to(device)
            label = sample[1]
            self.eval()
            with torch.no_grad():
                encoded_img  = self.encode(img)
            encoded_list.append([encoded_img.flatten().cpu().numpy(), label,])
        return np.asarray(encoded_list) 
    
        
    def generate_from_encoded_sample(self,encoded_sample):
        # Given an encoded sample the decoded image is returned.
        self.eval()
        with torch.no_grad():
            encoded_value = torch.tensor(encoded_sample).float().unsqueeze(0)
            new_img  = self.decode(encoded_value.to(device))
        return np.reshape(new_img.squeeze().cpu().numpy(), (28, 28))
    
    def images_smooth_sampling(self,encoded_images,start_digit,stop_digit, number_steps):
        # Extract the representation in the latent space of the examples in the dataset of the chosen digits. 
        sample_start_digit = encoded_images[ encoded_images[:,1] == start_digit ]
        sample_end_digit = encoded_images[ encoded_images[:,1] == stop_digit ]
        # Compute the centroids. 
        start_centroid = np.stack(sample_start_digit[:,0]).mean(axis=0)
        end_centroid = np.stack(sample_end_digit[:,0]).mean(axis=0)
        # Evenly spaced sequences between the two extrema. 
        middle_points = np.array([ np.linspace(start_centroid[i], end_centroid[i], 
                                               number_steps) for i in range(start_centroid.shape[0])])

        fig, ax = plt.subplots(1,number_steps, figsize=(20,6))
        for i in range(number_steps):
            image = self.generate_from_encoded_sample(encoded_sample=middle_points[:,i])
            ax[i].imshow(image, cmap='Greys') 
            ax[i].tick_params(labelsize=15)
        
        plt.savefig('./plots/images_sequence.png',bbox_inches="tight")

            
def get_list_rand_search(params_dict):
    inf_lim_exp_lr = params_dict["lr"][0]
    sup_lim_exp_lr = params_dict["lr"][1]
    inf_lim_drop = params_dict["dropout"][0]
    sup_lim_drop = params_dict["dropout"][1]
    inf_lim_exp_penalty = params_dict["penalty"][0]
    sup_lim_exp_penalty = params_dict["penalty"][1]
    optimizer = params_dict["optimizer"]
    ncombinations = params_dict["ncombinations"]
    inf_lim_n_epochs = params_dict["n_epochs"][0]
    sup_lim_n_epochs = params_dict["n_epochs"][1]

    optimizer = np.random.choice(optimizer, ncombinations)
    dropout = np.random.uniform(inf_lim_drop, sup_lim_drop, ncombinations)
    n_epochs = np.random.randint(low=inf_lim_n_epochs, high=sup_lim_n_epochs, size= ncombinations)
    lr_vals = 10**(np.random.uniform(inf_lim_exp_lr, sup_lim_exp_lr, ncombinations))
    L2_vals = 10**(np.random.uniform(inf_lim_exp_penalty, sup_lim_exp_penalty, ncombinations))   

    combinations = [ [optimizer[i], dropout[i], lr_vals[i], L2_vals[i],n_epochs[i]] for i in range(ncombinations[0])]

    return combinations