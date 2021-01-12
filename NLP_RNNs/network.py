"""
NEURAL NETWORKS AND DEEP LEARNING
PHYSICS OF DATA - Department of Physics and Astronomy
A.A. 2019/20 (6 CFU)
Dr. Alberto Testolin, Dr. Federico Chiariotti


Author: Matteo Guida
Date of creation : 25/10/2020


Description:

Recurrent neural network in PyTorch for natural language processing on Charles Dickens books.

- The program implement the Network class with which the model is defined and trained. By the functions generate_words and softmax_extraction given the initial seed, each following
word is obtained by sampling from the last layer of the network according with a softmax distribution.

"""

from torch import nn
import torch.nn.functional as F
import torch,string,pickle,dataset
import numpy as np
from tqdm import trange
from scipy.special import softmax
from preprocessing import clear_symbols,preprocessing



class Network(nn.Module):
    
    def __init__(self, vocab_size, emb_dim, hidden_units, layers_num, dropout_prob=0, linear_size=512):
                 
    
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        w2vweight = torch.load('./model/embedding.torch')
        w2vweight = torch.FloatTensor(w2vweight)
                
        self.embedding = nn.Embedding(vocab_size, emb_dim).from_pretrained(w2vweight)
        self.embedding.weight.requires_grad=False  

        self.rnn = nn.LSTM(input_size=emb_dim, 
                           hidden_size=hidden_units,
                           num_layers=layers_num,
                           dropout=dropout_prob,
                           batch_first=True).to(self.device)
        
        self.l1 = nn.Linear(hidden_units,linear_size).to(self.device)
        self.out = nn.Linear(linear_size,vocab_size).to(self.device) 

        self.loss_fn = nn.CrossEntropyLoss()
        
    def forward(self, x, state=None):

        # Embedding of x.
        x = self.embedding(x.to(self.device))
                
        # propagate through the LSTM.
        x, state = self.rnn(x.to(self.device), state)
                
        # Linear layer.
        x = F.leaky_relu(self.l1(x).to(self.device)).to(self.device)

        # Output layer.
        output = self.out(x).to(self.device)

        return output, state

    ### Training function
    def train_epoch(self, dataloader, optimizer):
        # Training
        self.train()
        train_loss_batch = []
        for batch_sample in dataloader:
            sample_batch = batch_sample['encoded'].to(self.device)
            # Extract data and move tensors to the selected device
            X, y = sample_batch[:, :-1].to(self.device), sample_batch[:, 1:].to(self.device)
            optimizer.zero_grad()
            # Forward pass
            predicted_word, _ = self(X)
            # Evaluate loss only for last output
            loss = self.loss_fn(predicted_word.transpose(1, 2), y)
            # Backward pass
            loss.backward()
            optimizer.step()
            train_loss_batch.append(loss.data.cpu().numpy())
        train_loss_batch = np.asarray(train_loss_batch)
        return np.mean(train_loss_batch)



    def test_epoch(self, test_dataloader):
        # Validation
        self.eval() # Evaluation mode (e.g. disable dropout)
        val_loss_batch = []
        with torch.no_grad(): # No need to track the gradients
            for batch_sample in test_dataloader:
                sample_batch = batch_sample['encoded'].to(self.device)
                X, y = sample_batch[:, :-1].to(self.device), sample_batch[:, 1:].to(self.device)
                predicted_word, _ = self(X)
                loss = self.loss_fn(predicted_word.transpose(1, 2), y)
                val_loss_batch.append(loss.data.cpu().numpy())
        val_loss_batch = np.asarray(val_loss_batch)
        return np.mean(val_loss_batch)

    def trainin(self, train_dataloader, test_dataloader, optimizer,n_epochs, early_stopping = [True,1], verbose = False, return_log=True):
        train_loss_log = []
        val_loss_log = []
        optimizer = eval(optimizer)(filter(lambda p: p.requires_grad, self.parameters()),lr=1e-3, weight_decay=5e-4)

        for epoch in trange(n_epochs):
            ### Training
            train_loss = self.train_epoch(dataloader=train_dataloader, optimizer=optimizer) 
            train_loss_log.append(train_loss)
            ### Validation
            val_loss = self.test_epoch(test_dataloader=test_dataloader) 
            val_loss_log.append(val_loss)
            if verbose is True:
                print('\n\nEPOCH', epoch + 1)
                print("TRAINING LOSS",round(float(train_loss),3))
                print("VALIDATION LOSS",round(float(val_loss),3))
            torch.cuda.empty_cache()
            if early_stopping[0]:

                if (np.mean(val_loss_log[-int(n_epochs*0.2):]) - np.mean(val_loss_log[-int(n_epochs*0.4):-int(n_epochs*0.2)]))/np.mean(val_loss_log[-int(n_epochs*0.4):-int(n_epochs*0.2)])*100> -early_stopping[1]:
                    print("EARLY STOPPED AT EPOCH : ",epoch )
                    break 
                else:
                    continue
        torch.save(self, './model/best_network.dat')
        if return_log is False:
            return train_loss,val_loss
        else:
            return train_loss_log,val_loss_log,train_loss,val_loss
    
    def generate_words(self,input_seed,n_words):
        
        seed = preprocessing(input_seed, verbose=False,paragraph=False)
        translator=str.maketrans('','',string.punctuation)
        seed = seed.translate(translator).split() 

        word2index = pickle.load(open("./model/word2index.p", "rb"))          
        index2word = pickle.load(open("./model/index2word.p", "rb"))  
        word2wordwithpunct = pickle.load(open("./model/word2word.p", "rb"))    

        # Evaluation mode
        self.eval() 
        next_encoded = dataset.encode_text(word2index, seed)
        next_encoded = next_encoded.unsqueeze(0).to(self.device)
        print(input_seed, end='', flush=True)

        for i in range(n_words):
            with torch.no_grad():
                net_out, net_state = self(next_encoded)
                next_encoded = self.softmax_extraction(net_out[:, -1, :].cpu().numpy())
                next_decoded = index2word[next_encoded]
                print(word2wordwithpunct[next_decoded], end='', flush=True)
                next_encoded = torch.LongTensor([next_encoded])
                next_encoded = next_encoded.unsqueeze(0).to(self.device)



    def softmax_extraction(self,x):
        out = softmax(x)
        vocab = np.arange(out.shape[1])
        sampling = out.reshape(-1,)
        predicted = np.random.choice(vocab, p=sampling)
        return predicted.item()

        