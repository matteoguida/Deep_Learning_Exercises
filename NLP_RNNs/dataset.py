"""
NEURAL NETWORKS AND DEEP LEARNING
PHYSICS OF DATA - Department of Physics and Astronomy
A.A. 2019/20 (6 CFU)
Dr. Alberto Testolin, Dr. Federico Chiariotti


Author: Matteo Guida
Date of creation : 25/10/2020


Description:

Recurrent neural network in PyTorch for natural language processing on Charles Dickens books.

"""

import gensim,pickle,torch,string



class Dataset():
    def __init__(self, text, emb_dim=100, cut_value=20,load=False):
        
        par_list = self.select_paragraphs(text=text,cut_value=cut_value)
        self.data_list = self.get_vectors_from_paragraph(par_list=par_list,cut_value=cut_value)

        self.word2vector_model(data_list= self.data_list,
                               emb_dim=emb_dim, cut_value=cut_value,load=load )
    
    def select_paragraphs(self, text,cut_value):
        translator=str.maketrans('','',string.punctuation)
        # For each paragraph a list composed by its words is created, without punctuation 
        par_list = [i.translate(translator).split() for i in text]
        # We hold the the sub-lists (i.e. paragraphs) only if they are composed by more than cut_value words. 
        par_list = [x for x in par_list if len(x) >= cut_value]
        return par_list
        
    def get_vectors_from_paragraph(self, par_list,cut_value):
        data_list = []
        for sublist in par_list:
            chunks = [sublist[x:x+cut_value] for x in range(0, len(sublist), cut_value)]
            for x in chunks:
                if len(x) >= cut_value:
                    data_list.append(x)
        return data_list
    
    
    def word2vector_model(self, data_list,emb_dim,cut_value,load):
        if load is False:
            
            word_model = gensim.models.Word2Vec(self.data_list, size=emb_dim, min_count=1, iter=200)
            words = list(word_model.wv.vocab)
            weights = torch.FloatTensor(word_model.wv.vectors)
            torch.save(obj=weights, f = './model/embedding.torch')

            self.word2index = {w: i for i, w in enumerate(word_model.wv.index2word)}           
            self.index2word = {word_model.wv.vocab[w].index : w for w in words}  
            self.word2wordwithpunct = { w : ' '+ w for w in words }

            self.word2wordwithpunct['commapunctuation'] = ','
            self.word2wordwithpunct['pointpunctuation'] = '.'
            self.word2wordwithpunct['questionpunctuation'] = '?'
                 
            pickle.dump(self.word2index, open("./model/word2index.p", "wb"))  
            pickle.dump(self.index2word, open("./model/index2word.p", "wb"))  
            pickle.dump(self.word2wordwithpunct, open("./model/word2word.p", "wb"))  
            
        else:
            
            self.word2index = pickle.load(open("./model/word2index.p", "rb"))
            self.index2word = pickle.load(open("./model/index2word.p", "rb"))
            self.word2wordwithpunct = pickle.load(open("./model/word2word.p", "rb"))
    
    def len_vocab(self):
        return len(self.index2word)
    
    def __len__(self):
        return len(self.data_list)
        
    def __getitem__(self, idx):

        text = self.data_list[idx]
        encoded = encode_text(self.word2index,text)
        cutsent2encod = {'text': text, 'encoded': encoded}

        return cutsent2encod

def encode_text(dictionary,text):
    encoded = torch.LongTensor([dictionary.get(x) for x in text])
    return encoded



def decode_text(dictionary,code):
    text = [dictionary[str(i)] for i in code]
    return text