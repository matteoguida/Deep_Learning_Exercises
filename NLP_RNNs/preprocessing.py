"""
NEURAL NETWORKS AND DEEP LEARNING
PHYSICS OF DATA - Department of Physics and Astronomy
A.A. 2019/20 (6 CFU)
Dr. Alberto Testolin, Dr. Federico Chiariotti


Author: Matteo Guida
Date of creation : 25/10/2020


Description:

Recurrent neural network in PyTorch for natural language processing on Charles Dickens books.

 - With the function loading we load the all .txt files present in filepath.
 - With the function preprocessig the loaded texts are cleared to start the embedding. 


"""




#%%
import re
import string
import os


def delate_chapters_heading(text):
    text = re.sub(r'CHAPTER.*\n+', '\n\n', text)
    return text

def clear_symbols(text):
    text = re.sub(r'\n+(?=\n)', '\n', text)
    chars_to_remove = ["'", '(', ')','*','/',"_","—","-","“","”","‘","’","£","—","\ufeff","&", '[', ']']
    rx = '[' + re.escape(''.join(chars_to_remove)) + ']'
    text = re.sub(rx, ' ', text)
    return text

def get_paragraph(text, sep='.\n'):
    sentences = re.split(sep, text)
    sequences = []
    for s in sentences:
        sequences.append(s + ' .\n')
    return sequences


def loading(filepath,titles):
    files = {}
    for f in os.listdir(filepath):    
        if f.endswith('.txt'):         
            with open(os.path.join(filepath, f), "r",encoding="utf8") as file:
                files[f] = file.read() 
    titles = [i +".txt" for i in titles]
    all_dickens = ""
    for i in titles:
        all_dickens+=files[i]
    return all_dickens

def preprocessing(text, verbose=True,paragraph=True):
    """
    
    
    """    
    text = clear_symbols(text)
    text = delate_chapters_heading(text)
    fullstops = [";", '!', ",", ".", "?"]
    fullstops_changes = [".", ".", " commapunctuation ", " pointpunctuation ", " questionpunctuation " ] 
    for i,j in zip(fullstops,fullstops_changes):
        text = text.replace(i,j)
    text = re.sub('(?<=[^.])\n'," ",text)
    text = text.lower()
    if verbose is True:
        alphabet = list(set(text))
        alphabet.sort()
        print('Found letters:', alphabet)
    if paragraph is True:
        paragraphs = get_paragraph(text)
        return paragraphs
    else:
        return text
