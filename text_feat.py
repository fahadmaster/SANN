#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 17:09:40 2019

@author: acer
"""
####create vocabulary from transcription
import os
import sys
import csv
import pickle
import numpy as np
from nlp_util import *
from nltk.tokenize import word_tokenize
lines = []
with open('/home/acer/Desktop/open_smile/processed_tran1.csv') as f:
# with open('../data/processed/IEMOCAP/processed_tran_fromG.csv') as f:
    read = csv.reader(f)
    lines = [ x[1] for x in read]
    
token_lines = [ word_tokenize(x) for x in lines]
token_lines_lower = [ [t.lower() for t in x] for x in token_lines]

sent_len = [ len(x) for x in token_lines]
print (np.max(sent_len))
print (np.min(sent_len))
print (np.mean(sent_len))
print (np.std(sent_len))

def read_data(dic, lines):

    for tokens in lines:
        
        for token in tokens:
            token = token.lower()
            
            if token in dic :
                dic[token] += 1
            else:
                dic[token] = 1
        
    return dic
dic_count = {}
dic_count = read_data(dic_count, token_lines_lower)
print ('dic size : ' + str(len(dic_count)))

dic = {}
dic['_PAD_'] = len(dic)
dic['_UNK_'] = len(dic)

for word in dic_count.keys():
    dic[word] = len(dic)    
print (len(dic))
print (dic['_PAD_'])

with open('/home/acer/Desktop/open_smile/dic.pkl', 'w') as f:
# with open('../data/processed/IEMOCAP/dic_G.pkl', 'w') as f:
    pickle.dump( dic, f )
    
    
 ###create index with vocabulary   
lines = []
with open('/home/acer/Desktop/open_smile/processed_tran1.csv') as f:
# with open('../data/processed/IEMOCAP/processed_tran_fromG.csv') as f:
    read = csv.reader(f)
    lines = [ x[1] for x in read]
    
token_lines = [ word_tokenize(x) for x in lines]
token_lines_lower = [ [t.lower() for t in x] for x in token_lines]

sent_len = [ len(x) for x in token_lines]
print (np.max(sent_len))
print (np.min(sent_len))
print (np.mean(sent_len))
print (np.std(sent_len))

# convert to index
index_lines = [ [ dic[t] for t in x ] for x in token_lines_lower ]
#save as numpy
np_trans = np.zeros( [1819, 128], dtype=np.int)
np.shape(np_trans)

for i in range( len(index_lines) ):
    
    if len( index_lines[i] ) > 127:
        np_trans[i][:] = index_lines[i][:128]
    else:
        np_trans[i][:len(index_lines[i])] = index_lines[i][:]
        
np.save('/home/acer/Desktop/open_smile/processed_trans1.npy', np_trans)