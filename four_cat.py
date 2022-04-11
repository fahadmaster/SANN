#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 15:54:01 2019
@author: acer
"""
import csv
import pandas as pd
import numpy as np
from file_util import   create_folder

lines = []
with open('/home/acer/Desktop/open_smile/label4.csv') as f :
    csv_reader = csv.reader(f)
    lines = [x for x in csv_reader]
    
with open('/home/acer/Desktop/open_smile/processed_label4.txt', 'w') as f2:
    for line in lines:
        if line[1] == 'ang':
            f2.write(line[0]+','+line[2]+','+line[3]+','+line[4]+','+'0\n')
        elif line[1] == 'hap':
            f2.write(line[0]+','+line[2]+','+line[3]+','+line[4]+','+'1\n')
        elif line[1] == 'exc':
            f2.write(line[0]+','+line[2]+','+line[3]+','+line[4]+','+'1\n')
        elif line[1] == 'sad':
            f2.write(line[0]+','+line[2]+','+line[3]+','+line[4]+','+'2\n')
        elif line[1] == 'neu':
            f2.write(line[0]+','+line[2]+','+line[3]+','+line[4]+','+'3\n')
        else :
            f2.write(line[0]+','+line[2]+','+line[3]+','+line[4]+','+'-1\n')
    
target_path = 'four_category_vowel'
df = pd.read_csv('/home/acer/Desktop/open_smile/processed_label4.txt', header=None)
data = df.values
full_label = data[:,4].astype(None)

create_folder('/home/acer/Desktop/open_smile/'+target_path)

# extract label
with open('/home/acer/Desktop/open_smile/' + target_path + '/FC_label4.txt', 'w') as f:
    for i, label in enumerate(full_label):
        if(label != -1):
            f.write(data[i][0]+','+
                     str(data[i][1])+','+
                     str(data[i][2])+','+
                     str(data[i][3])+','+
                     str(data[i][4])+'\n')

#extract feature
total_num = len([x for x in full_label if int(x)>-1])
feature_data = np.load('/home/acer/Desktop/open_smile/processed_emobase20104.npy')
new_data = np.zeros([total_num, 1582], dtype=np.float)
index = 0
for i, label in enumerate(full_label):
    if(label != -1):
        new_data[index] = feature_data[i]
        index += 1

np.save('/home/acer/Desktop/open_smile/four_category_vowel/FC_emobase20104.npy', new_data)
