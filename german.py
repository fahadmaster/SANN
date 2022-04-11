#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 11:07:27 2020

@author: acer
"""
import csv
import pandas as pd
import numpy as np
import os
import sys
emotion_class_map = {'W' : 0, 'F' : 1, 'N' : 2, 'T' : 3, 'A' : 4, 'L' : 5, 'E' : 6 }
speaker_class_map= {'03' : 0, '08' : 1, '09' : 2, '10' : 3, '11' : 4, '12' : 5, '13' : 6, '14' : 7, '15' : 8, '16' : 9 }
Y_train_emo = []
Y_test_emo=[]
Y_train_spk = []
Y_test_spk=[]
out_file = '/home/acer/Desktop/open_smile/german_csv/test.csv' 
OPENSMILE_CONFIG_PATH = '/home/acer/Desktop/open_smile/modified_emobase2010.conf'
folder_path_train = "/home/acer/Desktop/fahad_data/german_emo/train"
#folder_path_test = "C:/Users/Shreya/Desktop/Minor/ntest/" 
for filename in os.listdir(folder_path_train):
        print(filename)
        in_file = folder_path_train + '/' + filename
#        #print(filepath[-8] )
        emotion = filename[5]
        print(emotion)
        speaker = filename[0:2]
        print(speaker)
        emotion_class = emotion_class_map[emotion]
        print (emotion_class)
        speaker_class = speaker_class_map[speaker]
        Y_train_emo.append(emotion_class)
        Y_train_spk.append(speaker_class)
        cmd = 'SMILExtract -C ' + OPENSMILE_CONFIG_PATH + ' -I ' + in_file + ' -csvoutput ' + out_file   # emobase2010
                
        os.system(cmd) 
        sys.stdout.flush()
np.save('/home/acer/Desktop/open_smile/german_csv/Y_train_emo.npy', Y_train_emo)
np.save('/home/acer/Desktop/open_smile/german_csv/Y_train_spk.npy', Y_train_spk)
data = pd.read_csv('/home/acer/Desktop/open_smile/german_csv/test.csv' )
lines = []
with open('/home/acer/Desktop/open_smile/german_csv/test.csv' ) as f:
    csv_reader = csv.reader(f)
    lines = [x for x in csv_reader]

lines = [x[0].split(';') for x in lines]
lines = [ [float(t) for t in lines[i]] for i in range(1,len(lines))]
len(lines)
np_emobase = np.zeros( [165, 1582], dtype=np.float)
np.shape(np_emobase)
for i in range( len(np_emobase) ):
    np_emobase[i] = lines[i][1:]  
np.save('/home/acer/Desktop/open_smile/german_csv/X_test.npy', np_emobase)

#np.save('/home/acer/Desktop/open_smile/german_csv/Y_train_emo.npy', Y_train_emo)
#np.save('/home/acer/Desktop/open_smile/german_csv/Y_train_emo.npy', Y_train_emo)
    
                    