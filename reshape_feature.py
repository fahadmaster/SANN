#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 16:22:21 2019

@author: acer
"""
import csv
import numpy as np
import pandas as pd
data = pd.read_csv('/home/acer/Desktop/open_smile/emobase20104.csv')
lines = []
with open('/home/acer/Desktop/open_smile/emobase20104.csv') as f:
    csv_reader = csv.reader(f)
    lines = [x for x in csv_reader]

lines = [x[0].split(';') for x in lines]
lines = [ [float(t) for t in lines[i]] for i in range(1,len(lines))]
len(lines)
np_emobase = np.zeros( [2170, 1582], dtype=np.float)
np.shape(np_emobase)
for i in range( len(np_emobase) ):
    np_emobase[i] = lines[i][1:]  
np.save('/home/acer/Desktop/open_smile/processed_emobase20104.npy', np_emobase)
tmp = np.load( '/home/acer/Desktop/open_smile/processed_emobase20104.npy')
np.shape(tmp)