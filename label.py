#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 14:16:57 2019

@author: acer
"""
import csv
import os
from file_util import file_search
import sys
out_file = '/home/acer/Desktop/open_smile/processed_tran.csv'
os.system('rm ' + out_file)

list_category = [
                'ang',
                'hap',
                'sad',
                'neu',
                'fru',
                'exc',
                'fea',
                'sur',
                'dis',
                'oth',
                'xxx'
                ]
###extract text
def extract_trans( list_in_file, ses):
    out_file = '/home/acer/Desktop/open_smile/processed_tran' +str(ses)+'.csv'
    lines = []
    
    for in_file in list_in_file:
        cnt = 0
        with open(in_file, 'r') as f:
            lines = f.readlines()
        with open(out_file, 'a') as f:
            csv_writer = csv.writer( f )
            lines = sorted(lines)                  # sort based on first element
            for line in lines:
                name = line.split(':')[0].split(' ')[0].strip()
                # unwanted case 
                if name[:3] != 'Ses':             # noise transcription such as reply  M: sorry
                    continue
                elif name[-3:-1] == 'XX':        # we don't have matching pair in label
                    continue
                trans = line.split(':')[1].strip()
                
                cnt += 1
                csv_writer.writerow([name, trans])
                          
for x in range(5):
    list_files = []
    sess_name = 'Session' + str(x+1)
    path = '/home/acer/Desktop/fahad_data/IEMOCAP_full_release/'+ sess_name + '/dialog/transcriptions/'
    file_search(path, list_files)
    list_files = sorted(list_files)
    extract_trans(list_files, x)
    print (sess_name + ", #sum files: " + str(len(list_files)))

category = {}
for c_type in list_category:
    if c_type in category:
       None
    else:
        category[c_type] = len(category)
        
def find_category(lines):
    is_target = True
    
    id = ''
    c_label = ''
    list_ret = []
    for line in lines: 
        if is_target == True:
            try:
                id = line.split('\t')[1].strip()  #  extract ID
                c_label  = line.split('\t')[2].strip()  #  extract category
                print(c_label)
                if c_label not in category:
                    print("ERROR nokey" + c_label)
                    sys.exit()
                
                list_ret.append( [id, c_label] )
                is_target = False
            except:
                print("ERROR " + line)
                sys.exit()
        else:
            if line == '\n':
                is_target = True
    return list_ret
##extract label

def extract_labels(list_in_file, ses ) :
    out_file = '/home/acer/Desktop/open_smile/label' +str(ses)+'.csv'
    id = ''
    lines = []
    list_ret = []
    
    for in_file in list_in_file:
        with open(in_file, 'r') as f:
            lines = f.readlines()
            lines = lines[2:]                           # remove head
            list_ret = find_category(lines)
        list_ret = sorted(list_ret)                   # sort based on first element
    
        with open(out_file, 'a') as f:
            csv_writer = csv.writer( f )
            csv_writer.writerows(list_ret ) 
            
# [schema] ID, label [csv]
list_avoid_dir = ['Attribute', 'Categorical', 'Self-evaluation']

for x in range(5):
    list_files = []
    sess_name = 'Session' + str(x+1)
    path = '/home/acer/Desktop/fahad_data/IEMOCAP_full_release/' + sess_name + '/dialog/EmoEvaluation/'
    file_search(path, list_files, list_avoid_dir)
    list_files = sorted(list_files)
    extract_labels(list_files, x)
    print(sess_name + ", #sum files: " + str(len(list_files)))
