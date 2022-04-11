#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 13:07:48 2019

@author: acer
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 15:25:36 2019

@author: acer
"""

import os
import sys
import csv

from file_util import   file_search


# emobase2010
#OPENSMILE_CONFIG_PATH = '/home/acer/Downloads/opensmile-2.3.0/config/modified_emobase2010.conf'    
OPENSMILE_CONFIG_PATH = '/home/acer/Desktop/open_smile/modified_emobase2010.conf'
out_file = '/home/acer/Desktop/open_smile/emobase2010' 

def extract_feature( list_in_file, o_file, ses) :
        out_file = o_file+str(ses)+'.csv'
        attr_out = '/home/acer/Desktop/open_smile/attr'+str(ses)+'.csv'
        cnt = 0    
        for in_file in list_in_file: 
            if in_file.endswith('.wav'):
                if(in_file.find("_impro")>0 and in_file.find("_M")>0):
                    with open(attr_out,'a') as f:
                        csv_writer = csv.writer(f)
                        csv_writer.writerow(['1','1',str(2*ses+1)])
                elif(in_file.find("_script")>0 and in_file.find("_M")>0):
                    with open(attr_out,'a') as f:
                        csv_writer = csv.writer(f)
                        csv_writer.writerow(['2','1',str(2*ses+1)])
                elif(in_file.find("_script")>0 and in_file.find("_F")>0):
                    with open(attr_out,'a') as f:
                        csv_writer = csv.writer(f)
                        csv_writer.writerow(['2','2',str(2*ses+2)])
                elif(in_file.find("_impro")>0 and in_file.find("_F")>0):
                    with open(attr_out,'a') as f:
                        csv_writer = csv.writer(f)
                        csv_writer.writerow(['1','2',str(2*ses+2)])
            
#               cmd = 'SMILExtract -C ' + OPENSMILE_CONFIG_PATH + ' -I ' + in_file + ' -O ' + out_file + ' -headercsv 0'  #MFCC12EDAZ, prosody
#               cmd = 'SMILExtract -C ' + OPENSMILE_CONFIG_PATH + ' -I ' + in_file + ' -csvoutput ' + out_file + ' -headercsv 0'   # MFCC12EDA
               # cmd = 'SMILExtract -C ' + OPENSMILE_CONFIG_PATH + ' -I ' + in_file + ' -csvoutput ' + out_file   # emobase2010
                #print(cmd)
               # os.system(cmd)
        
                cnt += 1
                if cnt % 1000 == 0:
                    print (str(cnt) + " / " + str( len(list_in_file) ))
                    sys.stdout.flush()
            


for x in range(5):
    list_files = []
    sess_name = 'Session' + str(x+1)
    path = '/home/acer/Desktop/fahad_data/IEMOCAP_full_release/' + sess_name + '/sentences/wav/'
    file_search(path, list_files)
    list_files = sorted(list_files)
    extract_feature( list_files, out_file, x)
    print(sess_name + ", #sum files: " + str(len(list_files)))
    

