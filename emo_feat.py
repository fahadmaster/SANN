"""
Created on Fri Oct  4 15:25:36 2019
@author: acer
"""
import os
import sys
from file_util import file_search
# emobase2010
#OPENSMILE_CONFIG_PATH = '/home/acer/Downloads/opensmile-2.3.0/config/modified_emobase2010.conf'    
OPENSMILE_CONFIG_PATH = '/home/acer/Desktop/open_smile/modified_emobase2010.conf'
out_file = '/home/acer/Desktop/open_smile/emobase2010' 

def extract_feature( list_in_file, o_file, ses) :
        out_file = o_file+str(ses)+'.csv'
        cnt = 0    
        for in_file in list_in_file: 
            if in_file.endswith('.wav'):
#               cmd = 'SMILExtract -C ' + OPENSMILE_CONFIG_PATH + ' -I ' + in_file + ' -O ' + out_file + ' -headercsv 0'  #MFCC12EDAZ, prosody
#               cmd = 'SMILExtract -C ' + OPENSMILE_CONFIG_PATH + ' -I ' + in_file + ' -csvoutput ' + out_file + ' -headercsv 0'   # MFCC12EDA
                cmd = 'SMILExtract -C ' + OPENSMILE_CONFIG_PATH + ' -I ' + in_file + ' -csvoutput ' + out_file   # emobase2010
                #print(cmd)
                os.system(cmd)
                cnt += 1
                if cnt % 1000 == 0:
                    print (str(cnt) + " / " + str( len(list_in_file) ))
                    sys.stdout.flush()
            
for x in range(5):
    list_files = []
    sess_name = 'Session' + str(x+1)
    #path = '/home/acer/Desktop/fahad_data/IEMOCAP_full_release/' + sess_name + '/sentences/wav/'
    path ='/home/acer/Desktop/fahad_data/IEMOCAP_vowel/'+sess_name 
    file_search(path, list_files)
    list_files = sorted(list_files)
    extract_feature( list_files, out_file, x)
    print(sess_name + ", #sum files: " + str(len(list_files)))
          


    

