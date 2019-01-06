#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 14:27:53 2018
to run
python scripname.py file_directory file_name
python computeThroughput.py '../data/tstatDataOneFile/' '../data/interData/preProc/'
run  ./computeThroughput "../data/tstatDataOneFile/" "../data/interData/preProc/"
@author: alazar
"""

import tarfile
import pandas as pd
from io import StringIO
import sys
import os
import numpy as np

#dataPath = '../data/tstatDataOneFile/'
#fileName = 'tstat.dtn05.nersc.gov.testb'

def file_read(dataPath,fileName):
    fileExt='.all.csv.tar.gz'
    tar = tarfile.open(dataPath+fileName+fileExt, "r:gz")
    f = tar.extractfile(fileName)
    Data = f.read()
    newDF = pd.read_csv(StringIO(Data.decode('utf-8')),sep=',',header='infer')
    newDF = newDF.dropna(axis='rows')
    f.close()
    tar.close()
    return newDF

def make_tarfile(inputFile, outputFile,file_name):
    with tarfile.open(outputFile, "w:gz") as tar:
        tar.add(inputFile,arcname=file_name) 
    tar.close()

    
def compute_throughput(dataPath,outPath,fileNames):
    fileExt = '.tar.gz'
    for fileName in fileNames:
        print fileName
        workDF = file_read(dataPath,fileName)
        workDF['first_dt:105']=(pd.to_numeric(workDF['first:29'])/1000).round(0).astype(int)
        workDF['first_dt:105']=pd.to_datetime(workDF['first_dt:105'],unit='s')    
        workDF['last_dt:106']=(pd.to_numeric(workDF['last:30'])/1000).round(0).astype(int)
        workDF['last_dt:106']=pd.to_datetime(workDF['last_dt:106'],unit='s')  
        workDF=workDF.set_index(pd.DatetimeIndex(workDF['first_dt:105'])).sort_index()
        workDF.loc[:,'s_throughput_Mbps:107']=((workDF['s_bytes_all:23']*8) / (workDF['durat:31']) / 1000)
        #workDF.loc[:,'s_throughput_Mbps:107']=workDF['s_throughput_Mbps:107'].round(2)    
        workDF.loc[:,'c_throughput_Mbps:108']=((workDF['c_bytes_all:9']*8) / (workDF['durat:31']) / 1000)
        #workDF.loc[:,'c_throughput_Mbps:108']=workDF['c_throughput_Mbps:108'].round(2)    
        workDF.loc[:,'t_throughput_Mbps:109']=(((workDF['s_bytes_all:23']+workDF['c_bytes_all:9'])*8) \
        / (workDF['durat:31']) / 1000)
        #workDF.loc[:,'t_throughput_Mbps:109']=workDF['t_throughput_Mbps:109'].round(2)    
        workDF.loc[:,'throughput_Log:110'] = workDF['t_throughput_Mbps:109']+1
        workDF.loc[:,'throughput_Log:110'] = workDF['throughput_Log:110'].apply(np.log10)
        workDF.to_csv(outPath+fileName+ '.all.csv',sep=',',header=True,index=False)
        output_file = outPath + fileName + '.all.csv'   
        make_tarfile(output_file,output_file+fileExt,fileName)
        os.remove(output_file)
    return 

    
def main():
    dataPath = sys.argv[1]
    outPath = sys.argv[2]
    fileNames = ['tstat.dtn01.nersc.gov', 'tstat.dtn02.nersc.gov', 'tstat.dtn03.nersc.gov',\
                 'tstat.dtn04.nersc.gov', 'tstat.dtn05.nersc.gov', 'tstat.dtn06.nersc.gov',\
                 'tstat.dtn07.nersc.gov', 'tstat.dtn08.nersc.gov', 'dtn-tstat-2018.01','dtn-tstat-2018.07',\
                 'tstat.dtn05.nersc.gov.testb', 'tstat.dtn05.nersc.gov.tests'] 
    #fileNames = ['dtn-tstat-2018.07']
    #dDF = file_read(dataPath,fileName)
    compute_throughput(dataPath,outPath,fileNames)
    #outputFile = sys.argv[3] + fileName + '.through'
    #dDF.to_csv(outputFile,sep=',',header=True,mode='w',index=False)
    
if __name__== "__main__":
  main()

