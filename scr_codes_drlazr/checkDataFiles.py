#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 14:27:53 2018
to run
python scriptname.py file_directory outputdir
python checkDataFiles.py '../data/interData/preProc/'  '../results/'
run  ./checkDataFiles '../data/interData/preProc/'  '../results/'
@author: alazar
"""

import tarfile
import pandas as pd
from io import StringIO
import sys
import numpy as np
from tabulate import tabulate

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

def make_tarfile(inputFile, outputFile, fileName):
    with tarfile.open(outputFile, "w:gz") as tar:
        tar.add(inputFile,arcname=fileName) 
    tar.close()

def update_files(dataPath,fileNames):
    for fileName in fileNames:
        print fileName
        workDF = file_read(dataPath,fileName)
        workDF['first_dt:105']=(pd.to_numeric(workDF['first:29'])/1000).round(0).astype(int)
        workDF['first_dt:105']=pd.to_datetime(workDF['first_dt:105'],unit='s')    
        workDF['last_dt:106']=(pd.to_numeric(workDF['last:30'])/1000).round(0).astype(int)
        workDF['last_dt:106']=pd.to_datetime(workDF['last_dt:106'],unit='s')  
        workDF=workDF.set_index(pd.DatetimeIndex(workDF['first_dt:105'])).sort_index()
    workDF.to_csv(dataPath+fileNames,sep=',',header=True,index=True)

def make_table(dataPath,outputPath,fileNames,columnNames,time_intervals):
    fileSave = 'counts.csv'
    df_ = pd.DataFrame(index=None, columns=columnNames)
    df_ = df_.fillna(0) # with 0s rather than NaNs
    df_['file_names'] = fileNames
    df_=df_.set_index(['file_names'])
    for fileName in fileNames:
        print fileName
        workDF = file_read(dataPath,fileName)
        df_.loc[fileName]['total_rows'] = len(workDF)
        workDF=workDF.set_index(pd.DatetimeIndex(workDF['first:29'])).sort_index()
        df_['first'] = df_['first'].astype(str)
        df_.loc[fileName]['first']  = str(workDF['first:29'].iloc[0])
        df_['last'] = df_['last'].astype(str)
        df_.loc[fileName]['last'] = str(workDF['first:29'].iloc[-1])
        for time_i in time_intervals: 
            grouped = workDF.groupby(pd.Grouper(freq=time_i))
            df_.loc[fileName][time_i+'_avg']= np.round(np.mean(grouped.size()),decimals=2)
            listSizes = grouped.size()
            df_.loc[fileName][time_i+'_100win']= np.sum(listSizes[0:100])
            df_.loc[fileName][time_i+'_#win']= len(grouped)
    of=outputPath+fileSave
    df_.to_csv(of,sep=',',header=True,index=True)
    df_.reset_index(level=0,inplace=True)
    return df_
 

def main():
    dataPath = sys.argv[1]   
    outputPath = sys.argv[2]
    fileNames = ['tstat.dtn01.nersc.gov', 'tstat.dtn02.nersc.gov', 'tstat.dtn03.nersc.gov',\
                 'tstat.dtn04.nersc.gov', 'tstat.dtn05.nersc.gov', 'tstat.dtn06.nersc.gov',\
                 'tstat.dtn07.nersc.gov', 'tstat.dtn08.nersc.gov', 'dtn-tstat-2018.01','dtn-tstat-2018.07',\
                 'tstat.dtn05.nersc.gov.testb', 'tstat.dtn05.nersc.gov.tests'] 
    #fileNames = ['tstat.dtn05.nersc.gov.tests']
    columnNamesGen = ['file_names','total_rows','first','last','5min_avg',\
                      '1h_avg','1d_avg','5min_100win','1h_100win','1d_100win',\
                      '5min_#win','1h_#win','1d_#win']   
    table1 = ['file_names','total_rows','first','last']
    table2 = ['file_names','total_rows','5min_avg','1h_avg','1d_avg']
    table3 = ['file_names','5min_100win','1h_100win','1d_100win']
    table4 = ['file_names','5min_#win','1h_#win','1d_#win']
    time_intervals = ['5min','1h','1d']
    df = make_table(dataPath,outputPath,fileNames,columnNamesGen,time_intervals)
    
   
    df = pd.read_csv(outputPath + 'counts.csv',sep=',',header='infer')
    
    fileToShow = 'table.txt'
    with open(outputPath + fileToShow,'w') as f:
        f.write("Number of transfers and Time:\n")
        f.write(tabulate(df[table1], headers=table1,tablefmt='psql',floatfmt=('.4f')))
        f.write("\n\nAverage transfers per window:\n")
        f.write(tabulate(df[table2], headers=table2,tablefmt='psql',floatfmt=('.4f')))
        f.write("\n\nTotal number of transfers in the first 100 windows:\n")
        f.write(tabulate(df[table3], headers=table3,tablefmt='psql',floatfmt=('.4f')))
        f.write("\n\nNumber of windows:\n")
        f.write(tabulate(df[table4], headers=table4,tablefmt='psql',floatfmt=('.4f')))
        f.close()
        
        
if __name__== "__main__":
  main()

dataPath = '../data/interData/preProc/'  
outputPath = '../results/'
