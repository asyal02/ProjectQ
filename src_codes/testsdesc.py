# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 15:37:11 2018

@author: asthasyal
"""

import csv            
import pandas as pd
from tabulate import tabulate
from io import StringIO
import sys
import numpy as np

def file_read(dataPath,fileName):
    with open (dataPath+fileName+'.csv', mode='r') as f:
      newDF = pd.read_csv(f,sep=',',header='infer')
      newDF = newDF.dropna(axis='rows')
    f.close()
    return newDF

def main():
    np.set_printoptions(suppress=True,formatter={'float_kind':'{0.4f}'.format})
    #dataPath = sys.argv[1]   
    #outputPath = sys.argv[2]
    dataPath = r'C:\\Users\\asthasyal\\Documents\\TstatProject\\data\\'
    outpath=r'C:\\Users\\asthasyal\\Documents\\TstatProject\\result\\summary_tests.txt'
    #fileNames = ['tstat.dtn01.nersc.gov', 'tstat.dtn02.nersc.gov', 'tstat.dtn03.nersc.gov',
                # 'tstat.dtn04.nersc.gov', 'tstat.dtn05.nersc.gov', 'tstat.dtn06.nersc.gov', 
                 #'tstat.dtn07.nersc.gov', 'tstat.dtn08.nersc.gov', 'dtn-tstat-2018.01','dtn-tstat-2018.07',
                # 'tstat.dtn05.nersc.gov.testb', 'tstat.dtn05.nersc.gov.tests']
    fileName = 'tstat.dtn05.nersc.gov.tests'
    workDF = file_read(dataPath,fileName)
    summaryT = workDF.describe()
    summaryT = summaryT.drop(summaryT.columns[0],axis=1)
    with open(outpath, 'w') as f:
        f.write('Summary for: '+ fileName+'\n')
        j=0
        for i in range(9,111,9):
            whatP = summaryT.iloc[:,j:i]
            f.write(tabulate( whatP, headers= whatP.columns,tablefmt='psql',floatfmt=('.4f')))
            f.write('\n\n')
            j = i
    f.close()
