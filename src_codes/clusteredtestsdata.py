# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 12:15:35 2018

@author: asthasyal
"""
import tarfile             
import pandas as pd
from tabulate import tabulate
from io import StringIO
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy
import sklearn
from sklearn import preprocessing
from sklearn.preprocessing import scale
from sklearn.preprocessing import MinMaxScaler
from sklearn import manifold, datasets
from sklearn.cluster import KMeans
import datetime
from time import time

def file_read(dataPath,fileName):
     with open (dataPath+fileName+'.csv', mode='r') as f:
      newDF = pd.read_csv(f,sep=',',header='infer')
      newDF = newDF.dropna(axis='rows')
      f.close()
      return newDF
     
    
def main():
    np.set_printoptions(suppress=True,formatter={'float_kind':'{0.4f}'.format})
    dataPath = r'C:\\Users\\asthasyal\\Documents\\TstatProject\\data\\'
    outpath=r'C:\\Users\\asthasyal\\Documents\\TstatProject\\result\\summary_tests.txt'

fileName = 'tstat.dtn05.nersc.gov.tests'
newDF = file_read(dataPath,fileName)
list(newDF)

newDF['first:29']=pd.to_numeric(newDF['first:29']/1000).round(0).astype(int) 
newDF['first:29']=pd.to_datetime(newDF['first:29'],unit='s')
 
newDF['last:30']=pd.to_numeric(newDF['last:30']/1000).round(0).astype(int)
newDF['last:30']=pd.to_datetime(newDF['last:30'],unit='s')
 
newDF=newDF.set_index(pd.DatetimeIndex(newDF['first:29'])).sort_index()
dateFrom = datetime.datetime(2018, 1, 1, 5, 0, 0) #8 to 7 30
dateTo = datetime.datetime(2018, 1, 1, 23, 0, 0)

 
dateFrom = newDF['first:29'].iloc[0] # first date and time in the column
dateTo = newDF['first:29'].iloc[-1]# last date and time
 
selDF=newDF.set_index(pd.DatetimeIndex(newDF['first:29'])).sort_index().loc[dateFrom:dateTo]
selDF['throughput']=(selDF['s_bytes_all:23']*8) / (selDF['durat:31'] / 1000)
selDF['throughput']=selDF['throughput'].round(2)
 
selDF['throughput'] = selDF['throughput']+1
selDF['throughput'] = selDF['throughput'].apply(np.log10)

# computing short lived and long lived flows
selDF['flows_permin']=(selDF['throughput']/8) *(60*1000)
selDF['flows_permin']=selDF['flows_permin'].round(2)
threshold=selDF['flows_permin'].mean()


features=['s_bytes_all:23','durat:31','throughput','flows_permin','c_rtt_avg:45','c_rtt_min:46','c_rtt_max:47','c_rtt_std:48',\
          'c_rtt_cnt:49','s_rtt_avg:52', 's_rtt_min:53','s_rtt_max:54', 's_rtt_std:55', 's_rtt_cnt:56']

dataDF=selDF[features]
scaler = MinMaxScaler()
scaler.fit(dataDF) 
scaleDF=scaler.transform(dataDF)

X = scaleDF
y_pred = pd.DataFrame(KMeans(n_clusters=2, random_state=10).fit_predict(X))
y_pred.columns=['cluster']
n_neighbors = 10
n_components = 2

y_pred = y_pred.reset_index(drop=True)  
dataDF = dataDF.reset_index(drop=True) 

dataDF['cluster']= y_pred['cluster'].values

type(dataDF)
type(y_pred)
clusterDF1= dataDF.loc[dataDF['cluster']==0]
clusterDF2= dataDF.loc[dataDF['cluster']==1]
summaryDF1=clusterDF1.describe()
summaryDF2=clusterDF2.describe()
outpath1=r'C:\\Users\\asthasyal\\Documents\\TstatProject\\result\\summaryDF1tests.txt'
outpath2=r'C:\\Users\\asthasyal\\Documents\\TstatProject\\result\\summaryDF2tests.txt'
with open(outpath1, 'w') as f:
        f.write('Summary for cluster 0 of: '+ fileName+'\n')
        j=0
        for i in range(9,111,9):
            whatP = summaryDF1.iloc[:,j:i]
            f.write(tabulate( whatP, headers= whatP.columns,tablefmt='psql',floatfmt=('.4f')))
            f.write('\n\n')
            j = i
f.close()

with open(outpath2, 'w') as f1:
        f1.write('Summary for cluster 1 of: '+ fileName+'\n')
        j=0
        for i in range(9,111,9):
            whatP = summaryDF2.iloc[:,j:i]
            f1.write(tabulate( whatP, headers= whatP.columns,tablefmt='psql',floatfmt=('.4f')))
            f1.write('\n\n')
            j = i
f1.close()
