# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 19:44:59 2018

@author: asthasyal
"""
import tarfile
import pandas as pd
from tabulate import tabulate
from io import StringIO
import datetime
from time import time
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error 
from math import sqrt 
import scipy.stats as stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
import numpy as np
 
dataPath = r'C:\\Users\\asthasyal\\Documents\\TstatProject\\data\\'
outputPath = r'C:\\Users\\asthasyal\\Documents\\TstatProject\\result\\labelleddata.txt'
fileName = 'tstat.dtn05.nersc.gov.testb'
fileExt = '.all.csv.tar.gz'
tar = tarfile.open(dataPath+fileName+fileExt, "r:gz")
f = tar.extractfile(fileName+'.all.csv')
Data = f.read()
newDF = pd.read_csv(StringIO(Data.decode('utf-8')),sep=',',header='infer')
del Data
print (len(newDF))

newDF.index=pd.RangeIndex(len(newDF.index))
 
newDF['first:29']=pd.to_numeric(newDF['first:29']/1000).round(0).astype(int)
newDF['first:29']=pd.to_datetime(newDF['first:29'],unit='s')
 
newDF['last:30']=pd.to_numeric(newDF['last:30']/1000).round(0).astype(int)
newDF['last:30']=pd.to_datetime(newDF['last:30'],unit='s')
 
newDF=newDF.set_index(pd.DatetimeIndex(newDF['first:29'])).sort_index()
dateFrom = datetime.datetime(2018, 1, 1, 5, 0, 0) #8 to 7 30
dateTo = datetime.datetime(2018, 6, 1, 17, 0, 0)
 
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
selDF['cluster']=0

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

selDF['cluster']= y_pred['cluster'].values
dataDF['cluster']= y_pred['cluster'].values

clusterDF1= dataDF.loc[dataDF['cluster']==0]
clusterDF2= dataDF.loc[dataDF['cluster']==1]
summaryDF1=clusterDF1.describe()
summaryDF2=clusterDF2.describe()
outpath1=r'C:\\Users\\asthasyal\\Documents\\TstatProject\\result\\Flowspermin_Cluster1.txt'
outpath2=r'C:\\Users\\asthasyal\\Documents\\TstatProject\\result\\Flowspermin_Cluster2.txt'
with open(outpath1, 'w') as f:
        f.write('Summary for cluster 1 of: '+ fileName+'\n')
        j=0
        for i in range(9,111,9):
            whatP = summaryDF1.iloc[:,j:i]
            f.write(tabulate( whatP, headers= whatP.columns,tablefmt='psql',floatfmt=('.4f')))
            f.write('\n\n')
            j = i
f.close()

with open(outpath2, 'w') as f1:
        f1.write('Summary for cluster 2 of: '+ fileName+'\n')
        j=0
        for i in range(9,111,9):
            whatP = summaryDF2.iloc[:,j:i]
            f1.write(tabulate( whatP, headers= whatP.columns,tablefmt='psql',floatfmt=('.4f')))
            f1.write('\n\n')
            j = i
f1.close()
