# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 15:37:11 2018

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
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


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
features=['s_bytes_all:23','durat:31','c_rtt_avg:45','c_rtt_min:46','c_rtt_max:47','c_rtt_std:48',\
          'c_rtt_cnt:49','s_rtt_avg:52', 's_rtt_min:53','s_rtt_max:54', 's_rtt_std:55', 's_rtt_cnt:56', 'throughput','flows_permin' ]
newDF = file_read(dataPath,fileName)
newDF.index=pd.RangeIndex(len(newDF.index))
 
newDF['first:29']=pd.to_numeric(newDF['first:29']/1000).round(0).astype(int)
newDF['first:29']=pd.to_datetime(newDF['first:29'],unit='s')
 
newDF['last:30']=pd.to_numeric(newDF['last:30']/1000).round(0).astype(int)
newDF['last:30']=pd.to_datetime(newDF['last:30'],unit='s')
 
newDF=newDF.set_index(pd.DatetimeIndex(newDF['first:29'])).sort_index()

newDF['throughput']=(newDF['s_bytes_all:23']*8) / (newDF['durat:31'] / 1000)
newDF['throughput']=newDF['throughput'].round(2)
 
newDF['throughput'] = newDF['throughput']+1
newDF['throughput'] = newDF['throughput'].apply(np.log10)

# computing short lived and long lived flows
newDF['flows_permin']=(newDF['throughput']/8) *(60*1000)
newDF['flows_permin']=newDF['flows_permin'].round(2)
threshold=newDF['flows_permin'].mean()
newDF['cluster']=0

dataDF=newDF[features]
scaler = MinMaxScaler()
scaler.fit(dataDF) 
scaleDF=scaler.transform(dataDF)

X = scaleDF
y_pred = pd.DataFrame(KMeans(n_clusters=2, random_state=10).fit_predict(X))
y_pred.columns=['cluster']
n_neighbors = 10
n_components = 2

X= newDF[features].values
X_embedded = TSNE(n_components=2).fit_transform(X)
X_embedded.shape
X_embedded.show()

y_pred = y_pred.reset_index(drop=True)
dataDF = dataDF.reset_index(drop=True)

newDF['cluster']= y_pred['cluster'].values
dataDF['cluster']= y_pred['cluster'].values
#seperating both the clusters in two different dataframes
clusterDF1= dataDF.loc[dataDF['cluster']==0]
clusterDF2= dataDF.loc[dataDF['cluster']==1]

#Using different graphs to identify the relation between the flow types and rttmin values
Throughput=dataDF['throughput'].values
flowspermin= dataDF['flows_permin'].values

#Greater the number of flows permin, higher is throughput
plt.plot(Throughput,flowspermin)
plt.xlabel('Throughput')
plt.ylabel('flowspermin')
plt.show()

features1=['throughput', 'flows_permin']
regDF=dataDF[features1]
#Performing Logistic Regression
from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
regDF=datasets.load_iris()
x,y= regDF.data, regDF.target
#y=reg1DF.drop(['throughput'], axis=1)
#x_data=regDF.drop(['flows_permin'], axis=1)
#x = (x_data -np.min(x_data))/(np.max(x_data)-np.min(x_data)).values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0, random_state=0)

x_train = x_train.T
x_test = x_test.T
y_train = y_train.T
y_test = y_test.T

print("x train: ",x_train.shape)
print("x test: ",x_test.shape)
print("y train: ",y_train.shape)
print("y test: ",y_test.shape)

logreg = LogisticRegression()
logreg.fit(x_train, y_train)
pred = logreg.predict(x_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(x_test, y_test)))

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, logreg.predict(x_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(x_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()