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

def file_read(dataPath,fileName):
    listRec_ = []
    fileExt='.all.csv.tar.gz'
    tar = tarfile.open(dataPath+fileName+fileExt, "r:gz")
    f = tar.extractfile(fileName+'.all.csv')
    Data = f.read()
    newDF = pd.read_csv(StringIO(Data.decode('utf-8')),sep=',',header='infer')
    newDF = newDF.dropna(axis='rows')
    newDF['cluster'] = 1
    listRec_.append(newDF)
    newDF= pd.concat(listRec_)
    f.close()
    tar.close()
    return newDF
    
def main():
    np.set_printoptions(suppress=True,formatter={'float_kind':'{0.4f}'.format})
    dataPath = r'C:\\Users\\asthasyal\\Documents\\TstatProject\\data\\'
    outpath=r'C:\\Users\\asthasyal\\Documents\\TstatProject\\result\\scaleddata.txt'
    fileName = 'tstat.dtn05.nersc.gov.testb'
    workDF = file_read(dataPath,fileName)
    list(workDF)

features=['c_pkts_all:3', 'c_rst_cnt:4','c_ack_cnt:5', 'c_ack_cnt_p:6','c_bytes_uniq:7', 'c_pkts_data:8', 'c_bytes_all:9', \
          'c_pkts_retx:10', 'c_bytes_retx:11', 'c_pkts_ooo:12', 'c_syn_cnt:13','c_fin_cnt:14', 's_port:16', 's_pkts_all:17', \
          's_rst_cnt:18','s_ack_cnt:19', 's_ack_cnt_p:20', 's_bytes_uniq:21', 's_pkts_data:22','s_bytes_all:23', \
          's_pkts_retx:24','s_bytes_retx:25',  's_pkts_ooo:26', 's_syn_cnt:27', 's_fin_cnt:28','first:29', 'last:30', 'durat:31',
          'c_first:32', 's_first:33', 'c_last:34','s_last:35', 'c_first_ack:36', 's_first_ack:37','c_isint:38',\
          's_isint:39', 'c_iscrypto:40','s_iscrypto:41','con_t:42',  'p2p_t:43','http_t:44','c_rtt_avg:45','c_rtt_min:46','c_rtt_max:47', \
          'c_rtt_std:48', 'c_rtt_cnt:49', 'c_ttl_min:50', 'c_ttl_max:51', 's_rtt_avg:52', 's_rtt_min:53', \
          's_rtt_max:54', 's_rtt_std:55', 's_rtt_cnt:56', 's_ttl_min:57', 's_ttl_max:58','c_f1323_opt:59','c_tm_opt:60', \
           'c_win_scl:61','c_sack_opt:62','c_sack_cnt:63', 'c_mss:64', 'c_mss_max:65', 'c_mss_min:66', 'c_win_max:67', 'c_win_min:68', \
          'c_win_0:69','c_cwin_max:70', 'c_cwin_min:71', 'c_cwin_ini:72', 'c_pkts_rto:73','c_pkts_fs:74', 'c_pkts_reor:75', \
          'c_pkts_dup:76','c_pkts_unk:77', 'c_pkts_fc:78', 'c_pkts_unrto:79', 'c_pkts_unfs:80', 'c_syn_retx:81', 's_f1323_opt:82',\
          's_tm_opt:83','s_win_scl:84', 's_sack_opt:85', 's_sack_cnt:86', 's_mss:87', 's_mss_max:88', \
          's_mss_min:89', 's_win_max:90', 's_win_min:91', 's_win_0:92','s_cwin_max:93', 's_cwin_min:94','s_cwin_ini:95',\
          's_pkts_rto:96', 's_pkts_fs:97', 's_pkts_reor:98', 's_pkts_dup:99', 's_pkts_unk:100', 's_pkts_fc:101',\
          's_pkts_unrto:102', 's_pkts_unfs:103', 's_syn_retx:104', 'cluster' ]

dataDF=workDF[features]
scaler = MinMaxScaler()
scaler.fit(dataDF) 
scaleDF=scaler.transform(dataDF)

X = scaleDF
y_pred = pd.DataFrame(KMeans(n_clusters=2, random_state=10).fit_predict(X))
y_pred.columns=['cluster']
n_neighbors = 10
n_components = 2

scaleDF[:,101]= y_pred['cluster']
type(scaleDF)
type(y_pred)
scaledataframe= pd.DataFrame(data=scaleDF,columns= features)
scaleDF1= scaledataframe.loc[scaledataframe['cluster']==0]
scaleDF2=scaledataframe.loc[scaledataframe['cluster']==1]
summaryDF1=scaleDF1.describe()
summaryDF2=scaleDF2.describe()
outpath1=r'C:\\Users\\asthasyal\\Documents\\TstatProject\\result\\allcolsDF1testb.txt'
outpath2=r'C:\\Users\\asthasyal\\Documents\\TstatProject\\result\\allDF2testb.txt'
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
