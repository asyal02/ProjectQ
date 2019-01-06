#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 09:12:53 2018

@author: alazar
"""

import pandas as pd
import datetime
import tarfile
import numpy as np 
from io import StringIO
from sklearn.cluster import KMeans
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.spatial.distance
from scipy.optimize import linear_sum_assignment

tar = tarfile.open("../data/esnet/tstat/data/tstat-nersc-160922.tar.gz", "r:gz")
fileName = "tstat_nersc_globus_extended.csv"

f = tar.extractfile(fileName)
Data = f.read()
newDF = pd.read_csv(StringIO(Data.decode('utf-8')),sep=',',header='infer')
del Data
newDF = newDF[newDF['throughput'] != 0.0]
newDF['first_time'] = pd.to_datetime(newDF['first_time'])
#newDF['last_time'] = pd.to_datetime(newDF['last_time'])

dateFrom = datetime.datetime(2016, 4, 16, 8, 30, 0)
dateTo = datetime.datetime(2016, 5, 16, 8, 30, 0)

newDF=newDF.set_index(pd.DatetimeIndex(newDF['first_time'])).sort_index()

selDF=newDF.set_index(pd.DatetimeIndex(newDF['first_time'])).sort_index().loc[dateFrom:dateTo]

del newDF

selDF=selDF[['first_time','pkts_all','throughput']]

summary = selDF.groupby(pd.Grouper(freq='30Min')).count()

summary.mean()

selDF['pkts_all'] = selDF['pkts_all']+1
selDF['throughput'] = selDF['throughput']+1

selDF['pkts_all'] = selDF['pkts_all'].apply(np.log10)
selDF['throughput'] = selDF['throughput'].apply(np.log10)

max1 = selDF['pkts_all'].max()
max2 = selDF['throughput'].max()

min1 = selDF['pkts_all'].min()
min2 = selDF['throughput'].min()

grouped = selDF.groupby(pd.Grouper(freq='1D'))

timeArray = grouped.agg({'first_time' : 'first'})['first_time']
timeArray=timeArray.reset_index(drop=True)
timeArray=timeArray.values #.AddMinutes(-1)


avgThrough = grouped['throughput'].mean()
resultHist=plt.hist(avgThrough, bins='auto') 

verySmall = avgThrough[avgThrough < resultHist[1][1]]

mpl.style.use('seaborn')
f, axarr = plt.subplots(1, 1, sharex=True)
f.set_size_inches(24, 4)

axarr.plot(avgThrough, 'bs-', markersize=1, linewidth=1, label='Mean Throughput')
axarr.plot(verySmall, 'rs-', markersize=5, linestyle="None")
#axarr[0].set_title('npkt KS test statistic')
axarr.text(.5,.8,'Mean Throughput', horizontalalignment='center', transform=axarr[0].transAxes)
axarr.set_ylim([0,1.2])
axarr.yaxis.set_ticks((0,0.5,1.0))




pairsTime = zip(timeArray, timeArray[1:])

groupList = map(lambda (x,y) : selDF.loc[x:y], pairsTime)

groupListP = map(lambda x : x[['pkts_all','throughput']].values, groupList)

avgT = map(lambda x : np.mean(x[:,1]), groupListP)
minT = map(lambda x : min(x[:,1]), groupListP)
maxT = map(lambda x : max(x[:,1]), groupListP)

pairsKs = zip(groupListP,groupListP[1:])

ksListP = map(lambda (x,y) : stats.ks_2samp(x[:,0],y[:,0]), pairsKs)
ksListT = map(lambda (x,y) : stats.ks_2samp(x[:,1],y[:,1]), pairsKs)
pkt_s = map(lambda x : x[1],ksListP)
tput_s = map(lambda x : x[1],ksListT)

kmeansLabels = [None] * len(groupListP)
kmeansCentroids=[None] * len(groupListP)
i=0
for x in groupListP:
    if i==0:
        kmeansRes = KMeans(n_clusters=4, random_state=0).fit(x)
    else:
        kmeansRes = KMeans(n_clusters=4, random_state=0,init=kmeansCentroids[i-1]).fit(x)
    kmeansLabels[i]=kmeansRes.labels_
    kmeansCentroids[i]=kmeansRes.cluster_centers_
    i = i+1


#kmeansRes = map(lambda x : KMeans(n_clusters=4, random_state=0).fit(x),groupListP)
#kmeansLabels = map(lambda x : x.labels_,kmeansRes)
#kmeansCentroids = map(lambda x : x.cluster_centers_,kmeansRes)


pairsCentroids = zip(kmeansCentroids,kmeansCentroids[1:])

cost =  map(lambda x : scipy.spatial.distance.cdist(x[0],x[1]),pairsCentroids)
ind = map(lambda x : linear_sum_assignment(x), cost) 

reorderP = zip(pairsCentroids,ind)
reorderedCenters = map(lambda (x,y) : x[1][y[1]],reorderP)
pairsCentroids = zip(kmeansCentroids,reorderedCenters)

delta = map(lambda (x,y) : np.linalg.norm(x-y), pairsCentroids)



pairsDL = zip(groupList,kmeansLabels)
toPlot = list(map(lambda (x,y): x.assign(labelCol=y),pairsDL))

colors = ['red','green','blue','purple']


fig, axes = plt.subplots(4, 4, sharex='all', sharey='all',figsize=(15,15))
axesL=list(np.array(axes).reshape(-1,))
pairsPlot = zip(axesL,toPlot[0:16])
plotsArray = list(map(lambda (x,y): x.scatter(x='pkts_all', y='throughput',data=y,c='labelCol',\
                      cmap=mpl.colors.ListedColormap(colors),alpha=0.5),pairsPlot))
fig

index=range(47)
pkt_s = np.insert(pkt_s, 0, 0)
tput_s = np.insert(tput_s, 0, 0)
delta = np.insert(delta, 0, 0)
# Just a figure and one subplot
f, axarr = plt.subplots(4, 1, sharex=True)
#my_xticks = ['(1,2)', '', '(3,4)', '', '(5,6)', '', '(7,8)', '', '(9,10)', '', '(11,12)', '', '(13,14)', '', '(15,16)']
#plt.xticks(index, my_xticks, fontsize=13) #, rotation=45)

f.set_size_inches(24, 12)

#axarr[0].plot( pkt_p, label='pvalue of npkt')
#axarr[0].set_title('npkt p-value')
#axarr[1].plot( tput_p, label='pvalue of tput')
#axarr[1].set_title('tput p-value')

axarr[0].plot( pkt_s, 'ks-', markersize=10, linewidth=2, label='statistics of npkt')
#axarr[0].set_title('npkt KS test statistic')
axarr[0].text(.5,.8,'(a) npkt KS test statistic', horizontalalignment='center', transform=axarr[0].transAxes)
axarr[0].set_ylim([0,1.2])
axarr[0].axhline(np.mean(pkt_s),c='green',linewidth=0.5,zorder=0)
axarr[0].yaxis.set_ticks((0,0.5,1.0))

axarr[1].plot( tput_s, 'bD-', markersize=10, linewidth=2, label='statistics of tput')
#axarr[1].set_title('max_tput KS test statistic')
axarr[1].text(.5,.8,'(b) tput KS test statistic', horizontalalignment='center', transform=axarr[1].transAxes)
axarr[1].set_ylim([0,.2])
axarr[1].axhline(np.mean(tput_s),c='green',linewidth=0.5,zorder=0)
axarr[1].yaxis.set_ticks((0,0.05,.2))

axarr[2].plot( avgT, 'bD-', markersize=10, linewidth=2, label='statistics of tput')
axarr[2].plot( minT, 'rD-', markersize=10, linewidth=2)
axarr[2].plot( maxT, 'gD-', markersize=10, linewidth=2)
#axarr[1].set_title('max_tput KS test statistic')
axarr[2].text(.5,.8,'(c) tput statistics', horizontalalignment='center', transform=axarr[2].transAxes)
axarr[2].set_ylim([0,.1])
axarr[2].yaxis.set_ticks((0,0.05,.1))

axarr[3].plot( delta, 'ro-', markersize=10, linewidth=2, label='degree of change')
#axarr[2].set_title('degree of change')
axarr[3].text(.5,.8,'(d) degree of changes', horizontalalignment='center', transform=axarr[3].transAxes)
axarr[3].set_ylim([0,1.4])
axarr[3].axhline(np.mean(delta),c='green',linewidth=0.5,zorder=0)
axarr[3].yaxis.set_ticks(np.arange(0,.1,1.4))

#axarr[3].plot( delta_median, 'r*-', markersize=10, linewidth=2, label='degree of change')
##axarr[3].set_title('degree of changes')
#axarr[3].text(.5,.8,'(d) degree of change (median of multiple runs)', horizontalalignment='center', transform=axarr[3].transAxes)
#axarr[3].set_ylim([0,9])
#axarr[3].axhline(np.mean(delta_median),c='green',linewidth=0.5,zorder=0)
#axarr[3].yaxis.set_ticks(np.arange(0,8,3))

#plt.legend()
#plt.show()
#fig.tight_layout()
f.savefig('result_graph_subplot.png', bbox_inches='tight')