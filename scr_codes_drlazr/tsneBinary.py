# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 10:12:20 2018

@author: alazar
"""


import pandas as pd
#import numpy as np
from sklearn.preprocessing import MinMaxScaler
from time import time
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#from matplotlib.ticker import NullFormatter
import seaborn as sns
from sklearn import manifold, datasets
from sklearn.cluster import KMeans
#from matplotlib.colors import LinearSegmentedColormap

path = '../data/labeledData/'
fileNames = ['logtcpexp1.csv','logtcpexp2.csv','logtcpexp3.csv','logtcpexp4.csv',
             'logtcpexp5.csv','logtcpexp6.csv','logtcpexp7.csv','logtcpexp8.csv']
classesM = ['normal','no-flow','loss-1%','loss-5%',
           'duplicate-1%','duplicate-5%',
           'reordering25%-50%','reordering50%-50%']
classesB = ['normal','abnormal']

frame = pd.DataFrame()
listRec_ = []
for file_ in fileNames:
    df = pd.read_csv(path+file_,sep=',',index_col=None, header='infer',encoding="utf-8")
    df['label'] = fileNames.index(file_)
    listRec_.append(df)
dataDF = pd.concat(listRec_)
list(dataDF)

features = ['Average rtt C2S', 'Average rtt S2C','max seg size.1','min seg size.1', \
            'win max.1','win min.1','win zero.1','cwin max.1','cwin min.1', \
            'initial cwin.1','rtx RTO.1','rtx FR.1','reordering.1','net dup.1', \
            'max seg size','min seg size','win max','win min','win zero', \
            'cwin max','cwin min','initial cwin','rtx RTO', 'rtx FR','reordering','net dup']

features = ['Average rtt C2S', 'Average rtt S2C','max seg size.1','min seg size.1',\
            'win max.1','rtx RTO.1','rtx FR.1','reordering.1','net dup.1', \
            'max seg size','min seg size','win max',\
            'cwin max','cwin min','initial cwin','rtx RTO', 'rtx FR','reordering','net dup']

color = dataDF['label']
color.loc[color>=1]='No'
color.loc[color==0]='Yes'
color=pd.DataFrame(color)
color = color.reset_index(drop=True)

datadf = dataDF[features]
scaler = MinMaxScaler()
scaler.fit(datadf) 
scaleDF=scaler.transform(datadf)

X = scaleDF
y_pred = pd.DataFrame(KMeans(n_clusters=2, random_state=10).fit_predict(X))
y_pred.columns=['cluster']
n_neighbors = 10
n_components = 2

t0 = time()
Y = manifold.Isomap(n_neighbors, n_components).fit_transform(X)
t1 = time()
print("Isomap: %.2g sec" % (t1 - t0))
#ax = fig.add_subplot(151)
#plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=myCmap,label=classes,alpha=0.5, edgecolors='none')
Y=pd.DataFrame(Y)
Y.columns = ['x', 'y']
Y=pd.concat([Y,color], axis=1)
Y=pd.concat([Y,y_pred], axis=1)
myp=sns.lmplot(data=Y, x='x', y='y',hue='label',palette="Set1",markers=["o", "x"],
                 fit_reg=False, legend=True, legend_out=True)
new_title = 'Normal'
myp._legend.set_title(new_title)
plt.title("Isomap (%.2g sec)" % (t1 - t0))
#ax.xaxis.set_major_formatter(NullFormatter())
#ax.yaxis.set_major_formatter(NullFormatter())
plt.axis('tight')
#fig = myp.get_figure()
myp.savefig('dimRed.pdf') 

t0 = time()
mds = manifold.MDS(n_components, max_iter=100, n_init=1)
Y = mds.fit_transform(X)
t1 = time()
print("MDS: %.2g sec" % (t1 - t0))
#ax = fig.add_subplot(152)
#plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=myCmap,label=classes,alpha=0.5, edgecolors='none')
Y=pd.DataFrame(Y)
Y.columns = ['x', 'y']
Y=pd.concat([Y,color], axis=1)
Y=pd.concat([Y,y_pred], axis=1)
myp=sns.lmplot(data=Y, x='x', y='y',hue='label',palette="Set1",markers=["o", "x"],
                 fit_reg=False, legend=True, legend_out=True)
# title
new_title = 'Normal'
myp._legend.set_title(new_title)
plt.title("MDS (%.2g sec)" % (t1 - t0))
#ax.xaxis.set_major_formatter(NullFormatter())
#ax.yaxis.set_major_formatter(NullFormatter())
plt.axis('tight')
#fig = myp.get_figure()
myp.savefig('dimRed.pdf') 

t0 = time()
se = manifold.SpectralEmbedding(n_components=n_components,
                                n_neighbors=n_neighbors)
Y = se.fit_transform(X)
t1 = time()
print("SpectralEmbedding: %.2g sec" % (t1 - t0))
#ax = fig.add_subplot(153)
#plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=myCmap,label=classes,alpha=0.5, edgecolors='none')
Y=pd.DataFrame(Y)
Y.columns = ['x', 'y']
Y=pd.concat([Y,color], axis=1)
Y=pd.concat([Y,y_pred], axis=1)
myp=sns.lmplot(data=Y, x='x', y='y',hue='label',palette="Set1",markers=["o", "x"],
                 fit_reg=False, legend=True, legend_out=True)
# title
new_title = 'Normal'
myp._legend.set_title(new_title)
plt.title("SpectralEmbedding (%.2g sec)" % (t1 - t0))
#ax.xaxis.set_major_formatter(NullFormatter())
#ax.yaxis.set_major_formatter(NullFormatter())
plt.axis('tight')
#fig = myp.get_figure()
myp.savefig('dimRed.pdf') 

t0 = time()
tsne = manifold.TSNE(n_components=n_components, init='pca', random_state=0)
Y = tsne.fit_transform(X)
t1 = time()
print("t-SNE: %.2g sec" % (t1 - t0))
#ax = fig.add_subplot(154)
#tsnep=plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=myCmap,alpha=0.5, edgecolors='none')
Y=pd.DataFrame(Y)
Y.columns = ['x', 'y']
Y=pd.concat([Y,color], axis=1)
Y=pd.concat([Y,y_pred], axis=1)
myp=sns.lmplot(data=Y, x='x', y='y',hue='label',palette="Set1",markers=["o", "x"],
                 fit_reg=False, legend=True, legend_out=True)
# title
new_title = 'Normal'
myp._legend.set_title(new_title)
plt.title("t-SNE (%.2g sec)" % (t1 - t0))
#ax.xaxis.set_major_formatter(NullFormatter())
#ax.yaxis.set_major_formatter(NullFormatter())
plt.axis('tight')
#plt.legend(title='Normal',scatterpoints=1,loc=9, bbox_to_anchor=(1.7, 1))
fig = myp.get_figure()
fig.savefig('dimRed.pdf') 

#ax = fig.add_subplot(155)
#plt.axis('off')
#plt.legend(title='labels')

#plt.show()

perplexities = (2, 5, 30, 50, 70, 90, 100)
for perp in perplexities:
    t0 = time()
    tsne = manifold.TSNE(n_components=n_components, init='pca', random_state=0, perplexity=perp)
    Y = tsne.fit_transform(X)
    t1 = time()
    print("t-SNE: %.2g sec" % (t1 - t0))
    #ax = fig.add_subplot(154)
    #tsnep=plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=myCmap,alpha=0.5, edgecolors='none')
    Y=pd.DataFrame(Y)
    Y.columns = ['x', 'y']
    Y=pd.concat([Y,color], axis=1)
    Y=pd.concat([Y,y_pred], axis=1)
    myp=sns.lmplot(data=Y, x='x', y='y',hue='label',palette="Set1",markers=["o", "x"],
                     fit_reg=False, legend=True, legend_out=True)
    # title
    new_title = 'Normal'
    myp._legend.set_title(new_title)
    plt.title("t-SNE (%.2g sec)" % (t1 - t0))
    #ax.xaxis.set_major_formatter(NullFormatter())
    #ax.yaxis.set_major_formatter(NullFormatter())
    plt.axis('tight')
    #fig = myp.get_figure()
    myp.savefig('dimRed.pdf') 