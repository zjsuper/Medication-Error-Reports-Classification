# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 16:50:20 2017

@author: zjgsw
"""

from sklearn import pipeline
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_selection import SelectKBest
import sklearn.grid_search

print (sklearn.__version__)
df = pd.read_csv('event3stage_tfidf_250features.csv',header = 0)
#print (df)

X = df.iloc[:,1:]
k = df.iloc[:,0]
#y = label_binarize(y, classes=[0, 1, 2])
X = np.array(X)
#X_scaled = preprocessing.scale(X)

#print (y)
y = []
for i in k:
    if i == 'prescribe/order':
        y.append(0)
    elif i == 'transcribe/prepare/dispense':
        y.append(1)
    elif i == 'administer/monitor':
        y.append(2)
y=np.array(y)
print (X.shape,y.shape[0])

print (y)
random_state = np.random.RandomState(42) 

#clf = svm.SVC(kernel='rbf', decision_function_shape='ovo',probability=False,random_state = random_state)

clf = svm.libsvm.fit(X,y)
