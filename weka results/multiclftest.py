# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 00:00:17 2017

@author: zjgsw
"""

from sklearn.cross_validation import StratifiedShuffleSplit
import pandas as pd
import numpy as np
from sklearn import svm    
from sklearn.model_selection import train_test_split
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
df = pd.read_csv('event3stage_tfidf_250features.csv',header = 0)

#print (df)

X = df.iloc[:,1:]
y = df.iloc[:,0]
#y = label_binarize(y, classes=[0, 1, 2])
X = np.array(X)
print (y)
k = []
for i in y:
    if i == 'prescribe/order':
        k.append(0)
    elif i == 'transcribe/prepare/dispense':
        k.append(1)
    elif i == 'administer/monitor':
        k.append(2)
print (X.shape)
k =np.array(k)
X_train, X_test, k_train, k_test = train_test_split(X, k, test_size=.2)

clf_names = ["SVM RBF","Decision Tree","Random Forest"]

clfs = [SVC(kernel="poly",gamma=0.1,C = 1000), DecisionTreeClassifier(),
        RandomForestClassifier(max_depth = 5,max_features = 'sqrt', 
                               n_estimators = 10, random_state = 42)]
        
for clf in clfs:
    clf.fit(X_train, k_train)
    predictions = clf.predict(X_test)
    f1 = metrics.f1_score(k_test,predictions,labels=[0,1,2],average='weighted')
    print(predictions)
    print(k_test)
    print(f1)
    print(clf.score(X_test,k_test))