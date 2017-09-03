#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 10:42:24 2017

@author: SichengZhou
"""

from __future__ import print_function
from sklearn import pipeline
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import sklearn.grid_search
from sklearn import preprocessing
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.naive_bayes import GaussianNB
print(__doc__)
df = pd.read_csv('event3stage_tfidf_250features.csv',header = 0)
#print (df)

X = df.iloc[:,1:]
k = df.iloc[:,0]
#y = label_binarize(y, classes=[0, 1, 2])
X = np.array(X)
X_scaled = preprocessing.scale(X)


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

from sklearn.decomposition import PCA

pca_k = PCA(n_components=200)
X_pca = pca_k.fit_transform(X,y)
# To apply an classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
n_samples = 3

# Split the dataset in two equal parts
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)

# Set the parameters by cross-validation
tuned_parameters = [{'priors':[[0.25,0.5,0.25]]}]

scores = ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(GaussianNB(), tuned_parameters, cv=StratifiedShuffleSplit(n_splits=10, test_size=0.25, random_state=0),
                       scoring='%s_macro' % score)
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()

# Note the problem is too easy: the hyperparameter plateau is too flat and the
# output model is the same for precision and recall with ties in quality.