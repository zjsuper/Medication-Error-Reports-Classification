# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 00:32:49 2017

@author: zjgsw
"""

from __future__ import print_function
from sklearn import preprocessing
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
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
pca_k = PCA(n_components=247)
X_pca = pca_k.fit_transform(X,y)
n_samples = 3
parameters = dict(feature_selection__k=[200,247], 
              svm__C=[0.5,1, 2, 4,8,16],
              svm__gamma=[2**-11,2**-9,2**-7,2**-5,2**-3],svm__kernel=['rbf','poly'])

# Split the dataset in two equal parts
X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y, test_size=0.5, random_state=0)

# Set the parameters by cross-validation
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [2**-11,2**-9,2**-7,2**-5,2**-3],
                     'C': [1, 2, 4,8,16, 10]},
                    {'kernel': ['poly'], 'gamma': [2**-11,2**-9,2**-7,2**-5,2**-3],
                     'C': [1, 2, 4,8,16, 10]}]

scores = ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(SVC(), tuned_parameters, cv=StratifiedShuffleSplit(n_splits=10, test_size=0.25, random_state=0),
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