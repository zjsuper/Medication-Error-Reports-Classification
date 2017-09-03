#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 14:48:22 2017

@author: SichengZhou
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
clf_names = ["Naive Bayes","SVM RBF","Decision Tree","KMeans",
             "Random Forest"]

#clfs = [sklearn.naive_bayes.GaussianNB(),svm.SVC(kernel="rbf",gamma=0.1,C = 1000),
        #sklearn.ensemble.RandomForestClassifier(max_depth = 5,max_features = 'sqrt', 
           #                    n_estimators = 10, random_state = 42)]
clf = svm.SVC(kernel='rbf', decision_function_shape='ovo',probability=False,random_state = random_state)
select = SelectKBest(k=240)
steps = [('feature_selection', select),('svm', clf)]

pipeline = pipeline.Pipeline(steps)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=random_state)

pipeline.fit( X_train, y_train )
#y_prediction = pipeline.predict( X_test )
target_names = ['class 0', 'class 1', 'class 2']
#report = classification_report( y_test, y_prediction, target_names=target_names)
#print(report)
svm__C=[0.5,1, 2, 4,8,16]
svm__gamma=[2**-15,2**-13,2**-11,2**-9,2**-7,2**-5,2**-3]
parameters = dict(feature_selection__k=[100, 200,240], 
              svm__C=[0.5,1, 2, 4,8,16],
              svm__gamma=[2**-15,2**-13,2**-11,2**-9,2**-7,2**-5,2**-3],svm__kernel=['rbf','poly'])

cv = sklearn.grid_search.GridSearchCV(pipeline, param_grid=parameters)

cv.fit(X_train, y_train)
y_predictions = cv.predict(X_test)
report = sklearn.metrics.classification_report( y_test, y_predictions ,target_names=target_names)

print(report)
print("Best estimator found by grid search:")
print(cv.best_estimator_)
print (cv.cv_results_)
print (cv.best_params_)







#def plot_grid_search(cv_results, grid_param_1, grid_param_2, name_param_1, name_param_2):
#
## Get Test Scores Mean and std for each grid search
#    scores_mean = cv_results['mean_test_score']
#    scores_mean = np.array(scores_mean).reshape(len(grid_param_2),len(grid_param_1))
#
#    scores_sd = cv_results['std_test_score']
#    scores_sd = np.array(scores_sd).reshape(len(grid_param_2),len(grid_param_1))
#
## Plot Grid search scores
#    _, ax = plt.subplots(1,1)
#
## Param1 is the X-axis, Param 2 is represented as a different curve (color line)
#    for idx, val in enumerate(grid_param_2):
#        ax.plot(grid_param_1, scores_mean[idx,:], '-o', label= name_param_2 + ': ' + str(val))
#
#    ax.set_title("Grid Search Scores", fontsize=20, fontweight='bold')
#    ax.set_xlabel(name_param_1, fontsize=16)
#    ax.set_ylabel('CV Average Score', fontsize=16)
#    ax.legend(loc="best", fontsize=15)
#    ax.grid('on')
#
## Calling Method 
#plot_grid_search(cv.cv_results_, svm__C, svm__gamma, 'C', 'gamma')