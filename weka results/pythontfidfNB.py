#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 14:11:37 2017

@author: SichengZhou
"""

from __future__ import print_function
import numpy as np
#from numpy import toarray
from pprint import pprint
from time import time
import logging
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_selection import SelectKBest

print(__doc__)

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

categories = ['prescribe/order','transcribe/prepare/dispense',
              'administer/monitor']


df = pd.read_csv('full.csv')

detail = df['detail']
stage = df['eventoriginatestage']
#print(detail)

vec = CountVectorizer(max_df =1,lowercase = True,max_features=50000,
                         ngram_range = (1,3))
de_vec = vec.fit_transform(detail)
#print (de_vec)
print (de_vec.shape)
tfidf= TfidfTransformer(use_idf = True,norm = 'l2')
de_vec_tfidf = tfidf.fit_transform(de_vec)

de_vec_tfidf = de_vec_tfidf.toarray()
print (de_vec_tfidf.shape)
X = de_vec_tfidf

from sklearn.decomposition import PCA

#selectk = SelectKBest(k=100)
pca_k = PCA(n_components=500)

y = []
for i in stage.iloc[:]:
    if i == 'prescribe/order':
        y.append(0)
    elif i == 'transcribe/prepare/dispense':
        y.append(1)
    elif i == 'administer/monitor':
        y.append(2)
#print (type(de_vec_tfidf))

#X_select = selectk.fit_transform(X,y)
X_pca = pca_k.fit_transform(X,y)

X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y, test_size=0.3, random_state=0)

print(X_pca.shape)
pipeline = Pipeline([
#    ('vect', CountVectorizer()),
#    ('tfidf', TfidfTransformer()),
#    ('toarray',toarray()),
   ('clf', GaussianNB())])
    
tuned_parameters = [{'priors': [None]}]


scores = ['precision', 'recall']
#
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
# uncommenting more parameters will give better exploring power but will
# increase processing time in a combinatorial way
#parameters = {
#    'vect__max_df': ([1.0]),
#    'vect__lowercase':([True]),
#    'vect__max_features': (None, 10000, 30000),
#    'vect__ngram_range': ((1, 1), (1, 2),(1,3)),  # unigrams or bigrams
#    'tfidf__use_idf': ([True]),
#    'tfidf__norm': ('l1', 'l2'),
#    'clf__C': (1, 2, 4,8,10),
#    'clf__gamma': (2**-8,2**-7,2**-6,2**-5,2**-4),
#    'clf__kernel': ('rbf','poly')
#}
parameters = {
    #'vect__max_df': ([1.0]),
    #'vect__lowercase':([True]),
    #'vect__max_features': (None, 10000, 30000),
    #'vect__ngram_range': ((1, 1), (1, 2),(1,3)),  # unigrams or bigrams
    #'tfidf__use_idf': ([True]),
    #'tfidf__norm': ('l1', 'l2'),
    'clf__priors': ([0.25,0.5,0.25])
}

#if __name__ == "__main__":
#    # multiprocessing requires the fork to happen in a __main__ protected
#    # block
#
#    # find the best parameters for both the feature extraction and the
#    # classifier
#    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1,
#                               cv=StratifiedShuffleSplit(n_splits=10, 
#                                                         test_size=0.25, 
#                                                         random_state=0))
#
#    print("Performing grid search...")
#    print("pipeline:", [name for name, _ in pipeline.steps])
#    print("parameters:")
#    pprint(parameters)
#    t0 = time()
#    grid_search.fit(de_vec_tfidf, stage)
#    print("done in %0.3fs" % (time() - t0))
#    print(grid_search.cv_results_)
#
#    print("Best score: %0.3f" % grid_search.best_score_)
#    print("Best parameters set:")
#    best_parameters = grid_search.best_estimator_.get_params()
#    for param_name in sorted(parameters.keys()):
#        print("\t%s: %r" % (param_name, best_parameters[param_name]))