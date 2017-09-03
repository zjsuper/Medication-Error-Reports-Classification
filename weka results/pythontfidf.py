#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 14:11:37 2017

@author: SichengZhou
"""

from __future__ import print_function
import numpy as np
from numpy import toarray
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
print(__doc__)

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

categories = ['prescribe/order','transcribe/prepare/dispense',
              'administer/monitor']
# Uncomment the following to do the analysis on all the categories
#categories = None

print("Loading 20 newsgroups dataset for categories:")
print(categories)

#data = fetch_20newsgroups(subset='train', categories=categories)
#print("%d documents" % len(data.filenames))
#print("%d categories" % len(data.target_names))
#print(data)

df = pd.read_csv('full.csv')

detail = df['detail']
stage = df['eventoriginatestage']
print(detail)
pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('toarray',toarray()),
    ('clf', GaussianNB())
])

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
    'vect__max_df': ([1.0]),
    'vect__lowercase':([True]),
    'vect__max_features': (None, 10000, 30000),
    'vect__ngram_range': ((1, 1), (1, 2),(1,3)),  # unigrams or bigrams
    'tfidf__use_idf': ([True]),
    'tfidf__norm': ('l1', 'l2'),
    'clf__priors': ([0.25,0.5,0.25])
}

if __name__ == "__main__":
    # multiprocessing requires the fork to happen in a __main__ protected
    # block

    # find the best parameters for both the feature extraction and the
    # classifier
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1,
                               cv=StratifiedShuffleSplit(n_splits=10, 
                                                         test_size=0.25, 
                                                         random_state=0))

    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    pprint(parameters)
    t0 = time()
    grid_search.fit(detail, stage)
    print("done in %0.3fs" % (time() - t0))
    print(grid_search.cv_results_)

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))