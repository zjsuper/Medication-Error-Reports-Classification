# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 01:34:28 2017

@author: zjgsw
"""
from sklearn.cross_validation import StratifiedShuffleSplit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
import sklearn


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

#def test_classifier(clf, labels, features, folds = 1000):
#    #data = featureFormat(dataset, feature_list, sort_keys = True)
#    #labels, features = targetFeatureSplit(data)
#    cv = StratifiedShuffleSplit(labels, folds, random_state = 42)
#    true_negatives = 0
#    false_negatives = 0
#    true_positives = 0
#    false_positives = 0
#    for train_idx, test_idx in cv: 
#        features_train = []
#        features_test  = []
#        labels_train   = []
#        labels_test    = []
#        for ii in train_idx:
#            features_train.append( features[ii] )
#            labels_train.append( labels[ii] )
#        for jj in test_idx:
#            features_test.append( features[jj] )
#            labels_test.append( labels[jj] )
#        
#        ### fit the classifier using training set, and test on test set
#        clf.fit(features_train, labels_train)
#        predictions = clf.predict(features_test)
#        for prediction, truth in zip(predictions, labels_test):
#            if prediction == 0 and truth == 0:
#                true_negatives += 1
#            elif prediction == 0 and truth != 1:
#                false_negatives += 1
#            elif prediction == 1 and truth == 0:
#                false_positives += 1
#            elif prediction == 1 and truth == 1:
#                true_positives += 1
#            else:
#                print ("Warning: Found a predicted label not == 0 or 1.")
#                print ("All predictions should take value 0 or 1.")
#                print ("Evaluating performance for processed predictions:")
#                break
#    try:
#        total_predictions = true_negatives + false_negatives + false_positives + true_positives
#        accuracy = 1.0*(true_positives + true_negatives)/total_predictions
#        precision = 1.0*true_positives/(true_positives+false_positives)
#        recall = 1.0*true_positives/(true_positives+false_negatives)
#        f1 = 2.0 * true_positives/(2*true_positives + false_positives+false_negatives)
#        f2 = (1+2.0*2.0) * precision*recall/(4*precision + recall)
#        print (clf)
#        print (PERF_FORMAT_STRING.format(accuracy, precision, recall, f1, f2, display_precision = 5))
#        print (RESULTS_FORMAT_STRING.format(total_predictions, true_positives, false_positives, false_negatives, true_negatives))
#        print ("")
#    except:
#        print ("Got a divide by zero when trying out:", clf)
#        print ("Precision or recall may be undefined due to a lack of true positive predicitons.")
#





#k = label_binarize(k, classes=[0, 1, 2])
n_classes = 3
random_state = np.random.RandomState(0)                                    
X_train, X_test, k_train, k_test = train_test_split(X, k, test_size=.20, random_state=0)

#classifier = OneVsRestClassifier(svm.libsvm(kernel='sigmoid', probability=1))
#k_score = classifier.fit(X_train, k_train).decision_function(X_test)
clf = svm.libsvm()
clf.fit()
clf.predict(k_test)





#test_classifier(clf,k,X)
#k_score = clf.fit(X_train, k_train).decision_function(X_test)
# Compute ROC curve and ROC area for each class
#fpr = dict()
#tpr = dict()
#roc_auc = dict()
#for i in range(n_classes):
#    fpr[i], tpr[i], _ = roc_curve(k_test[:, i], k_score[:, i])
#    roc_auc[i] = auc(fpr[i], tpr[i])
#
## Compute micro-average ROC curve and ROC area
#fpr["micro"], tpr["micro"], _ = roc_curve(k_test.ravel(), k_score.ravel())
#roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
#
#plt.figure()
#lw = 2
#plt.plot(fpr[2], tpr[2], color='darkorange',
#         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
#plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
#plt.xlim([0.0, 1.0])
#plt.ylim([0.0, 1.05])
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
#plt.title('Receiver operating characteristic example')
#plt.legend(loc="lower right")
#plt.show()
#
## Compute macro-average ROC curve and ROC area
#
## First aggregate all false positive rates
#all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
#
## Then interpolate all ROC curves at this points
#mean_tpr = np.zeros_like(all_fpr)
#for i in range(n_classes):
#    mean_tpr += interp(all_fpr, fpr[i], tpr[i])
#
## Finally average it and compute AUC
#mean_tpr /= n_classes
#
#fpr["macro"] = all_fpr
#tpr["macro"] = mean_tpr
#roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
#
## Plot all ROC curves
#plt.figure()
#plt.plot(fpr["micro"], tpr["micro"],
#         label='micro-average ROC curve (area = {0:0.2f})'
#               ''.format(roc_auc["micro"]),
#         color='deeppink', linestyle=':', linewidth=4)
#
#plt.plot(fpr["macro"], tpr["macro"],
#         label='macro-average ROC curve (area = {0:0.2f})'
#               ''.format(roc_auc["macro"]),
#         color='navy', linestyle=':', linewidth=4)
#
#colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
#for i, color in zip(range(n_classes), colors):
#    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
#             label='ROC curve of class {0} (area = {1:0.2f})'
#             ''.format(i, roc_auc[i]))
#
#plt.plot([0, 1], [0, 1], 'k--', lw=lw)
#plt.xlim([0.0, 1.0])
#plt.ylim([0.0, 1.05])
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
#plt.title('Some extension of Receiver operating characteristic to multi-class')
#plt.legend(loc="lower right")
#plt.show()
#
#from sklearn.metrics import precision_recall_curve
#from sklearn.metrics import average_precision_score
## Compute Precision-Recall and plot curve
#precision = dict()
#recall = dict()
#average_precision = dict()
#for i in range(n_classes):
#    precision[i], recall[i], _ = precision_recall_curve(k_test[:, i],
#                                                        k_score[:, i])
#    average_precision[i] = average_precision_score(k_test[:, i], k_score[:, i])
#
## Compute micro-average ROC curve and ROC area
#precision["micro"], recall["micro"], _ = precision_recall_curve(k_test.ravel(),
#    k_score.ravel())
#average_precision["micro"] = average_precision_score(k_test, k_score,
#                                                     average="micro")
#
#
## Plot Precision-Recall curve
#plt.clf()
#plt.plot(recall[0], precision[0], lw=lw, color='navy',
#         label='Precision-Recall curve')
#plt.xlabel('Recall')
#plt.ylabel('Precision')
#plt.ylim([0.0, 1.05])
#plt.xlim([0.0, 1.0])
#plt.title('Precision-Recall example: AUC={0:0.2f}'.format(average_precision[0]))
#plt.legend(loc="lower left")
#plt.show()
#
## Plot Precision-Recall curve for each class
#plt.clf()
#plt.plot(recall["micro"], precision["micro"], color='gold', lw=lw,
#         label='micro-average Precision-recall curve (area = {0:0.2f})'
#               ''.format(average_precision["micro"]))
#for i, color in zip(range(n_classes), colors):
#    plt.plot(recall[i], precision[i], color=color, lw=lw,
#             label='Precision-recall curve of class {0} (area = {1:0.2f})'
#                   ''.format(i, average_precision[i]))
#
#plt.xlim([0.0, 1.0])
#plt.ylim([0.0, 1.05])
#plt.xlabel('Recall')
#plt.ylabel('Precision')
#plt.title('Extension of Precision-Recall curve to multi-class')
#plt.legend(loc="lower right")
#plt.show()