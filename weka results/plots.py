# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 18:21:20 2017

@author: zjgsw
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from sklearn import cross_validation
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier

df = pd.read_csv('event3stage_tfidf_250features.csv',header = 0)
#print (df)

X = df.iloc[:,1:]
y = df.iloc[:,0]
#y = label_binarize(y, classes=[0, 1, 2])
X = np.array(X)
X_scaled = preprocessing.scale(X)

#print (y)
k = []
for i in y:
    if i == 'prescribe/order':
        k.append(0)
    elif i == 'transcribe/prepare/dispense':
        k.append(1)
    elif i == 'administer/monitor':
        k.append(2)
k=np.array(k)
print (X.shape,k.shape)

from sklearn import svm



n_classes = 3
random_state = np.random.RandomState(0)                                    
X_train, X_test, k_train, k_test = train_test_split(X_scaled, k, test_size=.25, random_state=0)

#clf = svm.SVC(kernel='rbf', gamma='auto', C = 2.0,decision_function_shape='ovo',probability=False,random_state = random_state)
clf = MLPClassifier(activation = 'tanh',max_iter = 2000, solver='sgd',
                    learning_rate_init=0.001)
clf.fit(X_train,k_train)
y_predicted = clf.predict(X_test)

print ("Classification report for %s" % clf)

print (metrics.classification_report(k_test, y_predicted))

print ("Confusion matrix")
print (metrics.confusion_matrix(k_test, y_predicted))

###with cross-validation
 # build a svm classifier
scores = cross_validation.cross_val_score(clf, X_scaled, k, cv = 10) # calculate the accuracy
scores2 = cross_validation.cross_val_score(clf, X_scaled, k, cv = 10,scoring='f1_macro') # calculate the f1
  
print ('clf2 res:\n')
print (scores.mean())
print ('clf2 f1:\n')
print (scores2.mean())
print(clf)


cv = ShuffleSplit(n_splits=10, test_size=0.25, random_state=0)
score3 = cross_val_score(clf, X_scaled, k, cv=cv)
 
print (score3)

import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier

from scipy import interp

#k = label_binarize(k, classes=[0, 1, 2])
X_train, X_test, k_train, k_test = train_test_split(X_scaled, k, test_size=.20, random_state=0)
n_classes = 3

classifier = OneVsOneClassifier(clf)
k_score = classifier.fit(X_train, k_train).decision_function(X_test)


k_train = label_binarize(k_train, classes=[0, 1, 2])
k_test = label_binarize(k_test, classes=[0, 1, 2])

fpr = dict()
tpr = dict()
roc_auc = dict()

#print (n_classes)
#print (k_test)
#print(k_score)
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(k_test[:, i], k_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(k_test.ravel(), k_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

#Plot of a ROC curve for a specific class
plt.figure()
lw = 2
plt.plot(fpr[2], tpr[2], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


# Compute macro-average ROC curve and ROC area

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()

#accuracy and recall 

import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize


color_ar = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])

# Compute Precision-Recall and plot curve
precision = dict()
recall = dict()
average_precision = dict()

for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(k_test[:, i],
                                                        k_score[:, i])
    average_precision[i] = average_precision_score(k_test[:, i], k_score[:, i])
    
# Compute micro-average ROC curve and ROC area
precision["micro"], recall["micro"], _ = precision_recall_curve(k_test.ravel(),
    k_score.ravel())
average_precision["micro"] = average_precision_score(k_test, k_score,
                                                     average="micro")
# Plot Precision-Recall curve
plt.clf()
plt.plot(recall[0], precision[0], lw=lw, color='navy',
         label='Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall example: AUC={0:0.2f}'.format(average_precision[0]))
plt.legend(loc="lower left")
plt.show()


# Plot Precision-Recall curve for each class
plt.clf()
plt.plot(recall["micro"], precision["micro"], color='gold', lw=lw,
         label='micro-average Precision-recall curve (area = {0:0.2f})'
               ''.format(average_precision["micro"]))
for i, color in zip(range(n_classes), color_ar):
    plt.plot(recall[i], precision[i], color=color, lw=lw,
             label='Precision-recall curve of class {0} (area = {1:0.2f})'
                   ''.format(i, average_precision[i]))

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Extension of Precision-Recall curve to multi-class')
plt.legend(loc="lower right")
plt.show()

from sklearn import linear_model, decomposition, datasets
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import numpy as np

pca = decomposition.PCA()
pipe = Pipeline(steps=[('pca', pca), ('logistic', clf)])

pca.fit(X)
plt.figure(1, figsize=(4, 3))
plt.clf()
plt.axes([.2, .2, .7, .7])
plt.plot(pca.explained_variance_, linewidth=2)
plt.axis('tight')
plt.xlabel('n_components')
plt.ylabel('explained_variance_')


n_components = [40, 80,120, 140,160,180,200,220,247]


estimator = GridSearchCV(pipe,
                         dict(pca__n_components=n_components))
estimator.fit(X, k)

plt.axvline(estimator.best_estimator_.named_steps['pca'].n_components,
            linestyle=':', label='n_components chosen')
plt.legend(prop=dict(size=12))
