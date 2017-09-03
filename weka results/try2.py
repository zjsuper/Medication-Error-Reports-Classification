# Author: Olivier Grisel <olivier.grisel@ensta.org>
# License: Simplified BSD


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
#k = label_binarize(k, classes=[0, 1, 2])
cv = StratifiedKFold(n_splits=6)
n_classes = 3
random_state = np.random.RandomState(1)                                    
X_train, X_test, k_train, k_test = train_test_split(X, k, test_size=.1)
classifier = svm.SVC(gamma=0.07,C=1,verbose=True,decision_function_shape='ovr',
              kernel='rbf',random_state=random_state,probability=True) 
mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)
colors = cycle(['cyan', 'indigo', 'seagreen', 'yellow', 'blue', 'darkorange'])
lw = 2
i = 0
for (train, test), color in zip(cv.split(X, k), colors):
    probas_ = classifier.fit(X[train], k[train]).predict_proba(X[test])
    # Compute ROC curve and area the curve
    print (probas_)
#    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
#    mean_tpr += interp(mean_fpr, fpr, tpr)
#    mean_tpr[0] = 0.0
#    roc_auc = auc(fpr, tpr)
#    plt.plot(fpr, tpr, lw=lw, color=color,
#             label='ROC fold %d (area = %0.2f)' % (i, roc_auc))
#
#    i += 1
#plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k',
#         label='Luck')
#
#mean_tpr /= cv.get_n_splits(X, y)
#mean_tpr[-1] = 1.0
#mean_auc = auc(mean_fpr, mean_tpr)
#plt.plot(mean_fpr, mean_tpr, color='g', linestyle='--',
#         label='Mean ROC (area = %0.2f)' % mean_auc, lw=lw)
#
#plt.xlim([-0.05, 1.05])
#plt.ylim([-0.05, 1.05])
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
#plt.title('Receiver operating characteristic example')
#plt.legend(loc="lower right")
#plt.show()  
#clf.fit(X_train,k_train)

#pred = clf.predict(X_test)    
from sklearn.metrics import accuracy_score    
#scores = accuracy_score(k_test, pred)

#print (scores)