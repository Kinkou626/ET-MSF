# -*- coding: utf-8 -*-
# @Time    : 2021/9/12 10:12
# @Author  : Yi-Zheng Wang
# @Email   : wyz020@126.com
# @File    : etc-MSF-softVoting.py


import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC as SVM
from sklearn.neighbors import KNeighborsClassifier as KNN
from xgboost import XGBClassifier as XGBOOST
from sklearn.ensemble import RandomForestClassifier as RF
from lightgbm import LGBMClassifier as LGBM
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from evaluate import evaluate


electrons = np.load("dataset/PsePSSM/electron,lag=8.npy")
transports = np.load("dataset/PsePSSM/transport,lag=8.npy")
x = np.vstack((electrons, transports))

positive_label = np.zeros(len(electrons))
negative_label = np.ones(len(transports))
y = np.append(positive_label, negative_label)

train_data, test_data, train_label, test_label = train_test_split(x,
                                                                  y,
                                                                  random_state=1,
                                                                  train_size=0.85,
                                                                  test_size=0.15,
                                                                  stratify=y)

classifiers = [
    SVM(C=19,
        kernel='rbf',
        gamma=0.018,
        probability=True),

    KNN(n_neighbors=1,
        weights='uniform',
        algorithm='auto',
        leaf_size=30,
        metric='manhattan'),

    XGBOOST(max_depth=7,
            min_child_weight=2,
            learning_rate=0.2,
            n_estimators=500,
            max_delta_step=0.2,
            gamma=0.2,
            subsample=0.6,
            colsample_bytree=0.9,
            reg_alpha=1,
            reg_lambda=0.05),
    #
    # RF(n_estimators=100,
    #    random_state=1),

    # LGBM(num_leaves=83)
]

X_train_voting = np.zeros(len(train_data))
X_test_voting = np.zeros(len(test_data))

for i, classifier in enumerate(classifiers):
    print("classifier：{}".format(classifier))
    classifier.fit(train_data, train_label)
    y = classifier.predict_proba(test_data)[:, 1]
    X_test_voting = X_test_voting + y

for i in range(len(X_test_voting)):
    if X_test_voting[i] >= 1.5:
        X_test_voting[i] = 1
    else:
        X_test_voting[i] = 0

print("Performance Measures:")
e = evaluate(test_label, X_test_voting)
e.print_all_evaluation()
e.show_confusion_matrix()
