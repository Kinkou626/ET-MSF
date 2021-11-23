# -*- coding: utf-8 -*-
# @Time    : 2021/8/17 12:04
# @Author  : Yizheng Wang
# @Email   : wyz020@126.com
# @File    : MSF.py

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
        metric='manhattan',
        ),

    XGBOOST(max_depth=7,
            min_child_weight=2,
            learning_rate=0.2,
            n_estimators=500,
            max_delta_step=0.2,
            gamma=0.2,
            subsample=0.6,
            colsample_bytree=0.9,
            reg_alpha=1,
            reg_lambda=0.05,
            ),

    RF(n_estimators=100,
       random_state=1,
       ),

    # LGBM(num_leaves=83)
]

X_train_stack = np.zeros((train_data.shape[0], len(classifiers)))
X_test_stack = np.zeros((test_data.shape[0], len(classifiers)))

n_folds = 10
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=1)

for i, classifier in enumerate(classifiers):
    print("classifier：{}".format(classifier))
    X_stack_test_n = np.zeros((test_data.shape[0], n_folds))
    for j, (train_index, test_index) in enumerate(skf.split(train_data, train_label)):
        x_train = train_data[train_index]
        y_train = train_label[train_index]
        classifier.fit(x_train, y_train)
        X_train_stack[test_index, i] = classifier.predict_proba(train_data[test_index])[:, 1]
        X_stack_test_n[:, j] = classifier.predict_proba(test_data)[:, 1]
        print("\rCross-validation (", j+1, "/", n_folds, ")", end='')
    print("")
    X_test_stack[:, i] = X_stack_test_n.mean(axis=1)


# lr = LogisticRegression(solver="lbfgs")
lr = SVM(C=6, gamma=0.012, probability=True)
lr.fit(X_train_stack, train_label)
p = lr.predict(X_test_stack)
print("Performance Measures:")
e = evaluate(test_label, p)
e.print_all_evaluation()
e.show_confusion_matrix()
