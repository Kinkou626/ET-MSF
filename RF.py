# -*- coding: utf-8 -*-
# @Time    : 2021/8/15 20:36
# @Author  : Yizheng Wang
# @Email   : wyz020@126.com
# @File    : RF.py

import numpy as np
import sklearn
from evaluate import evaluate
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# ------------------ 加载数据 ------------------  #
electrons = np.load("dataset/PsePSSM/electron,lag=8.npy")
transports = np.load("dataset/PsePSSM/transport,lag=8.npy")
x = np.vstack((electrons,transports))

positive_label = np.zeros(len(electrons))
negative_label = np.ones(len(transports))
y = np.append(positive_label, negative_label)

train_data, test_data, train_label, test_label = sklearn.model_selection.train_test_split(
    x, y,
    random_state=1,
    train_size=0.85,
    test_size=0.15,
    stratify=y)


# ------------------ 网格搜索 ------------------  #
def grid_search():
    parameters = {
        'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
        'n_estimators': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
    }

    classifier = RandomForestClassifier()
    search = GridSearchCV(classifier, param_grid=parameters, scoring='accuracy', cv=5, verbose=2)
    search.fit(train_data, train_label)

    print("Best score: %0.3f" % search.best_score_)
    print("Best parameters set:")
    best_parameters = search.best_estimator_.get_params()
    for param_name in parameters.keys():
        print("\t%s: %r" % (param_name, best_parameters[param_name]))


# ------------------ 范围搜索 ------------------  #
def range_search():
    for i in range(10, 1000, 10):
        classifier = RandomForestClassifier(n_estimators=i, random_state=0)
        classifier.fit(train_data, train_label)
        y_p = classifier.predict(test_data)
        e = evaluate(test_label, y_p)
        e.print_all_evaluation()
        e.show_confusion_matrix()
        print(i, " ", e.Accuracy)


# ------------------ 独立测试 ------------------  #
def independent_test(n_estimators):
    classifier =RandomForestClassifier(n_estimators=n_estimators, random_state=0)
    classifier.fit(train_data, train_label)
    y_p = classifier.predict(test_data)
    e = evaluate(test_label, y_p)
    e.print_all_evaluation()
    e.show_confusion_matrix()


if __name__ == "__main__":
    # grid_search()
    # range_search()
    # cross_validation_search()
    independent_test(n_estimators=100)

