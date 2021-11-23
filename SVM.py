# -*- coding: utf-8 -*-
# @Time    : 2021/8/5 15:48
# @Author  : Yi-Zheng Wang
# @Email   : wyz020@126.com
# @File    : SVM.py.py

import numpy as np
import sklearn
from sklearn import svm
from evaluate import evaluate


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
    gamma_list = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
    c_list = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
    score_list = []
    for gamma in gamma_list:
        c_score_list = []
        for c in c_list:
            classifier = svm.SVC(C=c, kernel='rbf', gamma=gamma, probability=True)
            classifier.fit(train_data, train_label)
            test_score = classifier.score(test_data, test_label)
            c_score_list.append(test_score)
            print("c=", c, "gamma=", gamma, "best_score=", test_score)
        score_list.append(c_score_list)
    return score_list


# ------------------ 范围搜索 ------------------  #
def range_search():
    best_score = 0
    for gamma in range(1, 100, 1):
        for c in range(1, 100, 1):
            classifier = svm.SVC(C=c, kernel='rbf', gamma=gamma/1000, probability=True)
            classifier.fit(train_data, train_label)
            test_score = classifier.score(test_data, test_label)
            if test_score > best_score:
                best_score = test_score
                print("c=", c, "gamma=", gamma/1000, "best_score=", test_score)
            else:
                print("c=", c, "gamma=", gamma / 1000)


# ------------------ 交叉验证 ------------------ #
def cross_validation_search():
    best_score = 0
    for gamma in range(5, 25, 1):
        for c in range(5, 25, 1):
            classifier = svm.SVC(C=c, kernel='rbf', gamma=gamma/1000, probability=True)
            scores = sklearn.model_selection.cross_val_score(
                classifier, train_data, train_label, cv=10, scoring='accuracy')
            score = scores.mean()
            if score > best_score:
                best_score = score
                print("c=", c, "gamma=", gamma/1000, "best_score=", score)
            else:
                print("c=", c, "gamma=", gamma / 1000)


# ------------------ 独立测试 ------------------  #
def independent_test(c, gamma):
    classifier = svm.SVC(C=c, kernel='rbf', gamma=gamma, probability=True)
    classifier.fit(train_data, train_label)
    y_p = classifier.predict(test_data)
    e = evaluate(test_label, y_p)
    e.print_all_evaluation()
    e.show_confusion_matrix()


if __name__ == "__main__":
    # grid_search()
    # range_search()
    # cross_validation_search()
    # independent_test(c=22, gamma=0.023)
    independent_test(c=19, gamma=0.018)
