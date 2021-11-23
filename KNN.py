# -*- coding: utf-8 -*-
# @Time    : 2021/8/15 9:41
# @Author  : Yizheng Wang
# @Email   : wyz020@126.com
# @File    : KNN.py.py

import numpy as np
import sklearn
from evaluate import evaluate
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

electrons = np.load("dataset/PsePSSM/electron,lag=8.npy")
transports = np.load("dataset/PsePSSM/transport,lag=8.npy")
x = np.vstack((electrons, transports))

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
        'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25],

        'weights': ['uniform', 'distance'],

        'algorithm': ['auto'],

        'leaf_size': [10, 20, 30, 40, 50],

        'metric': ['manhattan'],

    }

    classifier = KNeighborsClassifier()
    search = GridSearchCV(classifier, param_grid=parameters, scoring='accuracy', cv=5, verbose=2)
    search.fit(train_data, train_label)

    print("Best score: %0.3f" % search.best_score_)
    print("Best parameters set:")
    best_parameters = search.best_estimator_.get_params()
    for param_name in parameters.keys():
        print("\t%s: %r" % (param_name, best_parameters[param_name]))


# ------------------ 交叉验证 ------------------ #
def cross_validation_search():
    best_score = 0
    for n_neighbors in range(1, 30, 1):
        for leaf_size in range(10, 100, 10):
            classifier = KNeighborsClassifier(n_neighbors=n_neighbors,
                                              weights='uniform',
                                              algorithm='auto',
                                              leaf_size=leaf_size,
                                              metric='manhattan'
                                              )
            scores = sklearn.model_selection.cross_val_score(
                classifier, train_data, train_label, cv=10, scoring='accuracy')
            score = scores.mean()
            if score > best_score:
                best_score = score
                print("n=", n_neighbors, "leaf_size=", leaf_size, "best_score=", score)
            else:
                print("n=", n_neighbors, "leaf_size=", leaf_size)


# ------------------ 独立测试 ------------------  #
def independent_test(n_neighbors, weights, algorithm, leaf_size, metric):
    classifier = KNeighborsClassifier(n_neighbors=n_neighbors,
                                      weights=weights,
                                      algorithm=algorithm,
                                      leaf_size=leaf_size,
                                      metric=metric,
                                      )
    classifier.fit(train_data, train_label)
    y_p = classifier.predict(test_data)
    e = evaluate(test_label, y_p)
    e.print_all_evaluation()
    e.show_confusion_matrix()


if __name__ == "__main__":
    # grid_search()
    # cross_validation_search()
    independent_test(n_neighbors=1,
                     weights='uniform',
                     algorithm='auto',
                     leaf_size=30,
                     metric='manhattan')
