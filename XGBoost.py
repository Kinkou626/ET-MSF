# -*- coding: utf-8 -*-
# @Time    : 2021/8/9 17:38
# @Author  : Yizheng Wang
# @Email   : wyz020@126.com
# @File    : XGBoost.py.py

import numpy as np
import sklearn
from xgboost import XGBClassifier
from evaluate import evaluate
from sklearn.model_selection import GridSearchCV

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
        'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],  # 重要影响

        'min_child_weight': [1, 2, 3, 4, 5, 6],  # 重要影响

        'learning_rate': [0.01, 0.05, 0.1, 0.2],

        'n_estimators': [50, 100, 200, 300, 400, 500, 600],

        'max_delta_step': [0, 0.2, 0.6, 1, 2],

        'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],

        'subsample': [0.6, 0.7, 0.8, 0.9],  #比例，次要影响

        'colsample_bytree': [0.6, 0.7, 0.8, 0.9],  #比例，次要影响

        'reg_alpha': [0.05, 0.1, 1, 2],

        'reg_lambda': [0.05, 0.1, 1, 2],

        # 'scale_pos_weight': [0.2, 0.4, 0.6, 0.8, 1]
        # 'scale_pos_weight': [0.6]
    }

    classifier = XGBClassifier()
    search = GridSearchCV(classifier, param_grid=parameters, scoring='accuracy', cv=5, verbose=2)
    search.fit(train_data, train_label)

    print("Best score: %0.3f" % search.best_score_)
    print("Best parameters set:")
    best_parameters = search.best_estimator_.get_params()
    for param_name in parameters.keys():
        print("\t%s: %r" % (param_name, best_parameters[param_name]))


# ------------------ 独立测试 ------------------  #
def independent_test(max_depth, min_child_weight, learning_rate, n_estimators, max_delta_step, gamma, subsample,
                     colsample_bytree, reg_alpha, reg_lambda):
    classifier = XGBClassifier(max_depth=max_depth,
                               min_child_weight=min_child_weight,
                               learning_rate=learning_rate,
                               n_estimators=n_estimators,
                               max_delta_step=max_delta_step,
                               gamma=gamma,
                               subsample=subsample,
                               colsample_bytree=colsample_bytree,
                               reg_alpha=reg_alpha,
                               reg_lambda=reg_lambda)
    classifier.fit(train_data, train_label)
    y_p = classifier.predict(test_data)
    e = evaluate(test_label, y_p)
    e.print_all_evaluation()
    e.show_confusion_matrix()


if __name__ == "__main__":
    # grid_search()
    independent_test(max_depth=7,
                     min_child_weight=2,
                     learning_rate=0.2,
                     n_estimators=500,
                     max_delta_step=0.2,
                     gamma=0.2,
                     subsample=0.6,
                     colsample_bytree=0.9,
                     reg_alpha=1,
                     reg_lambda=0.05)
