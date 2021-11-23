# -*- coding: utf-8 -*-
# @Time    : 2021/8/8 17:59
# @Author  : Yi-Zheng Wang
# @Email   : wyz020@126.com
# @File    : evaluate.py

import numpy as np


class evaluate():
    def __init__(self, y, y_p):
        self.label = y
        self.predict = y_p
        self.TP = 0
        self.FN = 0
        self.FP = 0
        self.TN = 0
        for i in range(len(self.predict)):
            if self.label[i] == 0 and self.predict[i] == 0:
                self.TP += 1
            if self.label[i] == 0 and self.predict[i] == 1:
                self.FP += 1
            if self.label[i] == 1 and self.predict[i] == 0:
                self.FN += 1
            if self.label[i] == 1 and self.predict[i] == 1:
                self.TN += 1

        # 真正类率：模型预测为正类且实际也为正类中占所有实际为正类的比例
        self.TPR = self.TP / (self.TP + self.FN)
        # 敏感性：
        self.Sensitivity = self.TPR
        # 召回率
        self.Recall = self.TPR

        # 真负类率：模型预测为负类且实际也为负类中占所有实际为负类的比例
        self.TNR = self.TN / (self.TN + self.FP)
        # 特异性：
        self.Specificity = self.TNR

        # 准确率：
        self.Accuracy = (self.TP + self.TN) / (self.TP + self.FN + self.FP + self.TN)

        # 精确率：正确预测正样本占实际预测为正样本的比例
        self.Precision = self.TP / (self.TP + self.FP)

        # F1值
        self.F1 = (2 * self.Precision * self.Recall) / (self.Precision + self.Recall)

        # 马修斯相关系数(MCC):用以测量二分类的分类性能的指标
        self.MCC = ((self.TP * self.TN) - (self.FP * self.FN)) / (
                    ((self.TP + self.FP) * (self.TP + self.FN) * (self.TN + self.FP) * (self.TN + self.FN)) ** 0.5)

    def show_confusion_matrix(self):  # 参数为实际分类和预测分类
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(self.label, self.predict)
        import matplotlib.pyplot as plt
        plt.matshow(cm, cmap=plt.cm.Greens)
        plt.colorbar()
        for x in range(len(cm)):
            for y in range(len(cm)):
                plt.annotate(cm[x, y], xy=(x, y), horizontalalignment='center', verticalalignment='center')
        plt.ylabel('True label')  # 坐标轴标签
        plt.xlabel('predicted label')  # 坐标轴标签
        plt.title('Confusion Matrix')
        plt.show()

    def print_all_evaluation(self):
        print("Sensitivity =", self.Sensitivity)
        print("Specificity =", self.Specificity)
        print("Accuracy =", self.Accuracy)
        print("Precision =", self.Precision)
        print("F1 =", self.F1)
        print("MCC =", self.MCC)


if __name__ == "__main__":
    p = np.append(np.zeros(183+46), np.ones(25+666))
    y = np.append(np.append(np.zeros(183), np.ones(46)), np.append(np.zeros(25), np.ones(666)))
    k = evaluate(y, p)
    k.print_all_evaluation()
    k.show_confusion_matrix()

