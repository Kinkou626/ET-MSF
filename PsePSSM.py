# -*- coding: utf-8 -*-
# @Time    : 2021/8/5 21:29
# @Author  : Yi-Zheng Wang
# @Email   : wyz020@126.com
# @File    : PsePssm.py
# 通过PSSM文件提取L*20矩阵，通过L*20矩阵计算得到PsePSSM


import os
import numpy as np


# ----------- 超参数定义 ----------- #
Lag = 8  # 生成PsePSSM的间隔


# ----------- 提取PSSM矩阵 ----------- #
def extract_matrix_from_pssm_file(dir):
    """
    通过pssm文件提取L*20矩阵
    :param dir: pssm文件路径
    :return: L*20矩阵
    """
    with open(dir) as f:
        lines = f.readlines()
    start_line = 3
    end_line = len(lines) - 7
    matrix = np.zeros((end_line - start_line + 1, 20))
    for i in range(start_line, end_line + 1):
        values = lines[i].strip().split()[2:22]
        for j in range(20):
            matrix[i - start_line][j] = int(values[j])
    return matrix


# ----------- 生成PsePSSM矩阵 ----------- #
def create_pse(pssm, lag):
    """
    通过L*20矩阵，根据公式，计算得出pse
    :param pssm: L*20矩阵
    :param lag: 超参数Lag，代表间隔
    :return:生成的20+20*lag维度的pse
    """
    N = len(pssm)

    # 根据公式将P(i,j)转化为P'(i,j)
    mean = pssm.mean(axis=1).reshape([N, 1])
    std = pssm.std(axis=1, ddof=1).reshape([N, 1])
    pssm = (pssm - mean) / std  # 这里使用broadcast机制完成矩阵相减和相除

    # 将0/0的情况产生的nan设置为1
    for i in range(len(pssm)):
        for j in range(len(pssm[0])):
            if np.isnan(pssm[i][j]):
                pssm[i][j] = 1

    # 计算前20维度
    ave = pssm.mean(axis=0)  # 一维数组
    Re = np.zeros(0)
    Tw = np.zeros(20)

    # 计算lag*10维度
    for l in range(1, lag+1):
        for j in range(20):
            s = 0
            for i in range(N - l):  # 求和
                s = s + (pssm[i, j] - pssm[i + l, j]) ** 2
            else:
                Tw[j] = s / (N - l)
        else:
            Re = np.hstack([Re, Tw])
    Re = np.hstack([Re, ave])
    return Re


if __name__ == "__main__":
    electron_dir = "dataset/PSSM/electron/"  # 正例electron PSSM文件目录
    transport_dir = "dataset/PSSM/transport/"  # 负例transport PSSM文件目录
    pse_save_dir = "dataset/PsePSSM/"

    electrons = os.listdir(electron_dir)
    transports = os.listdir(transport_dir)

    electron_pse = []
    j = 0
    for i in electrons:
        file = electron_dir + i
        pssm = extract_matrix_from_pssm_file(file)
        pse = create_pse(pssm, Lag)
        electron_pse.append(pse)
        j = j + 1
        print('\r' + "正在构建electron的PsePSSM矩阵 " + str(100 * j // len(electrons)) + '%', end='')
    np.save(pse_save_dir + "electron2,lag=" + str(Lag), electron_pse)

    transports_pse = []
    j = 0
    for i in transports:
        file = transport_dir + i
        pssm = extract_matrix_from_pssm_file(file)
        pse = create_pse(pssm, Lag)
        transports_pse.append(pse)
        j = j + 1
        print('\r' + "正在构建transport的PsePSSM矩阵 " + str(100 * j // len(transports)) + '%', end='')
    np.save(pse_save_dir + "transport2,lag=" + str(Lag), transports_pse)