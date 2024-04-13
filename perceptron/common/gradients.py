# coding: utf-8

import numpy as np


# 梯度
def numerical_gradient_no_batch(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]
        # 计算f(x+h)
        x[idx] = tmp_val + h
        fxh1 = f(x)

        # 计算f(x-h)
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        # 还原x值
        x[idx] = tmp_val
    return grad


def numerical_gradient11(f, X):
    if X.ndim == 1:
        return numerical_gradient_no_batch(f, X)
    else:
        grad = np.zeros_like(X)
        for idx, num in enumerate(X):
            grad[idx] = numerical_gradient_no_batch(f, num)


def numerical_gradient(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)  # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2 * h)

        x[idx] = tmp_val  # 还原值
        it.iternext()

    return grad


# 数值微分 # 导数
def numerical_diff(f, x):
    h = 1e-4
    return (f(x + h) - f(x - h)) / (2 * h)


def tangent_line(f, x):
    # 斜率（导数）
    d = numerical_gradient(f, x)
    print(d)
    y = f(x) - d * x
    return lambda t: t * x + y


# 梯度下降
"""
init_x： 初始x大小
lr： 学习速率
steps_num： 梯度法重复的次数
"""


def gradient_decent(f, init_x, lr=0.01, steps_num=100):
    x = init_x
    for i in range(steps_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
    return x
