# coding: utf-8
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T
    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))


# 均方误差
def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)


# 交叉熵误差
def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))


# 交叉熵误差（mini-batch）
def cross_entropy_error_mini_batches(y, t):
    delta = 1e-7
    if y.ndim == 1:
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y) + delta) / batch_size


# 如果监督是数字形式，如2，7，非one-hot形式
def cross_entropy_error_num(y, t):
    delta = 1e-7
    if y.ndim == 1:
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)
        batch_size = y.shape[0]
        return -np.sum(np.log(y[np.arange(batch_size), t] + delta)) / batch_size


def function_1(x):
    return 0.01 * x ** 2 + 0.1 * x


def function_2(x):
    return x[0] ** 2 + x[1] ** 2


# 数值微分 # 导数
def numerical_diff(f, x):
    h = 1e-4
    return (f(x + h) - f(x - h)) / (2 * h)


# 梯度
def numerical_gradient(f, x):
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
