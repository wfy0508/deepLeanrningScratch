# coding: utf-8
import sys
import os

sys.path.append(os.pardir)

import numpy as np
import pickle
from dataset.mnist import load_mnist
from common.functions import sigmoid
from common.functions import softmax


def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test


# 获取已经训练完成的神经网络权重和偏置
def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network_1 = pickle.load(f)
    return network_1


# 对测试图像和测试标签进行预测
def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    result = softmax(a3)

    return result


x, t = get_data()
network_test = init_network()
accuracy_cnt = 0
for i in range(len(x)):
    y = predict(network_test, x[i])
    p = np.argmax(y)  # 获取概率最高的元素的索引
    print(p, t[i])
    if p == t[i]:
        accuracy_cnt += 1

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
