import numpy as np
from summer.common.functions import *
from summer.common.gradients import numerical_gradient


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 初始化权重和偏置
        self.params = dict()
        self.params['w1'] = np.random.randn(input_size, hidden_size) * weight_init_std
        self.params['b1'] = np.zeros(hidden_size)
        self.params['w2'] = np.random.randn(hidden_size, output_size) * weight_init_std
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        # 取出权重和偏置
        w1, b1 = self.params['w1'], self.params['b1']
        w2, b2 = self.params['w2'], self.params['b2']

        # 计算
        a1 = np.dot(x, w1) + b1
        # 使用sigmoid激活函数计算第1层输出
        z1 = sigmoid(a1)
        # 第二层输出
        a2 = np.dot(z1, w2) + b2
        # 使用softmax计算预测值
        y = softmax(a2)
        return y

    def loss(self, x, t):
        y = self.predict(x)
        # 使用交叉熵误差计算损失函数
        return cross_entropy_error(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        # 如果两者的最大值索引一致，表示预测正确
        acc = np.sum(y == t) / len(t)
        return acc

    def numerical_gradient(self, x, t):
        loss_w = lambda w: self.loss(x, t)
        grad = dict()
        # 计算损失函数在权重和偏置处的梯度
        grad['w1'] = numerical_gradient(loss_w, self.params['w1'])
        grad['b1'] = numerical_gradient(loss_w, self.params['b1'])
        grad['w2'] = numerical_gradient(loss_w, self.params['w2'])
        grad['b2'] = numerical_gradient(loss_w, self.params['b2'])
        return grad

    def gradient(self, x, t):
        w1, w2 = self.params['w1'], self.params['w2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}

        batch_num = x.shape[0]

        # forward
        a1 = np.dot(x, w1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, w2) + b2
        y = softmax(a2)

        # backward
        dy = (y - t) / batch_num
        grads['w2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)

        da1 = np.dot(dy, w2.T)
        dz1 = sigmoid_grad(a1) * da1
        grads['w1'] = np.dot(x.T, dz1)
        grads['b1'] = np.sum(dz1, axis=0)

        return grads


if __name__ == '__main__':
    net = TwoLayerNet(input_size=784, hidden_size=100, output_size=10, weight_init_std=0.01)
    print(net.params['w1'].shape)
    print(net.params['b1'].shape)
    print(net.params['w2'].shape)
    print(net.params['b2'].shape)

    x = np.random.randn(100, 784)
    print(net.predict(x))
