import numpy as np

from perceptron.common.functions import softmax, cross_entropy_error
from perceptron.common.gradients import numerical_gradient, numerical_gradient_no_batch


class SimpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)
        return loss


net = SimpleNet()
print("权重：\n", net.W)

x = np.array([0.6, 0.9])
p = net.predict(x)
print("predict: \n", p)
print("最大值索引：", np.argmax(p))

# 正确解标签
t = np.array([0, 0, 1])
f = lambda w: net.loss(x, t)
print("损失函数: ", net.loss(x, t))


f = lambda w: net.loss(x, t)


dW = numerical_gradient(f, net.W)
print(dW)
