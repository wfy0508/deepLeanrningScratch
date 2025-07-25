# coding: utf-8
from perceptron.ch05.layer_naive import MulLayer, AddLayer

apple = 100
orange = 150
apple_num = 2
orange_num = 3
tax = 1.1

apple_price_layer = MulLayer()
orange_price_layer = MulLayer()
price_layer = AddLayer()
price_tax_layer = MulLayer()

# forward
apple_price = apple_price_layer.forward(apple, apple_num)
orange_price = orange_price_layer.forward(orange, orange_num)
all_price = price_layer.forward(apple_price, orange_price)
all_price_tax = price_tax_layer.forward(all_price, tax)
print(apple_price, orange_price, all_price, all_price_tax)

# backward
# 初始输入信号
dprice = 1

dall_price, dtax = price_tax_layer.backward(dprice)
dapple_price, dorange_price = price_layer.backward(dall_price)
dapple, dapple_num = apple_price_layer.backward(dapple_price)
dorange, dorange_num = orange_price_layer.backward(dorange_price)
print(dapple, dapple_num, dorange, dorange_num, dapple_price, dorange_num, dall_price, dtax)
