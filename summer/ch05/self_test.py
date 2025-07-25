# encoding = uth-8

# 定义乘法器
class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        return x * y

    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x
        return dx, dy


# 定义加法器
class AddLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        return x + y

    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1
        return dx, dy


apple_price = 100
apple_num = 2
orange_price = 150
orange_num = 3
tax = 1.1

apple_layer = MulLayer()
orange_layer = MulLayer()
apple_orange_layer = AddLayer()
tax_layer = MulLayer()

# 计算总金额
apple_amt = apple_layer.forward(apple_price, apple_num)
orange_amt = orange_layer.forward(orange_price, orange_num)
apple_orange_amt = apple_orange_layer.forward(apple_amt, orange_amt)
total_amt = tax_layer.forward(apple_orange_amt, tax)
print(apple_amt, orange_amt, apple_orange_amt, round(total_amt))

# 反向传播计算导数
dout = 1
d_total_amt, d_tax = tax_layer.backward(dout)
d_apple_amt, d_orange_amt = apple_orange_layer.backward(d_total_amt)
d_apple, d_apple_num = apple_layer.backward(d_apple_amt)
d_orange, d_orange_num = orange_layer.backward(d_orange_amt)
print(d_total_amt, d_tax, round(d_apple), round(d_apple_num), d_orange, d_orange_num)