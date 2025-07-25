import numpy as np


def and_gate(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.7
    tmp = np.sum(w1 * x1 + w2 * x2)
    if tmp <= theta:
        return 0
    elif tmp > theta:
        return 1


def or_gate(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.5
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


def nand_gate(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    tmp = np.sum(x * w) + b
    if tmp <= 0:
        return 0
    else:
        return 1


if __name__ == '__main__':
    print("x1 \t x2 \t result")
    for x1 in range(0, 2):
        for x2 in range(0, 2):
            and_result = and_gate(x1, x2)
            or_result = or_gate(x1,x2)
            nand_result = nand_gate(x1,x2)
            print(str(x1) + "\t" + str(x2) + "\t" + str(and_result))
