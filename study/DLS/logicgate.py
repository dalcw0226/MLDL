# 논리 게이트를 구현하기
def AND(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.7
    tmp = x1*w1 + x2*w2
    if tmp <= theta:
        return 0
    else:
        return 1

# print(AND(1,1))

# theta를 b로 생각하기 -> AND Gate
import numpy as np
# x = np.array([0,1])
# w = np.array([0.5, 0.5])
# b = -0.7
# print(np.sum(x*w)+b)
def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(x*w)+b
    if tmp <= 0:
        return 0
    else:
        return 1

# print(AND(1, 0))

# NAND Gate
def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7

    tmp = np.sum(x*w)+b
    if tmp <= 0:
        return 0
    else:
        return 1

# OR Gate
def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    tmp = np.sum(w*x)+b
    if tmp <= 0:
        return 0
    else:
        return 1