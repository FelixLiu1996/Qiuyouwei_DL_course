import numpy as np
import matplotlib.pyplot as plt

# threshold function
def threshold_function(x):
    y = x > 0
    return y.astype(int)

# x = np.array([1, -1, 0])
# print(threshold_function(x))


# tanh function  或者 直接调用 np.tanh()
def tangent_function(x):
    return (1 - np.exp(-2 * x)) / (1 + np.exp(-2 * x))

# x = np.array([-1, 1, 2])
# print(tangent_function(x))
# print(np.tanh(x))

# ReLU function
def relu_function(x):
    return np.maximum(0, x)


# Cross Entropy
def cross_entropy(y_hat, y):
    delta = 1e-8  # 为了防止log(0)的出现
    return -np.sum(y * np.log(y_hat + delta))

# 绘制切线函数

# 原函数
def func(x):
    return x ** 2

# 导数函数
def dfunc(f, x):
    h = 1e-4
    return (f(x + h) - f(x)) / h

# 切线函数
def tdfunc(f, x, t):
    d = dfunc(f, x)
    y = f(x) - d * x
    return d * t + y

x = np.arange(-6, 6, 0.01)
y = func(x)
plt.plot(x, y)

x2 = np.arange(0, 5, 0.01)
y2 = tdfunc(func, 3, x2)
plt.plot(x2, y2)
plt.show()
