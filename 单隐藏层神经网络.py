# coding=utf-8
import numpy as np
import math
import matplotlib.pyplot as plt
#有本地数据
def y_data(y):
    e = np.zeros((10, 1))
    e[y] = 1.0
    return e

#sigmoid函数
def sigmoid(x):
    return 1/(1+np.exp(-x))

#sigmoid导数
def sigmoid_thea(x):
    return (sigmoid(x))*(1 - sigmoid(x))

#数据提取
def load():
    mnist="mnist.npz"
    f = np.load(mnist)
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    f.close()
    y = np.zeros((60000,10))
    for t in range(60000):
        y[t][y_train[t]] = 1
    y1 = np.zeros((10000,10))
    for t in range(10000):
        y1[t][y_test[t]] = 1
    y_train = y.T
    y_test =y1.T
    x_train = x_train/255
    x_test = x_test/255
    x_train = x_train.reshape((60000, 784))
    x_test = x_test.reshape((10000, 784))
    return x_train, x_test,  y_train, y_test

'''
x_train, x_test,  y_train, y_test = load()
#print(x_train[0])#展示第一张图片，训练集和标签

x_train = x_train.reshape((60000,784))

print(x_train[0])#第一张图片的向量
'''
#初始化参数
def init_parameters(n_x, n_h, n_y):
    # x维度 隐藏层维度 y维度
    W1 = np.random.randn(n_h, n_x) * 0.01  # 输入层到隐层的权重
    # W1 = np.zeros((n_h, n_x))
    b1 = np.zeros((n_h, 1))                # 隐层的阈值
    W2 = np.random.randn(n_y, n_h) * 0.01  # 隐层到输出层的权重
    # W2 = np.zeros((n_y, n_h))
    b2 = np.zeros((n_y, 1))                # 输出层的阈值

    parameters = {
        "W1" : W1,
        "b1" : b1,
        "W2" : W2,
        "b2" : b2}

    return parameters

#前向传播
def forward_propagation(X, paras):
    W1 = paras["W1"]
    b1 = paras["b1"]
    W2 = paras["W2"]
    b2 = paras["b2"]
    Z1 = np.dot(W1, X) + b1
    A1 = sigmoid(Z1)            # 隐层输出
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)            #
    cache = {
        "Z1" : Z1,
        "A1" : A1,
        "Z2" : Z2,
        "A2" : A2}
    return cache

#计算损失函数
#反向传播计算梯度
def backpropagation(paras, cache, X, Y):
    thea1 = cache["Z1"]
    thea2 = cache["Z2"]
    W2 = paras["W2"]
    A1 = cache["A1"]
    A2 = cache["A2"]
    m = X.shape[1]  # 样本数
    # 反向传播误差计算梯度
    dZ2 = A2 - Y
    dW2 = np.dot(dZ2, thea1.T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m
    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
    dW1 = np.dot(dZ1, X.T) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m

    grads = {
        "dW1" : dW1,
        "db1" : db1,
        "dW2" : dW2,
        "db2" : db2}
    return grads

#更新梯度
def update_parameters(paras, grads, learning_rate):
    # 当前参数
    W1 = paras["W1"]
    b1 = paras["b1"]
    W2 = paras["W2"]
    b2 = paras["b2"]
    # 梯度值
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]
    # 梯度下降法更新
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    parameters = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2}
    return  parameters


def compute_cost(X, Y, Y_hat):
    m = Y.shape[1]  # 样本数量
    logprobs = Y * np.log(Y_hat) + (1 - Y) * np.log(1 - Y_hat)  #交叉熵
    cost = -np.sum(logprobs) / m
    return cost

def model(X, Y, n_h, iterations, learning_rate):
    n_x = X.shape[0]
    n_y = Y.shape[0]
    # 初始化参数w和b
    parameters = init_parameters(n_x, n_h, n_y)
    cost_list = []  # 存储每一次迭代的损失值
    # 优化w和b
    for i in range(iterations):
        # 前向传播，计算Y_hat、cost和grads
        cache = forward_propagation(X, parameters)  # 前向迭代计算各层输出
        cost = compute_cost(X, Y, cache["A2"])   # 计算损失值
        cost_list.append(cost)
        grads = backpropagation(parameters, cache, X, Y)  # 计算梯度
        parameters = update_parameters(parameters, grads, learning_rate)  # 更新网络参数

        if i % 100 == 0:
            print("The %sth iterations, cost is %s" % (str(i), str(cost)))

    return parameters



def predict(paras, X):
    W1 = paras["W1"]
    b1 = paras["b1"]
    W2 = paras["W2"]
    b2 = paras["b2"]
    A1 = sigmoid(np.dot(W1, X) + b1)  # 隐层输出值
    Y = sigmoid(np.dot(W2, A1) + b2)  # 预测值
    return Y

if __name__ =="__main__":
    x_train, x_test, y_train, y_test = load()
    cl = model(x_train.T, y_train, 1000, 2001, 0.01)

    y = predict(cl, x_test.T)

    tt = 0
    for i in range(10000):
        a = y[:, i]
        m = np.argmax(a)
        v = np.argmax(y_test[:, i])

        if (m == v):
            tt = tt + 1
    print(tt / 10000)




