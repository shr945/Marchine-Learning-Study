# coding=utf-8
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
#有本地数据




def softmax(x):
    return np.exp(x)/np.sum(np.exp(x))
#数据提取
def load():
    x_train = pd.read_csv("train.csv",header=-1,usecols = [0,1])
    y_train = pd.read_csv("train.csv",header=-1,usecols = [2])
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test= pd.read_csv("test.csv",header=-1,usecols = [0,1])
    y_test = pd.read_csv("test.csv",header=-1,usecols = [2])
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    return x_train, y_train, x_test, y_test


def train(digits, labels, maxIter = 100, alpha = 0.07):
    weights = np.random.randn(3, 2) * 0.01
    for iter in range(maxIter):
        for i in range(len(digits)):
            x = digits[i].reshape(-1, 1)
            y = np.zeros((3, 1))
            y[labels[i]] = 1
            y_ = softmax(np.dot(weights, x))
            weights -= alpha * (np.dot((y_ - y), x.T))
    return weights

def predict(digit,w):   #预测函数
    return np.argmax(np.dot(w, digit))   #返回softmax中概率最大的值


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = load();
    w = train(x_train, y_train, maxIter=100) #训练
    accuracy = 0
    N = len(x_test) #总共多少测试样本
    for i in range(N):
        digit = x_test[i]   #每个测试样本
        label = y_test[i][0]    #每个测试标签
        predict1 = predict(digit, w)  #测试结果
        if (predict1 == label):
            accuracy += 1
        print("predict:%d, actual:%d"% (predict1, label))
    print("accuracy:%.1f%%" %(accuracy / N * 100))


