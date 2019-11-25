from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

def load():
    datas = datasets.load_breast_cancer()
    X = datas.data
    y1 = datas.target
    y=np.zeros((len(y1),1))
    for i in range(len(y1)):
        y[i,0] = y1[i]
    print(X)
    print(y)
    return X, y


def sigmoid(inx):
    return 0.5 * (1 + np.tanh(0.5 * inx))


def LR_train( X, y_true, loopNum, learningRate):
    n_samples, n_features = X.shape
    weights = np.zeros((n_features, 1))
    costs = []
    for i in range(loopNum):
        y_predict = sigmoid(np.dot(X, weights))
        cost = y_true - y_predict
        #print(cost)
        weights = weights + learningRate * np.dot(X.transpose() , cost)
        cost = np.sum(cost)
        costs.append(np.sum(cost))
        if i % 100 == 0:
            print(f"Cost after iteration {i}: {cost}")
    return weights, costs

#print(X)
def acc(y_predict,y_test):
    e = 0
    for i in range(len(y_predict)):
        if (y_predict[i][0] != y_test[i][0]):
            e = e + 1
    return 1 - e / len(y_predict)


def main():
    X, y = load()
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    w_trained, costs = LR_train(X_train, y_train, loopNum=600, learningRate=0.009)
    y_predict = sigmoid(np.dot(X_test, w_trained))
    acc1 = acc(y_predict,y_test)
    print("自己写的函数准确率：",acc1)

    from sklearn.linear_model import LogisticRegression
    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)

    y_predict = classifier.predict(X_test)
    e = 0
    for i in range(len(y_predict)):
        if (y_predict[i] != y_test[i][0]):
            e = e + 1
    print("sklearn调用函数准确率",1 - e / len(y_predict))

if __name__ == '__main__':
    main()