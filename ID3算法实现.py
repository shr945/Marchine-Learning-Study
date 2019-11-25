import numpy as np
import math

def calcShannonEnt(dataSet):
    numEntries = len(dataSet)  # 样本数
    labelCounts = {}
    for featVec in dataSet:  # 遍历每个样本
        currentLabel = featVec[-1]  # 当前样本的类别
        if currentLabel not in labelCounts.keys():  # 生成类别字典
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:  # 计算信息熵
        prob = float(labelCounts[key]) / numEntries
        shannonEnt = shannonEnt - prob * math.log(prob, 2)
    return shannonEnt

# 划分数据集，axis:按第几个属性划分，value:要返回的子集对应的属性值
def split(data, axis, value):
    retDataSet = []
    for featVec in data:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1:])
            # 把每个样本特征堆叠在一起，变成一个子集合
            retDataSet.append(reducedFeatVec)

    return retDataSet


def chooseBestFeatureToSplit(data):
    num = len(data[0]) - 1
    ent_all = calcShannonEnt(data)
    chosenfeature = 0
    bestinfogain = 0
    for i in range(num):
        list = [vector[i] for vector in data]
        uniqueVals = set(list)
        newent = 0
        for value in uniqueVals:
            subdata = split(data, i, value)
            sum = len(subdata) / float(len(data))
            newent += sum * calcShannonEnt(subdata)
        infogain = ent_all - newent
        if (infogain > bestinfogain):
            bestinfogain = infogain
            chosenfeature = i
    return chosenfeature


def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


# ent=ccalcShannonEntal_ent(data)
# print(ent)
def CreateTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    # 如果划分的数据集只有一个类别，则返回此类别
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 如果使用完所有特征属性之后，类别标签仍不唯一，则使用majorityCnt函数，多数表决法，哪种类别标签多，则分为此类别
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    del (labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = CreateTree(split(dataSet, bestFeat, value), subLabels)
    return myTree

def judge(data, Tree):
    res = []
    Tree = Tree['有自己的房子']
    for featVec in data:
        if(featVec[2] == '是'):
            res.append('是')
        else:
            if(featVec[1]=='是'):
                res.append('是')
            else:
                res.append('否')
    return res



def main():
    data = [['青年', '否', '否', '一般', '否'],
            ['青年', '否', '否', '好', '否'],
            ['青年', '是', '否', '好', '是'],
            ['青年', '是', '是', '一般', '是'],
            ['青年', '否', '否', '一般', '否'],
            ['中年', '否', '否', '一般', '否'],
            ['中年', '否', '否', '好', '否'],
            ['中年', '是', '是', '好', '是'],
            ['中年', '否', '是', '非常好', '是'],
            ['中年', '否', '是', '非常好', '是'],
            ['老年', '否', '是', '非常好', '是'],
            ['老年', '否', '是', '好', '是'],
            ['老年', '是', '否', '好', '是'],
            ['老年', '是', '否', '非常好', '是'],
            ['老年', '否', '否', '一般', '否']
            ]
    label = ['年龄', '有工作', '有自己的房子', '信贷情况', '类别']
    Tree = CreateTree(data, label)
    print(Tree)
    print(judge(data, Tree))




if __name__ == '__main__':
    main()