#coding:utf-8
'''
Created on 2015年4月17日
kNN: k Nearest Neighbors

Input:   inX: vector to compare to existing dataset(1XN)
         dataSet: size m data set of known vectors(NXM)
         labels: data set labels (1XM vector)
         k: number of neighbors to use for comparison (should be an odd number)

Output:  the most popular class label

@author: Richard
'''
from numpy import *
import operator

def CreateDataset():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    
    return group, labels

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis = 1)
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()
    #argsort()得到的是排序后的数据原来位置的下标
    
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key = operator.itemgetter(1), reverse = True)
    
    return sortedClassCount[0][0]
    