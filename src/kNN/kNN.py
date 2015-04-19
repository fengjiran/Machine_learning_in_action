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

    
def file2matrix(filename):
    love_dictionary = {'largeDoses':3, 'smallDoses':2, 'didntLike':1}
    fr = open(filename)
    arrayOLines = fr.readlines()  #readlines()自动将文件内容分析成一个行的列表
    numberOfLines = len(arrayOLines)
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    
    #strip()函数原型
    #声明：s为字符串，rm为要删除的字符序列
    #s.strip(rm)        删除s字符串中开头、结尾处，位于 rm删除序列的字符
    #s.lstrip(rm)       删除s字符串中开头处，位于 rm删除序列的字符
    #s.rstrip(rm)      删除s字符串中结尾处，位于 rm删除序列的字符
    #注意：
    #1. 当rm为空时，默认删除空白符(包括'\n', '\r',  '\t',  ' ')
    #2. 这里的rm删除序列是只要边（开头或结尾）上的字符在删除序列内，就删除掉。
    
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        if (listFromLine[-1].isdigit()):
            classLabelVector.append(int(listFromLine[-1]))
        else:
            classLabelVector.append(love_dictionary.get(listFromLine[-1]))
        index += 1
        
    return returnMat, classLabelVector