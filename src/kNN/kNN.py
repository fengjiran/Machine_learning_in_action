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

