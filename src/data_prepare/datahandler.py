import os
import sys
import numpy

InputSize = 39

def nameSplit(name):
    part = name.split('_')
    return (part[0] + '_' + part[1]), int(part[2])

def findEndIndx(dataName):
    endIndxList = []
    curName, curNum = nameSplit(dataName[0])
    dataLen = len(dataName)
    for i in xrange(dataLen):
        nextName, nextNum = nameSplit(dataName[i])
        if(curName != nextName):
            endIndxList.append(i)
            curName = nextName
    endIndxList.append(dataLen)
    return endIndxList

def normalization(dataXYN):
    dataX, dataY, dataName = dataXYN
    dataX -= numpy.mean(dataX, axis = 0)
    dataX /= numpy.sqrt(numpy.var(dataX, axis = 0))
    newDataXYN = [dataX, dataY, dataName]
    return newDataXYN

def prepareSplice(dataXYN, SpliceRange = 4):
    dataX, dataY, dataName = dataXYN
    endIndxList = findEndIndx(dataName)
    GroupNum = len(endIndxList)
    newDataX = []
    newDataY = []
    newDataName = []

    for i in xrange(GroupNum):
        if i == 0:
            curFirst = 0
            curEnd = endIndxList[i]
        else:
            curFirst = endIndxList[i-1]
            curEnd = endIndxList[i]
        curGroupDataX = dataX[curFirst:curEnd]
        curGroupDataY = dataY[curFirst:curEnd]
        curGroupDataName = dataName[curFirst:curEnd]
        for j in xrange(SpliceRange):
            curGroupDataX = numpy.insert(numpy.append(curGroupDataX, dataX[curEnd-1]), 0, dataX[curFirst])
            curGroupDataY = numpy.insert(numpy.append(curGroupDataY, -1), 0 ,-1)
            curGroupDataName = [dataName[curFirst]] + curGroupDataName + [dataName[curEnd-1]]
        newDataX.append(curGroupDataX)
        newDataY.append(curGroupDataY)
        newDataName.append(curGroupDataName)

    newDataXYN = [newDataX, newDataY, newDataName]
    return newDataXYN

