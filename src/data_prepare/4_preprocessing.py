import cPickle
import numpy as np
import random
import sys
import utils

random.seed(1234)
NORMALIZE = True
SPLICE = True
datasetFilename = '../pkl/fbank_dataset_without_preprocessing.pkl'
outputFilename = '../pkl/fbank_dataset.pkl'
dim = 69

def normalization(dataXY, borrow = True):
    dataX, dataY, dataName = dataXY
    dataX -= np.mean(dataX, axis = 0)
    dataX /= np.sqrt(np.var(dataX, axis = 0))
    newDataXY = [dataX, dataY, dataName]
    return newDataXY

def splice(dataXY, borrow = True):
    dataX, dataY, dataName = dataXY
    newDataX = np.zeros((len(dataX), dim))
    curName, number = utils.namepick(dataName[0])
    endIndxGroup = utils.findEndIndxGroup(dataName)
    indexOfGroup = 0
    curFirst = 0
    curEnd = endIndxGroup[indexOfGroup]
    for i in xrange(len(dataX)):
        if i == curEnd:
            curFirst = curEnd
            curEnd = endIndxGroup[indexOfGroup + 1]
            indexOfGroup += 1
        if i < curFirst+4:
            newDataX[i] = dataX[i] * 9
        elif i >= curEnd-4:
            newDataX[i] = dataX[i] * 9
        else:
            for j in xrange(-4, 5, 1):
                newDataX[i] += dataX[i + j]

    newDataXY = newDataX, dataY, dataName
    return newDataXY

if __name__ == '__main__':
    trainSet, validSet, testSet = utils.load_pkl(datasetFilename)
    if NORMALIZE:
        print '...normalization'
        trainSet = normalization(trainSet)
        validSet = normalization(validSet)
        testSet = normalization(testSet)
    if SPLICE:
        print '...splice'
        trainSet = splice(trainSet)
        validSet = splice(validSet)
        testSet = splice(testSet)
    dataset = [trainSet, validSet, testSet]
    utils.makePkl(dataset, outputFilename)
