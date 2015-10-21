import numpy as np
import random
import cPickle
from operator import itemgetter, attrgetter
from numpy import genfromtxt
from utils import makePkl, namepick
import gzip

random.seed(1234)
dirPath = '../data/fbank_valid/'
trainArkFilename   = dirPath + 'train.ark'
trainLabelFilename = dirPath + 'train.lab'
validArkFilename   = dirPath + 'dev.ark'
validLabelFilename = dirPath + 'dev.lab'
testArkFilename    = dirPath + 'test.ark'
outputPklFilename  = '../pkl/fbank_dataset_without_preprocessing.pkl'
dim = 69

def countLineNum(fileArkName):
    f = open(fileArkName, 'rb')
    num = 0
    for i in f:
        num += 1
    f.close()
    return num

def covertData(fileArkName, LineNum, fileLabelName = None, existY = True):
    dataX = []
    dataY = []
    npDataX = np.zeros((LineNum, dim))
    npDataY = np.zeros((LineNum,))
    dataName = [''] *  LineNum
    fileForX = open(fileArkName, 'rb')
    for curLine in fileForX:
        curLine = curLine.strip()
        curLine = curLine.split()
        name, number = namepick(curLine[0])
        feature = np.ndarray(dim, dtype=float)
        for i in xrange(dim):
            feature[i] = np.float32(curLine[i+1])
        dataX.append([name, number,feature])
    if fileLabelName is not None:
        dataX = sorted(dataX, key=itemgetter(0,1))  
    
    if existY:
        fileForY = open(fileLabelName, 'rb')
        for curLine in fileForY:
            curLine = curLine.strip()
            curLine = curLine.split(',')
            name, number = namepick(curLine[0])
            label = int(curLine[1])
            dataY.append([name, number, label])
        if existY:
            dataY = sorted(dataY, key=itemgetter(0,1))
    else:
        dataY = np.zeros((len(dataX), 3))
    dataset = []
    for i in xrange(len(dataX)):
        npDataX[i] = dataX[i][2]
        npDataY[i] = dataY[i][2]
        dataName[i] = dataX[i][0] + '_' + str(dataX[i][1])

    return npDataX, npDataY, dataName

if __name__ == '__main__':
    print '... covert training set'
    trainLineNum = countLineNum(trainArkFilename)
    trainSet = covertData(fileArkName = trainArkFilename, fileLabelName = trainLabelFilename, LineNum = trainLineNum)
    print '... covert valid set'
    validLineNum = countLineNum(validtArkFilename)
    validSet = covertData(fileArkName = validArkFilename, LineNum = validLineNum, existY = True)
    print '... covert test set'
    testLineNum = countLineNum(testArkFilename)
    testSet = covertData(fileArkName = testArkFilename, LineNum = testLineNum, existY = False)
    print '... make pkl file'
    makePkl([trainSet, validSet, testSet], outputPklFilename)
