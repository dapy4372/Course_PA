import numpy as np
import random
import cPickle
from operator import itemgetter, attrgetter
from numpy import genfromtxt
from utils import makePkl, namepick
import gzip
#from StringIO import StringIO
DEBUG = False
NNET = False
random.seed(1234)

featureType     = 'fbank'
trainArkDir     = '../data/' + featureType + '/train.ark'
trainLabelDir   = '../data/label/train_int.lab'
testArkDir      = '../data/' + featureType + '/test.ark'
datasetFilename = '../pkl/' + featureType + '_dataset_without_preprocessing.pkl'
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
        dataset.append([dataX[i][2], dataY[i][2], dataX[i][0]+'_'+str(dataX[i][1])]) 
        """
        npDataX[i] = dataX[i][2]
        npDataY[i] = dataY[i][2]
        dataName[i] = dataX[i][0] + '_' + str(dataX[i][1])
        """
    return dataset

if __name__ == '__main__':
    print '... covert training set'
    trainLineNum = countLineNum(trainArkDir)
    trainSet = covertData(fileArkName = trainArkDir, fileLabelName = trainLabelDir, LineNum = trainLineNum)
    print '... covert test set'
    testLineNum = countLineNum(testArkDir)
    testSet = covertData(fileArkName = testArkDir, LineNum = testLineNum, existY = False)
    print '... make pkl file'
    makePkl([trainSet, testSet], datasetFilename)
