import theano
import cPickle
import numpy as np
import utils
import random
import sys
random.seed(1234)
CREATE_VALID_SET = True
NORMALIZE = True
SPLICE = True
datasetFilename = sys.argv[1]
dataDim = sys.argv[2]
outputFilename = sys.argv[3]
"""
def normalization(dataXY, borrow = True):
    dataX, dataY, dataName = dataXY
    dataX -= np.mean(dataX, axis = 0)
    dataX /= np.sqrt(np.var(dataX, axis = 0))
    newDataXY = [dataX, dataY, dataName]
    return newDataXY
"""
def normalizing(dataset, borrow = True):
    for i in xrange(len(dataset)):
        tmp = np.asarray(dataset[i])
        tmp = tmp.T[0]
        print np.mean(tmp,axis=0) 
#tmppp -= np.mean(tmppp)
#print np.mean(tmppp)
#tmp[0] -= np.sqrt(np.var(tmp[0]))
    return dataset

"""
def slice(dataXY, borrow = True):
    dataX, dataY, dataName = dataXY
    newDataX = np.zeros((len(dataX), dim))
    curName, number = namepick(dataName[0])
    endIndxGroup = findEndIndxGroup(dataName)
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
"""
def splice(dataset, borrow = True):
    for s in xrange(len(dataset)):
        thisSetNum = len(dataset[s][0])
        newDataX = np.zeros((thisSetNum, dim))
        curName, number = utils.namepick(dataset[s][2][0])
        endIndxGroup = utils.findEndIndxGroup(dataset[s][2])
        indexOfGroup = 0
        curFirst = 0
        curEnd = endIndxGroup[indexOfGroup]
        for i in xrange(thisSetNum):
            if i == curEnd:
                curFirst = curEnd
                curEnd = endIndxGroup[indexOfGroup + 1]
                indexOfGroup += 1
            if i < curFirst+4:
                newDataX[i] = dataset[s][0][i] * 9
            elif i >= curEnd-4:
                newDataX[i] = dataset[s][0][i] * 9
            else:
                for j in xrange(-4, 5, 1):
                    newDataX[i] += dataset[s][0][i + j]
            dataset[s][0] = newDataX
        return dataset

def createValidSet(trainSet):
    npTrainSet = np.asarray(trainSet)
#npTrainSet = trainSet
    speakerInterval = utils.findSpeakerInterval(npTrainSet.T[2])
    random.shuffle(speakerInterval)
    totalSpeakerNum = len(speakerInterval)
    validSpeakerNum = totalSpeakerNum / 10
    
    validSet = []
    for i in speakerInterval[:validSpeakerNum]:
        validSet += trainSet[i[0]:i[1]+1]

    trainSetnew = []
    for i in speakerInterval[validSpeakerNum:]:
        trainSetnew += trainSet[i[0]:i[1]+1]
    return trainSetnew, validSet

if __name__ == '__main__':
    dataset = utils.load_pkl(datasetFilename)
    if CREATE_VALID_SET:
        print '...create valid set'
        trainSet, validSet = createValidSet(dataset[0])
        testSet = dataset[1]
        print len(validSet)
        print len(trainSet)
        dataset = [trainSet, validSet, testSet]
    if NORMALIZE:
        print '...normalizing'
        dataset = normalizing(dataset)
    if SPLICE:
        print '...splicing'
        dataset = splice(dataset)
    utils.makePkl(dataset, outputFilename)
