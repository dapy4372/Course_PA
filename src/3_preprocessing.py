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
def normalization(dataXY, borrow = True):
    dataX, dataY, dataName = dataXY
    dataX -= np.mean(dataX, axis = 0)
    dataX /= np.sqrt(np.var(dataX, axis = 0))
    newDataXY = [dataX, dataY, dataName]
    return newDataXY
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

def createValidSet(trainSet):
    npTrainSet = np.asarray(trainSet)
    speakerInterval = utils.findSpeakerInterval(trainSet[2])
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
        print trainSet[0]
        print trainSet[1]
        print trainSet[2]
    if NORMALIZE:
        print '...normalizing'
        dataset = normalizing(dataset)
    if SPLICE:
        print '...splicing'
        dataset = splice(dataset)
    utils.makePkl(dataset, outputFilename)
