import cPickle
import numpy
import random
import sys
import utils

random.seed(1234)
NORMALIZE = True
SPLICE = True
datasetFilename = '../../pkl/fbank_dataset_without_preprocessing.pkl'
outputFilename = '../../pkl/fbank_dataset.pkl'
dim = 69
width = 4
def normalization(dataXY, borrow = True):
    dataX, dataY, dataName = dataXY
    dataX -= numpy.mean(dataX, axis = 0)
    dataX /= numpy.sqrt(numpy.var(dataX, axis = 0))
#newDataXY = [dataX, dataY, dataName]
    return dataX, dataY,dataName

def splice(dataset, borrow = True):
    dataX, dataY, dataName = dataset
    speakerIntervalList = utils.findSpeakerInterval(dataName)
    speakerIntervalListLen = len(speakerIntervalList)

    def spliceCore(start, end, w, spliceDataX, borrow=True):
        spliceDataX = dataX[0]
        print spliceDataX.eval()

        for i in xrange(width):
            thisIntervalData = numpy.vstack((thisIntervalData, thisIntervalData[totalLen-1]))
        for i in xrange(width):
            thisIntervalData = numpy.vstack((thisIntervalData[0], thisIntervalData))
        for i in xrange(totalLen):
            spliceDataX.append(thisIntervalData[i:(width*2+1)+i].flatten())

    spliceDataX = []
    for i in xrange(speakerIntervalListLen):
        thisIntervalLen = speakerIntervalList[i][1] - speakerIntervalList[i][0]
        spliceCore(speakerIntervalList[i][0], speakerIntervalList[i][1], width, spliceDataX)
    spliceDataX = numpy.array(spliceDataX)
    print spliceDataX[0]
    return spliceDataX, dataY, dataName 
"""
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
"""
