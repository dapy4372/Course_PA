import cPickle
import gzip
import os
import sys
import timeit
import numpy
import theano
import theano.tensor as T

def load_pkl(filename):
    f = open(filename, 'rb')
    temp = cPickle.load(f)
    f.close()
    return temp

def namepick(name):
    part = name.split('_')
    return (part[0] + '_' + part[1]), int(part[2])

def findEndIndxGroup(dataName):
    endIndxGroup = []
    curName, number = namepick(dataName[0])
    dataLen = len(dataName)
    for i in xrange(dataLen):
        nextName, nextNum = namepick(dataName[i])
        if(curName != nextName):
            endIndxGroup.append(i)
            curName = nextName
    endIndxGroup.append(dataLen)
    return endIndxGroup

def load_data(filename, totalSetNum):

    print '... loading data'
    # Load the dataset
    f = open(filename, 'rb')
    dataset = cPickle.load(f)
    f.close()
    dataset = numpy.asarray(dataset)
    
    def sharedDataset(data, borrow=True):
        dataX, dataY, dataName = data.T
        sharedX = theano.shared(numpy.asarray(dataX, dtype=theano.config.floatX), borrow=borrow)
        #TODO does't work in GPU for sharedY
        sharedY = theano.shared(numpy.asarray(dataY, dtype=theano.config.floatX), borrow=borrow)
        return (sharedX, T.cast(sharedY, 'int32'), dataName)
    
    for i in xrange(totalSetNum):
        dataset[i] = sharedDataset(dataset[i])
    return dataset

def makePkl(dataset, filename):
    f = open(filename, 'wb')
    cPickle.dump(dataset, f, protocol=2)
    f.close()

def readFile(filename):
    f = open(filename, 'r')
    name = []
    label = []
    for i in f:
        part = i.split(',')
        name.append(part[0])
        label.append(part[1])
    f.close()
    return name, label

def readModelPkl(filename):
    f = open(filename, 'rb')
    bestModel = cPickle.load(f)
    return bestModel

def pickResultFilename(resultFilename):
    tmp = resultFilename.split('/')
    return tmp[len(tmp)-1]

def findSpeakerInterval(speakerNameList):
    prevName, dummy = namepick(speakerNameList[0])
    print "prev"
    print prevName
    start = 0
    end = 0
    speakerInterval = []
    speakerNameListLen = len(speakerNameList)
    for i in xrange(speakerNameListLen):
        curName, dummy = namepick(speakerNameList[i])
        if (prevName != curName):
            end = i - 1
            speakerInterval.append((start, end))
            start = i
        prevName = curName
    speakerInterval.append((start, speakerNameListLen - 1))
    return speakerInterval
