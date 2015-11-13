import sys
import numpy
import utils
import datahandler

datasetFilename = sys.argv[1]
outputFilename = sys.argv[2]

trainSet, validSet, testSet = utils.loadPkl(datasetFilename)
print '...normalization'
trainSet = datahandler.normalization(trainSet)
validSet = datahandler.normalization(validSet)
testSet  = datahandler.normalization(testSet)

print '...prepare for splice'
trainSet = datahandler.prepareSplice(trainSet)
validSet = datahandler.prepareSplice(validSet)
testSet  = datahandler.prepareSplice(testSet)
datasets = [trainSet, validSet, testSet]
utils.makePkl(datasets, outputFilename)
