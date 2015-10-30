import numpy
import utils
from data_prepare import datahandler

datasetFilename = '../pkl/fbank_dataset_without_preprocessing.pkl'
outputFilename = '../pkl/fbank_dataset.pkl'

if __name__ == '__main__':
    trainSet, validSet, testSet = utils.load_pkl(datasetFilename)
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
