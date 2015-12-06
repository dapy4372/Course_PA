import sys
import numpy
import cPickle
target_name = sys.argv[1]

def read_examples(filename, sparm): # add in "sparm" later
    datasets = loadDataset(filename = filename, totalSetNum = 3)
    trainSet = datasets[0]
    print len(trainSet[2])
    ret = [] # initializing the return item, an empty list
    for idx in xrange(len(trainSet[2])):
        dummy_tuple = (trainSet[0][idx] , trainSet[1][idx])
        ret.append(dummy_tuple)
    return ret

def loadDataset(filename, totalSetNum):
    print '... loading data'
    # Load the dataset
    f = open(filename, 'rb')
    datasets = cPickle.load(f)
    f.close()
    return datasets

"""if __name__ == '__main__':
    print "hello!"
    ret = read_examples(target_name)
    print ret[0]
    print ret[1]
    print len(ret)
    # First dimension -> train/valid/test
    # Second dimension-> mfcc/label/speaker_sentence_frame
    # Third dimension -> """
