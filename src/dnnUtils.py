import utils
import numpy
import theano
def EvalandResult(Model, indexList, modelType):
    result = []
    Losses = []
    for i in xrange(len(indexList)):
        thisLoss, thisResult = Model(indexList[i][0], indexList[i][1])
        result += thisResult.tolist()
        Losses.append(thisLoss)
    FER = numpy.mean(Losses)
    print ((modelType + ' FER,%f') % (FER * 100))
    return result

def writeResult(result, filename, setNameList):
    f = open(filename, 'w')
    for i in xrange(len(result)):
        f.write(setNameList[i] + ',' + str(result[i]) + '\n')
    f.close()

def makeBatch(totalSize, batchSize = 32):
    numBatchSize = totalSize / batchSize
    indexList = [[i * batchSize, (i + 1) * batchSize] for i in xrange(numBatchSize)]
    indexList.append([numBatchSize * batchSize, totalSize * batchSize - 1])
    return indexList

def getParamsValue(nowParams):
    params = []
    for i in xrange(len(nowParams)):
        params.append(nowParams[i].get_value())
    return params

def setParamsValue(preParams, nowParams):
    for i in xrange(len(preParams)):
        nowParams[i].set_value(preParams[i])

def Dropout(rng, input, inputNum, D = None, dropoutProb = 1):
    D_values = numpy.asarray(
              rng.binomial( size = (inputNum,), n = 1, p = dropoutProb ),
              dtype=theano.config.floatX )
    D = theano.shared( value=D_values, name='D', borrow=True )
    return input * D

def findSpliceIdxList(dataY):
    spliceIdxList = []
    print dataY
    for i in xrange(len(dataY)):
        if dataY[i] == -1:
            continue
        else:
            spliceIdxList.append(i)
    return spliceIdxList

def splice(dataset, w):
    dataX, dataY, dataName = dataset
    spliceIdxList = findSpliceIdxList(dataY)
#x = T.fmatrix('x')
#i = T.lscalar('i')
#s = T.concatenate([x[j] for j in xrange(2*w+1)], axis = 0)
#c = T.theano.shared(numpy.array(([[0,0],[1,1],[2,2],[3,3],[4,4],[5,5],[6,6],[7,7],[8,8],[9,9]]),dtype=theano.config.floatX))
    spliceDataX = []
    spliceDataY = []
    spliceDataName = []
    for j in spliceIdxList:
        spliceDataX.append(numpy.concatenate( [dataX[j+i] for i in xrange(-w, w+1)], axis = 0))
        spliceDataY.append(dataY[j])
        spliceDataName.append(dataName[j])
    spliceDataX = numpy.array(spliceDataX, dtype=theano.config.floatX)
    spliceDataY = numpy.array(spliceDataY, dtype=theano.config.floatX)
    return spliceDataX, spliceDataY, spliceDataName

class Parameters(object):
    def __init__(self, filename):
       title, parameter           = utils.readFile2(filename)
       self.SHUFFLE               = bool(parameter[title.index('shuffle')])
       self.momentum              = float(parameter[title.index('momentum')])
       self.dnnWidth              = int(parameter[title.index('width')])
       self.dnnDepth              = int(parameter[title.index('depth')])
       self.batchSizeForTrain     = int(parameter[title.index('batchSize')])
       self.learningRate          = float(parameter[title.index('learningRate')])
       self.dropoutHiddenProb     = float(parameter[title.index('hiddenProb')])
       self.dropoutInputProb      = float(parameter[title.index('inputProb')])
       self.datasetFilename       = parameter[(title.index('dataSetFilename'))].strip('\n')
       self.datasetType           = parameter[(title.index('dataSetType'))].strip('\n')
       self.maxEpoch              = int(parameter[title.index('maxEpoch')])
       self.inputDimNum           = int(parameter[title.index('inputDimNum')])
       self.outputPhoneNum        = int(parameter[title.index('outputPhoneNum')])
       self.seed                  = int(parameter[title.index('seed')])
       self.earlyStop             = int(parameter[title.index('earlyStop')])
       self.L1Reg                 = float(parameter[title.index('L1Reg')])
       self.L2Reg                 = float(parameter[title.index('L2Reg')])
       self.outputFilename = (str(self.datasetType) 
                              + '_s_' + str(self.SHUFFLE)
                              + '_m_' + str(self.momentum)
                              + '_dw_'+ str(self.dnnWidth)
                              + '_dd_'+ str(self.dnnDepth)
                              + '_b_' + str(self.batchSizeForTrain)
                              + '_lr_'+ str(self.learningRate)
                              + '_di_'+ str(self.dropoutInputProb)
                              + '_dh_'+ str(self.dropoutHiddenProb) )
       self.bestModelFilename   = '../model/' + self.outputFilename
       self.testResultFilename  = '../result/test_result/' + self.outputFilename + '.csv'
       self.validResultFilename = '../result/valid_result/' + self.outputFilename + '.csv'
       self.validSmoothedResultFilename = '../result/smoothed_valid_result/' + self.outputFilename + '.csv' 
       self.testSmoothedResultFilename  = '../result/smoothed_test_result/' + self.outputFilename + '.csv' 
       self.logFilename = '../log/' + self.outputFilename + '.log'
       self.rng = numpy.random.RandomState(self.seed)


