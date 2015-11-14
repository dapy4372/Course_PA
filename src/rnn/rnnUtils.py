import utils
import numpy as np
import theano
import theano.tensor as T
import updateMethod

# SpeakerNameList should be a total setName. (e.g. trainSetName)
# It will return the idex of each sentence interval. ( e.g. (200, 456) )
def findSentenceInterval(datasetName):
    prevName, _ = utils.namepick(datasetName[0])
    start = 0
    end = 0
    sentenceInterval = []
    for i in xrange(1, len(datasetName)):
        curName, _ = utils.namepick(datasetName[i])
        if(prevName != curName):
            end = i+1
            sentenceInterval.append( (start, end) )
            start = i
        prevName = curName
    sentenceInterval.append( (start, len(datasetName)-1) )
    return sentenceInterval

def toBeSentence(subset, interval):
    sentencedSubset = []
    for i in xrange(len(interval)):
        sentencedSubset.append( subset[ interval[i][0] : (interval[i][1]+1) ] )  # Cuz the index will be -1, plusing 1 to avoid.
    return sentencedSubset

def cutSentence(Set,size):
    finalSet=[]
    for i in range (3):
        finalSet.append([])
    for i in range (len(Set[0])):
        for j in range((len(Set[0][i])/size)):
            finalSet[0].append(Set[0][i][j*size:(j+1)*size])
            finalSet[1].append(Set[1][i][j*size:(j+1)*size])
            finalSet[2].append(Set[2][i][j*size:(j+1)*size])
        if len(Set[0][i])%size!=0:
            j=(len(Set[0][i])/size)

            finalSet[0].append(Set[0][i][j*size:len(Set[0][i])])
            finalSet[1].append(Set[1][i][j*size:len(Set[0][i])])
            finalSet[2].append(Set[2][i][j*size:len(Set[0][i])])
    return finalSet

def toBeClipSentence(subset, interval, clipSize = 20):
    sentencedSubset = []
    for i in xrange(len(interval)):
        clipNum = (interval[i][1] - interval[i][0] + 1) / clipSize
        for j in xrange(clipNum):
            sentencedSubset.append( subset[ interval[i][0] + (j*clipSize) : (interval[i][1]+1) + ((j+1) * clipSize) ] )
        sentencedSubset.append( subset[ interval[i][0] + clipNum * clipSize : interval[i][1] + (clipNum+1) * clipSize ] )
    return sentencedSubset

# make original data sentenced
def makeDataSentence(dataset):
    datasetX, datasetY, datasetName = dataset
    sentenceInterval = findSentenceInterval(datasetName)
    return toBeSentence(datasetX, sentenceInterval), toBeSentence(datasetY, sentenceInterval), toBeSentence(datasetName, sentenceInterval)

def sharedDataXY(dataX, dataY, borrow=True):
    sharedX = theano.shared(np.asarray(dataX, dtype=theano.config.floatX), borrow=True)
    #TODO does't work in GPU for sharedY
    sharedY = theano.shared(np.asarray(dataY, dtype=theano.config.floatX), borrow=True)
    return [sharedX, sharedY, T.cast(sharedY,'int32')]

def clearSharedDataXY(sharedX, sharedY):
    sharedX.set_value([[]])
    sharedY.set_value([])

def chooseUpdateMethod(grads, params, P):
    if P.updateMethod == 'Momentum':
        return updateMethod.Momentum(grads, params, P.momentum)
    if P.updateMethod == 'RMSProp':
        return updateMethod.RMSProp(grads, params)
    if P.updateMethod == 'Adagrad':
        return updateMethod.Adagrad(grads, params)

# GP means gradient and parameter(W and b).
def printGradsParams(grads, params, rnnDepth):
    for i in xrange(0, (3 * rnnDepth), 6):
        print ( '================ Layer %d ================' % (i/3 + 1))
        printNpArrayMeanStdMaxMin("Gradient of Wi1", grads[i])
        printNpArrayMeanStdMaxMin("Gradient of Wh1", grads[i+1])
        printNpArrayMeanStdMaxMin("Gradient of bh1", grads[i+2])
        printNpArrayMeanStdMaxMin("Gradient of Wi2", grads[i+3])
        printNpArrayMeanStdMaxMin("Gradient of Wh2", grads[i+4])
        printNpArrayMeanStdMaxMin("Gradient of bh2", grads[i+5])
        printNpArrayMeanStdMaxMin("Wi1 ", params[i])   
        printNpArrayMeanStdMaxMin("Wh1 ", params[i+1])
        printNpArrayMeanStdMaxMin("bh1 ", params[i+2])
        printNpArrayMeanStdMaxMin("Wi1 ", params[i+3])  
        printNpArrayMeanStdMaxMin("Wh1 ", params[i+4])
        printNpArrayMeanStdMaxMin("bh1 ", params[i+5])
    print ( '================ Output Layer ================' )
    printNpArrayMeanStdMaxMin("Gradient of Wo", grads[6*rnnDepth])
    printNpArrayMeanStdMaxMin("Gradient of bo", grads[6*rnnDepth+1])
    printNpArrayMeanStdMaxMin("Wo ", params[6*rnnDepth]) 
    printNpArrayMeanStdMaxMin("bo ", params[6*rnnDepth+1])

def printNpArrayMeanStdMaxMin(name, npArray):
    print(" #%s \t mean = %f \t std = %f \t max = %f \t min = %f" % (name, np.mean(npArray), np.std(npArray), np.amax(npArray), np.amin(npArray) ))
"""
def printNpArrayMeanStdMaxMin(name, npArray):
    print(" #%s" % (name))
    print "======= mean ======"
    print np.mean(npArray, axis=1)
    print "======= std  ======"
    print np.std(npArray, axis=1)
    print "======= max  ======"
    print np.amax(npArray, axis=1)
    print "======= min ======"
    print np.amin(npArray, axis=1)
"""


def EvalandResult(Model, totalSentNum, setX, setY, modelType):
    result = []
    Losses = []
    for i in xrange(totalSentNum):
        thisLoss, thisResult = Model( np.array(setX[i]).astype(dtype='float32'), np.array(setY[i]).astype(dtype='int32') )
        result.append(thisResult.tolist())
        Losses.append(thisLoss)
    FER = np.mean(Losses)
    print ((modelType + ' FER,%f') % (FER * 100))
    return result

def writeResult(result, filename, setNameList):
    f = open(filename, 'w')
    print result
    for i in xrange(len(result)):
        for j in xrange(len(result[i])):
            f.write(setNameList[i][j] + ',' + str(result[i][j]) + '\n')
    f.close()

def makeBatch(totalSize, batchSize = 32):
    numBatchSize = totalSize / batchSize
    indexList = [[i * batchSize, (i + 1) * batchSize] for i in xrange(numBatchSize)]
    indexList.append([numBatchSize * batchSize, totalSize])
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
    D_values = np.asarray(
              rng.binomial( size = (inputNum,), n = 1, p = dropoutProb ),
              dtype=theano.config.floatX )
    D = theano.shared( value=D_values, name='D', borrow=True )
    return input * D

def findCenterIdxList(dataY):
    spliceIdxList = []
    for i in xrange(len(dataY)):
        if dataY[i] == -1:
            continue
        else:
            spliceIdxList.append(i)
    return spliceIdxList

def splicedX(x, idx):
    spliceWidth = 4
    return T.concatenate([ (T.stacklists([x[j+i] for j in [idx] ])) for i in xrange(-spliceWidth, spliceWidth+1)])

def splicedY(y, idx):    
    return T.concatenate([y[i] for i in [idx]])
    
class Parameters(object):
    def __init__(self, filename):
       title, parameter           = utils.readSetting(filename)
       self.shuffle               = int(parameter[title.index('shuffle')])
       self.momentum              = float(parameter[title.index('momentum')])
       self.rnnWidth              = int(parameter[title.index('width')])
       self.rnnDepth              = int(parameter[title.index('depth')])
       self.cutSentSize           = int(parameter[title.index('cutSentSize')])
       self.clipRange             = float(parameter[title.index('clipRange')])
#       self.batchSizeForTrain     = int(parameter[title.index('batchSize')])
       self.learningRate          = float(parameter[title.index('learningRate')])
       self.learningRateDecay     = float(parameter[title.index('learningRateDecay')])
       self.datasetFilename       = parameter[(title.index('dataSetFilename'))].strip('\n')
       self.datasetType           = parameter[(title.index('dataSetType'))].strip('\n')
       self.updateMethod          = parameter[(title.index('updateMethod'))].strip('\n')
       self.maxEpoch              = int(parameter[title.index('maxEpoch')])
       self.inputDimNum           = int(parameter[title.index('inputDimNum')])
       self.outputPhoneNum        = int(parameter[title.index('outputPhoneNum')])
       self.seed                  = int(parameter[title.index('seed')])
       self.earlyStop             = int(parameter[title.index('earlyStop')])
       self.L1Reg                 = float(parameter[title.index('L1Reg')])
       self.L2Reg                 = float(parameter[title.index('L2Reg')])
       self.outputFilename = (str(self.datasetType) + '_' + (str(self.updateMethod))
                              + '_s_' + str(self.shuffle)
                              + '_m_' + str(self.momentum)
                              + '_dw_'+ str(self.rnnWidth)
                              + '_dd_'+ str(self.rnnDepth)
                              + '_cs_'+ str(self.cutSentSize)
                              + '_cr_'+ str(self.clipRange)
#+ '_b_' + str(self.batchSizeForTrain)
                              + '_lr_'+ str(self.learningRate)
                              + '_lrd_' + str(self.learningRateDecay) )
       self.bestModelFilename   = '../model/' + self.outputFilename
       self.trainProbFilename   = '../prob/' + self.outputFilename + '.ark'
       self.testResultFilename  = '../result/test_result/' + self.outputFilename + '.csv'
       self.validResultFilename = '../result/valid_result/' + self.outputFilename + '.csv'
       self.validSmoothedResultFilename = '../result/smoothed_valid_result/' + self.outputFilename + '.csv' 
       self.testSmoothedResultFilename  = '../result/smoothed_test_result/' + self.outputFilename + '.csv' 
       self.logFilename = '../log/' + self.outputFilename + '.log'
       self.rng = np.random.RandomState(self.seed)


