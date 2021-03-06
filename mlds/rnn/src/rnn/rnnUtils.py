import utils
import numpy as np
import theano
import theano.tensor as T
import updateMethod
import cPickle

# Save model
def saveModelPkl(model, P, modelFilename):
    modelPkl = [model, P]
    utils.makePkl(modelPkl, modelFilename)

# Read model
def readModelPkl(filename):
    f = open(filename, 'rb')
    model, P = cPickle.load(f)
    return model, P

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
            end = i + 1 # +1 Cuz of being idx 
            sentenceInterval.append( (start, end) )
            start = i
        prevName = curName
    sentenceInterval.append( (start, len(datasetName)+1) ) # +1 Cuz of being idx 
    return sentenceInterval

def toBeSentence(subset, interval):
    sentencedSubset = []
    for i in xrange(len(interval)):
        sentencedSubset.append( subset[ interval[i][0] : (interval[i][1]) -1] )  # Cuz the index will be -1, plusing 1 to avoid.
    return sentencedSubset

# make original data sentenced
def makeDataSentence(dataset):
    datasetX, datasetY, datasetName = dataset
    sentenceInterval = findSentenceInterval(datasetName)
    return toBeSentence(datasetX, sentenceInterval), toBeSentence(datasetY, sentenceInterval), toBeSentence(datasetName, sentenceInterval)

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

def cutSentenceAndFill(Set,size):
    finalSet = []
    for i in range (4):
        finalSet.append([])
    for i in xrange(len(Set[0])):
        sentLen = len(Set[0][i])
        for j in xrange(sentLen/size):
            finalSet[0].append(Set[0][i][j*size:(j+1)*size])
            finalSet[1].append(Set[1][i][j*size:(j+1)*size])
            finalSet[2].append(Set[2][i][j*size:(j+1)*size])
            finalSet[3].append(np.ones(size))
        if sentLen % size != 0:
            j = sentLen / size
            tmpSize = size - (sentLen % size)
            tmpSet = [ Set[0][i][j*size:sentLen], Set[1][i][j*size:sentLen], Set[2][i][j*size:sentLen], np.ones(sentLen%size) ]
            zs = np.zeros(48).astype(dtype = theano.config.floatX)
            for k in xrange(tmpSize):
                tmpSet[0] = np.vstack((tmpSet[0], zs))
                tmpSet[1] = np.append(tmpSet[1], 0)
                tmpSet[2] = np.append(tmpSet[2], "null")
                tmpSet[3] = np.append(tmpSet[3], 0)
            finalSet[0].append(tmpSet[0])
            finalSet[1].append(tmpSet[1])
            finalSet[2].append(tmpSet[2])
            finalSet[3].append(tmpSet[3])
    return finalSet

def cutSentenceAndSlide(Set, size, move):
    finalSet = []
    for i in range (4):
        finalSet.append([])
    for i in xrange(len(Set[0])):
        sentLen = len(Set[0][i])
        j = 0
        while (j+size) <= sentLen:
            finalSet[0].append(Set[0][i][j:(j+size)])
            finalSet[1].append(Set[1][i][j:(j+size)])
            finalSet[2].append(Set[2][i][j:(j+size)])
            finalSet[3].append(np.ones(size))
            j += move
        if j < sentLen:
            tmpSize = j + size - sentLen
            tmpSet = [ Set[0][i][j:sentLen], Set[1][i][j:sentLen], Set[2][i][j:sentLen], np.ones(sentLen-j) ]
            zs = np.zeros(48).astype(dtype = theano.config.floatX)
            for k in xrange(tmpSize):
                tmpSet[0] = np.vstack((tmpSet[0], zs))
                tmpSet[1] = np.append(tmpSet[1], 0)
                tmpSet[2] = np.append(tmpSet[2], "null")
                tmpSet[3] = np.append(tmpSet[3], 0)
            finalSet[0].append(tmpSet[0])
            finalSet[1].append(tmpSet[1])
            finalSet[2].append(tmpSet[2])
            finalSet[3].append(tmpSet[3])
    return finalSet

def fillBatch(Set, batchSize):
    sentAmount = len(Set[0])
    sentLen = len(Set[0][0])
    if sentAmount % batchSize != 0:
        tmpSent = [[], [], [], []]
        tmpSent[0] = np.zeros((sentLen,48)).astype(dtype = theano.config.floatX)
        tmpSent[1] = tmpSent[3] = np.zeros(sentLen).astype(dtype = theano.config.floatX)
        tmpSent[2] = ["null"] * sentLen
        for i in xrange(batchSize - (sentAmount % batchSize)):
            Set[0].append(tmpSent[0])
            Set[1].append(tmpSent[1])
            Set[2].append(tmpSent[2])
            Set[3].append(tmpSent[3])
    return Set

# Used to bulid model
def chooseUpdateMethod(grads, params, P):
    if P.updateMethod == 'Momentum':
        return updateMethod.Momentum(grads, params, P.momentum)
    if P.updateMethod == 'NAG':
        return updateMethod.NAG(grads, params, P.momentum)
    if P.updateMethod == 'RMSProp':
        return updateMethod.RMSProp(grads, params)
    if P.updateMethod == 'Adagrad':
        return updateMethod.Adagrad(grads, params)

# Used to debug
def printGradsParams(grads, params, rnnDepth):
    for i in xrange(0, rnnDepth):
        print ( '================ Layer %d ================' % i)
        printNpArrayMeanStdMaxMin("Gradient of Wi1", grads[i])
        printNpArrayMeanStdMaxMin("Gradient of Wh1", grads[i+1])
        printNpArrayMeanStdMaxMin("Gradient of bh1", grads[i+2])
        printNpArrayMeanStdMaxMin("Gradient of Wi2", grads[i+3])
        printNpArrayMeanStdMaxMin("Gradient of Wh2", grads[i+4])
        printNpArrayMeanStdMaxMin("Gradient of bh2", grads[i+5])
        printNpArrayMeanStdMaxMin("Gradient of a0 ", grads[i+6])
        printNpArrayMeanStdMaxMin("Gradient of a0r", grads[i+7])
        printNpArrayMeanStdMaxMin("Wi1 ", params[i])   
        printNpArrayMeanStdMaxMin("Wh1 ", params[i+1])
        printNpArrayMeanStdMaxMin("bh1 ", params[i+2])
        printNpArrayMeanStdMaxMin("Wi1 ", params[i+3])  
        printNpArrayMeanStdMaxMin("Wh1 ", params[i+4])
        printNpArrayMeanStdMaxMin("bh1 ", params[i+5])
        printNpArrayMeanStdMaxMin("a0  ", params[i+6])
        printNpArrayMeanStdMaxMin("a0r ", params[i+7])
    print ( '================ Output Layer ================' )
    printNpArrayMeanStdMaxMin("Gradient of Wo", grads[8*rnnDepth])
    printNpArrayMeanStdMaxMin("Gradient of bo", grads[8*rnnDepth+1])
    printNpArrayMeanStdMaxMin("Wo ", params[8*rnnDepth]) 
    printNpArrayMeanStdMaxMin("bo ", params[8*rnnDepth+1])

# Used to debug
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

# Used in getResult
def EvalnSaveResult(Model, totalBatchNum, oriSetX, oriSetY, oriSetN, oriSetM, filename, batchSize, modelType):
    result = []
    Losses = []
    f = open(filename, 'w')
    for i in xrange(totalBatchNum):
        setX = oriSetX[i * batchSize : (i+1) * batchSize]
        setY = oriSetY[i * batchSize : (i+1) * batchSize]
        setN = oriSetN[i * batchSize : (i+1) * batchSize]
        setM = oriSetM[i * batchSize : (i+1) * batchSize]
        setX = np.array(setX).astype(dtype='float32')
        setY = np.array(setY).astype(dtype='int32')
        setM = np.array(setM).astype(dtype='int32')
        setX = np.transpose(setX, (1, 0, 2))
        setY = np.transpose(setY, (1, 0))
        setM = np.transpose(setM, (1, 0))
        setN = np.asarray(setN)

        thisLoss, thisResult = Model( setX, setY, setM )
        thisResult = np.array(thisResult)
        thisResult = np.transpose(thisResult, (1,0))
        for i in xrange(len(setN)):
            for j in xrange(len(setN[i])):
                if setN[i][j] != "null":
                    f.write(setN[i][j] + ',' + str(thisResult[i][j]) + '\n')
        Losses.append(thisLoss)
    FER = np.mean(Losses)
    print ((modelType + ' FER,%f') % (FER * 100))

# Used to get the current parameters of the model   
def getParamsValue(nowParams):
    params = []
    for i in xrange(len(nowParams)):
        params.append(nowParams[i].get_value())
    return params

# Used to set the parameters of the model   
def setParamsValue(preParams, nowParams):
    for i in xrange(len(preParams)):
        nowParams[i].set_value(preParams[i])

class Parameters(object):
    def __init__(self, filename):
       title, parameter           = utils.readSetting(filename)
       self.shuffle               = int(parameter[title.index('shuffle')])
       self.momentum              = float(parameter[title.index('momentum')])
       self.rnnWidth              = int(parameter[title.index('width')])
       self.rnnDepth              = int(parameter[title.index('depth')])
       self.cutSentSize           = int(parameter[title.index('cutSentSize')])
       self.clipRange             = float(parameter[title.index('clipRange')])
       self.move                  = int(parameter[title.index('move')])
       self.batchSize             = int(parameter[title.index('batchSize')])
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
                              + '_MaxE_' + str(self.maxEpoch)
                              + '_mv_'+ str(self.move)
                              + '_b_' + str(self.batchSize)
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


