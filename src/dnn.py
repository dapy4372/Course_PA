import os
import sys
import timeit
import random
import numpy
import math
import theano
from theano import function
import theano.tensor as T
import sys
import utils
from theano.tensor.shared_randomstreams import RandomStreams

DEBUG = False
MODELEXIST = False
modelFilename = '../model/test_s_1_dw_1024_dd_5_b_32_lr_0.005_dh_0.1_di_0.1.model'
parameterFilename = sys.argv[1]

# TODO RMSprop
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

class HiddenLayer(object):
    def __init__(self, rng, input, inputNum, outputNum, dnnWidth, W = None, b = None, dropoutProb = 1.0, DROPOUT = False):
    #def __init__(self, input, P, W = None, b = None, DROPOUT = False):
        if DROPOUT == True:
            self.input = Dropout( rng = rng, input = input, inputNum = inputNum, dropoutProb = dropoutProb )
        else:
            self.input = input * dropoutProb
        if W is None:
            W_values = numpy.asarray(
                        rng.uniform( low=-numpy.sqrt(6. / (inputNum + outputNum)),
                        high = numpy.sqrt(6. / (inputNum + outputNum)),
                        size = (inputNum, outputNum) ), dtype=theano.config.floatX )
            W = theano.shared(value = W_values, name = 'W', borrow = True)
        else:
            W = theano.shared( value = numpy.array(W, dtype = theano.config.floatX), name='W', borrow = True )

        if b is None:
            #b_values = numpy.ones( (outputNum,), dtype=theano.config.floatX)
            b_values = numpy.asarray( rng.uniform( low = -1, high = 1, size = (outputNum,)), dtype=theano.config.floatX)
            b = theano.shared(value = b_values, name = 'b', borrow = True)
        else:
            b = theano.shared( value = numpy.array(b, dtype = theano.config.floatX), name='b', borrow = True )
        self.W = W
        self.b = b
        
        z = T.dot(self.input, self.W) + self.b
        
        # Maxout
        zT= z.dimshuffle(1,0)
        self.output = T.maximum(zT[0:dnnWidth/2],zT[dnnWidth/2:]).dimshuffle(1,0)
        
        # parameters of the model
        self.params = [self.W, self.b]

class OutputLayer(object):

    def __init__(self, input, inputNum, outputNum, rng, W = None, b = None):
    #def __init__(self, input, P, W = None, b = None):
        if W is None:
            W_values = numpy.asarray(
                        rng.uniform( low=-numpy.sqrt(6. / (inputNum + outputNum)),
                        high = numpy.sqrt(6. / (inputNum + outputNum)),
                        size = (inputNum, outputNum) ), dtype=theano.config.floatX )
            W = theano.shared(value = W_values, name = 'W', borrow = True)
        else:
            W = theano.shared( value = numpy.array(W, dtype = theano.config.floatX), name='W', borrow=True )

        if b is None:
            b_values = numpy.asarray( rng.uniform( low = -1, high = 1, size = (outputNum,)), dtype=theano.config.floatX)
            b = theano.shared(value = b_values, name = 'b', borrow = True)
            #b = theano.shared( value = numpy.ones( (outputNum,), dtype=theano.config.floatX ), name='b', borrow=True )
        else:
            b = theano.shared( value = numpy.array(b, dtype = theano.config.floatX), name='b', borrow=True )
        self.W = W
        self.b = b
        
        z = T.dot(input, self.W) + self.b
        
        # Softmax
        absZ = T.abs_(z)
        maxZ = T.max(absZ, axis=1)
        maxZ = T.reshape(maxZ, (maxZ.shape[0], 1))
        expZ = T.exp(z * 10 / maxZ)
        expZsum = T.sum(expZ, axis=1)
        expZsum = T.reshape(expZsum, (expZsum.shape[0], 1))
        self.p_y_given_x = (expZ / expZsum)
        
        # Find larget y_i
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input
    
    # Cross entropy
    def crossEntropy(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        # Check y and y_pred dimension
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # Check if y is the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()
    def getPredit():
         return self.y_pred

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
       self.outputFilename = (str(self.datasetType) 
                              + '_s_' + str(self.SHUFFLE)
                              + '_m_' + str(self.momentum)
                              + '_dw_'+ str(self.dnnWidth)
                              + '_dd_'+ str(self.dnnDepth)
                              + '_b_' + str(self.batchSizeForTrain)
                              + '_lr_'+ str(self.learningRate)
                              + '_di_'+ str(self.dropoutInputProb)
                              + '_dh_'+ str(self.dropoutHiddenProb) )
       self.bestModelFilename = '../model/' + self.outputFilename
       self.testResultFilename = '../result/test_result/' + self.outputFilename + '.csv'
       self.validResultFilename = '../result/valid_result/' + self.outputFilename + '.csv'
       self.validSmoothedResultFilename = '../result/smoothed_valid_result/' + self.outputFilename + '.csv' 
       self.testSmoothedResultFilename = '../result/smoothed_test_result/' + self.outputFilename + '.csv' 
       self.logFilename = '../log/' + self.outputFilename + '.log'
       self.rng = numpy.random.RandomState(1234)

class DNN(object):
    #def __init__(self, rng, input, inputNum, dnnWidth, dnnDepth, outputNum, params = None, DROPOUT = False):
    def __init__(self, input, P, params = None, DROPOUT = False):
        
        # Create Hidden Layers
        self.hiddenLayerList=[]
        self.hiddenLayerList.append(
            HiddenLayer(
                rng = P.rng,
                input = input,
                inputNum = P.inputDimNum,
                outputNum = P.dnnWidth,
                dnnWidth = P.dnnWidth,
                dropoutProb = P.dropoutHiddenProb,
                W = params[0],
                b = params[1],
                DROPOUT = DROPOUT ) )

        for i in xrange (P.dnnDepth - 1):
            self.hiddenLayerList.append(
                HiddenLayer(
                    input = self.hiddenLayerList[i].output,
                    rng = P.rng,
                    inputNum = P.dnnWidth/2,
                    outputNum = P.dnnWidth,
                    dnnWidth = P.dnnWidth,
                    dropoutProb = P.dropoutHiddenProb,
                    W = params[2 * (i + 1)],
                    b = params[2 * (i + 1) + 1],
                    DROPOUT = DROPOUT ) )

        # Output Layer
        self.outputLayer = OutputLayer(
              input = self.hiddenLayerList[P.dnnDepth - 1].output,
              inputNum = P.dnnWidth/2, 
              outputNum = P.outputPhoneNum,
              rng = P.rng,
              W = params[2 * P.dnnDepth],
              b = params[2 * P.dnnDepth + 1])
        
        # Weight decay
        # L1 norm ; one regularization option is to enforce L1 norm to be small
        self.L1 = 0
        for i in xrange(P.dnnDepth):
             self.L1 += abs(self.hiddenLayerList[i].W).sum()
        self.L1 += abs(self.outputLayer.W).sum()
        # square of L2 norm ; one regularization option is to enforce square of L2 norm to be small
        self.L2_sqr = 0
        for i in xrange(P.dnnDepth):
            self.L2_sqr += (self.hiddenLayerList[i].W ** 2).sum()
        self.L2_sqr += (self.outputLayer.W ** 2).sum()

        # CrossEntropy
        self.crossEntropy = ( self.outputLayer.crossEntropy )
        
        # Same holds for the function computing the number of errors
        self.errors = self.outputLayer.errors

        # Get the predict int for test set output
        self.yPred = self.outputLayer.y_pred
        
        # Parameters of all DNN model
        self.params = self.hiddenLayerList[0].params
        for i in xrange(1, P.dnnDepth):
            self.params += self.hiddenLayerList[i].params
        self.params += self.outputLayer.params

        # keep track of model input
        self.input = input

def trainDNN(datasets, P, L1_reg = 0.00, L2_reg = 0.0002):
    
    trainSetX, trainSetY, trainSetName = datasets[0]
    validSetX, validSetY, validSetName = datasets[1]
    
    lr = P.learningRate
    
    ###############
    # BUILD MODEL #
    ###############
    print '... building the model'

    start = T.lscalar()  # index to a [mini]batch
    end = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')    # the data is presented as rasterized images
    y = T.ivector('y')   # the labels are presented as 1D vector of
                         # [int] labels

    dummyParams = [None] * (2 * (P.dnnDepth + 1))
    
    # build the DNN object for training
    classifier = DNN(
                  input = x,
                  params    = dummyParams,
                  P = P,
                  DROPOUT = True )
    
    # build the DNN object for Validation
    predicter = DNN(
                  input = x,
                  P = P,
                  params    = dummyParams )
    # valid model
    validModel = theano.function(
                 inputs  = [start, end],
                 outputs = predicter.errors(y),
                 givens  = { x: validSetX[start : end], y: validSetY[start : end] } )
    
    # Cost function 1.cross entropy 2.weight decay
    cost = (
          classifier.crossEntropy(y)
            + L1_reg * classifier.L1
            + L2_reg * classifier.L2_sqr )
    
    # Momentum        
    def initialVelocitys():
        v = []
        v.append(theano.shared(numpy.zeros( (P.inputDimNum, P.dnnWidth), dtype = theano.config.floatX ), borrow = True))
        v.append(theano.shared(numpy.zeros( (P.dnnWidth,), dtype = theano.config.floatX ), borrow = True) )
        for i in xrange(P.dnnDepth - 1):
           v.append(theano.shared(numpy.zeros( (P.dnnWidth/2,P.dnnWidth), dtype = theano.config.floatX ), borrow = True) )
           v.append(theano.shared(numpy.zeros( (P.dnnWidth,), dtype = theano.config.floatX ), borrow = True) )
        v.append(theano.shared(numpy.zeros( (P.dnnWidth/2, P.outputPhoneNum), dtype = theano.config.floatX ), borrow = True) )
        v.append(theano.shared(numpy.zeros( (P.outputPhoneNum,), dtype = theano.config.floatX ), borrow = True) )
        return v
    flag = True
    grads = [T.grad(cost, param) for param in classifier.params]
    def myUpdates(grads, params, velocitys, first = flag):
        if(first):
            velocitys = [velocity - lr * grad for velocity, grad in zip(velocitys, grads)]
            flag = False
        else:
            velocitys = [ P.momentum * velocity - lr * (1 - P.momentum) * grad for velocity, grad in zip(velocitys, grads) ]
        params_update = [ (param, param + velocity) for param, velocity in zip(params, velocitys) ]
        return params_update
    
    #training model
    velocitys = initialVelocitys()
    trainModel = theano.function(
                inputs  = [start, end],
#outputs = [classifier.errors(y), classifier.outputLayer.a, classifier.outputLayer.c],
                outputs = classifier.errors(y),
                updates = myUpdates(grads = grads, params = classifier.params, velocitys = velocitys),
                givens={ x: trainSetX[start : end], y: trainSetY[start : end] } )

    ###################
    # TRAIN DNN MODEL #
    ###################

    print '... start training'
    print ('epoch,\ttrain,\tvalid')
    validFrequency = 1000
    prevFER = numpy.inf
    prevModel = None
    nowModel = None

    startTime  = timeit.default_timer()

    epoch = 0
    doneLooping = False
    
    # Total data size
    totalTrainSize = trainSetX.get_value(borrow=True).shape[0]
    totalValidSize = validSetX.get_value(borrow=True).shape[0]
    
    # make batch list
    numTrainBatches = totalTrainSize / P.batchSizeForTrain
    indexForTrainList = makeBatch(totalTrainSize, P.batchSizeForTrain)
     
    batchSizeForValid = 9192
    numValidBatches = totalValidSize / batchSizeForValid
    indexForValidList = makeBatch(totalValidSize, batchSizeForValid)

    random.seed(P.seed)
    
    while (epoch < P.maxEpoch) and (not doneLooping):
        epoch = epoch + 1

        if P.SHUFFLE:
            random.shuffle(indexForTrainList)

        trainLosses = []
    
        if DEBUG:    
            startTime1  = timeit.default_timer()        
        # the core of training
        for i in xrange(numTrainBatches):
            thisBatchTrainLosses = trainModel(indexForTrainList[i][0], indexForTrainList[i][1])
            trainLosses.append(thisBatchTrainLosses)
        if DEBUG:    
            endTime1 = timeit.default_timer()
            print (('time %.2fm' % ((endTime1 - startTime1) / 60.)))

        # Set the now train model's parameters to valid model
        nowModel = getParamsValue(classifier.params)
        setParamsValue(nowModel, predicter.params)
        
        validLosses = [validModel(indexForValidList[i][0], indexForValidList[i][1]) for i in xrange(numValidBatches + 1)]
        validFER = numpy.mean(validLosses)
        
        if validFER < prevFER:
            prevFER = validFER
            prevModel = nowModel
        else:
            if lr > 0.00000001:
                lr = lr/2
                epoch = epoch - 1
                setParamsValue(prevModel, classifier.params)
                trainFER = numpy.mean(trainLosses)
                print (('== half ==,%i,\t%f,\t%f') % (epoch, trainFER * 100, validFER * 100. ))
                continue
            else:
                doneLooping = True
        trainFER = numpy.mean(trainLosses)
        print (('%i,\t%f,\t%f') % (epoch, trainFER * 100, validFER * 100. ))
    # end of training
        
    endTime = timeit.default_timer()
    print (('time %.2fm' % ((endTime - startTime) / 60.)))
    
    return prevModel

def getResult(bestModel, datasets, P):
    
    validSetX, validSetY, validSetName = datasets[1]
    testSetX, testSetY, testSetName = datasets[2]
    
    # allocate symbolic variables for the data
    start = T.lscalar()  # index to a [mini]batch
    end = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')    # the data is presented as rasterized images
    y = T.ivector('y')   # the labels are presented as 1D vector of

    # bulid best DNN model
    predicter = DNN(
                  input = x,
                  P = P,
                  params = bestModel )
    
    testBatchSize = 9192
    totalTestSize = testSetX.get_value(borrow = True).shape[0]
    testIndexList = makeBatch(totalTestSize, testBatchSize)
    
    validBatchSize = 9192
    totalValidSize = validSetX.get_value(borrow = True).shape[0]
    validIndexList = makeBatch(totalValidSize, validBatchSize)
    
    # bulid valid model
    validModel = theano.function(
                inputs  = [start, end],
                outputs = [predicter.errors(y), predicter.yPred],
                givens  = { x: validSetX[start:end], y: validSetY[start:end] })
    
    # bulid test model
    testModel = theano.function(
                inputs  = [start, end],
                outputs = [predicter.errors(y), predicter.yPred],
                givens  = { x: testSetX[start:end], y: testSetY[start:end] })

    validResult = EvalandResult(validModel, validIndexList, 'valid')    
    testResult = EvalandResult(testModel, testIndexList, 'test')
    writeResult(validResult, P.validResultFilename, validSetName)
    writeResult(testResult, P.testResultFilename, testSetName)

if __name__ == '__main__':
    datasets = utils.load_data(filename = datasetFilename, totalSetNum = 3)
    if not MODELEXIST:
        bestModel = trainDNN(datasets = datasets)
        utils.makeModelPkl(bestModel, bestModelFilename, P)
        getResult(datasets = datasets, bestModel = bestModel)
    else:
        model = utils.readModelPkl(modelFilename)
        getResult(datasets = datasets, bestModel = model)

    
