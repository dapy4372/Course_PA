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
from dnnArchitecture import HiddenLayer, OutputLayer, DNN
from dnnUtils import EvalandResult, writeResult, makeBatch, getParamsValue,setParamsValue, Dropout, Parameters
parameterFilename = sys.argv[1]

# TODO RMSprop
def trainDNN(datasets, P):
    
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
                  input  = x,
                  params = dummyParams,
                  P = P,
                  DROPOUT = True )
    
    # build the DNN object for Validation
    predicter = DNN(
                  input = x,
                  P = P,
                  params = dummyParams )
    # valid model
    validModel = theano.function(
                 inputs  = [start, end],
                 outputs = predicter.errors(y),
                 givens  = { x: validSetX[start : end], y: validSetY[start : end] } )
    
    # Cost function 1.cross entropy 2.weight decay
    cost = (
          classifier.crossEntropy(y)
            + P.L1Reg * classifier.L1
            + P.L2Reg * classifier.L2_sqr )
    
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
    
    # make training  batch list
    indexForTrainList = makeBatch(totalTrainSize, P.batchSizeForTrain)
    # make validation batch list 
    indexForValidList = makeBatch(totalValidSize, 18348)

    random.seed(P.seed)
    curEarlyStop = 0
    while (epoch < P.maxEpoch) and (not doneLooping):
        epoch = epoch + 1

        if P.SHUFFLE:
            random.shuffle(indexForTrainList)
    
        # Training
        trainLosses = [trainModel(indexForTrainList[i][0], indexForTrainList[i][1]) for i in xrange(len(indexForValidList))]
        # Evaluate training FER 
        trainFER = numpy.mean(trainLosses)

        # Set the now train model's parameters to valid model
        nowModel = getParamsValue(classifier.params)
        setParamsValue(nowModel, predicter.params)
        
        # Evaluate validation FER
        validLosses = [validModel(indexForValidList[i][0], indexForValidList[i][1]) for i in xrange(len(indexForValidList))]
        validFER = numpy.mean(validLosses)
        
        if validFER < prevFER:
            prevFER = validFER
            prevModel = nowModel
            curEarlyStop = 0
        else:
            if curEarlyStop < P.earlyStop:
                lr = lr/2
                epoch -= 1
                setParamsValue(prevModel, classifier.params)
                print (('== half ==,%i,\t%f,\t%f') % (epoch, trainFER * 100, validFER * 100. ))
                curEarlyStop += 1
                continue
            else:
                doneLooping = True
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
