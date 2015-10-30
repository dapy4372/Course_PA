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
from dnnUtils import EvalandResult, writeResult, makeBatch, getParamsValue,setParamsValue, Dropout, Parameters, splice, shareDataXY
from activation import initialVelocitys, momentum
parameterFilename = sys.argv[1]

def trainDNN(datasets, P):
    
    trainSetX, trainSetY, trainSetName = splice(datasets[0])
    validSetX, validSetY, validSetName = splice(datasets[1])
    shareTrainSetX, shareTrainSetY = shareDataXY(trainSetX, trainSetY)
    shareValidSetX, shareValidSetY = shareDataXY(validSetX, validSetY)

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
    classifier = DNN( input = x, params = dummyParams, P = P, DROPOUT = True )
    
    # build the DNN object for Validation
    predicter = DNN( input = x, P = P, params = dummyParams )

    # validation model
    validModel = theano.function(
                 inputs  = [start, end],
                 outputs = predicter.errors(y),
                 givens  = { x: shareValidSetX[start : end], y: shareValidSetY[start : end] } )
    
    # Cost function 1.cross entropy 2.weight decay
    cost = ( clssifier.crossEntropy(y) + P.L1Reg * classifier.L1 + P.L2Reg * classifier.L2_sqr )
    
    #training model
    flag = [True] # for first momentum or not
    grads = [T.grad(cost, param) for param in classifier.params]
    velocitys = activation.initialVelocitys(P)

    trainModel = theano.function(
                inputs  = [start, end],
                outputs = classifier.errors(y),
                updates = activation.momentum(grads, classifier.params, velocitys, flag),
                givens={ x: shareTrainSetX[start : end], y: shareTrainSetY[start : end] } )

    ###################
    # TRAIN DNN MODEL #
    ###################

    print '... start training'
    print ('epoch,\ttrain,\tvalid')

    # Training parameter
    epoch = 0
    prevModel = None
    nowModel = None
    curEarlyStop = 0
    prevFER = numpy.inf
    validFrequency = 1000
    lr = P.learningRate
    doneLooping = False
    random.seed(P.seed)
    
    # Total data size
    totalTrainSize = trainSetX.get_value(borrow=True).shape[0]
    totalValidSize = validSetX.get_value(borrow=True).shape[0]
    
    # make training and validation batch list
    trainBatchIdxList = makeBatch(totalTrainSize, P.batchSizeForTrain)
    validBatchIdxList = makeBatch(totalValidSize, 16384)
    
    startTime  = timeit.default_timer()
    while (epoch < P.maxEpoch) and (not doneLooping):
        epoch = epoch + 1

        if P.SHUFFLE:
            p = numpy.random.permutation(totalTrainSize)
            shareTrainSetX.set_value([[]])
            shareTrainSetY.set_value([[]])
            shareTrainSetX, shareTrainSetY = shareDataXY(trainSetX[p], trainSetY[p])
         
        # Training
        trainLosses = [trainModel(trainBatchIdxList[i][0], trainBatchIdxList[i][1]) for i in xrange(len(trainIdxList))]
        # Evaluate training FER 
        trainFER = numpy.mean(trainLosses)

        # Set the now train model's parameters to valid model
        nowModel = getParamsValue(classifier.params)
        setParamsValue(nowModel, predicter.params)
        
        # Evaluate validation FER
        validLosses = [validModel(validBatchIdxList[i][0], validBatchIdxList[i][1]) for i in xrange(len(validIdxList))]
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
    
    totalTestSize = testSetX.get_value(borrow = True).shape[0]
    totalValidSize = validSetX.get_value(borrow = True).shape[0]
    
    testBatchIdxList = makeBatch(totalTestSize, 16384)
    validBatchIdxList = makeBatch(totalValidSize, 16384)
    
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

    validResult = EvalandResult(validModel, validBatchIdxList, 'valid')    
    testResult = EvalandResult(testModel, testBatchIdxList, 'test')
    writeResult(validResult, P.validResultFilename, validSetName)
    writeResult(testResult, P.testResultFilename, testSetName)
