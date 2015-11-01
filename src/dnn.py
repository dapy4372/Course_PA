import os
import sys
import timeit
import random
import numpy
import math
import theano
import theano.tensor as T
import activation
import globalParam
import dnnUtils
from utils import sharedDataXY, setSharedDataXY
from dnnArchitecture import HiddenLayer, OutputLayer, DNN
from dnnUtils import Parameters
parameterFilename = sys.argv[1]
numpy.set_printoptions(threshold=numpy.nan)

def trainDNN(datasets, P):
    
    trainSetX, trainSetY, trainSetName = dnnUtils.splice(datasets[0], 4)
    validSetX, validSetY, validSetName = dnnUtils.splice(datasets[1], 4)
    sharedTrainSetX = theano.shared(numpy.asarray(trainSetX, dtype=theano.config.floatX), borrow=True) 
    sharedTrainSetY = theano.shared(numpy.asarray(trainSetY, dtype=theano.config.floatX), borrow=True)
    castSharedTrainSetY = T.cast(sharedTrainSetY, 'int32')
    sharedValidSetX = theano.shared(numpy.asarray(validSetX, dtype=theano.config.floatX), borrow=True) 
    sharedValidSetY = theano.shared(numpy.asarray(validSetY, dtype=theano.config.floatX), borrow=True) 
    castSharedValidSetY = T.cast(sharedValidSetY, 'int32')

#shareTrainSetX, shareTrainSetY, castSharedTrainSetY = sharedDataXY(trainSetX, trainSetY)
#shareValidSetX, shareValidSetY, castSharedValidSetY = sharedDataXY(validSetX, validSetY)

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
                 givens  = { x: sharedValidSetX[start : end], y: castSharedValidSetY[start : end] } )
    
    # Cost function 1.cross entropy 2.weight decay
    cost = ( classifier.crossEntropy(y) + P.L1Reg * classifier.L1 + P.L2Reg * classifier.L2_sqr )
    
    #training model
    globalParam.initGlobalLearningRate(P)
    globalParam.initGlobalFlag()
    globalParam.initGlobalVelocitys()
    globalParam.initGlobalSigmas()
    globalParam.initGlobalgradSqrs()
    
    grads = [T.grad(cost, param) for param in classifier.params]
    myOutputs = [classifier.errors(y)] + grads + classifier.params
    trainModel = theano.function(
                inputs  = [start, end],
                outputs = myOutputs,
#updates = activation.momentum(grads, classifier.params, P),
#updates = activation.RMSProp(grads, classifier.params),
                updates = activation.Adagrad(grads, classifier.params),
                givens={ x: sharedTrainSetX[start : end], y: castSharedTrainSetY[start : end] } )

    ###################
    # TRAIN DNN MODEL #
    ###################

    print '... start training'
    print ('epoch,\ttrain,\tvalid')

    # Training parameter
    epoch = 0
    curEarlyStop = 0
    prevModel = None
    nowModel = None
    doneLooping = False
    prevFER = numpy.inf
    random.seed(P.seed)
    
    # Total data size
    totalTrainSize = len(trainSetX)
    totalValidSize = len(validSetX)

    # make training and validation batch list
    trainBatchIdxList = dnnUtils.makeBatch(totalTrainSize, P.batchSizeForTrain)
    validBatchIdxList = dnnUtils.makeBatch(totalValidSize, 16384)
    
    startTime  = timeit.default_timer()
    while (epoch < P.maxEpoch) and (not doneLooping):
        epoch = epoch + 1

        if P.SHUFFLE:
            p = numpy.random.permutation(totalTrainSize)
            sharedTrainSetX.set_value(trainSetX[p])
            sharedTrainSetY.set_value(trainSetY[p])
            castSharedTrainSetY = T.cast(sharedTrainSetY, 'int32')

        # Training
        trainLosses=[]
        s = 2*(P.dnnDepth+1)
        for i in xrange(len(trainBatchIdxList)):
            out = trainModel(trainBatchIdxList[i][0], trainBatchIdxList[i][1])
            trainLosses.append(out[0])
            """
            if i == 0:
                for j in [0,2]:
                    print ('grad %d' % (j+1) )
                    print numpy.array_str(out[j+1])
            """
        # Evaluate training FER 
        trainFER = numpy.mean(trainLosses)

        # Set the now train model's parameters to valid model
        nowModel = dnnUtils.getParamsValue(classifier.params)
        dnnUtils.setParamsValue(nowModel, predicter.params)
        
        # Evaluate validation FER
        validLosses = [validModel(validBatchIdxList[i][0], validBatchIdxList[i][1]) for i in xrange(len(validBatchIdxList))]
        validFER = numpy.mean(validLosses)
        
        if validFER < prevFER:
            prevFER = validFER
            prevModel = nowModel
            curEarlyStop = 0
        else:
            if curEarlyStop < P.earlyStop:
                globalParam.lr = globalParam.lr/2
                epoch -= 1
                dnnUtils.setParamsValue(prevModel, classifier.params)
                print (('== half ==,%i,\t%f,\t%f') % (epoch, trainFER * 100, validFER * 100. ))
                curEarlyStop += 1
                continue
            else:
                doneLooping = True
                continue
        print (('%i,\t%f,\t%f') % (epoch, trainFER * 100, validFER * 100. ))
    # end of training
        
    endTime = timeit.default_timer()
    print (('time %.2fm' % ((endTime - startTime) / 60.)))
    
    sharedValidSetX.set_value([[]])
    sharedValidSetY.set_value([]) 
    sharedTrainSetX.set_value([[]])
    sharedTrainSetY.set_value([]) 
    return prevModel

def getResult(bestModel, datasets, P):
    
    validSetX, validSetY, validSetName = dnnUtils.splice(datasets[1], 4)
    testSetX, testSetY, testSetName = dnnUtils.splice(datasets[2], 4)

    sharedValidSetX = theano.shared(numpy.asarray(validSetX, dtype=theano.config.floatX), borrow=True) 
    sharedValidSetY = theano.shared(numpy.asarray(validSetY, dtype=theano.config.floatX), borrow=True) 
    castSharedValidSetY = T.cast(sharedValidSetY, 'int32')
    
    sharedTestSetX = theano.shared(numpy.asarray(testSetX, dtype=theano.config.floatX), borrow=True) 
    sharedTestSetY = theano.shared(numpy.asarray(testSetY, dtype=theano.config.floatX), borrow=True)
    castSharedTestSetY = T.cast(sharedTestSetY, 'int32')
    
    # allocate symbolic variables for the data
    start = T.lscalar()  # index to a [mini]batch
    end = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')    # the data is presented as rasterized images
    y = T.ivector('y')   # the labels are presented as 1D vector of

    # bulid best DNN model
    predicter = DNN( input = x, P = P, params = bestModel )
    
    # Total data size
    totalTestSize = len(testSetX)
    totalValidSize = len(validSetX)
    
#totalTestSize = testSetX.get_value(borrow = True).shape[0]
#totalValidSize = validSetX.get_value(borrow = True).shape[0]
    
    testBatchIdxList = dnnUtils.makeBatch(totalTestSize, 16384)
    validBatchIdxList = dnnUtils.makeBatch(totalValidSize, 16384)
    
    # validation model
    validModel = theano.function(
                 inputs  = [start, end],
                 outputs = [predicter.errors(y), predicter.yPred],
                 givens  = { x: sharedValidSetX[start : end], y: castSharedValidSetY[start : end] } )
    
    # bulid test model
    testModel = theano.function(
                inputs  = [start, end],
                outputs = [predicter.errors(y), predicter.yPred],
                givens  = { x: sharedTestSetX[start:end], y: castSharedTestSetY[start:end] })

    validResult = dnnUtils.EvalandResult(validModel, validBatchIdxList, 'valid')    
    testResult = dnnUtils.EvalandResult(testModel, testBatchIdxList, 'test')
    
    dnnUtils.writeResult(validResult, P.validResultFilename, validSetName)
    dnnUtils.writeResult(testResult, P.testResultFilename, testSetName)
    
    sharedValidSetX.set_value([[]])
    sharedValidSetY.set_value([]) 
    sharedTestSetX.set_value([[]])
    sharedTestSetY.set_value([])

    # For getting prob
    trainSetX, trainSetY, trainSetName = dnnUtils.splice(datasets[0], 4)
    sharedTrainSetX = theano.shared(numpy.asarray(trainSetX, dtype=theano.config.floatX), borrow=True) 
    sharedTrainSetY = theano.shared(numpy.asarray(trainSetY, dtype=theano.config.floatX), borrow=True)
    castSharedTrainSetY = T.cast(sharedTrainSetY, 'int32')
    
    # training model
    trainModel = theano.function(
                inputs  = [start, end],
                outputs = predicter.p_y_given_x,
                givens={ x: sharedTrainSetX[start : end], y: castSharedTrainSetY[start : end] }, on_unused_input='warn')
    
    totalTrainSize = len(trainSetX)
    trainBatchIdxList = dnnUtils.makeBatch(totalTrainSize, P.batchSizeForTrain)

    trainProb = dnnUtils.getProb(trainModel, trainBatchIdxList) 
    dnnUtils.writeProb(trainProb, P.trainProbFilename, trainSetName)
    
    sharedTrainSetX.set_value([[]])
    sharedTrainSetY.set_value([]) 
