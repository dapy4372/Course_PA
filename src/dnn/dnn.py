import os
import sys
import timeit
import random
import numpy
import theano
import theano.tensor as T
import dnnUtils
import globalParam
from dnnArchitecture import HiddenLayer, OutputLayer, DNN
from dnnUtils import Parameters, sharedDataXY, setSharedDataXY, clearSharedDataXY
parameterFilename = sys.argv[1]
numpy.set_printoptions(threshold=numpy.nan) # for print numpy array

def trainDNN(datasets, P):

    trainSetX, trainSetY, trainSetName = datasets[0]
    validSetX, validSetY, validSetName = datasets[1]
    sharedTrainSetX, sharedTrainSetY, castSharedTrainSetY = sharedDataXY(trainSetX, trainSetY)
    sharedValidSetX, sharedValidSetY, castSharedValidSetY = sharedDataXY(validSetX, validSetY)

    ###############
    # BUILD MODEL #
    ###############
    print '... building the model'
    i = T.ivector('i')
    x = T.tensor3('x')    # the data is presented as rasterized images
    y = T.ivector('y')   # the labels are presented as 1D vector of
                         # [int] labels

    # For create a new model
    dummyParams = [None] * (2 * (P.dnnDepth + 1))
    
    # Build the DNN object for training
    classifier = DNN( input = x, params = dummyParams, P = P, DROPOUT = True )
    
    # Build the DNN object for Validation
    predicter = DNN( input = x, P = P, params = dummyParams )

    # Validation model
    validModel = theano.function( inputs = [x,y], outputs = predicter.errors(y), givens ={x:sharedTrainSetX})
    
    # Cost function 1.cross entropy 2.weight decay
    cost = ( classifier.crossEntropy(y) + P.L1Reg * classifier.L1 + P.L2Reg * classifier.L2_sqr )
   
    # Global parameters setting
    globalParam.initGlobalLearningRate(P)
    globalParam.initGlobalFlag()
    globalParam.initGlobalVelocitys()
    globalParam.initGlobalSigmas()
    globalParam.initGlobalgradSqrs()
    
    # Training model
    grads = [T.grad(cost, param) for param in classifier.params]
    myOutputs = [classifier.errors(y)] + grads + classifier.params
    myUpdates = dnnUtils.chooseUpdateMethod(grads, classifier.params, P)
    trainModel = theano.function( inputs = [x, y], outputs = myOutputs, updates = myUpdates)

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

    # Make mini-Batch
    trainBatchIdx = dnnUtils.makeBatch(totalTrainSize, P.batchSizeForTrain)
    validBatchIdx = dnnUtils.makeBatch(totalValidSize, 16384)
   
    # Center Indx
    trainCenterIdx = dnnUtils.findCenterIdxList(trainSetY)
    validCenterIdx = dnnUtils.findCenterIdxList(validSetY)

    startTime  = timeit.default_timer()
    while (epoch < P.maxEpoch) and (not doneLooping):
        epoch = epoch + 1

        if P.SHUFFLE:
            random.shuffle(trainCenterIdx)
#p = numpy.random.permutation(totalTrainSize)
#sharedTrainSetX, sharedTrainSetY, castSharedTrainSetY = setSharedDataXY(sharedTrainSetX, sharedTrainSetY, trainSetX[p], trainSetY[p])

        # Training
        trainLosses=[]
        s = 2*(P.dnnDepth+1)
        for i in xrange(len(trainBatchIdx)):
            outputs = trainModel(dnnUtils.spliceInput(sharedTrainSetX, castSharedTrainSetY, trainCenterIdx[ trainBatchIdx[i][0]:trainBatchIdx[i][1] ]))
            trainLosses.append(outputs[0])


            # print value for debug
#    if i == 0 and P.DEBUG:
#               dnnUtils.printGradsParams(outputs[1:], P.dnnDepth)


        # Evaluate training FER 
        trainFER = numpy.mean(trainLosses)

        # Set the now train model's parameters to valid model
        nowModel = dnnUtils.getParamsValue(classifier.params)
        dnnUtils.setParamsValue(nowModel, predicter.params)
        
        # Evaluate validation FER
        validLosses = [validModel(dnnUtils.spliceInput(sharedValidSetX, castSharedValidSetY, validCenterIdx[ validBatchIdx[i][0]:validBatchIdx[i][1] ])) for i in xrange(len(validBatchIdx))]

        validFER = numpy.mean(validLosses)
        if P.updateMethod != 'Momentum':
            prevFER = validFER
            prevModel = nowModel
        else:
            if validFER < prevFER:
                prevFER = validFER
                prevModel = nowModel
                curEarlyStop = 0
            else:
                if curEarlyStop < P.earlyStop:
                    globalParam.lr = globalParam.lr * P.learningRateDecay
                    epoch -= 1
                    dnnUtils.setParamsValue(prevModel, classifier.params)
                    print (('====,%i,\t%f,\t%f') % (epoch, trainFER * 100, validFER * 100. ))
                    curEarlyStop += 1
                    continue
                else:
                    doneLooping = True
                    continue
        print (('%i,\t%f,\t%f') % (epoch, trainFER * 100, validFER * 100. ))
    # end of training
        
    endTime = timeit.default_timer()
    print (('time %.2fm' % ((endTime - startTime) / 60.)))

    clearSharedDataXY(sharedTrainSetX, sharedTrainSetY)
    clearSharedDataXY(sharedValidSetX, sharedValidSetY)

    return prevModel

def getResult(bestModel, datasets, P):

    print "...getting result"

    validSetX, validSetY, validSetName = dnnUtils.splice(datasets[1], 4)
    testSetX, testSetY, testSetName = dnnUtils.splice(datasets[2], 4)

    sharedValidSetX, sharedValidSetY, castSharedValidSetY = sharedDataXY(validSetX, validSetY)
    sharedTestSetX, sharedTestSetY, castSharedTestSetY = sharedDataXY(testSetX, testSetY)
    
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
    
    clearSharedDataXY(sharedTestSetX, sharedTestSetY)
    clearSharedDataXY(sharedValidSetX, sharedValidSetY)


def getProb(bestModel, datasets, P):

    print "...getting probability"
    # For getting prob
    trainSetX, trainSetY, trainSetName = dnnUtils.splice(datasets[0], 4)
    sharedTrainSetX, sharedTrainSetY, castSharedTrainSetY = sharedDataXY(trainSetX, trainSetY)

    # allocate symbolic variables for the data
    start = T.lscalar()  # index to a [mini]batch
    end = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')    # the data is presented as rasterized images
    y = T.ivector('y')   # the labels are presented as 1D vector of

    # bulid best DNN model
    predicter = DNN( input = x, P = P, params = bestModel )
    
    # training model
    trainModel = theano.function(
                inputs  = [start, end],
                outputs = predicter.p_y_given_x,
                givens={ x: sharedTrainSetX[start : end], y: castSharedTrainSetY[start : end] }, on_unused_input='ignore')
    
    totalTrainSize = len(trainSetX)
    trainBatchIdxList = dnnUtils.makeBatch(totalTrainSize, 16384)

    dnnUtils.writeProb(trainModel, trainBatchIdxList, trainSetName, P.trainProbFilename) 
    
    clearSharedDataXY(sharedTrainSetX, sharedTrainSetY)
