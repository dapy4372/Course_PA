import os
import sys
import timeit
import random
import numpy
import theano
import theano.tensor as T
import filler
import rnnUtils
import globalParam
from rnnUtils import Parameters
from rnnArchitecture import HiddenLayer, OutputLayer, RNN

DEBUG = False 
clipRange = 0.1
clipSentSize = 50

parameterFilename = sys.argv[1]
numpy.set_printoptions(threshold=numpy.nan) # for print numpy array

def trainDNN(datasets, P):

    
#trainSetX, trainSetY, trainSetName, trainMask = filler.fillerCore(datasets[0])
#validSetX, validSetY, validSetName, validMask = filler.fillerCore(datasets[1])
    trainSetX, trainSetY, trainSetName = rnnUtils.makeDataSentence(datasets[0])
    validSetX, validSetY, validSetName = rnnUtils.makeDataSentence(datasets[1])
#print numpy.array(trainSetY[0]).shape
#print numpy.array(trainSetY[1]).shape
#trainSetX, trainSetY, trainSetName = filler.cut([trainSetX, trainSetY, trainSetName], clipSentSize)
#validSetX, validSetY, validSetName = filler.cut([validSetX, validSetY, validSetName], clipSentSize)
#print numpy.array(trainSetY[i]).shape
#sharedTrainSetX, sharedTrainSetY, castSharedTrainSetY = rnnUtils.sharedDataXY(trainSetX, trainSetY)
#sharedValidSetX, sharedValidSetY, castSharedValidSetY = rnnUtils.sharedDataXY(validSetX, validSetY)
    ###############
    # BUILD MODEL #
    ###############
    print '... building the model'
    idx = T.iscalar('i')
    x = T.matrix(dtype=theano.config.floatX)
    y = T.ivector()

    # For create a new model
    dummyParams = [None] * (3 * (P.rnnDepth) + 2)
    
    # Build the RNN object for training
    classifier = RNN( input = x, params = dummyParams, P = P)
    
    # Build the DNN object for Validation
    predicter = RNN( input = x, P = P, params = dummyParams )
    
   
    # Global parameters setting
    globalParam.initGlobalLearningRate(P)
    globalParam.initGlobalFlag()
    globalParam.initGlobalVelocitys()
    globalParam.initGlobalSigmas()
    globalParam.initGlobalgradSqrs()
    
    # Cost function 1.cross entropy 2.weight decay
    cost = ( classifier.crossEntropy(y) + P.L1Reg * classifier.L1 + P.L2Reg * classifier.L2_sqr )
    
#grads = [(T.grad(cost, param)).clip(-0.1, 0.1) for param in classifier.params]
#grads = [ (T.grad(cost, param)).clip(-0.5,0.5) for param in classifier.params]
    grads = [ (T.grad(cost, param)).clip(-1 * clipRange, clipRange) for param in classifier.params]
    myOutputs = [classifier.errors(y)] +[cost]+ [classifier.hiddenLayerList[0].z_seq] + [classifier.hiddenLayerList[0].alp] +classifier.hiddenLayerList[0].output + [classifier.p_y_given_x] + [classifier.yPred] + grads + classifier.params 
    myUpdates = rnnUtils.chooseUpdateMethod(grads, classifier.params, P)

    # Training mode
    """
    trainModel = theano.function( inputs = [idx], outputs = myOutputs, updates = myUpdates, 
                                  givens = {x:sharedTrainSetX[idx], y:castSharedTrainSetY[idx]})
    """
    trainModel = theano.function( inputs = [x, y], outputs = myOutputs, updates = myUpdates )
    # Validation model
    """
    validModel = theano.function( inputs = [idx], outputs = predicter.errors(y),
                                  givens = {x:sharedValidSetX[idx], y:castSharedValidSetY[idx]})
    """
    validModel = theano.function( inputs = [x, y], outputs = predicter.errors(y))

    ###################
    # TRAIN DNN MODEL #
    ###################

    print '... start training'
    print ('epoch,    train,    valid')

    # Training parameter
    epoch = 0
    curEarlyStop = 0
    prevModel = None
    nowModel = None
    doneLooping = False
    prevFER = numpy.inf
    random.seed(P.seed)
    
    # Adagrad, RMSProp
    prevGradSqrs = None
    prevSigmaSqrs = None
   
    # Center Index
#trainCenterIdx = rnnUtils.findCenterIdxList(trainSetY)
#validCenterIdx = rnnUtils.findCenterIdxList(validSetY)
    
    # Total Center Index
#totalTrainSize = len(trainCenterIdx)
#totalValidSize = len(validCenterIdx)
    
    totalTrainSentSize = len(trainSetX)
    totalValidSentSize = len(validSetX)

    startTime  = timeit.default_timer()
    while (epoch < P.maxEpoch) and (not doneLooping):
        epoch = epoch + 1

#random.shuffle(trainCenterIdx)
        sent = 1
        # Training
        trainLosses=[]
        for i in xrange(totalTrainSentSize):
#outputs = trainModel(i)
            outputs = trainModel(numpy.array(trainSetX[i]).astype(dtype='float32'), numpy.array(trainSetY[i]).astype(dtype='int32'))
            trainLosses.append(outputs[0])
#print outputs[2]
#print '========================'
#sys.stdin.read(1)
            """
            trainFER = numpy.mean(trainLosses)
            print (('   %i,\t%f\t, cost = %f') % (sent, trainFER * 100, outputs[1]))
            """
#           sent += 1
#rnnUtils.printNpArrayMeanStdMaxMin('hidden output', outputs[1])
            """
            print '========== In order ==========='
            print outputs[3]
            """
            """
            print '========== In reverse ==========='
            print outputs[2]
            """
            """
            print '==========================================='
            print outputs[4]
            print '==========================================='
            """
            """
            print '====== p y given x ====='
            print outputs[2]
            """
            """
            print '====== y pred ====='
            print outputs[3]
            """
            # Print parameter value for debug
            if (i == 0 and DEBUG):
                rnnUtils.printGradsParams(outputs[5:], P.rnnDepth)
        if (i != 0 and epoch % 30 == 0):
            globalParam.lr *= 0.5
        # Evaluate training FER 
        trainFER = numpy.mean(trainLosses)

        # Set the now train model's parameters to valid model
        nowModel = rnnUtils.getParamsValue(classifier.params)
        rnnUtils.setParamsValue(nowModel, predicter.params)
        
        # Evaluate validation FER
        validLosses = [validModel(numpy.array(validSetX[i]).astype(dtype='float32'), numpy.array(validSetY[i]).astype(dtype='int32')) for i in xrange(totalValidSentSize)]
        validFER = numpy.mean(validLosses)
        prevModel = nowModel
        """
        if validFER < prevFER:
            prevFER = validFER
            prevModel = nowModel
            curEarlyStop = 0
        else:
            if curEarlyStop < P.earlyStop:
                epoch -= 1
                rnnUtils.setParamsValue(prevModel, classifier.params)
                print (('====,%i,\t%f,\t%f') % (epoch, trainFER * 100, validFER * 100. ))
                curEarlyStop += 1
                if P.updateMethod == 'Momentum':
                    globalParam.lr = globalParam.lr * P.learningRateDecay
                elif P.updateMethod == 'Adagrad':
                    globalParam.gradSqrs = prevGradSqrs
                elif P.updateMethod == 'RMSProp':
                    globalParam.sigmaSqrs = prevSigmaSqrs
                continue
            else:
                doneLooping = True
                continue
        """
        print (('%i,\t%f,\t%f') % (epoch, trainFER * 100, validFER * 100. ))

        # Record the Adagrad, RMSProp parameter
        if P.updateMethod == 'Adagrad':
            prevGradSqrs = globalParam.gradSqrs
        if P.updateMethod == 'RMSProp':
            prevSigmaSqrs = globalParam.sigmaSqrs

    # end of training
        
    endTime = timeit.default_timer()
    print (('time %.2fm' % ((endTime - startTime) / 60.)))

#    rnnUtils.clearSharedDataXY(sharedTrainSetX, sharedTrainSetY)
#    rnnUtils.clearSharedDataXY(sharedValidSetX, sharedValidSetY)

    return prevModel

def getResult(bestModel, datasets, P):

    print "...getting result"

    validSetX, validSetY, validSetName = datasets[1]
    testSetX, testSetY, testSetName = datasets[2]
    sharedValidSetX, sharedValidSetY, castSharedValidSetY = rnnUtils.sharedDataXY(validSetX, validSetY)
    sharedTestSetX, sharedTestSetY, castSharedTestSetY = rnnUtils.sharedDataXY(testSetX, testSetY)
    
    print "...buliding model"
    idx = T.ivector('i')
    sX = T.matrix(dtype=theano.config.floatX)
    sY = T.ivector()
    
    # bulid best DNN  model
    predicter = RNN( input = sX, P = P, params = bestModel )
    
    # Center Index
    validCenterIdx = rnnUtils.findCenterIdxList(validSetY)
    testCenterIdx = rnnUtils.findCenterIdxList(testSetY)
    
    # Total Center Index
#totalValidSize = len(validCenterIdx)
#totalTestSize = len(testCenterIdx)
    
    # Make mini-Batch
#validBatchIdx = rnnUtils.makeBatch(totalValidSize, 16384)
#testBatchIdx = rnnUtils.makeBatch(totalTestSize, 16384)
    totalTestSentSize = len(testSetX)
    totalValidSentSize = len(validSetX)
    
    # Validation model
    validModel = theano.function( inputs = [idx], outputs = [predicter.errors(sY),predicter.yPred],
                                  givens={sX:sharedValidSetX, sY:castSharedValidSetY})
    
    # bulid test model
    testModel = theano.function( inputs = [idx], outputs = [predicter.errors(sY),predicter.yPred],
                                  givens={sX:sharedTestSetX, sY:castSharedTestSetY})
    
    validResult = rnnUtils.EvalandResult(validModel, totalValidSize, 'valid') 
    testResult = rnnUtils.EvalandResult(testModel, totalTestSize, 'test')
    
    rnnUtils.writeResult(validResult, P.validResultFilename, validSetName)
    rnnUtils.writeResult(testResult, P.testResultFilename, testSetName)
    
    rnnUtils.clearSharedDataXY(sharedTestSetX, sharedTestSetY)
    rnnUtils.clearSharedDataXY(sharedValidSetX, sharedValidSetY)


def getProb(bestModel, dataset, probFilename):

    print "...getting probability"
    # For getting prob
    setX, setY, setName = dataset
    sharedSetX, sharedSetY, castSharedSetY = rnnUtils.sharedDataXY(setX, setY)

    idx = T.ivector('i')
    sX = T.matrix(dtype=theano.config.floatX)
    sY = T.ivector()

    # bulid best DNN model
    predicter = DNN( input = rnnUtils.splicedX(sX, idx), P = P, params = bestModel )

    Model = theano.function( inputs = [idx], outputs = predicter.p_y_given_x, 
                                  givens={sX:sharedSetX, sY:castSharedSetY}, on_unused_input='ignore')
    
    # Center Index
    centerIdx = rnnUtils.findCenterIdxList(setY)

    # Total Center Index
    totalSize = len(centerIdx)
    
    # Make mini-Batch
    batchIdx = rnnUtils.makeBatch(totalSize, 16384)
    
    # Writing Probability
    rnnUtils.writeProb(Model, batchIdx, centerIdx, setName, probFilename)

    rnnUtils.clearSharedDataXY(sharedSetX, sharedSetY)
