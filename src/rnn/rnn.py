import os
import sys
import timeit
import random
import numpy as np
import theano
import theano.tensor as T
import rnnUtils
import globalParam
from rnnUtils import Parameters
from rnnArchitecture import HiddenLayer, OutputLayer, RNN

# Print Weight matrix and bias
DEBUG = False

# Print PER for each training iteration
FER_PER_SENT = False

# Pause traing for each training iteration
PAUSE = False
OUTPUT_DETAIL = False
THRESHOLD = 0.35

# For learning rate decay threshould
THRESHOLD = 0.3 

# TODO
"""OUTPUT_DETAIL = False"""

parameterFilename = sys.argv[1]
np.set_printoptions(threshold=np.nan) # for print np array

def trainRNN(datasets, P):
    
    # make data to be sentence
    trainSetX, trainSetY, trainSetName = rnnUtils.makeDataSentence(datasets[0])
    validSetX, validSetY, validSetName = rnnUtils.makeDataSentence(datasets[1])
    
    # Cut sentence to be sub-sentence and fill with 0
    trainSetX, trainSetY, trainSetName, trainSetM = rnnUtils.cutSentenceAndSlide([trainSetX, trainSetY, trainSetName], P.cutSentSize, P.move)
    validSetX, validSetY, validSetName, validSetM = rnnUtils.cutSentenceAndFill([validSetX, validSetY, validSetName], P.cutSentSize)

    # make the total number of sub-sentence mod batchSize equals to zero 
    trantrainSetX, trainSetY, trainSetName, trainSetM = rnnUtils.fillBatch([trainSetX, trainSetY, trainSetName, trainSetM], P.batchSize)
    validSetX, validSetY, validSetName, validSetM = rnnUtils.fillBatch([validSetX, validSetY, validSetName, validSetM], P.batchSize)
    
    # make data to be numpy.array
    trainSetX = np.array(trainSetX)
    trainSetY = np.array(trainSetY)
    trainSetM = np.array(trainSetM)

    validSetX = np.array(validSetX)
    validSetY = np.array(validSetY)
    validSetM = np.array(validSetM)

    ###############
    # BUILD MODEL #
    ###############
    print '... building the model'
    x = T.tensor3()
    y = T.imatrix()
    m = T.imatrix()

    # For create a new model
    # 8 for W_i1, W_h1, b_h1, W_i2, W_h2, b_h2, a_0, a_0_reverse   
    # +2 for outputlayer W_o and b_o
    dummyParams = [None] * (8 * (P.rnnDepth) + 2) 
    
    # Build the RNN object for training
    classifier = RNN( input = x, params = dummyParams, P = P )
    
    # Build the DNN object for Validation
    predicter = RNN( input = x, params = dummyParams, P = P )
   
    # Global parameters setting
    globalParam.initGlobalLearningRate(P)
    globalParam.initGlobalFlag()
    globalParam.initGlobalVelocitys()
    globalParam.initGlobalSigmas()
    globalParam.initGlobalgradSqrs()

    # Cost function 1.cross entropy 2.weight decay
    cost = ( classifier.crossEntropy(y, m) + P.L1Reg * classifier.L1 + P.L2Reg * classifier.L2_sqr )
    
    grads = [ (T.grad(cost, param)).clip(-1 * P.clipRange, P.clipRange) for param in classifier.params]
    myOutputs = ( [classifier.errors(y, m)] +[cost]+ classifier.hiddenLayerList[0].output 
                 + [classifier.p_y_given_x] + [classifier.yPred] + grads + classifier.params )
    myUpdates = rnnUtils.chooseUpdateMethod(grads, classifier.params, P)

    # Training mode
    trainModel = theano.function( inputs = [x, y, m], outputs = myOutputs, updates = myUpdates )

    # Validation model
    validModel = theano.function( inputs = [x, y, m], outputs = predicter.errors(y, m) )

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
    prevFER = np.inf
    random.seed(P.seed)
    bestModelName = ''

    # Momentum, Adagrad, RMSProp parameters
    prevVelocitys = 0.
    prevGradSqrs  = 0.
    prevSigmaSqrs = 0.
   
    # Total number of sub-sentence
    totalTrainSentNum = len(trainSetX)
    totalValidSentNum = len(validSetX)

    # the number of batch
    totalTrainBatchNum = totalTrainSentNum / P.batchSize
    totalValidBatchNum = totalValidSentNum / P.batchSize

    startTime  = timeit.default_timer()
    while (epoch < P.maxEpoch) and (not doneLooping):
        epoch = epoch + 1

        if P.shuffle:
            p = np.random.permutation(totalTrainSentNum)

        trainLosses = []

        # For printing FER per sub-sentence
        sentNum = 0

        # Training
        for i in xrange(totalTrainBatchNum):
            setX = trainSetX[p][i * P.batchSize : (i+1) * P.batchSize]
            setY = trainSetY[p][i * P.batchSize : (i+1) * P.batchSize]
            setM = trainSetM[p][i * P.batchSize : (i+1) * P.batchSize]
            setX = np.array(setX).astype(dtype='float32')
            setY = np.array(setY).astype(dtype='int32')
            setM = np.array(setM).astype(dtype='int32')
            setX = np.transpose(setX, (1, 0, 2))
            setY = np.transpose(setY, (1, 0))
            setM = np.transpose(setM, (1, 0))
            outputs = trainModel(setX, setY, setM)
            trainLosses.append(outputs[0])
            
            break
            # TODO
            """if OUTPUT_DETAIL:
                  rnnUtils.printOutputDetail(outputs[])"""

            if FER_PER_SENT:
                trainFER = np.mean(trainLosses)
                print (('    %i,%f') % (i, trainFER * 100))
                sentNum+=1

            # Print parameter value for debug
            if DEBUG:
                rnnUtils.printGradsParams(outputs[7:7 + 8 * P.rnnDepth + 2], outputs[7 + 8 * P.rnnDepth:], P.rnnDepth)

            # Pause every iteration
            if PAUSE:
                sys.stdin.read(1)

        # Evaluate training FER 
        trainFER = np.mean(trainLosses)

        # Set the now train model's parameters to valid model
        nowModel = rnnUtils.getParamsValue(classifier.params)
        rnnUtils.setParamsValue(nowModel, predicter.params)
      

        validLosses = []
        for i in xrange(totalValidBatchNum):
            setX = validSetX[i * P.batchSize : (i+1) * P.batchSize]
            setY = validSetY[i * P.batchSize : (i+1) * P.batchSize]
            setM = validSetM[i * P.batchSize : (i+1) * P.batchSize]
            setX = np.array(setX).astype(dtype='float32')
            setY = np.array(setY).astype(dtype='int32')
            setM = np.array(setM).astype(dtype='int32')
            setX = np.transpose(setX, (1, 0, 2))
            setY = np.transpose(setY, (1, 0))
            setM = np.transpose(setM, (1, 0))
            validLosses.append( validModel(setX, setY, setM) )

        # Evaluate validation FER
        validFER = np.mean(validLosses)
        prevModel = nowModel

        if validFER < prevFER:
            if bestModelName != '':
                os.remove(bestModelName)
            prevFER = validFER
            prevModel = nowModel
            bestModelName = '../model/' + P.outputFilename + '_validFER_' + str(validFER * 100.) 
            rnnUtils.saveModelPkl( nowModel, P, bestModelName ) 
            curEarlyStop = 0
        else:
            curEarlyStop += 1

        # Learning rate decay
        if validFER < THRESHOLD and curEarlyStop > P.earlyStop:
            rnnUtils.setParamsValue(prevModel, classifier.params)
            print ((' ====,%i,\t%f,\t%f') % (epoch, trainFER * 100, validFER * 100. ))
            globalParam.lr = globalParam.lr * P.learningRateDecay
            if P.updateMethod == 'Momentum':
                globalParam.velocitys = prevVelocitys
            if P.updateMethod == 'NAG':
                globalParam.velocitys = prevVelocitys
            if P.updateMethod == 'Adagrad':
                globalParam.gradSqrs = prevGradSqrs
            if P.updateMethod == 'RMSProp':
                globalParam.sigmaSqrs = prevSigmaSqrs
            epoch -= 1
            continue

        # Record the Adagrad, RMSProp parameter
        if P.updateMethod == 'Momentum':
            prevVelocitys = globalParam.velocitys
        if P.updateMethod == 'NAG':
            prevVelocitys = globalParam.velocitys
        if P.updateMethod == 'Adagrad':
            prevGradSqrs = globalParam.gradSqrs
        if P.updateMethod == 'RMSProp':
            prevSigmaSqrs = globalParam.sigmaSqrs

        # Print the result of this epoch
        print (('%i,%f,%f') % (epoch, trainFER * 100, validFER * 100. ))

    # end of training
    endTime = timeit.default_timer()
    print (('time %.2fm' % ((endTime - startTime) / 60.)))

    return bestModelName

def getResult(bestModelName, datasets):
    print "... getting result"

    print "... load model"
    bestModel, P = rnnUtils.readModelPkl(bestModelName)     
    validSetX, validSetY, validSetN = rnnUtils.makeDataSentence(datasets[1])
    testSetX, testSetY, testSetN = rnnUtils.makeDataSentence(datasets[2])

    if P.cutSentSize > 0:
        validSetX, validSetY, validSetN, validSetM = rnnUtils.cutSentenceAndFill([validSetX, validSetY, validSetN], P.cutSentSize)
        testSetX, testSetY, testSetN, testSetM = rnnUtils.cutSentenceAndFill([testSetX, testSetY, testSetN], P.cutSentSize)
    
    validSetX, validSetY, validSetN, validSetM = rnnUtils.fillBatch([validSetX, validSetY, validSetN, validSetM], P.batchSize)
    testSetX, testSetY, testSetN, testSetM = rnnUtils.fillBatch([testSetX, testSetY, testSetN, testSetM], P.batchSize)

    validSetX = np.array(validSetX)
    validSetY = np.array(validSetY)
    validSetM = np.array(validSetM)

    testSetX = np.array(testSetX)
    testSetY = np.array(testSetY)
    testSetM = np.array(testSetM)

    print '... building the model'
    x = T.tensor3()
    y = T.imatrix()
    m = T.imatrix()
    
    # bulid best RNN  model
    predicter = RNN( input = x, P = P, params = bestModel )
    
    # Total number of sub-sentence
    totalValidSentNum = len(validSetX)
    totalTestSentNum = len(testSetX)

    # the number of batch
    totalValidBatchNum = totalValidSentNum / P.batchSize
    totalTestBatchNum = totalTestSentNum / P.batchSize

    # best model
    model = theano.function( inputs = [x, y, m], outputs = [predicter.errors(y, m), predicter.yPred] )
    
    rnnUtils.EvalnSaveResult(model, totalValidBatchNum, validSetX, validSetY, validSetN, validSetM, 
                             P.validResultFilename, P.batchSize, 'valid') 
    rnnUtils.EvalnSaveResult(model, totalTestBatchNum, testSetX, testSetY, testSetN, testSetM, 
                             P.testResultFilename, P.batchSize, 'test')
