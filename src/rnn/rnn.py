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

DEBUG = False 
FER_PER_SENT = True
PAUSE = False
OUTPUT_DETAIL = False
THRESHOLD = 0.27

parameterFilename = sys.argv[1]
np.set_printoptions(threshold=np.nan) # for print np array

def trainRNN(datasets, P):
    
    trainSetX, trainSetY, trainSetName = rnnUtils.makeDataSentence(datasets[0])
    validSetX, validSetY, validSetName = rnnUtils.makeDataSentence(datasets[1])
    if P.cutSentSize > 0:
        trainSetX, trainSetY, trainSetName = rnnUtils.cutSentence([trainSetX, trainSetY, trainSetName], P.cutSentSize)
        validSetX, validSetY, validSetName = rnnUtils.cutSentence([validSetX, validSetY, validSetName], P.cutSentSize)

    ###############
    # BUILD MODEL #
    ###############
    print '... building the model'
    idx = T.iscalar('i')
    x = T.matrix()
    y = T.ivector()

    # For create a new model
    dummyParams = [None] * (6 * (P.rnnDepth) + 2)  # +2 for outputlayer W_o and b_o
    
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
    
    grads = [ (T.grad(cost, param)).clip(-1 * P.clipRange, P.clipRange) for param in classifier.params]
    myOutputs = ( [classifier.errors(y)] +[cost]+ classifier.hiddenLayerList[0].output 
                 + [classifier.p_y_given_x] + [classifier.yPred] + grads + classifier.params )
    myUpdates = rnnUtils.chooseUpdateMethod(grads, classifier.params, P)

    # Training mode
    trainModel = theano.function( inputs = [x, y], outputs = myOutputs, updates = myUpdates )

    # Validation model
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
    prevFER = np.inf
    random.seed(P.seed)
    bestModelName = ''

    # Momentum, Adagrad, RMSProp parameters
    prevVelocitys = 0.
    prevGradSqrs  = 0.
    prevSigmaSqrs = 0.
   
    # Total Center Index
    totalTrainSentNum = len(trainSetX)
    totalValidSentNum = len(validSetX)

    # Sentence Index
    trainSentIdx = list(range(totalTrainSentNum))
    validSentIdx = list(range(totalValidSentNum))

    startTime  = timeit.default_timer()
    while (epoch < P.maxEpoch) and (not doneLooping):
        epoch = epoch + 1

        if P.shuffle:
            random.shuffle(trainSentIdx)
        # Training
        trainLosses=[]
        sentNum = 1
        for i in xrange(totalTrainSentNum):
            outputs = trainModel(np.array(trainSetX[trainSentIdx[i]]).astype(dtype='float32'), np.array(trainSetY[trainSentIdx[i]]).astype(dtype='int32'))
            trainLosses.append(outputs[0])
            
            # Print output detail
            """if OUTPUT_DETAIL:
#                  rnnUtils.printOutputDetail(outputs[])"""

            if FER_PER_SENT:
                trainFER = np.mean(trainLosses)
                print (('    %i,%f') % (sentNum, trainFER * 100))
                sentNum+=1
            # Print parameter value for debug
            if DEBUG:
                rnnUtils.printGradsParams(outputs[7:7 + 8 * P.rnnDepth + 2], outputs[7 + 8 * P.rnnDepth+2:], P.rnnDepth)

            # Pause every iteration
            if PAUSE:
                sys.stdin.read(1)

        # Evaluate training FER 
        trainFER = np.mean(trainLosses)

        # Set the now train model's parameters to valid model
        nowModel = rnnUtils.getParamsValue(classifier.params)
        rnnUtils.setParamsValue(nowModel, predicter.params)
        
        # Evaluate validation FER
        validLosses = [ validModel( np.array(validSetX[validSentIdx[i]]).astype(dtype='float32'), np.array(validSetY[validSentIdx[i]]).astype(dtype='int32') ) for i in xrange(totalValidSentNum)]
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
            if P.updateMethod == 'Adagrad':
                globalParam.gradSqrs = prevGradSqrs
            if P.updateMethod == 'RMSProp':
                globalParam.sigmaSqrs = prevSigmaSqrs
            epoch -= 1
            continue

        # Record the Adagrad, RMSProp parameter
        if P.updateMethod == 'Momentum':
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

    validSetX, validSetY, validSetName = rnnUtils.makeDataSentence(datasets[1])
    testSetX, testSetY, testSetName = rnnUtils.makeDataSentence(datasets[2])

    if P.cutSentSize > 0:
        validSetX, validSetY, validSetName = rnnUtils.cutSentence([validSetX, validSetY, validSetName], P.cutSentSize)
        testSetX, testSetY, testSetName = rnnUtils.cutSentence([testSetX, testSetY, testSetName], P.cutSentSize)
    
    print '... building the model'
    idx = T.iscalar('i')
    x = T.matrix()
    y = T.ivector()
    
    # bulid best RNN  model
    predicter = RNN( input = x, P = P, params = bestModel )
    
    # Total number of sentence
    totalValidSentNum = len(validSetX)
    totalTestSentNum = len(testSetX)

    # Sentence Index
    validSentIdx = list(range(totalValidSentNum))
    testSentIdx = list(range(totalTestSentNum))
    
    # best model
    model = theano.function( inputs = [x, y], outputs = [predicter.errors(y), predicter.yPred] )
    
    validResult = rnnUtils.EvalandResult(model, totalValidSentNum, validSetX, validSetY, 'valid') 
    testResult = rnnUtils.EvalandResult(model, totalTestSentNum, testSetX, testSetY, 'test')
    
    rnnUtils.writeResult(validResult, P.validResultFilename, validSetName)
    rnnUtils.writeResult(testResult, P.testResultFilename, testSetName)
