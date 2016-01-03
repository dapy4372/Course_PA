import os
import resource
import csv
import random
import numpy as np
import theano
import argparse

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Reshape, Merge, Dense, MaxoutDense, Dropout, Activation
from keras.layers.recurrent import LSTM
from keras.optimizers import SGD

from spacy.en import English

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-lstm_units', type=int, default=512)
    parser.add_argument('-lstm_layers', type=int, default=1)
    parser.add_argument('-mlp_units', type=int, default=1024)
    parser.add_argument('-mlp_layers', type=int, default=3)
    parser.add_argument('-mlp_activation', type=str, default='softplus')
    parser.add_argument('-dropout', type=float, default=0.5)
    parser.add_argument('-maxout', type=bool, default=False)
    parser.add_argument('-batch_size', type=int, default=128)
    parser.add_argument('-epochs', type=int, default=100)
    parser.add_argument('-lr', type=float, default=0.1)
    parser.add_argument('-momentum', type=float, default=0.9)
    return parser.parse_args()

img_dim = 4096
def getImageFeature(imageData, idList):
    batchSize = len(idList)
    imageMatrix = np.zeros((batchSize, img_dim), dtype = 'float32')
    for i in xrange(batchSize):
        imageMatrix[i,:] = imageData[ idList[i] ]
    return imageMatrix

word_vec_dim = 300
def getQuestionWordVector(questionData, idList, wordVectorModel):
    batchSize = len(idList)
    maxlen = 0
    tokens = []
    for i in xrange(batchSize):
        tokens.append( wordVectorModel( questionData[ idList[i] ].decode('utf8')) )
        maxlen = max(maxlen, len(tokens[i]))
    questionMatrix = np.zeros((batchSize, maxlen, word_vec_dim), dtype = 'float32')
    for i in xrange(batchSize):
        for j in xrange(len(tokens[i])):
            questionMatrix[i,j,:] = tokens[i][j].vector
    return questionMatrix

answer_group_num = 1000
def getAnswer(answerData, idList, categorical = True):
    batchSize = len(idList)
    if categorical:
        answerMatrix = np.zeros((batchSize, answer_group_num), dtype = 'int32')
        for i in xrange(batchSize):
            # answerMatrix[i][ answerData[ idList[i] ][0] ] = 1
            answerMatrix[i][ answerData[ idList[i] ] ] = 1
        return answerMatrix
    else:
        answerMatrix = np.zeros(batchSize, dtype = 'float32')
        for i in xrange(batchSize):
            # answerMatrix[i] = answerData[ idList[i] ][0]
            answerMatrix[i] = answerData[ idList[i] ]
        return answerMatrix

def testData():
    idMap = {}
    questionData = {}
    imageData = {}
    answerData = {}
    idMap[1002] = 100
    questionData[1002] = "Who are you?"
    imageData[100] = np.random.rand(4096)
    answerData[1002] = [2, 52, 2, 109, 400, 876]
    return idMap, questionData, imageData, answerData

def loadData():
    idMap = {}
    with open('/share/MLDS/preprocessed/id_train.txt', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter = ' ')
        for row in reader:
            idMap[int(row[1])] = int(row[0])

    questionIdList = idMap.keys()
    questionData = {}
    with open('/share/MLDS/preprocessed/questions_train.txt', 'r') as txtfile:
        questionsTrain = txtfile.read().splitlines()
        if (len(questionIdList) != len(questionsTrain)):
            print "*** load question error ***"
        for i in xrange(len(questionIdList)):
            questionData[questionIdList[i]] = questionsTrain[i]

    imageData = {}
    with open('/share/MLDS/final_img_feat.txt', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter = ' ')
        for row in reader:
            imageData[int(row[0])] = np.array(row[1:]).astype(dtype = 'float32')

    answerData = {}
    with open('/share/MLDS/cluster_results/answers_kmeans_train_1000.txt', 'r') as txtfile:
        answersTrain = txtfile.read().splitlines()
        if (len(questionIdList) != len(answersTrain)):
            print "*** load answer error ***"
        for i in xrange(len(questionIdList)):
            answerData[questionIdList[i]] = int(answersTrain[i])

    return idMap, questionData, imageData, answerData

def prepareIdList(idList, batchSize):
    questionNum = len(idList)
    batchNum = questionNum / batchSize
    random.shuffle(idList)
    idListInBatch = []
    for i in xrange(batchNum):
        idListInBatch.append( idList[i * batchSize : (i+1) * batchSize] )
    if questionNum % batchSize != 0:
        idListInBatch.append( idList[batchNum * batchSize :] )
        batchNum += 1
    return idListInBatch, batchNum

def limit_memory(maxsize):
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (maxsize, hard))

if __name__ == '__main__':
    arg = parseArgs()
    max_len = 30
    wordVectorModel = English()
    limit_memory(1.0 * 1e10)  # about 10GB

    # build model
    image_model = Sequential()
    image_model.add(Reshape(input_shape = (img_dim,), dims=(img_dim,)))

    language_model = Sequential()
    if arg.lstm_layers == 1:
        language_model.add(LSTM(output_dim = arg.lstm_units, input_shape = (max_len, word_vec_dim), return_sequences = False, activation = 'sigmoid', inner_activation = 'hard_sigmoid'))
    else:
        language_model.add(LSTM(output_dim = arg.lstm_units, input_shape = (max_len, word_vec_dim), return_sequences = True, activation = 'sigmoid', inner_activation = 'hard_sigmoid'))
        for i in xrange(arg.lstm_layers - 2):
            language_model,add(LSTM(output_dim = arg.lstm_units, return_sequences = True, activation = 'sigmoid', inner_activation = 'hard_sigmoid'))
        language_model.add(LSTM(output_dim = arg.lstm_units, return_sequences = False, activation = 'sigmoid', inner_activation = 'hard_sigmoid'))

    model = Sequential()
    model.add(Merge([image_model, language_model], mode = 'concat', concat_axis = 1))
    if arg.maxout is True:
        for i in xrange(arg.mlp_layers):
            model.add(MaxoutDense(output_dim = arg.mlp_units, nb_feature = 2, init = 'uniform'))
            model.add(Dropout(arg.dropout))
    else:
        for i in xrange(arg.mlp_layers):
            model.add(Dense(output_dim = arg.mlp_units, init = 'uniform'))
            model.add(Activation(arg.mlp_activation))
            model.add(Dropout(arg.dropout))
    model.add(Dense(output_dim = answer_group_num, init = 'uniform'))
    model.add(Activation('softmax'))

    sgd = SGD(lr = arg.lr, decay = 1e-6, momentum = arg.momentum, nesterov = True)
    model.compile(loss = 'categorical_crossentropy', optimizer = sgd)

    # read data
    print '*** load data ***'
    # idMap, questionData, imageData, answerData = testData()
    idMap, questionData, imageData, answerData = loadData()
    print idMap.items()[0]
    print questionData.items()[0]
    print imageData.items()[0]
    print answerData.items()[0]

    # training
    print '*** start training ***'
    for i in xrange(arg.epochs):
        questionIdList, batchNum = prepareIdList(idMap.keys(), arg.batch_size)
        for j in xrange(batchNum):
            imageIdListForBatch = [idMap[key] for key in questionIdList[j]]
            loss = model.train_on_batch(X = [ getImageFeature(imageData, imageIdListForBatch),
                                              getQuestionWordVector(questionData, questionIdList[j], wordVectorModel) ],
                                        y = getAnswer(answerData, questionIdList[j]) )
            print loss
        # predict = model.predict_on_batch(X)

    model.save_weights(model.save_weights('2016010400_LSTM_default_model.hdf5')

