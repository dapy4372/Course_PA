import os
import resource
import csv
import random
import numpy as np
import theano.tensor as T
import argparse

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Reshape, Merge, Dense, MaxoutDense, Dropout, Activation
from keras.layers.recurrent import LSTM
from keras.optimizers import SGD

# from spacy.en import English

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

# answer_group_num = 1000
# def getAnswer(answerData, idList, categorical = True):
#     batchSize = len(idList)
#     if categorical:
#         answerMatrix = np.zeros((batchSize, answer_group_num), dtype = 'int32')
#         for i in xrange(batchSize):
#             # answerMatrix[i][ answerData[ idList[i] ][0] ] = 1
#             answerMatrix[i][ answerData[ idList[i] ] ] = 1
#         return answerMatrix
#     else:
#         answerMatrix = np.zeros(batchSize, dtype = 'float32')
#         for i in xrange(batchSize):
#             # answerMatrix[i] = answerData[ idList[i] ][0]
#             answerMatrix[i] = answerData[ idList[i] ]
#         return answerMatrix

def getLanguageFeature(questionData, choiceData, idList):
    batchSize = len(idList)
    featureMatrix = np.zeros((batchSize, word_vec_dim*6), dtype = 'float32')
    for i in xrange(batchSize):
        featureMatrix[i,:] = np.hstack((questionData[idList[i]], choiceData[idList[i]]))
    return featureMatrix

def getAnswer(choiceData, answerData, idList):
    batchSize = len(idList)
    answerMatrix = np.zeros((batchSize, word_vec_dim), dtype = 'float32')
    for i in xrange(batchSize):
        answerMatrix[i,:] = choiceData[idList[i]][ answerData[idList[i]]*word_vec_dim : (answerData[idList[i]]+1)*word_vec_dim ]
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
    with open('/share/MLDS/preprocessed/id_train.csv', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter = ' ')
        for row in reader:
            idMap[int(row[1])] = int(row[0])

    questionData = {}
    with open('/share/MLDS/question_wordvector/spacy_avg_300_train.csv', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter = ' ')
        for row in reader:
            questionData[int(row[0])] = np.array(row[1:]).astype(dtype = 'float32')

    imageData = {}
    with open('/share/MLDS/image_feature/caffenet_4096_train.csv', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter = ' ')
        for row in reader:
            imageData[int(row[0])] = np.array(row[1:]).astype(dtype = 'float32')

    choiceData = {}
    with open('/share/MLDS/choice_wordvector/spacy_sent_1500_train.csv', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter = ' ')
        for row in reader:
            choiceData[int(row[0])] = np.array(row[1:]).astype(dtype = 'float32')

    answerData = {}
    with open('/share/MLDS/preprocessed/id_answer_category_train.csv', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter = ' ')
        for row in reader:
            answerData[int(row[0])] = int(row[1])

    return idMap, questionData, imageData, choiceData, answerData

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


def cos_sim(y_true, y_pred):
    dot = T.sum(y_true * y_pred, axis = 1)
    u = T.sqrt(T.sum(T.sqr(y_true), axis = 1))
    v = T.sqrt(T.sum(T.sqr(y_pred), axis = 1))
    return 1 - dot / (u * v + 0.0001)

if __name__ == '__main__':
    arg = parseArgs()
    max_len = 30
    # wordVectorModel = English()
    limit_memory(1.2 * 1e10)  # about 12GB

    # build model
    image_model = Sequential()
    image_model.add(Reshape(input_shape = (img_dim,), dims=(img_dim,)))

    language_model = Sequential()
    language_model.add(Reshape(input_shape = (word_vec_dim * 6,), dims=(word_vec_dim * 6,)))
    # if arg.lstm_layers == 1:
    #     language_model.add(LSTM(output_dim = arg.lstm_units, input_shape = (max_len, word_vec_dim), return_sequences = False, activation = 'sigmoid', inner_activation = 'hard_sigmoid'))
    # else:
    #     language_model.add(LSTM(output_dim = arg.lstm_units, input_shape = (max_len, word_vec_dim), return_sequences = True, activation = 'sigmoid', inner_activation = 'hard_sigmoid'))
    #     for i in xrange(arg.lstm_layers - 2):
    #         language_model,add(LSTM(output_dim = arg.lstm_units, return_sequences = True, activation = 'sigmoid', inner_activation = 'hard_sigmoid'))
    #     language_model.add(LSTM(output_dim = arg.lstm_units, return_sequences = False, activation = 'sigmoid', inner_activation = 'hard_sigmoid'))

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
    model.add(Dense(output_dim = word_vec_dim, init = 'uniform'))
    # model.add(Activation('softmax'))

    # sgd = SGD(lr = arg.lr, decay = 1e-6, momentum = arg.momentum, nesterov = True)
    model.compile(loss = cos_sim, optimizer = 'rmsprop')

    model_file_name = 'model/2016010502_MLP_vectorsum'
    open(model_file_name + '.json', 'w').write( model.to_json() )

    # read data
    print '*** load data ***'
    # idMap, questionData, imageData, answerData = testData()
    idMap, questionData, imageData, choiceData, answerData = loadData()
    print idMap.items()[0]
    print questionData.items()[0]
    print imageData.items()[0]
    print choiceData.items()[0]
    print answerData.items()[0]

    # training
    print '*** start training ***'
    tmp = 0
    for i in xrange(arg.epochs):
        print 'epoch #' + str(i+1)
        questionIdList, batchNum = prepareIdList(idMap.keys(), arg.batch_size)
        for j in xrange(batchNum):
            imageIdListForBatch = [idMap[key] for key in questionIdList[j]]
            loss = model.train_on_batch(X = [ getImageFeature(imageData, imageIdListForBatch),
                                              getLanguageFeature(questionData, choiceData, questionIdList[j]) ],
                                        y = getAnswer(choiceData, answerData, questionIdList[j]) )
            print loss
            tmp = loss[0]
        # predict = model.predict_on_batch(X)
        if (i+1) % 5 == 0:
            model.save_weights(model_file_name + '_epock_{:03d}_loss_{:.3f}.hdf5'.format(i+1, tmp))

