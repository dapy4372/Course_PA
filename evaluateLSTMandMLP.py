import sys
import resource
import csv
import string
import numpy as np
import argparse

from keras.models import model_from_json
# sys.path.insert(0,'../data/')
# sys.path.insert(0,'../scripts')

# python evaluateLSTMandMLP.py -model model/2016010401_LSTM_default_model.json -weights model/2016010401_LSTM_default_model_epock_025.hdf5 -choices /share/MLDS/cluster_results/choices_kmeans_test_1000.txt -results 2016010401_testing_result.txt

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-predict_type', type=str, default='test')
    parser.add_argument('-model', type=str, required=True)
    parser.add_argument('-weights', type=str, required=True)
    parser.add_argument('-choices', type=str, required=True)
    parser.add_argument('-answers', type=str, required=False)
    parser.add_argument('-results', type=str, required=True)
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

def getLanguageFeature(questionData, choiceData, idList):
    batchSize = len(idList)
    featureMatrix = np.zeros((batchSize, word_vec_dim * 6), dtype = 'float32')
    for i in xrange(batchSize):
        featureMatrix[i,:] = np.hstack((questionData[idList[i]], choiceData[idList[i]]))
    return featureMatrix


def loadData():
    idMap = {}
    with open('/share/MLDS/preprocessed/id_test.csv', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter = ' ')
        for row in reader:
            idMap[int(row[1])] = int(row[0])

    questionData = {}
    with open('/share/MLDS/question_wordvector/spacy_avg_300_test.csv', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter = ' ')
        for row in reader:
            questionData[int(row[0])] = np.array(row[1:]).astype(dtype = 'float32')

    imageData = {}
    with open('/share/MLDS/image_feature/caffenet_4096_test.csv', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter = ' ')
        for row in reader:
            imageData[int(row[0])] = np.array(row[1:]).astype(dtype = 'float32')

    choiceData = {}
    with open('/share/MLDS/choice_wordvector/spacy_sent_1500_test.csv', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter = ' ')
        for row in reader:
            choiceData[int(row[0])] = np.array(row[1:]).astype(dtype = 'float32')

    return idMap, questionData, imageData, choiceData


def prepareIdList(idList, batchSize):
    questionNum = len(idList)
    batchNum = questionNum / batchSize
    # random.shuffle(idList)
    idListInBatch = []
    for i in xrange(batchNum):
        idListInBatch.append( idList[i * batchSize : (i+1) * batchSize] )
    if questionNum % batchSize != 0:
        idListInBatch.append( idList[batchNum * batchSize :] )
        batchNum += 1
    return idListInBatch, batchNum

def cos_sim(y_true, y_pred):
    dot = T.sum(y_true * y_pred, axis = 0)
    u = T.sqrt(T.sum(T.sqr(y_true), axis = 0))
    v = T.sqrt(T.sum(T.sqr(y_pred), axis = 0))
    return 1 - dot / (u * v + 0.0001)

def main():
    args = parser.parse_args()
    arg = parseArgs()
    # nlp = English()

    print '*** load model ***'
    model = model_from_json( open(args.model).read() )
    model.load_weights(args.weights)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    origin_data_path = '/share/MLDS/'
    if args.predict_type == 'test':
        # questions_id_filename = origin_data_path + 'preprocessed/id_test.txt'
        # questions = open(origin_data_path + 'preprocessed/questions_test.txt', 'r').read().decode('utf8').splitlines()
    else:
        raise Exception("predict type error!")

    print '*** load data ***'
    # idMap, questionData, imageData, answerData = testData()
    idMap, questionData, imageData, choiceData = loadData()
    print idMap.items()[0]
    print questionData.items()[0]
    print imageData.items()[0]

    y_predict = []
    batchSize = 128
    idList = idMap.keys()
    questionIdList, batchNum = prepareIdList(idList, batchSize)
    for j in xrange(batchNum):
        imageIdListForBatch = [idMap[key] for key in questionIdList[j]]
        y_predict.extend(model.predict_classes([ getImageFeature(imageData, imageIdListForBatch),
                                                 getLanguageFeature(questionData, choiceData, questionIdList[j]) ],
                                               verbose=0))
    print y_predict[0]

    # choose answer
    label = ['A', 'B', 'C', 'D', 'E']
    answers_predict = []
    for i in xrange(len(idList)):
        loss = 10
        answer = -1
        for j in xrange(5):
            current_id = idList[i]
            current_loss = cos_sim(y_predict[i], questionData[ idList[i] ][ j*300:(j+1)*300 ])
            if current_loss < loss:
                answer = j
        answers_predict.append(label[answer])

    # write testing answer to file which will be uploaded
    if args.predict_type == 'test':
        with open(args.results, 'w') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(['q_id', 'ans'])
            for i in xrange(len(idList)):
                writer.writerow([idList[i], answers_predict[i]])


if __name__ == "__main__":
    main()
