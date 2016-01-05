import sys
import csv
import string
import numpy as np
import argparse

from keras.models import model_from_json
from spacy.en import English
import scipy.io

sys.path.insert(0,'../data/')
# from preprocess import get_answers
sys.path.insert(0,'../scripts')

# python evaluateLSTMandMLP.py -model model/2016010401_LSTM_default_model.json -weights model/2016010401_LSTM_default_model_epock_025.hdf5 -choices /share/MLDS/cluster_results/choices_kmeans_test_1000.txt -results 2016010401_testing_result.txt

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

def loadData():
    idMap = {}
    with open('/share/MLDS/preprocessed/id_test.txt', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter = ' ')
        for row in reader:
            idMap[int(row[1])] = int(row[0])

    questionIdList = idMap.keys()
    questionData = {}
    with open('/share/MLDS/preprocessed/questions_test.txt', 'r') as txtfile:
        questionsTrain = txtfile.read().splitlines()
        if (len(questionIdList) != len(questionsTrain)):
            print "*** load question error ***"
        for i in xrange(len(questionIdList)):
            questionData[questionIdList[i]] = questionsTrain[i]

    imageData = {}
    with open('/share/MLDS/test_id_feat_pair.txt', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter = ' ')
        for row in reader:
            imageData[int(row[0])] = np.array(row[1:]).astype(dtype = 'float32')

    return idMap, questionData, imageData

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', type=str, required=True)
    parser.add_argument('-weights', type=str, required=True)
    parser.add_argument('-results', type=str, required=False)
    parser.add_argument('-choices', type=str, required=True)
    parser.add_argument('-answers', type=str, required=False)
    parser.add_argument('-predict_type', type=str, default='test')

    args = parser.parse_args()

    model = model_from_json( open(args.model).read() )
    model.load_weights(args.weights)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    origin_data_path = '/share/MLDS/'
    if args.predict_type == 'test':
        questions_id_filename = origin_data_path + 'preprocessed/id_test.txt'
        questions = open(origin_data_path + 'preprocessed/questions_test.txt', 'r').read().decode('utf8').splitlines()
    else:
        raise Exception("predict type error!")

    questions_id = []
    with open(questions_id_filename, "r") as infile:
        reader = csv.reader(infile, delimiter = ' ')
        for row in reader:
            questions_id.append(int(row[1]))

    nlp = English()

    print '*** load data ***'
    # idMap, questionData, imageData, answerData = testData()
    idMap, questionData, imageData = loadData()
    print idMap.items()[0]
    print questionData.items()[0]
    print imageData.items()[0]

    y_predict = []
    batchSize = 128
    questionIdList, batchNum = prepareIdList(idMap.keys(), batchSize)

    for j in xrange(batchNum):
        imageIdListForBatch = [idMap[key] for key in questionIdList[j]]
        y_predict.extend(model.predict_classes([ getImageFeature(imageData, imageIdListForBatch),
                                                 getQuestionWordVector(questionData, questionIdList[j], nlp) ],
                                               verbose=0))
    print y_predict[0]

    # choose choice by cluster number
    answers_predict = []
    with open(args.choices, 'r') as infile:
        reader = csv.reader(infile, delimiter = ' ')
        cur_choice = 0
        for row in reader:
            row = map(int, row)
            if y_predict[cur_choice] in row:
                answers_predict.append(row.index(y_predict[cur_choice]))
            else:
                answers_predict.append(0)
            cur_choice += 1

    # write testing answer to file which will be uploaded
    if args.predict_type == 'test':
        with open(args.results, 'w') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(['q_id', 'ans'])
            for ans, qid in zip(answers_predict, questions_id):
                ans = str(ans).translate(string.maketrans("01234", "ABCDE"))
                writer.writerow([qid,ans])


if __name__ == "__main__":
    main()
