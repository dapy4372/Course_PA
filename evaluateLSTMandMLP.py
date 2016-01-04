import pdb
import sys
import csv
import string
import argparse

#from progressbar import Bar, ETA, Percentage, ProgressBar
from keras.models import model_from_json
from spacy.en import English
import numpy as np
import scipy.io
from sklearn.externals import joblib

sys.path.insert(0,'../data/')
# from preprocess import get_answers
sys.path.insert(0,'../scripts')

from itertools import izip_longest
def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return izip_longest(*args, fillvalue=fillvalue)

# python evaluateLSTMandMLP.py -model model/2016010401_LSTM_default_model.json -weights model/2016010401_LSTM_default_model_epock_025.hdf5 -choices /share/MLDS/cluster_results/choices_kmeans_test_1000.txt -results 2016010401_testing_result.txt
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
    if args.predict_type == 'train':
        questions_id_filename = origin_data_path + 'preprocessed/id_train.txt'
        questions = open(origin_data_path + 'preprocessed/questions_train.txt', 'r').read().decode('utf8').splitlines()
    elif args.predict_type == 'test':
        questions_id_filename = origin_data_path + 'preprocessed/id_test.txt'
        questions = open(origin_data_path + 'preprocessed/questions_test.txt', 'r').read().decode('utf8').splitlines()
    else:
        raise Exception("predict type error!")

    questions_id = []
    with open(questions_id_filename, "r") as infile:
        reader = csv.reader(infile, delimiter = ' ')
        for row in reader:
            questions_id.append(row[1])

    nlp = English()
    print 'loaded word2vec features'

    nb_classes = 3000
    y_predict = []
    batchSize = 128
    #widgets = ['Evaluating ', Percentage(), ' ', Bar(marker='#',left='[',right=']'), ' ', ETA()]
    #pbar = ProgressBar(widgets=widgets)

    for qu_batch in grouper(questions, batchSize, fillvalue=questions[0]):
        X_q_batch = get_questions_matrix_sum(qu_batch, nlp)
        X_batch = X_q_batch
        y_predict.extend(model.predict_classes(X_batch, verbose=0))
        #y_predict_text.extend(labelencoder.inverse_transform(y_predict))

    #pdb.set_trace()
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

    # check train correct rate
    if args.predict_type == 'train':
        # answers = get_answers(args.answers)
        nb_answers = answers.shape[0]
        results = np.all([answers, np.array(answers_predict, dtype = 'int32')], axis = 0)
        nb_correct = np.sum(results)
        correct_rate = float(nb_correct) / float(nb_answers)
        modelparams_str = (args.model.split('/')[-1]).split('.')[0] + '\n'
        eval_str = str.format("{0:.2f}", correct_rate*100) + ' ' + str(nb_answers) + ' ' + str(nb_correct) + '\n'
        with open('../log/train_correct_rate.log', "a") as outfile:
            outfile.write(modelparams_str)
            outfile.write(eval_str)
        print eval_str

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
