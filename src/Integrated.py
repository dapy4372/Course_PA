import dnn
import os
import sys
 
from utils import readFile2, load_data

specifying_textfile = sys.argv[1]

if __name__ == '__main__':
 
   P         = dnn.Parameters(specifying_textfile)
   datasets  = load_data(filename = P.datasetFilename, totalSetNum=3)
   bestModel = dnn.trainDNN(datasets = datasets, P=P)

   bestModelFilename = '../model/' + P.outputFilename + '.model'
   makePkl(bestModel, bestModeFilename)
   getResults(datasets = datasets, bestModel = bestModel)
