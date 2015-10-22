import dnn
import os
import sys
import postfrank as pf 
from utils import readFile2, load_data, makePkl, readFile, load_pkl

specifying_textfile = sys.argv[1]
import sys

USE_EXIST_MODEL = True
def smooth(filename):
    name, label = readFile(filename)
    endIndxGroup = pf.findEndIndxofGroup(name = name, label = label)
    label = pf.correctLabel(endIndxGroup = endIndxGroup, name = name, label = label)
    pf.writeFile(filename = P.testSmoothedResultFilename, name = name, label = label)

class Logger(object):
    def __init__(self, logFilename):
        self.terminal = sys.stdout
        self.log = open(logFilename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

if __name__ == '__main__':
    P = dnn.Parameters(specifying_textfile)
    datasets  = load_data(filename = P.datasetFilename, totalSetNum=3)
    if not USE_EXIST_MODEL: 
        sys.stdout = Logger(P.logFilename)
        bestModel = dnn.trainDNN(datasets = datasets, P=P)
        bestModelFilename = '../model/' + P.outputFilename + '.model'
        makePkl(bestModel, P.bestModelFilename)
    else:
        # TODO use filename to build P
        bestModelFilename = sys.argv[2]
        bestModel = load_pkl(bestModelFilename)

    dnn.getResult(datasets = datasets, bestModel = bestModel, P = P)
    smooth(P.testResultFilename)
    smooth(P.validResultFilename)
   
