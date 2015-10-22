import dnn
import os
import sys
import postfrank as pf 
from utils import readFile2, load_data, makePkl, readFile

specifying_textfile = sys.argv[1]
import sys

class Logger(object):
    def __init__(self, logFilename):
        self.terminal = sys.stdout
        self.log = open(logFilename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

if __name__ == '__main__':
 
   P = dnn.Parameters(specifying_textfile)
   sys.stdout = Logger(P.logFilename)

   datasets  = load_data(filename = P.datasetFilename, totalSetNum=3)
   bestModel = dnn.trainDNN(datasets = datasets, P=P)

   bestModelFilename = '../model/' + P.outputFilename + '.model'
   makePkl(bestModel, P.bestModelFilename)
   dnn.getResult(datasets = datasets, bestModel = bestModel, P = P)
   
   name, label = readFile(P.testResultFilename)
   endIndxGroup = pf.findEndIndxofGroup(name = name, label = label)
   label = pf.correctLabel(endIndxGroup = endIndxGroup, name = name, label = label)
   pf.writeFile(filename = P.testSmoothedResultFilename, name = name, label = label)
   
   name, label = readFile(P.validResultFilename)
   endIndxGroup = pf.findEndIndxofGroup(name = name, label = label)
   label = pf.correctLabel(endIndxGroup = endIndxGroup, name = name, label = label)
   pf.writeFile(filename = P.validSmoothedResultFilename, name = name, label = label)
