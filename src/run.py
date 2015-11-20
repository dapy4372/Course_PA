import os
import sys
import utils
import rnn.rnn as rnn
import postprocessing as pp
import postprocessing_shrink.post_shrink as ps 
import rnn.rnnUtils as rnnUtils
setting = sys.argv[1]
USE_EXIST_MODEL = False

def smooth(noSmoothedFilename, smoothedFilename):
    name, label = utils.readFile(noSmoothedFilename)
    endIndxGroup = pp.findEndIndxofGroup(name = name, label = label)
    label = pp.correctLabel(endIndxGroup = endIndxGroup, name = name, label = label)
    pp.writeFile(filename = smoothedFilename, name = name, label = label)

class Logger(object):
    def __init__(self, logFilename):
        self.terminal = sys.stdout
        self.log = open(logFilename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

if __name__ == '__main__':

    P = rnnUtils.Parameters(setting)
    print P.outputFilename
    datasets  = utils.loadDataset(filename = P.datasetFilename, totalSetNum=3)
   
    # Redirect stdout to log file
    sys.stdout = Logger(P.logFilename)

    # train RNN model
    bestModelFilename = rnn.trainRNN(datasets, P)
    
    # Get result
    rnn.getResult(bestModelFilename, datasets)

    # Smooth
    smooth(noSmoothedFilename = P.testResultFilename, smoothedFilename = P.testSmoothedResultFilename)
    smooth(noSmoothedFilename = P.validResultFilename, smoothedFilename = P.validSmoothedResultFilename)
  
    # Postprocessing

    smoothFinalResult = P.testSmoothedResultFilename
    outputFilename = '../result/final_result.csv'
    
    new_name, new_label = ps.readFile_shrink(smoothFinalResult)
    twoMap = ps.readMap_old('./postprocessing_shrink/48_39.map')
    word, alph = ps.readMap('./postprocessing_shrink/48_idx_chr.map_b')
    
    new_name, new_label = ps.shrinkPhonemes(new_name, new_label)
    label = ps.remap_48_39(new_name, new_label, twoMap)
    label = ps.remap_39_alph(new_name, new_label, word, alph)

    ps.writeFile2(outputFilename, new_name, new_label)
