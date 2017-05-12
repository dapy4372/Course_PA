import utils
import sys
import utils

def loadDataset(filename, totalSetNum):
    import sys
    import numpy
    import cPickle
    # print '... loading data'
    # Load the dataset
    f = open(filename, 'rb')
    datasets = cPickle.load(f)
    f.close()
    return datasets

def load_results(filename):
    f = open(filename, 'r')
    ret = []
    for sentence in f:
      this_sent = []
      part = sentence.split(',')
      sentLen = len(part)
      for frameidx in xrange(sentLen):
          if (frameidx == 0):
              dummy = part[frameidx].replace("[","")
              this_sent.append(dummy)
          elif (frameidx == (sentLen-1)):
              dummy = part[frameidx].replace("]","")
              dummy = dummy.replace(" ","")
              dummy = dummy.replace("\n","")
              this_sent.append(dummy)
          else:
              dummy = part[frameidx].replace(" ","")
              this_sent.append(dummy)
#print this_sent
      ret.append(this_sent)
    return ret

if __name__ == '__main__':
    input_file  = sys.argv[1]
    output_file = sys.argv[3]
    golden_file = sys.argv[2]
    results = load_results(input_file)
    datasets = loadDataset(filename = golden_file, totalSetNum = 3)
    testX, testY, testN = utils.makeDataSentence(datasets[2])
    f = open(output_file, 'w')
    for i in xrange(len(results)):
      for j in xrange(len(results[i])):
        f.write( str(testN[i][j]) + "," + (results[i][j]) + "\n")
#f.write("#############################")
    f.close()

