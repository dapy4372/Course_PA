import os
import sys
import subprocess
import glob
from os import path


def readMap(filename):
    f =  open(filename, 'r')
    index = []
    for i in f:
        part = i.split(',')
        index.append(int(part[1]))
    f.close()
    return index

def mapping(inputMap, inputVector, outputDim)
    if len(inputMap) == len(inputVector):
        output = []
        # Initialization of the output vector
        for i  in xrange(len(outputDim)):
            output.append(int(0))

        for i in xrange(len(inputVector))
            output[int(inputMap[i])] += vector[i]
        return output
        
    else:
        print: 'The input map/vector dimensions dont match...'
        return 0

    
