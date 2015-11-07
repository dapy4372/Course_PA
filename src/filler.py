import os,sys
import numpy
import theano
from utils import loadDataset,namepick


def maxLen(Set):
    max_sen='aa'
    maxLength=0
    for i in range(len(Set[2])):
        name,number=namepick(Set[2][i])
        if maxLength<number:
            maxLength=number
            max_sen=name
# print max_sen,maxLength
    return maxLength


def fillerCore(Set):
    maxLength=maxLen(Set)
    count=0
    fillSet=[]
    for i in range (4):
      fillSet.append([])

    seqNum=0
    
    finalSet=[]
    for i in range (4):
        finalSet.append([])

    currentName,a=namepick(Set[2][0])
    for i in range(len(Set[2])):
        count=count+1
        fillSet[0].append(Set[0][i])
        fillSet[1].append(Set[1][i])
        fillSet[2].append(Set[2][i])
        fillSet[3].append(1)
        name,b=namepick(Set[2][i])
        if currentName!=name:
            for j in range(maxLength-count):
                fillSet[0].append(numpy.zeros(len(Set[0][0])))
                fillSet[1].append(-1)
                fillSet[2].append(currentName+"_"+str(j+count))
                fillSet[3].append(-1)
            count=0
            currentName=name
    for i in range((len(fillSet[0])/maxLength)):
        finalSet[0].append(fillSet[0][i*maxLength:(i+1)*maxLength-1])
        finalSet[1].append(fillSet[1][i*maxLength:(i+1)*maxLength-1])
        finalSet[2].append(fillSet[2][i*maxLength:(i+1)*maxLength-1])
        finalSet[3].append(fillSet[3][i*maxLength:(i+1)*maxLength-1])
    return finalSet

if __name__ == '__main__':
    dataSet=loadDataset("../pkl/fbank_69_dataset_without_preprocessing.pkl",3)
    fill_trainSet=fillerCore(dataSet[0])
#fill_validSet=fillerCore(dataSet[1])
#fill_testSet =fillerCore(dataSet[2])
    a = numpy.array(fill_trainSet)
    print a[0]
    print a[0][0]
    print a[1].shape
    print a[2].shape
    print a[3].shape


