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
    print max_sen,maxLength
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
        name,b=namepick(Set[2][i])
        if currentName!=name:
            for j in range(maxLength-count+1):
                fillSet[0].append(numpy.zeros(len(Set[0][0])))
                fillSet[1].append(-1)
                fillSet[2].append(currentName+"_"+str(j+count+1))
                fillSet[3].append(-1)
            count=1
            currentName=name
        
        fillSet[0].append(Set[0][i])
        fillSet[1].append(Set[1][i])
        fillSet[2].append(Set[2][i])
        fillSet[3].append(1)

    for i in range((len(fillSet[0])/maxLength)):
        finalSet[0].append(fillSet[0][i*maxLength:(i+1)*maxLength-1])
        finalSet[1].append(fillSet[1][i*maxLength:(i+1)*maxLength-1])
        finalSet[2].append(fillSet[2][i*maxLength:(i+1)*maxLength-1])
        finalSet[3].append(fillSet[3][i*maxLength:(i+1)*maxLength-1])
    return finalSet

def filler():
    dataSet=loadDataset("../pkl/fbank_69_dataset.pkl",3)
    print dataSet[0][2][0:300]
    fill_trainSet=fillerCore(dataSet[0])
    fill_validSet=fillerCore(dataSet[1])
    fill_testSet =fillerCore(dataSet[2])
    return fill_trainSet,fill_validSet,fill_testSet

a,b,c=filler()

print a[1][0]
