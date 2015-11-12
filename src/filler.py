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
                fillSet[2].append(currentName+"_"+str(j+count))
                fillSet[3].append(-1)
            count=1
            currentName=name
        
        fillSet[0].append(Set[0][i])
        fillSet[1].append(Set[1][i])
        fillSet[2].append(Set[2][i])
        fillSet[3].append(1)
    #print "bbbbbbbbbbbbbb"
    #print fillSet[2]
    for i in range((len(fillSet[0])/maxLength)):
        finalSet[0].append(fillSet[0][i*maxLength:(i+1)*maxLength])
        finalSet[1].append(fillSet[1][i*maxLength:(i+1)*maxLength])
        finalSet[2].append(fillSet[2][i*maxLength:(i+1)*maxLength])
        finalSet[3].append(fillSet[3][i*maxLength:(i+1)*maxLength])
    #print "cccccccccccccccc"
    #print finalSet[2]    
    return finalSet

def filler():
    dataSet=loadDataset("../pkl/small_data.pkl",3)
    print dataSet[0][2]
    fill_trainSet=fillerCore(dataSet[0])
    fill_validSet=fillerCore(dataSet[1])
    fill_testSet =fillerCore(dataSet[2])
    return fill_trainSet,fill_validSet,fill_testSet

#dataSet=loadDataset("../pkl/small_data.pkl",3)
#print"aaaaaaaaaaaaaaaaaaaaaaaaaaaa"
#print dataSet[0][2]
#fillerCore(dataSet[0])


def cut(Set,size):
    finalSet=[]
    for i in range (3):
        finalSet.append([])
    for i in range (len(Set[0])):
        for j in range((len(Set[0][i])/size)-1):
            finalSet[0].append(Set[0][i][j*size:(j+1)*size])
            finalSet[1].append(Set[1][i][j*size:(j+1)*size])
            finalSet[2].append(Set[2][i][j*size:(j+1)*size])
        if len(Set[0][i])%size!=0:
            j=(len(Set[0][i])/size)

            finalSet[0].append(Set[0][i][j*size:len(Set[0][i])])
            finalSet[1].append(Set[1][i][j*size:len(Set[0][i])])
            finalSet[2].append(Set[2][i][j*size:len(Set[0][i])])
    return finalSet


dataSet=loadDataset("../pkl/small_data.pkl",3)
b=fillerCore(dataSet[0])
a=cut(b,20)
print a[1]
#print a[1][2] 
