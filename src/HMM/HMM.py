import os
import sys
import timeit
import random
import numpy 
import theano
import theano.tensor as T
import rnnUtils
import globalParam
from rnnUtils import Parameters
from HMM_core import HMM_core
from utils import loadDataset
import math

DEBUG = False
FER_PER_SENT = True
PAUSE = False
OUTPUT_DETAIL = False
THRESHOLD = 0.27

#parameterFilename = sys.argv[1]
#np.set_printoptions(threshold=np.nan) # for print np array

def trainHMM(datasets):

    trainSetX, trainSetY, trainSetName = rnnUtils.makeDataSentence(datasets[0])
    validSetX, validSetY, validSetName = rnnUtils.makeDataSentence(datasets[1])
    '''
    if P.cutSentSize > 0:
        trainSetX, trainSetY, trainSetName = rnnUtils.cutSentence([trainSetX, trainSetY, trainSetName], P.cutSentSize)
        validSetX, validSetY, validSetName = rnnUtils.cutSentence([validSetX, validSetY, validSetName], P.cutSentSize)

    '''
    ###############
    # BUILD MODEL #
    ###############
    
    print '... building the model'
    idx = T.iscalar('i')
    x = T.matrix()
    y = T.ivector()
    stateNum=len(trainSetX[0][0])#total state/phone
    pi=numpy.zeros(stateNum,dtype=theano.config.floatX)
    for i in xrange(len(trainSetY)):#count how many start
        pi[trainSetY[i][0]]+=1
    for i in xrange(stateNum):#initial prob for 1/totalStateNum
        pi[i]=pi[i]/float(len(trainSetY))
    print pi
    A=numpy.zeros((stateNum,stateNum),dtype=theano.config.floatX)
    totalNum=[]
    for i in xrange(stateNum):
        totalNum.append(0)
    for i in xrange(len(trainSetY)):#num of sentence
        for j in xrange(len(trainSetY[i])-1):#num of element transition in each sentence
            A[trainSetY[i][j],trainSetY[i][j+1]]+=1#count how many times for each transitions
            totalNum[int(trainSetY[i][j])]+=1 
    for i in xrange(stateNum):# for each element/totalNum of observed outputs
        for j in xrange(stateNum):
            A[j,i]=float(A[j,i])/float(totalNum[j]) 
    for i in xrange(stateNum):
        print A[i,:]
    #best parameters
    best_A=A
    best_pi=pi
    #end best parameter
    '''
    B_list=[]#use DNN to calculate conditional probability of each output
    for i in xrange(len(trainSetX)):#total num of sentences
        B_list.append([])
    for i in xrange(len(trainSetX)):
        for j in xrange(len(trainSetX[i])):  
            B_list[i].append(trainSetX[i][j]) #get_conditional_probability:use DNN
    '''
    #B=numpy.array(trainSetX[0],dtype=theano.config.floatX)
    #print B
    
 
    # For create a new model
    classifier=HMM_core(input=x,pi=pi,A=A)
    '''
    myOutputs=classifier.P_O_Q
    dA=T.grad(myOutputs,classifier.A)
    dpi=T.grad(myOutputs,classifier.pi)
                 
    trainModel=theano.function(inputs=[x],outputs=myOutputs,updates=myUpDates([classifier.A,classifier.pi],[dA,dpi])) 
    times=0#training times
    '''
    errorRate=1
    print "... start training"
    #validSet test
    
    totalElement=0
    accumulateWrong=0
    for i in xrange(len(validSetX)):
        B_matrix=numpy.array(validSetX[i],dtype=theano.config.floatX)
        B_matrix=B_matrix.T
        print B_matrix.shape
        y_result=classifier.Viterbi(B_matrix,stateNum)
        y_ref=validSetY[i]
        for j in xrange(len(y_result)):#calculate total wrong estimated element
            #print "(",y_result[j],",",y_ref[j],")"
            if y_result[j]!=y_ref[j]:
                accumulateWrong=accumulateWrong+1
        totalElement=totalElement+len(y_ref)#calculate total element    
        print "wrongNum:",accumulateWrong  
        print "totalNum:",totalElement  
    print "current_error:",#if new errorRate<current errorRate: renew parameter

    if (float(accumulateWrong)/float(totalElement))<errorRate:
        errorRate=float(accumulateWrong)/float(totalElement)
        best_A=classifier.A
        best_pi=classifier.pi
        print errorRate
    #for i in xrange(len(trainSetY)):
    '''
    print dataSet[0][0][40],dataSet[0][1][40]
    while (times<100) or (not((A==best_A) and (pi==best_pi))):#training
        for i in xrange(len(trainSetY)):
            B=numpy.array(B_list[i],dtype=theano.config.floatX)
            print B
            outputs=trainModel(B)
            print ("outputs=%d"%(outputs))
        #validSet test
        time=time+1
        totalElement=0
        accumulateWrong=0
        for i in xrange(len(ValidSetY)):
            B=[]
            for j in xrange(len(ValidSetX[i])):
                B.append(ValidSetX[i][j]) #get_conditional_probability:use DNN
            B_matrix=numpy.array(B,dtype=theano.config.floatX)
            y_result=classifier.Viterbi(B_matrix) 
            y_ref=ValidSetY[i]
            for j in xrange(len(y_ref)):#calculate total wrong estimated element
                if y_result[i]!=y_ref[i]:
                    accumulateWrong=accumulateWrong+1
            totalElement=totalElement+len(y_ref)#calculate total element
        print "current_error:%d",#if new errorRate<current errorRate: renew parameter

        if (float(accumulateWrong)/float(totalElement))<errorRate:
            errorRate=float(accumulateWrong)/float(totalElement) 
            best_A=classifier.A
            best_pi=classifier.pi   
    '''
def myUpDates(parameters,grads):
    parameters_updates=[(p,update_function(p,g))for (p,g) in izip(parameters,grads)]
    return parameters_updates
def update_function(p,g):#algorithm:  http://www.robots.ox.ac.uk:5000/~vgg/rg/papers/hmm.pdf  page10
    row,col=p.shape
    updates=numpy.zeros((row,col),dtype=theano.config.floatX)
    for i in xrange(col):#total setNum
        denominator=0
        for j in xrange(row):#total state
            updates[j,i]=p*g
            denominator=denominator+p*g
        for j in xrange(row):
            updates[j,i]=float(update[j,i])/float(denominator)
    return updates

'''
def loadProb(fileName):
    print "... loading conditional probability"
    f = open(fileName, 'rb')
    fileRead=f.read()
    fileSplit=fileRead.split('\n')# cut changing line to get each element
    fileProb=[]
    for i in xrange(len(fileSplit)):#initialize fileProb[] to fileProb[][] 
        fileProb.append([])
    for i in xrange(len(fileSplit)):
        store=fileSplit[i].split(' ')#split by ' ' to get name and each number of probability
        for j in xrange(len(store)):#fill in fileProb[given element][j]
            fileProb[i].append(store[j])
    for i in xrange(len(fileSplit)):
        for j in xrange(len(fileProb[i])-1):
            x=fileProb[i][j+1].split('e')#j+1 because fileProb[i][0] is filename, ex:file 2.33432e-03
            #change from character to float number
            if len(x)==1:
                fileProb[i][j+1]=float(x[0])
            else:
                fileProb[i][j+1]=float(x[0])*10**float(x[1])
    return fileProb
def get_conditional_prob(Name,File):#search each sequence by name
    returnArray=[]
    for i in xrange(len(File)):
        if File[i][0]==Name:#file[i][0] is name for each
            for j in xrange(len(File[i])-1):
                returnArray.append(File[i][j+1]) 
            break
    return returnArray  
'''           

def dataCorrectRate(Set):
    correct=0
    totalNum=0
    for i in xrange(len(Set[0])):
        maxNum=0
        maxState=-1
        for j in xrange(len(Set[0][i])):    
            if(Set[0][i][j]>maxNum):
                maxNum=Set[0][i][j]
                maxState=j
        if(Set[1][i]==maxState):
            correct=correct+1
        totalNum+=1
    print "correct rate:",float(correct)/float(totalNum)
    
if __name__ == '__main__':

    dataSet=loadDataset("../../pkl/t26v31.pkl",3)
    #dataCorrectRate(dataSet[0])
    trainHMM(dataSet)
