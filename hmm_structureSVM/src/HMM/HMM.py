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

filename="HMM_test_bear2"#output filename
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
    testSetX,  testSetY,  testSetName  = rnnUtils.makeDataSentence(datasets[2])
    '''
    SetX=[]
    SetY=[]
    
    for i in xrange(100):
        SetX.append(trainSetX[i])
        SetY.append(trainSetY[i])
    trainSetX=SetX
    trainSetY=SetY
    VSetX=[]
    VSetY=[]
    for i in xrange(20):
        VSetX.append(validSetX[i])
        VSetY.append(validSetY[i])
    validSetX=VSetX
    validSetY=VSetY
    '''
    '''
    if P.cutSentSize > 0:
        trainSetX, trainSetY, trainSetName = rnnUtils.cutSentence([trainSetX, trainSetY, trainSetName], P.cutSentSize)
        validSetX, validSetY, validSetName = rnnUtils.cutSentence([validSetX, validSetY, validSetName], P.cutSentSize)
    '''
    
    ###############
    # BUILD MODEL #
    ###############
    
    print '... building the model'
    stateNum=len(trainSetX[0][0])#total state/phone
    pi=numpy.zeros(stateNum,dtype=theano.config.floatX)
    for i in xrange(len(trainSetY)):#count how many start
        pi[trainSetY[i][0]]+=1

    for i in xrange(stateNum):#initial prob for 1/totalStateNum
        pi[i]=pi[i]/float(len(trainSetY))
    print pi
    A,countA=tran_prob_matrix_A(trainSetY,stateNum)
    '''
    for i in xrange(stateNum):
       print countA[i,:]
    '''
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
    classifier=HMM_core(pi=pi,A=A,countA=countA)
    errorRate=1
    #training...
    print "... start training"
    trainTimes=0
    #initial error rate
    #validSetX=B_process(validSetX,trainSetY)
    
    accumulateWrong,totalElement=validMeasure(classifier,validSetX,validSetY)
    errorRate=float(accumulateWrong)/float(totalElement)
    print "errorRate:",errorRate
    write_trans_prob(A,"./prob_A/A_0.1:"+str(errorRate))
     
    
    while(trainTimes<5):
        trainTimes+=1
        ViterbiLearning(classifier,trainSetX,trainSetY)
        #validSetX=B_process(validSetX,validSetY)
        
        #EMLearning(classifier,trainSetX)

        accumulateWrong,totalElement=validMeasure(classifier,validSetX,validSetY)
        print "totalWrong:",accumulateWrong
        print "totalNum:",totalElement
        print "train error rate:",float(accumulateWrong)/float(totalElement)
        if (float(accumulateWrong)/float(totalElement))<errorRate:
            errorRate=float(accumulateWrong)/float(totalElement)
            best_A=numpy.array(classifier.A,dtype=theano.config.floatX)
            best_pi=classifier.pi
            
            write_trans_prob(A,"./prob_A/A_with_error:"+str(errorRate))
        print "current_error:",errorRate
    #bestModel to decode testSet
    
    #testSetX=B_process(testSetX,trainSetY)
    bestClass=HMM_core(pi=best_pi,A=best_A,countA=countA)
    outputTest(bestClass,testSetX,testSetName,filename)

def get_result(fileName):
    A=numpy.array(load_trans_prob(fileName),dtype=theano.config.floatX)
    print A
   
def EMLearning(classifier,SetX):
    for i in xrange(len(SetX)):
        B_matrix=numpy.array(SetX[i],dtype=theano.config.floatX)
        B_matrix=B_matrix.T
 
        classifier.EM_learn(B_matrix)


def ViterbiLearning(classifier,SetX,SetY):
    resultSet=[]
    for i in xrange(len(SetY)):
        resultSet.append([])

    for i in xrange(len(SetY)):
        B_matrix=numpy.array(SetX[i],dtype=theano.config.floatX)
        B_matrix=B_matrix.T
        resultSet[i]=classifier.Viterbi(B_matrix)
        #print i
    stateNum=len(SetX[0][0])
    trans_prob_revise(classifier,SetY,resultSet,stateNum)
    for i in xrange(stateNum):
        print classifier.A[i,:]

def trans_prob_revise(classifier,SetY,resultSet,stateNum):
    #calc original tran_prob_matrix
    totalNum=[]
    for i in xrange(stateNum):
        totalNum.append(0)
    for i in xrange(stateNum):#num of sentence
        for j in xrange(stateNum):#num of element transition in each sentence
            totalNum[i]+=classifier.countA[i][j]
    #start to revise
    for i in xrange(len(SetY)):
        for j in xrange(len(SetY[i])-1):
            seqNum=j+1# 1st wrong belong to initial prob
            if (resultSet[i][seqNum]!=SetY[i][seqNum])&(classifier.countA[int(resultSet[i][seqNum-1]),int(resultSet[i][seqNum])]>0):
                classifier.countA[int(resultSet[i][seqNum-1]),int(resultSet[i][seqNum])]-=0.1
                classifier.countA[int(SetY[i][seqNum-1]),int(SetY[i][seqNum-1])]+=-0.1
                totalNum[int(resultSet[i][seqNum-1])]-=0.1
                totalNum[int(SetY[i][seqNum-1])]+=0.1
    for i in xrange(stateNum):# for each element/totalNum of observed outputs
        for j in xrange(stateNum):
            classifier.A[j,i]=float(classifier.countA[j,i])/float(totalNum[j])

def B_process(SetX,SetY):#P(O|S)=P[S|O](from DNN)*P[O](neglect)/P[S](what we do now:for /P[S])
    count_y=[]
    for i in xrange(len(SetX[0][0])):#state Num
        count_y.append(0)
    totalNum=0
    for i in xrange(len(SetY)):
        for j in xrange(len(SetY[i])):
            count_y[int(SetY[i][j])]+=1#count[i]=num of state i
            totalNum+=1
    for i in xrange(len(count_y)):
        count_y[i]=float(count_y[i])/float(totalNum)#count[i]/=totalNum->prob of state i
        print "count[",i,"]=",count_y[i]
    for i in xrange(len(SetX)):
        for j in xrange(len(SetX[i])):
            for k in xrange(len(SetX[i][j])):
                SetX[i][j][k]=float(SetX[i][j][k])/float(count_y[k])
    
    return SetX

def validMeasure(classifier,validSetX,validSetY):#given variable with validSet to measure error
    accumulateWrong=0
    totalElement=0

    for i in xrange(len(validSetX)):
        B_matrix=numpy.array(validSetX[i],dtype=theano.config.floatX)
        B_matrix=B_matrix.T
        #print B_matrix.shape

        y_result=classifier.Viterbi(B_matrix)
        y_ref=validSetY[i]
        
        for j in xrange(len(y_result)):#calculate total wrong estimated element
            #print "(",y_result[j],",",y_ref[j],")"
            if y_result[j]!=y_ref[j]:
                accumulateWrong=accumulateWrong+1
        totalElement=totalElement+len(y_ref)#calculate total element
        '''
        acc=editDistance(y_ref,y_result)
        accumulateWrong+=acc
        totalElement+=len(y_ref)
        '''
        #print "wrongNum:",accumulateWrong
        #print "totalNum:",totalElement
    return accumulateWrong,totalElement
def editDistance(y_ref,y_out):
    r = []	
    h = []
    y_col=y_ref.shape
    print y_col[0]
    for i in xrange (y_col[0]):
        r.append(y_ref[i])
        h.append(y_out[i])

    d = numpy.zeros((len(r)+1)*(len(h)+1), dtype=numpy.uint8)
    d = d.reshape((len(r)+1, len(h)+1))
    for i in range(len(r)+1):
	for j in range(len(h)+1):
            if i == 0:
               	d[0][j] = j
      	    elif j == 0:
                d[i][0] = i

    # computation
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitution = d[i-1][j-1] + 1
                insertion    = d[i][j-1] + 1
                deletion     = d[i-1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)

    #edit distance
    print d[len(r)][len(h)]
    return d[len(r)][len(h)]
    #error rate
    #print float(d[len(r)][len(h)])/float(len(r))
    
def tran_prob_matrix_A(SetY,stateNum):#given validSetY count probability transition matrix A
    A=numpy.zeros((stateNum,stateNum),dtype=theano.config.floatX)
    totalNum=[]
    for i in xrange(stateNum):
        totalNum.append(0)
    for i in xrange(len(SetY)):#num of sentence
        for j in xrange(len(SetY[i])-1):#num of element transition in each sentence
            A[SetY[i][j],SetY[i][j+1]]+=1#count how many times for each transitions
            totalNum[int(SetY[i][j])]+=1
    countA=numpy.zeros((stateNum,stateNum),dtype=theano.config.floatX)
    for i in xrange(stateNum):
        for j in xrange(stateNum):
            countA[i,j]=A[i,j]
       
    for i in xrange(stateNum):# for each element/totalNum of observed outputs
        for j in xrange(stateNum):
            A[j,i]=float(A[j,i])/float(totalNum[j])
    '''
    for i in xrange(stateNum):
            print A[i,:]
            count=0
            for j in xrange(stateNum):
                count+=A[i,j]
            print "count:",count
    '''
    return A,countA



def load_trans_prob(fileName):
    print "... loading conditional probability"
    f = open(fileName, 'rb')
    fileRead=f.read()
    fileSplit=fileRead.split('\n')# cut changing line to get each element
    fileSplit.pop()
    fileProb=[]
    for i in xrange(len(fileSplit)):#initialize fileProb[] to fileProb[][] 
        fileProb.append([])
    for i in xrange(len(fileProb)):
        store=fileSplit[i].split(' ')#split by ' ' to get name and each number of probability
        store.pop()
        for j in xrange(len(store)):#fill in fileProb[given element][j]
            fileProb[i].append(store[j])
    for i in xrange(len(fileSplit)):
        for j in xrange(len(fileProb[i])):
            x=fileProb[i][j].split('e')#j+1 because fileProb[i][0] is filename, ex:file 2.33432e-03
            #change from character to float number
            if len(x)==1:
                fileProb[i][j]=float(x[0])
            else:
                fileProb[i][j]=float(x[0])*10**float(x[1])
    return fileProb#fileProb[i][j]=A[i,j]
'''
def get_conditional_prob(Name,File):#search each sequence by name
    returnArray=[]
    for i in xrange(len(File)):
        if File[i][0]==Name:#file[i][0] is name for each
            for j in xrange(len(File[i])-1):
                returnArray.append(File[i][j+1]) 
            break
    return returnArray  
'''           
def outputTest(classifier,testSetX,testSetName,name):
    
    fout = open(name, 'w')
    for i in xrange(len(testSetX)):
        B_matrix=numpy.array(testSetX[i],dtype=theano.config.floatX)
        B_matrix=B_matrix.T
        #print B_matrix.shape
        y_result=classifier.Viterbi(B_matrix)
    
        for j in xrange(len(testSetX[i])):
            fout.write(testSetName[i][j])
        
            fout.write(",")
            fout.write(str(y_result[j]))
            fout.write("\n")


def write_trans_prob(A,name):
    stateNum,A_col=A.shape
    fout=open(name,'w')
    for i in xrange(stateNum):
       for j in xrange(stateNum):
           fout.write(str(A[i,j]))
           fout.write(" ")
       fout.write("\n")
    
    

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
    
    #dataSet=loadDataset("/home/roylu/share/DNNResult/t23.7v29.95/t23.7v29.95.pkl",3)
    dataCorrectRate(dataSet[0])
    trainHMM(dataSet)
    #get_result("A_with_error:0.281822988462")
