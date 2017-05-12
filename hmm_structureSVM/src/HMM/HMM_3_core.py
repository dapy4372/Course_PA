import os,sys
import numpy
import theano
import math
from utils import loadDataset,namepick

'''
def Viterbi(pi,A,B):#pi for initial distribution;A for transition matrix A(i,j):state i to state j; B for conditional probability:B[i,j]=P(jth Output|state i);O output observe sequence
    A_row,A_col=A.shape
    B_row,B_col=B.shape
    path=[]
    value=[]
    for i in xrange(A_row):
        path.append([])
    #initial state 
    for i in xrange(B_row):
        v[i]=pi[i]*B[i,0]
        path[i].append[i]
    #main part
    for i in xrange(B_col-1):
    #i th output
        seqNum=i-1
        for j in xrange(B_row):
            #to j th state
            array=[]
            for k in xrange(B_row):# k th last state
                array.append(value[k]*A(k,j))
            value[j],maxState=getMax(array)
            path[j].append(maxState)
    #find max path over, then trace back
    maxValue,maxState=getMax(value)
    y_sequence=[B_col]            
    for i in xrange(B_col):
        y_sequence[B_col-i]=path[maxState][B_col-i]
        maxState=path[maxState][B_col-i]
    return y_sequence    


def getMax(array,maxState):
    maximum=0
    for i in xrange(len(array)):
        if maximum<array[i]
            maximum=array[i]
    return maximum,maxState
'''

class HMM_core(object):
    def __init__(self,input,pi,A):
        #self.pi=theano.shared(value=numpy.array(pi,dtype=theano.config.floatX),name='Pi',borrow=True)#initial distribution
        #self.A=theano.shared(value=numpy.array(A,dtype=theano.config.floatX),name='A',borrow=True)#transition matrix
        self.A=A
        self.pi=pi
        self.P_O_Q=self.calc_P(input)
    def calc_P(self,B):
        #B_row,B_col=B.shape
        B_row=48
        B_col=10

        apha_current=[]
        apha_next=[]
        for i in xrange(B_row):
            apha_current.append([])
            apha_next.append([])
        for i in xrange(B_row):#initial
            apha_current[i]=self.pi[i]*B[i,1]
        for i in xrange(B_col-1):#for all observed output
            for j in xrange(B_row):#for alpha[j state]
                store_value=0
                for k in xrange(B_row):#sigma apha[i]*aij  
                    store_value=store_value+apha_current[k]*self.A[k,j]
                apha_next[j]=store_value*B[j,i+1]
            apha_current=apha_next
        P=0
        for i in xrange(B_row):#last sum up
           P=P+apha_current[i] 
        return P
                
        
    def Viterbi(self,B,stateNum):#pi for initial distribution;A for transition matrix A(i,j):state i to state j; B for conditional probability:B[i,j]=P(jth Output|state i);O output observe sequence
        A_row=stateNum
        B_row,B_col=B.shape
        #B_col=10
        path=[]
        value=[]#current&remember value
        value_next=[]#store new value
        for i in xrange(A_row):
            value_next.append(0)
            value.append(0)
            path.append([])
        #initial state
        for i in xrange(B_row):
            value[i]=self.getLog(self.pi[i])+self.getLog(B[i,0])
        #main part
        for i in xrange(B_col-1):
        #i th output
            seqNum=i+1
            for j in xrange(B_row):
                #to j th state
                array=[]
                for k in xrange(B_row):# k th last state
                    array.append(value[k]+self.getLog(self.A[k,j]*B[j,seqNum]))
                value_next[j],maxState=self.getMax(array)
                path[j].append(maxState)
            for j in xrange(B_row):# update value to next state
                value[j]=value_next[j]
            #print "max:",self.getMax(value)
        #find max path over, then trace back
        maxValue,maxState=self.getMax(value)
        print "out"
        #print path
        y_sequence=[]
        for i in xrange(B_col):#make y_sequence are [B_col][]
            y_sequence.append(0)
        for i in xrange(B_col):
            y_sequence[B_col-i-1]=maxState/3#because state*3,state*3+1,state*3+2-->state
            maxState=path[maxState][B_col-i-2]
        return y_sequence
    def getMax(self,array):
        maximum=-10000000
        maxState=0
        for i in xrange(len(array)):
            if array[i]>maximum:
                maximum=array[i]
                maxState=i
        return maximum,maxState
    def getLog(self,Num):
        if Num==0:
            return -100000
        else:
            return math.log(Num,10)
