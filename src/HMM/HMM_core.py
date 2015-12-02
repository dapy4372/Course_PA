import os,sys
import numpy
import theano
import math
from utils import loadDataset,namepick


class HMM_core(object):
    def __init__(self,pi,A,countA):
        #self.pi=theano.shared(value=numpy.array(pi,dtype=theano.config.floatX),name='Pi',borrow=True)#initial distribution
        #self.A=theano.shared(value=numpy.array(A,dtype=theano.config.floatX),name='A',borrow=True)#transition matrix
        self.A=A
        self.pi=pi
        self.countA=countA#A with counting number with every element(for Viterbi learning)
    def calc_Log_apha(self,B):#apha[state][time]
       apha=[]
       B_row,B_col=B.shape
       for i in xrange(B_row):
           apha.append([]) 

       for i in xrange(B_col):
           for j in xrange(B_row):
               if i==0:
                   apha[j].append(self.getLog(self.pi[j])+self.getLog(B[j,0]))
               else:
                   p=-1*10**20
                   for k in xrange(B_row):#apha[j][i]=sigma(k)[apha[k][i-1]*akj*B[j,i]
                       p=self.addLog(p,(apha[k][i-1]+self.getLog(self.A[k,j]))) 
                   apha[j].append(p+self.getLog(B[j,i]))
       return apha
                
    def calc_Log_beta(self,B):#beta[state][time]
        #initialize storage
        beta=[]
        B_row,B_col=B.shape
        for i in xrange(B_row):
            beta.append([])
        for i in xrange(B_row):
            for j in xrange(B_col):
                beta[i].append(0)
        #calculate
        for i in xrange(B_col):#i:ith
            seqNum=B_col-1-i
            for j in xrange(B_row):#j:state
                if (seqNum ==B_col-1):
                    beta[j][seqNum] = 0;
                else:#beta[j][i]=sigma(k):ajk*B[k,i+1]*beta[k][i+1]
                    p = -1*10**20
                    for k in xrange(B_row):
                        p= self.addLog(p,(beta[k][seqNum+1]+self.getLog(self.A[j,k])+self.getLog(B[k,seqNum+1])))
                    beta[j][seqNum]=p
        return beta
    def EM_learn(self,B):
        B_row,B_col=B.shape
        apha=self.calc_Log_apha(B)
        beta=self.calc_Log_beta(B)
        #print "apha:",apha
        #print "beta:",beta
        gamma=[]#pass through state i
        eta=[]#pass from i to j,eta[i][j][time],from state i to state j
        for i in xrange(B_row):
            gamma.append([])
            eta.append([])
            for j in xrange(B_row):
                eta[i].append([])
        #end initialize
        #calc gamma
        pg=-1*10**20
        for i in xrange(B_col):
            for j in xrange(B_row):
                pg=self.addLog(pg,apha[j][i]+beta[j][i])
            for j in xrange(B_row):
                gamma[j].append(apha[j][i]+beta[j][i]-pg)
        #calc eta
        pe=-1*10**20
        for i in xrange(B_col-1):
            for j in xrange(B_row):
                for k in xrange(B_row):
                    pe=self.addLog(pe,apha[j][i]+self.getLog(self.A[j,k]*B[k,i+1])+beta[k][i+1])
            for j in xrange(B_row):
                for k in xrange(B_row):
                    eta[j][k].append(apha[j][i]+self.getLog(self.A[j,k]*B[k,i+1])+beta[k][i+1]-pe)
            
        for i in xrange(B_row):
            for j in xrange(B_row):
                storeGamma=-1*10**20
                storeEta=-1*10**20
                for k in xrange(B_col-1):
                    storeGamma=self.addLog(storeGamma,gamma[i][k])
                    storeEta=self.addLog(storeEta,eta[i][j][k])
                self.A[i,j]=10**(storeEta-storeGamma)
        '''
        for i in xrange(B_row):
            print self.A[i,:]
            count=0
            for j in xrange(B_row):
                count+=self.A[i,j]
            print "count:",count
        '''
    def addLog(self,x,y):#from logX,logY to log(X+Y)
        
        if x<y:
            temp=y
            y=x
            x=temp
        diff=y-x
        if diff<(-9):
            return x
        else:
            z=10**diff
            return x+self.getLog(1+z)

    
    def Viterbi(self,B):#pi for initial distribution;A for transition matrix A(i,j):state i to state j; B for conditional probability:B[i,j]=P(jth Output|state i);O output observe sequence
        
        B_row,B_col=B.shape
        A_row=B_row
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
        #print "out"
        #print path
        y_sequence=[]
        for i in xrange(B_col):#make y_sequence are [B_col][]
            y_sequence.append(0)
        for i in xrange(B_col):
            y_sequence[B_col-i-1]=maxState
            maxState=path[maxState][B_col-i-2]
        return y_sequence
    def getMax(self,array):
        maximum=-1*(10**20)
        maxState=0
        for i in xrange(len(array)):
            if array[i]>maximum:
                maximum=array[i]
                maxState=i
        return maximum,maxState
    def getLog(self,Num):
        if Num==0:
            return -1*(10**19)
        else:
            return math.log(Num,10)
