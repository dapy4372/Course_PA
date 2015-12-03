import svmapi
import numpy
import utils

FEATURE_DIM = 2
LABEL_NUM = 2

def parse_parameters(sparm):
    sparm.arbitrary_parameter = 'I am an arbitrary parameter!'

def parse_parameters_classify(attribute, value):
    print 'Got a custom command line argument %s %s' % (attribute, value)

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

def read_examples(filename, sparm):
#datasets = loadDataset(filename = filename, totalSetNum = 3)
#trainX, trainY, trainN = utils.makeDataSentence(datasets[0])
    trainX = [ [[0.1, 0.1], [0.1, 0.1], [0.1, 0.1], [0.9, 0.9], [0.9, 0.9], [0.9, 0.9]], 
                    [[0.1, 0.1], [0.1, 0.1], [0.1, 0.1], [0.9, 0.9], [0.9, 0.9], [0.9, 0.9]] ]
    trainY = [ [0, 0, 0 , 1, 1, 1], [0, 0, 0, 1, 1, 1]]
#trainY = trainY[0:100]
#trainX = trainX[0:100]
    ret = []
    sentNum = len(trainY)
    for sentIdx in xrange(sentNum):
#       ret.append((trainX[sentIdx], trainY[sentIdx].astype(dtype='int32')))
        ret.append((trainX[sentIdx], trainY[sentIdx]))
    return ret

def init_model(sample, sm, sparm):
    sm.num_features = FEATURE_DIM
    sm.num_classes = LABEL_NUM
    sm.size_psi = FEATURE_DIM * LABEL_NUM + LABEL_NUM * LABEL_NUM

def init_constraints(sample, sm, sparm):
    if True:
        c, d = svmapi.Sparse, svmapi.Document
        return [(d([c([(1,1)])],slackid=len(sample)+1),   1), (d([c([0,0,0,1])],slackid=len(sample)+1),.2)]
    constraints = []
    for i in xrange(sm.size_psi):
        sparse = svmapi.Sparse([(i,1)])
        lhs = svmapi.Document([sparse], costfactor=1, slackid=i+1+len(sample))
        constraints.append((lhs, 0))
    return constraints

def find_most_violated_constraint_slack(x, y, sm, sparm):
    return find_most_violated_constraint(x, y, sm, sparm)

def find_most_violated_constraint_margin(x, y, sm, sparm):
    return find_most_violated_constraint(x, y, sm, sparm)


def classify_example(x, sm, sparm):
    yLen = len(x)
    score = [ [0] * LABEL_NUM ] * yLen
    backTrack = [ [0] * LABEL_NUM ] * yLen
    w_o_offset = LABEL_NUM * FEATURE_DIM
    for frameIdx in xrange(yLen):
        for labelIdx in xrange(LABEL_NUM):
            if(frameIdx == 0):
                score[frameIdx][labelIdx] = dot(sm.w[labelIdx * FEATURE_DIM:(labelIdx + 1) * FEATURE_DIM], x[frameIdx])
                backTrack[frameIdx][labelIdx] = labelIdx
            else:
                bestLabel = 0
                bestScore = 0
                for i in xrange(LABEL_NUM):
                    if( (sm.w[w_o_offset + i * LABEL_NUM + labelIdx] + score[frameIdx-1][i]) > bestScore):
                        bestScore = ( sm.w[w_o_offset + i * LABEL_NUM + labelIdx] + score[frameIdx-1][i] )
                        bestLabel = i
                score[frameIdx][labelIdx] = dot(sm.w[labelIdx * FEATURE_DIM:(labelIdx + 1) * FEATURE_DIM], x[frameIdx]) + bestScore
                backTrack[frameIdx][labelIdx] = bestLabel
    score_max = score[yLen - 1][0]
    label_max = 0
    for i in xrange(LABEL_NUM):
        if(score[yLen - 1][i] > score_max):
            score_max = score[yLen - 1][i]
            label_max = i
    ret = [label_max]
    prev = backTrack[yLen-1][label_max]
    for i in xrange(yLen-2, -1, -1):
        ret.insert(0, prev)
        prev = backTrack[i][prev]
    return ret

def dot(v1, v2):
    v1Len = len(v1)
    v2Len = len(v2)
    assert(v1Len == v2Len)
    ret = 0
    for i in xrange(v1Len):
        ret += v1[i] * v2[i]
    return ret

def find_most_violated_constraint(x, y, sm, sparm):
    yLen = len(y)
#loss_fraction = 1/yLen
    score = [ [0] * LABEL_NUM ] * yLen
    backTrack = [ [0] * LABEL_NUM ] * yLen
    w_o_offset = LABEL_NUM * FEATURE_DIM
    for frameIdx in xrange(yLen):
        for labelIdx in xrange(LABEL_NUM):
            if(frameIdx == 0):
                score[frameIdx][labelIdx] = dot(sm.w[labelIdx * FEATURE_DIM:(labelIdx + 1) * FEATURE_DIM], x[frameIdx])
                if(labelIdx != y[frameIdx]):
#score[frameIdx][labelIdx] += loss_fraction
                    score[frameIdx][labelIdx] += 1
                  
                backTrack[frameIdx][labelIdx] = labelIdx
            else:
                bestLabel = 0
                bestScore = 0
                for i in xrange(LABEL_NUM):
                    if( (sm.w[w_o_offset + i * LABEL_NUM + labelIdx] + score[frameIdx-1][i]) > bestScore):
                        bestScore = ( sm.w[w_o_offset + i * LABEL_NUM + labelIdx] + score[frameIdx-1][i] )
                        bestLabel = i
                score[frameIdx][labelIdx] = dot(sm.w[labelIdx * FEATURE_DIM:(labelIdx + 1) * FEATURE_DIM], x[frameIdx]) + bestScore
                if(labelIdx != y[frameIdx]):
#score[frameIdx][labelIdx] += loss_fraction
                    score[frameIdx][labelIdx] += 1
                backTrack[frameIdx][labelIdx] = bestLabel
    score_max = score[yLen - 1][0]
    label_max = 0
    for i in xrange(LABEL_NUM):
        if(score[yLen - 1][i] > score_max):
            score_max = score[yLen-1][i]
            label_max = i
    ret = [label_max]
    prev = backTrack[yLen-1][label_max]
    for i in xrange(yLen-2, -1, -1):
        ret.insert(0, prev)
        prev = backTrack[i][prev]
    return ret

def psi(x, y, sm, sparm):
    sentLen = len(x)
    observationLen = FEATURE_DIM * LABEL_NUM
    pvec = [0] * ( observationLen + LABEL_NUM * LABEL_NUM )
    prevY = -1
    for idx in xrange(sentLen):
        offset = FEATURE_DIM * y[idx]
        # observation vector
        for i in xrange(FEATURE_DIM):
            pvec[i + offset] += x[idx][i]
        # transition vector
        if(idx != 0):
            pvec[observationLen + prevY * LABEL_NUM + y[idx]] += 1  
        prevY = y[idx]
    
    pvec = svmapi.Sparse(pvec)
    return pvec

def loss(y, ybar, sparm):
#loss_fraction = len(y)
    ret = 0
    for i in xrange(len(y)):
        ret +=int(y[i] != ybar[i])
    return ret

def print_iteration_stats(ceps, cached_constraint, sample, sm, cset, alpha, sparm):
    print

def print_learning_stats(sample, sm, cset, alpha, sparm):
    print 'Model learned:',
    print '[',', '.join(['%g'%i for i in sm.w]),']'
    print 'Losses:',
    print [loss(y, classify_example(x, sm, sparm), sparm) for x,y in sample]

def print_testing_stats(sample, sm, sparm, teststats):
    print teststats

def eval_prediction(exnum, (x, y), ypred, sm, sparm, teststats):
    if exnum==0: teststats = []
    print 'on example',exnum,'predicted',ypred,'where correct is',y
    teststats.append(loss(y, ypred, sparm))
    return teststats

def write_model(filename, sm, sparm):
    import cPickle, bz2
    f = bz2.BZ2File(filename, 'w')
    cPickle.dump(sm, f)
    f.close()

def read_model(filename, sparm):
    import cPickle, bz2
    return cPickle.load(bz2.BZ2File(filename))

def write_label(fileptr, y):
    print>>fileptr,y

def print_help():
    print svmapi.default_help
    print "This is a help string for the learner!"

def print_help_classify():
    print "This is a help string for the classifer!"
