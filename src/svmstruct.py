import svmapi

FEATURE_SIZE = 69
LABEL_NUM = 48
LOSS_FACTOR = 1

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
    datasets = loadDataset(filename = filename, totalSetNum = 3)
    trainSet = datasets[0]
    print len(trainSet[2])
    ret = [] # initializing the return item, an empty list
    for idx in xrange(len(trainSet[2])):
        dummy_tuple = (trainSet[0][idx] , trainSet[1][idx])
        ret.append(dummy_tuple)
    return ret


def init_model(sample, sm, sparm):
    sm.num_features = FEATURE_SIZE
    sm.num_classes = LABEL_NUM
    sm.size_psi = FEATURE_SIZE * LABEL_NUM

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

def classification_score(x, y, sm, sparm):
    return sum([ sm.w[k] * v for k, v in psi(x, y, sm, sparm) ])

def classify_example(x, sm, sparm):
    scores = [(classification_score(x, c, sm, sparm), c) for c in xrange(sm.num_classes)]
    return max(scores)[1]

def find_most_violated_constraint(x, y, sm, sparm):
    scores = [(classification_score(x, c, sm, sparm) + loss(y, c, sparm), c) for c in xrange(sm.num_classes)]
    return max(scores)[1]

def psi(x, y, sm, sparm):
    offset = sm.num_features * y
    pvec = [(k + offset, v) for k, v in x]
    pvec = svmapi.Sparse(pvec)
    return pvec

def loss(y, ybar, sparm):
    return LOSS_FACTOR * int(y != ybar)

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
