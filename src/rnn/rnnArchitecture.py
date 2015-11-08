import numpy
import theano
import theano.tensor as T
def sigmoid(z):
    return 1/(1+T.exp(-z))
def softmax(z):
    # Softmax
    absZ = T.abs_(z)
    maxZ = T.max(absZ, axis=1)
    maxZ = T.reshape(maxZ, (maxZ.shape[0], 1))
    expZ = T.exp(z * 10 / maxZ)
    expZsum = T.sum(expZ, axis=1)
    expZsum = T.reshape(expZsum, (expZsum.shape[0], 1))
    return expZ / expZsum

class HiddenLayer(object):
    def __init__(self, rng, input, inputNum, outputNum, W_i = None, b_i = None, W_h = None, b_h = None):
        if W_i is None:
            W_i_values = rng.uniform( low = -numpy.sqrt(6./(inputNum+outputNum)), high = numpy.sqrt(6./(inputNum+outputNum)),
            size = (inputNum, outputNum) ).astype( dtype=theano.config.floatX )
            W_i = theano.shared(value = W_i_values, name = 'W', borrow = True)
        else:
            W_i = theano.shared( value = numpy.array(W, dtype = theano.config.floatX), name='W', borrow = True )

        if W_h is None:
            W_h_values = rng.uniform( low = -numpy.sqrt(6./(inputNum+outputNum)), high = numpy.sqrt(6./(inputNum+outputNum)),
            size = (outputNum, inputNum) ).astype( dtype=theano.config.floatX )
            W_h = theano.shared(value = W_h_values, name = 'W', borrow = True)
        else:
            W_h = theano.shared( value = numpy.array(W, dtype = theano.config.floatX), name='W', borrow = True )

        if b_i is None:
            b_values = rng.uniform( low = -1, high = 1, size = (outputNum,)).astype(dtype=theano.config.floatX)
            b_i = theano.shared(value = b_values, name = 'b', borrow = True)
        else:
            b_i = theano.shared( value = numpy.array(b, dtype = theano.config.floatX), name='b', borrow = True )

        if b_h is None:
            b_values = rng.uniform( low = -1, high = 1, size = (inputNum,)).astype(dtype=theano.config.floatX)
            b_h = theano.shared(value = b_values, name = 'b', borrow = True)
        else:
            b_h = theano.shared( value = numpy.array(b, dtype = theano.config.floatX), name='b', borrow = True )

        self.W_i = W_i
        self.b_i = b_i
        self.W_h = W_h
        self.b_h = b_h

        def step(x_t, z_tm1):
            return sigmoid( T.dot(x_t, self.W_i) + self.b_i + T.dot(z_tm1, self.W_h) + self.b_h )

        a_0 = theano.shared(numpy.zeros(outputNum).astype(dtype = theano.config.floatX), borrow = True)
        
        #self.output = []
        #self.output.append( scan(step, sequences = input, outputs_info = a_0, truncate_gradient = -1)[0] )

        a_seq, _ = theano.scan(step, sequences = input, outputs_info = a_0, truncate_gradient = -1)
        self.output = a_seq

        self.params = [self.W_i, self.b_i, self.W_h, self.b_h]
        
class OutputLayer(object):
    def __init__(self, input, inputNum, outputNum, rng, W = None, b = None):
        if W is None:
            W_values = rng.uniform( low = -numpy.sqrt(6./(inputNum+outputNum)), high = numpy.sqrt(6./(inputNum+outputNum)),
                                    size = (inputNum, outputNum) ).astype(dtype=theano.config.floatX )
            W_o = theano.shared(value = W_values, name = 'W', borrow = True)
        else:
            W_o = theano.shared( value = numpy.array(W, dtype = theano.config.floatX), name='W', borrow=True )

        if b is None:
            b_values = rng.uniform( low = -1, high = 1, size = (outputNum,)).astype(dtype=theano.config.floatX)
            b_o = theano.shared(value = b_values, name = 'b', borrow = True)
        else:
            b_o = theano.shared( value = numpy.array(b, dtype = theano.config.floatX), name='b', borrow=True )

        self.W_o = W_o
        self.b_o = b_o
        
        def step(x_t, y_tm1):
            return softmax( T.dot(x_t, self.W_o) )

        y_0 = theano.shared(numpy.zeros(outputNum).astype(dtype=theano.config.floatX), borrow=True)
        
        #self.output = []
        #self.output.append( theano.scan(step, sequences = input, outputs_info = y_0, truncate_gradient = -1)[0] )
        y_seq, _ = theano.scan(step, sequences = input, outputs_info = y_0, truncate_gradient = -1)[0]
      
        self.p_y_given_x = y_seq
        
        # Find larget y_i
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        self.params = [self.W, self.b]
    
    # Cross entropy
    def crossEntropy(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        # Check y and y_pred dimension
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # Check if y is the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

class RNN(object):
    def __init__(self, input, P, params = None, DROPOUT = False):
        
        self.hiddenLayerList = []
        self.hiddenLayerList.append(
            HiddenLayer( input = input, rng = P.rng, inputNum = P.inputDimNum, outputNum = P.rnnWidth, 
                         W_i = params[0], b_i = params[1], W_h = params[2], b_h = params[3] ))
        # Other hidden layers, 
        # Currently it's a Jordan type RNN, thus no other hidden layers are required
        '''
        for i in xrange (P.rnnDepth - 1):
            self.hiddenLayerList.append(
                HiddenLayer(
                    input = self.hiddenLayerList[i].output,
                    rng = P.rng,
                    inputNum = P.rnnW_idth/2,
                    outputNum = P.rnnW_idth,
                    rnnW_idth = P.rnnW_idth,
                    dropoutProb = P.dropoutHiddenProb,
                    W = params[2 * (i + 1)],
                    b = params[2 * (i + 1) + 1],
                    DROPOUT = DROPOUT ) )
        '''
        # Output Layer
        self.outputLayer = OutputLayer( input = self.hiddenLayerList[P.rnnDepth - 1].output, rng = P.rng, inputNum = P.rnnWidth, 
                                        outputNum = P.outputPhoneNum, W = params[4 * P.rnnDepth], b = params[4 * P.rnnDepth+1] )
        
        # Weight decay
        # L1 norm ; one regularization option is to enforce L1 norm to be small
        self.L1 = 0
        for i in xrange(P.rnnDepth):
             self.L1 += abs(self.hiddenLayerList[i].W).sum()
        self.L1 += abs(self.outputLayer.W).sum()

        # square of L2 norm ; one regularization option is to enforce square of L2 norm to be small
        self.L2_sqr = 0
        for i in xrange(P.rnnDepth):
            self.L2_sqr += (self.hiddenLayerList[i].W ** 2).sum()
        self.L2_sqr += (self.outputLayer.W ** 2).sum()

        # CrossEntropy
        self.crossEntropy = ( self.outputLayer.crossEntropy )
        
        # Same holds for the function computing the number of errors
        self.errors = self.outputLayer.errors

        # Get the predict int for test set output
        self.yPred = self.outputLayer.y_pred
        
        # Get the probability
        self.p_y_given_x = self.outputLayer.p_y_given_x

        # Parameters of all DNN model
        self.params = self.hiddenLayerList[0].params
        for i in xrange(1, P.rnnDepth):
            self.params += self.hiddenLayerList[i].params
        self.params += self.outputLayer.params
