import numpy
import theano
import theano.tensor as T
def sigmoid(z):
    return 1/(1+T.exp(-z))

def softmax(z):
    absZ = T.abs_(z)
    maxZ = T.max(absZ)
    expZ = T.exp(z * 10 / maxZ)
    expZsum = T.sum(expZ)
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
            size = (outputNum, outputNum) ).astype( dtype=theano.config.floatX )
            W_h = theano.shared(value = W_h_values, name = 'W', borrow = True)
        else:
            W_h = theano.shared( value = numpy.array(W, dtype = theano.config.floatX), name='W', borrow = True )

        if b_i is None:
            b_values = rng.uniform( low = -1, high = 1, size = (outputNum,)).astype(dtype=theano.config.floatX)
            b_i = theano.shared(value = b_values, name = 'b', borrow = True)
        else:
            b_i = theano.shared( value = numpy.array(b, dtype = theano.config.floatX), name='b', borrow = True )

        if b_h is None:
            b_values = rng.uniform( low = -1, high = 1, size = (outputNum,)).astype(dtype=theano.config.floatX)
            b_h = theano.shared(value = b_values, name = 'b', borrow = True)
        else:
            b_h = theano.shared( value = numpy.array(b, dtype = theano.config.floatX), name='b', borrow = True )

        self.W_i = W_i
        self.b_i = b_i
        self.W_h = W_h
        self.b_h = b_h
        self.output=[]

        def step(x_t, z_tm1):
            return sigmoid( T.dot(x_t, self.W_i) + self.b_i + T.dot(z_tm1, self.W_h) + self.b_h )
        
        a_0 = theano.shared(numpy.zeros(outputNum).astype(dtype = theano.config.floatX), borrow = True)
        
        a_seq, _ = theano.scan(step, sequences = input[0], outputs_info = a_0, truncate_gradient = -1)
        self.output.append(a_seq)

        reverse_a_seq, _ = theano.scan(step, sequences = input[1], outputs_info = a_0, truncate_gradient = -1)
        self.output.append(reverse_a_seq)

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
         
        avergeInput = (input[0] + input[1]) / 2

        def step(x_t, y_tm1):
            return softmax( T.dot(x_t, self.W_o) + b_o )

        y_0 = theano.shared(numpy.zeros(outputNum).astype(dtype=theano.config.floatX), borrow=True)
        y_seq, _ = theano.scan(step, sequences = avergeInput, outputs_info = y_0, truncate_gradient = -1)
      
        self.p_y_given_x = y_seq
        
        # Find larget y_i
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        self.params = [self.W_o, self.b_o]
    
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
        bidirectionalInput = []
        bidirectionalInput.append(input)
        bidirectionalInput.append(input[::-1])
        
        # First hidden layer
        self.hiddenLayerList.append(
            HiddenLayer( input = bidirectionalInput, rng = P.rng, inputNum = P.inputDimNum, outputNum = P.rnnWidth, 
                         W_i = params[0], b_i = params[1], W_h = params[2], b_h = params[3] ))
        
        # Other hidden layers 
        for i in xrange (P.rnnDepth - 1):
            self.hiddenLayerList.append(
                HiddenLayer( input = self.hiddenLayerList[i].output, rng = P.rng, inputNum = P.rnnWidth, outputNum = P.rnnWidth,
                             W_i = params[2 * (i + 1)], b_i = params[2 * (i + 1) + 1], W_h = params[2 * (i + 1) + 2], b_h = params[2 * (i + 1) + 3] ))
        # Output Layer
        self.outputLayer = OutputLayer( input = self.hiddenLayerList[P.rnnDepth - 1].output, rng = P.rng, inputNum = P.rnnWidth, 
                                        outputNum = P.outputPhoneNum, W = params[4 * P.rnnDepth], b = params[4 * P.rnnDepth+1] )
        
        # Weight decay
        # L1 norm ; one regularization option is to enforce L1 norm to be small
        self.L1 = 0
        for i in xrange(P.rnnDepth):
             self.L1 = self.L1 + abs(self.hiddenLayerList[i].W_i).sum() + abs(self.hiddenLayerList[i].W_h).sum()
        self.L1 += abs(self.outputLayer.W_o).sum()

        # square of L2 norm ; one regularization option is to enforce square of L2 norm to be small
        self.L2_sqr = 0
        for i in xrange(P.rnnDepth):
            self.L2_sqr = self.L2_sqr + (self.hiddenLayerList[i].W_i ** 2).sum() + (self.hiddenLayerList[i].W_h ** 2).sum()
        self.L2_sqr += (self.outputLayer.W_o ** 2).sum()

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
