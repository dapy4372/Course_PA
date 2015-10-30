import math
import theano
import numpy
# Momentum        
def initialVelocitys(P):
    v = []
    v.append(theano.shared(numpy.zeros( (P.inputDimNum, P.dnnWidth), dtype = theano.config.floatX ), borrow = True))
    v.append(theano.shared(numpy.zeros( (P.dnnWidth,), dtype = theano.config.floatX ), borrow = True) )
    for i in xrange(P.dnnDepth - 1):
       v.append(theano.shared(numpy.zeros( (P.dnnWidth/2,P.dnnWidth), dtype = theano.config.floatX ), borrow = True) )
       v.append(theano.shared(numpy.zeros( (P.dnnWidth,), dtype = theano.config.floatX ), borrow = True) )
    v.append(theano.shared(numpy.zeros( (P.dnnWidth/2, P.outputPhoneNum), dtype = theano.config.floatX ), borrow = True) )
    v.append(theano.shared(numpy.zeros( (P.outputPhoneNum,), dtype = theano.config.floatX ), borrow = True) )
    return v

def momentum(grads, params, velocitys, lr, flag):
    if(flag[0]):
        velocitys = [velocity - lr * grad for velocity, grad in zip(velocitys, grads)]
        flag[0] = [False]
    else:
        velocitys = [ P.momentum * velocity - lr * (1 - P.momentum) * grad for velocity, grad in zip(velocitys, grads) ]
    params_update = [ (param, param + velocity) for param, velocity in zip(params, velocitys) ]
    return params_update

def RMSProp(grads, params, sigma, lr, flag):
    if(flag[0]):
        sigma = grads.sum()
    else:
        sigma = math.sqrt(alpha * (sigma ** 2) - (1 - alpha)(grads.sum() ** 2) )
    params_update = [(param, param - (lr/sigma) * grad) for param, grad in zip(params, grads)]
    return params_update, sigma

