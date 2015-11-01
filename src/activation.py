import math
import theano
import theano.tensor as T
import numpy
import globalParam

# Momentum        
def momentum(grads, params, P):
    if(globalParam.flag):
        globalParam.velocitys = [ -globalParam.lr * grad for grad in grads ]
        globalParam.flag = False
    else:
        globalParam.velocitys = [ P.momentum * velocity - globalParam.lr * (1 - P.momentum) * grad for velocity, grad in zip(globalParam.velocitys, grads) ]
    paramsUpdate = [ (param, param + velocity) for param, velocity in zip(params, globalParam.velocitys) ]
    return paramsUpdate

# RMSProp        
def RMSProp(grads, params):
    alpha = 0.9
    epsilon = 1e-6
    if(globalParam.flag):
        globalParam.sigmas = [ g  for g in grads ]
        globalParam.flag = False
    else:
        globalParam.sigmas = [ T.sqrt( ( alpha * T.sqr(s) ) + ( (1 - alpha) * T.sqr(g) ) ) for s, g in zip(globalParam.sigmas, grads) ]
    paramsUpdate = [( p, p - ( globalParam.lr * g ) / ( s + epsilon ) ) for p, g, s in zip(params, grads, globalParam.sigmas)]
    return paramsUpdate

# Adagrad
def Adagrad(grads, params):
    epsilon = 1e-6
    if(globalParam.flag):
        globalParam.gradSqrs = [ g * g for g in grads ]
        globalParam.flag = False
    else:
        globalParam.gradSqrs = [ s + g * g for s, g in globalParam.gradSqrs, grads ]
    paramsUpdate = [ (p, p - ( globalParam.lr * g ) / (T.sqrt(s) + epsilon) ) for p, g, s in zip(params, grads, globalParam.gradSqrs) ]
    return paramsUpdate
