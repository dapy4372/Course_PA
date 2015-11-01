import math
import theano
import theano.tensor as T
import numpy
import settings

# Momentum        
def momentum(grads, params):
    if(settings.flag):
        settings.velocitys = [ -settings.lr * grad for grad in grads ]
        settings.flag = False
    else:
        settings.velocitys = [ P.momentum * velocity - settings.lr * (1 - P.momentum) * grad for velocity, grad in zip(settings.velocitys, grads) ]
    params_update = [ (param, param + velocity) for param, velocity in zip(params, settings.velocitys) ]
    return params_update

#RMSProp        
def RMSProp(grads, params):
    alpha = 0.9
    if(settings.flag):
        settings.sigmas = [ grad for grad in grads ]
        settings.flag = False
    else:
        settings.sigmas = [ T.sqrt( ( alpha * T.sqr(sigma) ) - ( (1 - alpha) * T.sqr(grad) ) ) for sigma, grad in zip(settings.sigmas, grads) ]
    params_update = [( param, T.clip( ( param - (settings.lr) * (grad / sigma) ), -0.5, 0.5 ) ) for param, grad, sigma in zip(params, grads, settings.sigmas)]
    return params_update

