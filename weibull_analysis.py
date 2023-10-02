#!/usr/bin/env python

import autograd.numpy as np

def sigmoid_function(c,a0,a1,k,c50):
    return a0+ (1-a0-a1)*1/(1+ np.exp(-k*(c-c50)))

def weibull_function(c,a0,a1,k,b):
    return a0+(1-a0-a1)*(1-np.exp(-(c/b)**k))

def sigmoid_function_reparam(c,a0,a1,q,s):
    return a0+ (1-a0-a1)*(1 - 2**(-(s*c)**q))

def sigmoid_function_n(n,c,a0,a1,q,s):
    return a0+ (1-a0-a1)*(1 - 2**(-n*(s*c)**q))

def weibull_function_n(n,c,a0,a1,k,b):
    return a0+(1-a0-a1)*(1-np.exp(-n*(c/b)**k))

def sigmoid_function_n_one_arg(x,a0,a1,q,s):
    n = x[:,0]
    c = x[:,1]
    return sigmoid_function_n(n,c,a0,a1,q,s)

def weibull_function_n_one_arg(x,a0,a1,q,s):
    n = x[:,0]
    c = x[:,1]
    return weibull_function_n(n,c,a0,a1,q,s)