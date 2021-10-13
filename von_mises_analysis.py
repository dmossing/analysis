#!/usr/bin/env python

import autograd.numpy as np
import scipy.optimize as sop
from autograd import grad

uangle = np.arange(0,360,45)

def von_mises_function(x,kappa,mu,ht):
    # x: directions in radians
    # kappa: tuning sharpness parameter
    # mu: preferred direction in radians
    # ht: height of non-preferred direction peak relative to preferred
    return (np.exp(kappa*np.cos(x-mu)) + ht*np.exp(kappa*np.cos(x-mu-np.pi)))/(np.exp(kappa) + ht*np.exp(-kappa))

def sym_von_mises_function(x,kappa,amplitude,offset=0):
    # constraining ht = 1 above
    # multiplies output of von_mises_function() by amplitude
    return amplitude*von_mises_function(x,kappa,0,1)+offset

def svm_fn(*args):#kappa,amplitude):
    # wrapper function, using fixed uangle in degrees
    return sym_von_mises_function(np.deg2rad(uangle),*args)#kappa,amplitude)

def svm_cost(x,data):
    # computes cost given parameters (kappa, amplitude[, optionally offset])
    return np.nansum((data-svm_fn(*x))**2)

def svm_cost_l1(x,data,lam=1):
    # computes cost given (kappa, amplitude[, optionally offset]), with an L1 penalty on kappa (exponential distribution prior on kappa)
    msk = ~np.isnan(data)
    return np.sum((data-svm_fn(*x))[msk]**2) + lam*x[0]
    #return np.nansum((data-svm_fn(*x))**2) + lam*x[0]

def optimize_kappa(data,x0=None,cost_fn=svm_cost,args=()):
    # given data, optimize (kappa, amplitude) inputs to svm_fn(), and return kappa
    if x0 is None:
        x0 = np.array((1,np.nanmean(data)))
    this_cost_fn = lambda x: cost_fn(x,data,*args)
    this_grad = grad(this_cost_fn)
    result = sop.minimize(this_cost_fn,x0,bounds=[(0,np.inf),(0,np.inf)],method='L-BFGS-B',jac=this_grad)
    kappa,amplitude = result.x
    return kappa,amplitude

def optimize_kappa_offset(data,x0=None,cost_fn=svm_cost,args=()):
    # given data, optimize (kappa, amplitude) inputs to svm_fn(), and return kappa
    if x0 is None:
        x0 = np.array((1,np.nanmax(data),0))
    result = sop.minimize(cost_fn,x0,args=(data,)+args,bounds=[(0,np.inf),(0,np.inf),(-np.inf,np.inf)])
    kappa = result.x[0]
    return kappa
