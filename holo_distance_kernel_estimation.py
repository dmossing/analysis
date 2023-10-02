#!/usr/bin/env python

import autograd.numpy as np
from autograd import grad
import scipy.optimize as sop

def pred_distance_effect(D,amplitudeE,sigmaE,amplitudeI,sigmaI,fudge=1e-3):
    # D: Nn non-target neurons x Ne ensembles x Nt targeted neurons/ensemble
    def gaussian(x,amplitude,sigma):
        s2 = sigma**2+fudge
        return amplitude*np.exp(-0.5*x**2/s2)/np.sqrt(2*np.pi*s2)
    Eeffect = np.sum(gaussian(D,amplitudeE,sigmaE),axis=-1)
    Ieffect = np.sum(gaussian(D,amplitudeI,sigmaI),axis=-1)
    return Eeffect - Ieffect

def fit_distance_kernel(D,dF):
    # D: Nn non-target neurons x Ne ensembles x Nt targeted neurons/ensemble
    # dF: Nn non-target neurons x Ne ensembles
    def cost(theta):
        modeled = pred_distance_effect(D,*theta)
        return np.sum((dF - modeled)**2)
    theta0 = np.array((1,100,1,100))
    bds = sop.Bounds(lb=np.zeros_like(theta0),ub=np.inf*np.ones_like(theta0))
    res = sop.minimize(cost,theta0,bounds=bds,jac=grad(cost))
    if res.success:
        return res.x
    else:
        return np.nan*np.ones_like(res.x)
