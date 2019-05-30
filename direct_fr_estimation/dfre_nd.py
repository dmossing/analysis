#!/usr/bin/env python

import numpy as np
from oasis.functions import deconvolve,estimate_parameters
import scipy.optimize as sop

#step_pulse = np.concatenate((np.zeros((nbefore-p,)),np.ones((stimlen,)),np.zeros((nbefore,))))

class data_obj(object):
    # this contains the preprocessed calcium data, and a first pass at deconvolution using OASIS
    def __init__(self,angle,F,S,dfof,nbefore,nafter):
        self.angle = angle.copy() # (y,N) shape, where y is the number of parameters varied
        if len(self.angle.shape)<2:
            self.angle = self.angle[np.newaxis]
        self.F = F.copy()
        if len(self.F.shape)<3: # (n,T,N) shape, where there are n neurons, T time points, N trials
            self.F = self.F[np.newaxis] 
        self.S = S.copy()
        if len(self.S.shape)<3:
            self.S = self.S[np.newaxis] 
        self.dfof = dfof.copy()
        if len(self.dfof.shape)<2:
            self.dfof = self.dfof[np.newaxis] 
        self.data_params = {}
        self.data_params['nbefore'] = nbefore
        self.data_params['nafter'] = nafter
        self.data_params['stimlen'] = self.F.shape[1]-nbefore-nafter

class fn_obj(object):
    # this is an API for writing custom firing rate models
    def __init__(self,rfunc,drfunc,compute_helper_vars):
        self.rfunc = rfunc
        # rate fn: input d dimensional theta, output T time points x N trials
        self.drfunc = drfunc
        # derivative of rate fn: input d dimensional theta, output d dimensions x T time points x N trials
        self.compute_helper_vars = compute_helper_vars
        # take as input 'data_obj' defined above. Output a dict. Needs to include at least bounds, theta_guess, but can include others
        self.helper_vars = {}
        self.helper_vars['bounds'] = None
        self.helper_vars['theta_guess'] = None
        #    suggested bounds: [(0,np.inf) for x in theta_guess]
        
class fit_obj(object):
    # this contains variables and methods necessary for performing general L-BFGS optimization on an inhomogeneous Poisson spiking model with autoregressive Ca dynamics, without specific methods specialized to particular firing rate models
    def __init__(self,data_obj,p=1,oversampled=0,t_offset=None,precomputed=None):
        # some fns. require 'precomputed', a dict with at least two keys theta_star (output of lbfgsb) and fn_obj used in the optimiztion of theta_star
        # t_offset: if oversampled, this is shape (N,), and gives the offset between stim trigger and frame (nbefore).
        self.data_obj = data_obj
        self.F = data_obj.F
        self.nroi = self.F.shape[0]
        self.p = p
        self.b = np.zeros((self.nroi,1,1))
        self.g = np.zeros((self.nroi,self.p,1))
        self.a = np.zeros((self.nroi,))
        self.sn
        fudge_factor = .97
        for i in range(self.nroi):
            _,s,self.b[i,0,0],gtemp,_  = deconvolve(data_obj.dfof[i].astype(np.float64),penalty=1,g=tuple([None]*self.p))
            self.g[i,:,0] = np.array(gtemp)
            self.a[i] = np.percentile(s,95)
            est = estimate_parameters(data_obj.dfof[i].astype(np.float64), p=self.p, fudge_factor=fudge_factor)
            self.sn[i] = est[1]
#        if not type(g) is tuple:
#            g = (g,)
#        self.g = np.array(g)
        #self.fn_obj = fn_obj
        #nangle = len(np.unique(data_obj.angle))
        self.noise = (self.sn**2*(1+(self.g**2).sum(1)))[:,np.newaxis,np.newaxis]
        self.smax = 10
        #self.fn_obj.compute_helper_vars(data_obj,self)
        self.pFs = [self.p_F_given_s(s) for s in range(self.smax)]
        self.oversampled = oversampled
        if self.oversampled:
            self.sampwt = np.ones((self.oversampled,1))/self.oversampled
            self.sampmat = np.zeros((self.oversampled*(self.F.shape[0]-1),self.F.shape[1]),dtype='bool')
            dig = np.floor(self.oversampled*t_offset).astype('<i2')
            for i in range(self.sampmat.shape[1]):
                self.sampmat[dig::self.oversampled,i] = 1
        if precomputed:
            theta_star = precomputed['theta_star']
            fn_obj = precomputed['fn_obj']
            self.rpre = np.zeros(np.array(self.F.shape)+np.array((0,-1,0))) # one fewer time point required
            for i in range(self.nroi):
                self.rpre[i] = fn_obj.rfunc(theta_star[i][0])
            
            
        #self.q = np.zeros((self.nparam,len(angle)),dtype='bool')
       
    def p_s_given_r(self,s,r):
        # bin-wise likelihood of r given s spikes in each bin
        # n x T x N
        if self.oversampled:
            rhat = snd.convolve(r,self.sampwt)[self.sampmat].reshape(s.shape)
            return np.exp(-rhat)*rhat**s/np.math.factorial(s)
        else:
            return np.exp(-r)*r**s/np.math.factorial(s)
    
    def p_F_given_s(self,s):
        # bin-wise likelihood of s spikes in each bin given df/f data F
        # n x T x N
        prev = np.array([self.g[:,k:k+1]*(self.F-self.b)[:,self.p-(k+1):-(k+1)] for k in range(self.p)])
        return np.exp(-0.5*((self.F-self.b)[self.p:]-self.a*s-prev.sum(1))**2/self.noise)
    
    def L_of_Theta(self,theta,fn_obj):
        # likelihood of parameters theta given df/f data
        r = fn_obj.rfunc(theta)
        # r: n neurons x T time points x N trials
        ell = np.array([self.pFs[s]*self.p_s_given_r(s,r) for s in range(self.smax)])
        # ell: smax possibilities x n neurons x T time points x N trials
        return np.sum(np.log(np.sum(ell,axis=0)))
        # sum over T time points, N trials
    
    def dLdTheta(self,theta,fn_obj,fudge=1e-3):
        # derivative of likelihood of theta given df/f data
        # r: n neurons x T time points x N trials
        # F: n neurons x T+p time points x N trials (p at beginning for AR(p) process)
        # noise, g: estimated by OASIS, n x 1 x 1, n x p x 1
        # rfunc: function handle for rate from parameters. Returns n x T x N
        # drfunc: function handle for derivative from parameters. Returns d x n x T x N for d fitting parameters
        # theta: d dimensions
        # returns d dimensions
        # fudge factor, units of events per bin, to avoid /0 errors
        
        r = fn_obj.rfunc(theta)
        
        ell = np.array([self.pFs[s]*self.p_s_given_r(s,r) for s in range(self.smax)])
        # ell: smax possibilities x n x T time points x N trials
        dell = np.array([s/(r+fudge)-1 for s in range(self.smax)])*ell
        # dell: smax possibilities x n x T time points x N trials
        dLdr = np.sum(dell,axis=0)/np.sum(ell,axis=0)
        # dLdr: n x T time points x N trials; likelihood of each data point
        return np.sum(np.sum(np.sum(fn_obj.drfunc(theta)*dLdr[np.newaxis],axis=-1),axis=-1),axis=-1)
    
    def fit(self,this_fn_obj,factr=1e4,epsilon=1e-2):
        #print(epsilon)
        this_fn_obj.compute_helper_vars(self)
        minusL = lambda x: -self.L_of_Theta(x,this_fn_obj)
        minusdL = lambda x: -self.dLdTheta(x,this_fn_obj)
        theta_guess = this_fn_obj.helper_vars['theta_guess']
        bounds = this_fn_obj.helper_vars['bounds']
        theta_star = sop.fmin_l_bfgs_b(minusL,theta_guess,fprime=minusdL,bounds=bounds,pgtol=1e-8,factr=factr,epsilon=epsilon)
        return theta_star
    
