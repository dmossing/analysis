#!/usr/bin/env python

import numpy as np
from oasis.functions import deconvolve,estimate_parameters
import scipy.optimize as sop

#step_pulse = np.concatenate((np.zeros((nbefore-p,)),np.ones((stimlen,)),np.zeros((nbefore,))))

class data_obj(object):
    # this contains the preprocessed calcium data, and a first pass at deconvolution using OASIS
    def __init__(self,angle,F,S,dfof,nbefore,nafter):
        self.angle = angle
        self.F = F
        self.S = S
        self.dfof = dfof
        self.data_params = {}
        self.data_params['nbefore'] = nbefore
        self.data_params['nafter'] = nafter
        self.data_params['stimlen'] = self.F.shape[0]-nbefore-nafter

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
    def __init__(self,data_obj,p=1,oversampled=0,t_offset=None):
        # t_offset: if oversampled, this is shape (N,), and gives the offset between stim trigger and frame (nbefore).
        self.data_obj = data_obj
        self.F = data_obj.F
        self.p = p
        c,s,b,g,_  = deconvolve(data_obj.dfof.astype(np.float64),penalty=1,g=tuple([None]*self.p))
        self.a = np.percentile(s,95)
        self.b = b
        fudge_factor = .97
        est = estimate_parameters(data_obj.dfof.astype(np.float64), p=self.p, fudge_factor=fudge_factor)
        self.sn = est[1]
        if not type(g) is tuple:
            g = (g,)
        self.g = np.array(g)
        #self.fn_obj = fn_obj
        #nangle = len(np.unique(data_obj.angle))
        self.noise = self.sn**2*(1+(self.g**2).sum())
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
            
            
        #self.q = np.zeros((self.nparam,len(angle)),dtype='bool')
       
    def p_s_given_r(self,s,r):
        # bin-wise likelihood of r given s spikes in each bin
        if self.oversampled:
            rhat = snd.convolve(r,self.sampwt)[self.sampmat].reshape(s.shape)
            return np.exp(-rhat)*rhat**s/np.math.factorial(s)
        else:
            return np.exp(-r)*r**s/np.math.factorial(s)
    
    def p_F_given_s(self,s):
        # bin-wise likelihood of s spikes in each bin given df/f data F
        prev = np.array([self.g[k]*(self.F-self.b)[self.p-(k+1):-(k+1)] for k in range(self.p)])
        return np.exp(-0.5*((self.F-self.b)[self.p:]-self.a*s-prev.sum(0))**2/self.noise)
    
    def L_of_Theta(self,theta,fn_obj):
        # likelihood of parameters theta given df/f data
        r = fn_obj.rfunc(theta)
        # r: T time points x N trials
        ell = np.array([self.pFs[s]*self.p_s_given_r(s,r) for s in range(self.smax)])
        # ell: smax possibilities x T time points x N trials
        return np.sum(np.log(np.sum(ell,axis=0)))
        # sum over T time points, N trials
    
    def dLdTheta(self,theta,fn_obj,fudge=1e-3):
        # derivative of likelihood of theta given df/f data
        # r: T time points x N trials
        # F: T+p time points x N trials (p at beginning for AR(p) process)
        # noise, g: estimated by OASIS
        # q: stimulus, d dimensions x N trials
        # rfunc: function handle for rate from parameters. Returns T x N
        # drfunc: function handle for derivative from parameters. Returns d x T x N
        # theta: d dimensions
        # returns d dimensions
        
        r = fn_obj.rfunc(theta)
        
        ell = np.array([self.pFs[s]*self.p_s_given_r(s,r) for s in range(self.smax)])
        # ell: smax possibilities x T time points x N trials
        dell = np.array([s/(r+fudge)-1 for s in range(self.smax)])*ell
        # dell: smax possibilities x T time points x N trials
        dLdr = np.sum(dell,axis=0)/np.sum(ell,axis=0)
        # dLdr: T time points x N trials; likelihood of each data point
        return np.sum(np.sum(fn_obj.drfunc(theta)*dLdr[np.newaxis],axis=-1),axis=-1) 
    
    def fit(self,this_fn_obj,factr=1e4,epsilon=1e-2):
        #print(epsilon)
        this_fn_obj.compute_helper_vars(self)
        minusL = lambda x: -self.L_of_Theta(x,this_fn_obj)
        minusdL = lambda x: -self.dLdTheta(x,this_fn_obj)
        theta_guess = this_fn_obj.helper_vars['theta_guess']
        bounds = this_fn_obj.helper_vars['bounds']
        theta_star = sop.fmin_l_bfgs_b(minusL,theta_guess,fprime=minusdL,bounds=bounds,pgtol=1e-8,factr=factr,epsilon=epsilon)
        return theta_star
    
    #def dLdTheta(theta,rfunc,drfunc):
#	# r: T time points x N trials
#	# F: T+2 time points x N trials (two at beginning for AR(2) process)
#	# sigma, g: estimated by OASIS
#	# q: stimulus, d dimensions x N trials
#	# rfunc: function handle for rate from parameters. Returns T x N
#	# drfunc: function handle for derivative from parameters. Returns d x T x N
#	# theta: d dimensions
#	# returns d dimensions
#	
#	r = rfunc(q,theta)
#	noise = sigma**2+(g**2).sum()
#	def p_s_given_r(s,r):
#		return np.exp(-r)*r**s/np.factorial(s)
#	def p_F_given_s(F,s):
#		prev = np.array([g[k]*F[p-(k+1):-(k+1)] for k in range(p)])
#		return np.exp(-0.5*(F[p:]-s-prev.sum())**2/noise)
#	ell = np.array([p_F_given_s(F,s)*p_s_given_r(s,r) for s in range(smax)])
#	# ell: smax possibilities x T time points x N trials
#	dell = np.array([s/r-1 for s in range(smax)])*ell
#	# dell: smax possibilities x T time points x N trials
#	dLdr = np.sum(dell,axis=0)/np.sum(ell,axis=0)
#	# dLdr: T time points x N trials; likelihood of each data point
#	return np.sum(np.sum(drfunc(q,theta)*dLdr[np.newaxis],axis=-1),axis=-1)
#	
#def r_step_T_nonparametric_theta(q,theta):
#	# theta: one per stim condition, plus one baseline in the -1 position
#	# timepart: T x 1
#	trialpart = theta[q]
#	# trialpart: 1 x N
#	return step_pulse[:,np.newaxis]*trialpart[np.newaxis,:] + theta[-1]
#		
#def d_r_step_T_nonparametric_theta(q,theta):
#	deriv = q[:,np.newaxis,:]*step_pulse[np.newaxis,:,np.newaxis]
#	deriv[-1] = 1
#	return deriv
