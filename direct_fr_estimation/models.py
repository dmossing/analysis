#!/usr/bin/env python

import direct_fr_estimation as dfre
#from importlib import reload
#reload(dfre)
import numpy as np

class step_T_nonparametric_theta(dfre.fn_obj):
    def __init__(self):
        #self = dfre.fn_obj(self.r_step_T_nonparametric_theta,self.dr_step_T_nonparametric_theta,self.compute_helper_vars)
        self = dfre.fn_obj(self.rfunc,self.drfunc,self.compute_helper_vars)
        
    def rfunc(self,theta): #r_step_T_nonparametric_theta(self,theta):
        # theta: one per stim condition, plus one baseline in the -1 position
        # timepart: T x 1
        qwhere = self.helper_vars['qwhere']
        step_pulse = self.helper_vars['step_pulse']
        trialpart = theta[qwhere]
        # trialpart: 1 x N
        return step_pulse[:,np.newaxis]*trialpart[np.newaxis,:] + theta[-1]
        
    def drfunc(self,theta): #d_r_step_T_nonparametric_theta(self,theta):
        deriv = self.helper_vars['deriv']
        return deriv
    
    def compute_helper_vars(self,fit_obj):
        self.helper_vars = {}
        data_obj = fit_obj.data_obj
        angle = data_obj.angle
        nangle = len(np.unique(angle))
        nbefore = data_obj.data_params['nbefore']
        nafter = data_obj.data_params['nafter']
        stimlen = data_obj.data_params['stimlen']
        p = fit_obj.p
        q = np.zeros((nangle+1,len(angle)),dtype='bool')
        for i in range(nangle):
            q[i] = angle==i
        qwhere = np.where(q.T)[1]
        step_pulse = np.concatenate((np.zeros((nbefore-p,)),np.ones((stimlen,)),np.zeros((nafter,))))
        deriv = q[:,np.newaxis,:]*step_pulse[np.newaxis,:,np.newaxis]
        deriv[nangle] = 1

        theta_guess = np.zeros((nangle+1,))
        for i in range(nangle):
            theta_guess[i] = data_obj.S[nbefore:-nafter][:,angle==i].mean()
        theta_guess[nangle] = data_obj.S[nbefore:].mean()
        
        bounds = [(0,np.inf) for x in theta_guess] 

        self.helper_vars['qwhere'] = qwhere
        self.helper_vars['step_pulse'] = step_pulse
        self.helper_vars['deriv'] = deriv
        self.helper_vars['theta_guess'] = theta_guess
        self.helper_vars['bounds'] = bounds
        self.helper_vars['nangle'] = nangle
        self.helper_vars['nbefore'] = nbefore

class step_transient_T_nonparametric_theta(dfre.fn_obj):
    def __init__(self):
        #self = dfre.fn_obj(self.r_step_T_nonparametric_theta,self.dr_step_T_nonparametric_theta,self.compute_helper_vars)
        self = dfre.fn_obj(self.rfunc,self.drfunc,self.compute_helper_vars)

    def rfunc(self,theta):
        # theta: one per stim condition, plus one baseline in the -1 position
        # timepart: T x 1
        qwhere = self.helper_vars['qwhere']
        step_pulse = self.helper_vars['step_pulse']
        transient = self.helper_vars['transient']
        nangle = self.helper_vars['nangle']

        trialpart = theta[qwhere]
        # trialpart: 1 x N
        baseline = theta[nangle]
        transient_ht = theta[nangle+1]
        return (step_pulse[:,np.newaxis]+transient_ht*transient[:,np.newaxis])*trialpart[np.newaxis,:] + baseline
    
    def drfunc(self,theta):
        qwhere = self.helper_vars['qwhere']
        step_pulse = self.helper_vars['step_pulse']
        transient = self.helper_vars['transient']
        nangle = self.helper_vars['nangle']
        deriv = self.helper_vars['deriv']
        deriv_transient_portion = self.helper_vars['deriv_transient_portion']

        trialpart = theta[qwhere]
        transient_ht = theta[nangle+1]
        deriv_transient = deriv.copy()
        deriv_transient = deriv_transient + transient_ht*deriv_transient_portion
        deriv_transient[nangle+1] = transient[:,np.newaxis]*trialpart[np.newaxis,:]
        return deriv_transient
         
    def compute_helper_vars(self,fit_obj):
        self.helper_vars = {}
        data_obj = fit_obj.data_obj
        angle = data_obj.angle
        nangle = len(np.unique(angle))
        nbefore = data_obj.data_params['nbefore']
        nafter = data_obj.data_params['nafter']
        stimlen = data_obj.data_params['stimlen']
        p = fit_obj.p
        q = np.zeros((nangle+2,len(angle)),dtype='bool')
        for i in range(nangle):
            q[i] = angle==i
        qwhere = np.where(q.T)[1]
        step_pulse = np.concatenate((np.zeros((nbefore-p,)),np.ones((stimlen,)),np.zeros((nafter,))))
        deriv = q[:,np.newaxis,:]*step_pulse[np.newaxis,:,np.newaxis]
        deriv[nangle] = 1

        theta_guess = np.zeros((nangle+2,))
        for i in range(nangle):
            theta_guess[i] = data_obj.S[nbefore:-nafter][:,angle==i].mean()
        theta_guess[nangle] = data_obj.S[nbefore:].mean()
        theta_guess[nangle+1] = 1
        
        bounds = [(0,np.inf) for x in theta_guess] 

        transient = np.concatenate((np.zeros((nbefore-p,)),(1,),np.zeros((stimlen-1+nafter,))))
        deriv_transient_portion = q[:,np.newaxis,:]*transient[np.newaxis,:,np.newaxis]

        self.helper_vars['qwhere'] = qwhere
        self.helper_vars['step_pulse'] = step_pulse
        self.helper_vars['deriv'] = deriv
        self.helper_vars['theta_guess'] = theta_guess
        self.helper_vars['bounds'] = bounds
        self.helper_vars['nangle'] = nangle
        self.helper_vars['transient'] = transient
        self.helper_vars['deriv_transient_portion'] = deriv_transient_portion
        self.helper_vars['nbefore'] = nbefore
        
class nonparametric_T_nonparametric_theta(dfre.fn_obj):
    def __init__(self,nt=10):
        # nt the number of bins for computing nonparametric time component
        dfre.fn_obj.__init__(self,self.rfunc,self.drfunc,self.compute_helper_vars)
        self.helper_vars['nt'] = nt

    def gen_timepart_trialpart(self,theta):
        timepartbase = self.helper_vars['timepartbase']
        qwhere = self.helper_vars['qwhere']
        nt = self.helper_vars['nt']
        nbefore = self.helper_vars['nbefore']
        nangle = self.helper_vars['nangle']
        p = self.helper_vars['p']
        trialpart = theta[qwhere]
        timepart = timepartbase.copy()
        timepart[nbefore-p:nbefore-p+nt] = theta[nangle+1:]
        #baseline = theta[nangle]
        return timepart, trialpart #, baseline
        
        
    def rfunc(self,theta): #r_step_T_nonparametric_theta(self,theta):
        # theta: one per stim condition, plus one baseline in the -1 position
        # timepart: T x 1
        #qwhere = self.helper_vars['qwhere']
        nangle = self.helper_vars['nangle']
        #nbefore = self.helper_vars['nbefore']
        #timepartbase = self.helper_vars['timepartbase']
        #nt = self.helper_vars['nt']
        timepart,trialpart = self.gen_timepart_trialpart(theta)
        #trialpart = theta[qwhere]
        #timepart = timepartbase.copy()
        #timepart[nbefore:nbefore+nt] = theta[nangle+1:]
        # trialpart: 1 x N
        baseline = theta[nangle]
        return timepart[:,np.newaxis]*trialpart[np.newaxis,:] + baseline
        
    def drfunc(self,theta): #d_r_step_T_nonparametric_theta(self,theta):
        q = self.helper_vars['q']
        nangle = self.helper_vars['nangle']
        tmatrix = self.helper_vars['tmatrix']
        timepart,trialpart = self.gen_timepart_trialpart(theta)
        deriv = q[:,np.newaxis,:]*timepart[np.newaxis,:,np.newaxis]
        deriv = deriv + tmatrix[:,:,np.newaxis]*trialpart[np.newaxis,np.newaxis,:]
        deriv[nangle] = 1
        return deriv
    
    def compute_helper_vars(self,fit_obj):
        data_obj = fit_obj.data_obj
        angle = data_obj.angle
        nangle = len(np.unique(angle))
        nbefore = data_obj.data_params['nbefore']
        nafter = data_obj.data_params['nafter']
        stimlen = data_obj.data_params['stimlen']
        nt = self.helper_vars['nt']
        p = fit_obj.p
        nttotal = nbefore+nafter+stimlen-p
        #nt = self.helper_vars['nt']
        q = np.zeros((nangle+1+nt,len(angle)),dtype='bool')
        for i in range(nangle):
            q[i] = angle==i
        qwhere = np.where(q.T)[1]
        timepartbase = np.zeros((nttotal,))
        tmatrix = np.zeros((q.shape[0],nttotal))
        tmatrix[nangle+1:][:,nbefore-p:nbefore-p+nt] = np.identity(nt)
        #step_pulse = np.concatenate((np.zeros((nbefore-p,)),np.ones((stimlen,)),np.zeros((nafter,))))
        #deriv = q[:,np.newaxis,:]*step_pulse[np.newaxis,:,np.newaxis]
        #deriv[nangle] = 1

        theta_guess = np.zeros((nangle+1+nt,))
        for i in range(nangle):
            theta_guess[i] = data_obj.S[nbefore:-nafter][:,angle==i].mean()
        theta_guess[nangle] = data_obj.S[nbefore:].mean()
        theta_guess[nangle+1:nangle+1+stimlen] = 1
        
        bounds = [(0,np.inf) for x in theta_guess] 

        self.helper_vars['p'] = p
        self.helper_vars['q'] = q
        self.helper_vars['qwhere'] = qwhere
        self.helper_vars['theta_guess'] = theta_guess
        self.helper_vars['bounds'] = bounds
        self.helper_vars['nangle'] = nangle
        self.helper_vars['timepartbase'] = timepartbase
        self.helper_vars['tmatrix'] = tmatrix
        self.helper_vars['nbefore'] = nbefore
