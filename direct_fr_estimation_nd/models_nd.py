#!/usr/bin/env python

import direct_fr_estimation_nd as dfre_nd
#from importlib import reload
#reload(dfre)
#import numpy as np

import autograd.numpy as np
from autograd import jacobian

class step_T_nonparametric_theta(dfre_nd.fn_obj):
    def __init__(self):
        #self = dfre.fn_obj(self.r_step_T_nonparametric_theta,self.dr_step_T_nonparametric_theta,self.compute_helper_vars)
        dfre_nd.fn_obj.__init__(self,self.rfunc,self.drfunc,self.compute_helper_vars)
        
    def rfunc(self,theta): #r_step_T_nonparametric_theta(self,theta):
        # theta: one per stim condition, plus one baseline in the -1 position
        # timepart: T x 1
        qwhere = self.helper_vars['qwhere'] # n x N
        step_pulse = self.helper_vars['step_pulse'] # T
        nroi = self.helper_vars['nroi']
        baseline = theta[-nroi:]
        trialpart = theta[qwhere] - baseline[:,np.newaxis] # n x N
        # trialpart: 1 x N
        return step_pulse[np.newaxis,:,np.newaxis]*trialpart[:,np.newaxis,:] + baseline[:,np.newaxis,np.newaxis]
        # return n x T x N
        
    def drfunc(self,theta): #d_r_step_T_nonparametric_theta(self,theta):
        # d x n x T x N
        deriv = self.helper_vars['deriv']
        return deriv
    
    def compute_helper_vars(self,fit_obj):
        data_obj = fit_obj.data_obj
        angle = data_obj.stim_id # y x N
        nangle = np.unique(angle,axis=1).shape[1] # need to decide how to handle multiple dimensions with this naive fn. Easiest way would be row-wise, but could build in orientation dependence explicitly
        ntrial = angle.shape[1]
        nbefore = data_obj.data_params['nbefore']
        nafter = data_obj.data_params['nafter']
        stimlen = data_obj.data_params['stimlen']
        nroi = fit_obj.nroi
        p = fit_obj.p
        q = np.zeros((nroi,nangle+1,ntrial),dtype='bool') # n x d x N
        qwhere = np.zeros((nroi,ntrial)) # n x N
        for j in range(nroi):
            for i in range(nangle):
                q[j,i] = angle[0]==i # label trials with a particular stim
            qwhere[j] = np.where(q[j].T)[1]+nangle+1 # index the relevant location on the optimized weight parameters; every nangle+1 parameters is a new ROI
        q[:,-1] = -1 #np.tile(np.identity(nroi),(1,1,ntrial))
        q = q.reshape((nroi*(nangle+1),ntrial))
        #qwhere = np.where(q.T)[1]
        step_pulse = np.concatenate((np.zeros((nbefore-p,)),np.ones((stimlen,)),np.zeros((nafter,))))
        deriv = q[:,np.newaxis,np.newaxis,:]*step_pulse[np.newaxis,np.newaxis,:,np.newaxis] # dn x n x T x N
        deriv[-nroi:] = 1
        deriv = np.tile(deriv,(1,nroi,1,1))

        theta_guess = np.zeros((nroi*(nangle+1),))
        for i in range(nangle):
            theta_guess[i::nangle+1] = data_obj.S[:,nbefore:-nafter][:,:,angle[0]==i].mean(1)
        theta_guess[nangle::nangle+1] = data_obj.S[:,:nbefore].mean(1)
        
        bounds = [(0,np.inf) for x in theta_guess] 

        self.helper_vars['qwhere'] = qwhere
        self.helper_vars['step_pulse'] = step_pulse
        self.helper_vars['deriv'] = deriv
        self.helper_vars['theta_guess'] = theta_guess
        self.helper_vars['bounds'] = bounds
        self.helper_vars['nangle'] = nangle
        self.helper_vars['nbefore'] = nbefore
        self.helper_vars['ntrial'] = ntrial

class step_transient_T_nonparametric_theta(dfre_nd.fn_obj):
    def __init__(self):
        #self = dfre.fn_obj(self.r_step_T_nonparametric_theta,self.dr_step_T_nonparametric_theta,self.compute_helper_vars)
        dfre_nd.fn_obj.__init__(self,self.rfunc,self.drfunc,self.compute_helper_vars)

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
        data_obj = fit_obj.data_obj
        angle = data_obj.stim_id
        nangle = len(np.unique(angle))
        nbefore = data_obj.data_params['nbefore']
        nafter = data_obj.data_params['nafter']
        stimlen = data_obj.data_params['stimlen']
        p = fit_obj.p
        q = np.zeros((nangle+2,len(angle)),dtype='bool')
        for i in range(nangle):
            q[i] = angle[0]==i
        qwhere = np.where(q.T)[1]
        step_pulse = np.concatenate((np.zeros((nbefore-p,)),np.ones((stimlen,)),np.zeros((nafter,))))
        deriv = q[:,np.newaxis,:]*step_pulse[np.newaxis,:,np.newaxis]
        deriv[nangle] = 1

        theta_guess = np.zeros((nangle+2,))
        for i in range(nangle):
            theta_guess[i] = data_obj.S[nbefore:-nafter][:,angle[0]==i].mean()
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
        
class nonparametric_T_nonparametric_theta(dfre_nd.fn_obj):
    def __init__(self,nt=10):
        # nt the number of bins for computing nonparametric time component
        dfre_nd.fn_obj.__init__(self,self.rfunc,self.drfunc,self.compute_helper_vars)
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
        angle = data_obj.stim_id
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
            q[i] = angle[0]==i
        qwhere = np.where(q.T)[1]
        timepartbase = np.zeros((nttotal,))
        tmatrix = np.zeros((q.shape[0],nttotal))
        tmatrix[nangle+1:][:,nbefore-p:nbefore-p+nt] = np.identity(nt)
        #step_pulse = np.concatenate((np.zeros((nbefore-p,)),np.ones((stimlen,)),np.zeros((nafter,))))
        #deriv = q[:,np.newaxis,:]*step_pulse[np.newaxis,:,np.newaxis]
        #deriv[nangle] = 1

        theta_guess = np.zeros((nangle+1+nt,))
        for i in range(nangle):
            theta_guess[i] = data_obj.S[nbefore:-nafter][:,angle[0]==i].mean()
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

class shared_kd_gain_with_coupling(dfre_nd.fn_obj):
    def __init__(self,kgain=1):
        # nt the number of bins for computing nonparametric time component
        dfre_nd.fn_obj.__init__(self,self.rfunc,self.drfunc,self.compute_helper_vars)
        self.helper_vars['kgain'] = kgain
         
    def compute_ch(self,theta):
        # theta: k global gain dimensions h + k coupling terms c; (N gain0 terms h, N gain1 terms h, ..., n gain0 coupling weights c, n gain1 coupling weights c, ...)
        kgain = self.helper_vars['kgain'] 
        ntrial = self.helper_vars['ntrial']
        nroi = self.helper_vars['nroi']
        h = theta[:kgain*ntrial].reshape((kgain,ntrial)) # k x N trialwise gain terms
        c = theta[kgain*ntrial:].reshape((kgain,nroi)) # k x n roiwise coupling weights
        return c,h

    def rfunc(self,theta): # theta: k global gain dimensions + k coupling terms; (N gain0 terms, N gain1 terms, ..., n gain0 coupling weights, n gain1 coupling weights, ...)
        rpre = self.helper_vars['rpre'] # n x T x N
        c,h = self.compute_ch(theta) # k x n, k x N
        return rpre*np.exp(np.sum(c[:,:,np.newaxis,np.newaxis]*h[:,np.newaxis,np.newaxis,:],0)) # k x n x T x N, summed over axis 0 
    
    def drfunc(self,theta): # (kN + kn) x n x T x N
        kgain = self.helper_vars['kgain'] 
        ntrial = self.helper_vars['ntrial']
        nroi = self.helper_vars['nroi']
        rcurr = self.rfunc(theta)
        c,h = self.compute_ch(theta)# k x n, k x N
        drdh = c[:,:,np.newaxis,np.newaxis]*rcurr[np.newaxis] # k x n x T x N
        drdc = h[:,np.newaxis,np.newaxis,:]*rcurr[np.newaxis] # k x n x T x N
        drdh = drdh[:,np.newaxis]*np.identity(ntrial)[np.newaxis,:,np.newaxis,np.newaxis,:] # k x N x n x T x N
        drdc = drdc[:,np.newaxis]*np.identity(nroi)[np.newaxis,:,:,np.newaxis,np.newaxis] # k x n x n x T x N
        drdh = drdh.reshape((drdh.shape[0]*drdh.shape[1],)+drdh.shape[2:])
        drdc = drdc.reshape((drdc.shape[0]*drdc.shape[1],)+drdc.shape[2:])
        deriv = np.concatenate((drdh,drdc),axis=0) # (kN + kn) x n x T x N
        return deriv
        
    def compute_helper_vars(self,fit_obj):
        kgain = self.helper_vars['kgain']
        ntrial = fit_obj.F.shape[2] # n x T x N
        nroi = fit_obj.F.shape[0]
        fudge = 0.1
        try:
            rpre = 0*fit_obj.rpre + fudge
        except:
            print('r not precomputed. Precompute r')
        
        theta_guess = np.ones((kgain*(ntrial+nroi),))
        theta_guess[:kgain*ntrial] = np.tile(fit_obj.F.mean(0).mean(0),(kgain,))
    
        bounds = [(-2,2) for x in theta_guess]

        self.helper_vars['kgain'] = kgain
        self.helper_vars['ntrial'] = ntrial
        self.helper_vars['nroi'] = nroi
        self.helper_vars['rpre'] = rpre
        self.helper_vars['theta_guess'] = theta_guess
        self.helper_vars['bounds'] = bounds

class gaussian_rf_step_T(dfre_nd.fn_obj):
    def __init__(self):
        dfre_nd.fn_obj.__init__(self,self.rfunc,self.drfunc,self.compute_helper_vars)
    
    def rfunc(self,theta):
        mu = theta[:2][:,np.newaxis] # (2,1), center of gaussian
        sa = theta[2] # elements of inverse covariance matrix
        sb = theta[3]
        sc = theta[4]
        A = theta[5] # amplitude
        b = theta[6] # offset
        xx = self.helper_vars['xx'] # (2,N), where N is the # of trials. Each is one of Ng grid locations
        step_pulse = self.helper_vars['step_pulse'] # (T,), where T is the number of time points
        siginv = np.array(((sa,sb),(sb,sc))) # (2,2)
        trialpart = (A-b)*np.exp(np.sum(-0.5*(xx-mu)*siginv.dot(xx-mu),axis=0)) # (2,2) dot (2,N) -> (2,N). result is (N,)
        return step_pulse[np.newaxis,:,np.newaxis]*trialpart[np.newaxis,np.newaxis,:] + b # (1,T,N)
    
    def drfunc(self,theta):
        mu = theta[:2][:,np.newaxis] # (2,1), center of gaussian
        sa = theta[2] # elements of inverse covariance matrix ((sa,sb),(sb,sc))
        sb = theta[3]
        sc = theta[4]
        A = theta[5] # amplitude
        b = theta[6] # offset
        xx = self.helper_vars['xx'] # (2,N), where Ng is the number of grid locations
        step_pulse = self.helper_vars['step_pulse'][np.newaxis,:,np.newaxis] # (1,T,1)
        siginv = np.array(((sa,sb),(sb,sc))) # (2,2)
        r = self.rfunc(theta) # (1,T,N)
        drdA = ((r - b)/(A - b))*step_pulse # (1,T,N)
        drdb = -drdA + 1 # (1,T,N)
        drdmu = siginv.dot(xx-mu)[:,np.newaxis,:]*r*step_pulse # (2,2) dot (2,N) = (2,N) * (T,N) -> (2,T,N)
        xxT = (xx-mu)[:,np.newaxis,:]*(xx-mu)[np.newaxis,:,:] # (2,2,N)
        xxTa = xxT[0:1,0:1,:] # (1,1,N)
        xxTb = xxT[0:1,1:2,:] # (1,1,N)
        xxTc = xxT[1:2,1:2,:] # (1,1,N)
        drdsa,drdsb,drdsc = [(-0.5*y*r*step_pulse) for y in (xxTa,xxTb,xxTc)] # (1,T,N)
        deriv = np.concatenate((drdmu,drdsa,drdsb,drdsc,drdA,drdb),axis=0) # (d,T,N) d = 7
        return deriv[:,np.newaxis,:,:] # (d,1,T,N)
        
    def compute_helper_vars(self,fit_obj): 
        data_obj = fit_obj.data_obj
        xx = data_obj.stim_id.astype(np.float64) # (2,N)
        ny = len(np.unique(xx[0]))
        nx = len(np.unique(xx[1]))
        nbefore = data_obj.data_params['nbefore']
        nafter = data_obj.data_params['nafter']
        stimlen = data_obj.data_params['stimlen']
        p = fit_obj.p
        nttotal = nbefore+nafter+stimlen-p
        step_pulse = np.concatenate((np.zeros((nbefore-p,)),np.ones((stimlen,)),np.zeros((nafter,))))
        eps = 5e-3
        bdmux = (-0.5,float(nx)-0.5)
        bdmuy = (-0.5,float(ny)-0.5)
        bdsiginvii = (eps,np.inf)
        bdsiginvij = (0,np.inf)
        bdA = (eps,np.inf)
        bdb = (eps,np.inf)
        bounds = [bdmux,bdmuy,bdsiginvii,bdsiginvij,bdsiginvii,bdA,bdb]
        guessy = (ny-1)/2
        guessx = (nx-1)/2
        guess_sa = 1/((ny-1)/2)**2
        guess_sb = 0
        guess_sc = 1/((nx-1)/2)**2
        prestim = data_obj.F[:,:nbefore].mean()
        during = data_obj.F[:,nbefore:-nafter].mean()
        if prestim < during:
            guessA = 1
            guessb = eps
        else:
            guessA = eps
            guessb = 1
        theta_guess = np.array([guessy,guessx,guess_sa,guess_sb,guess_sc,guessA,guessb])
    
        self.helper_vars['step_pulse'] = step_pulse
        self.helper_vars['xx'] = xx
        self.helper_vars['bounds'] = bounds
        self.helper_vars['theta_guess'] = theta_guess 


class step_T_nonparametric_theta_exp_kd_gain(dfre_nd.fn_obj):
    def __init__(self,global_gain=None,coupling_bds=(-2,2)):
        #self = dfre.fn_obj(self.r_step_T_nonparametric_theta,self.dr_step_T_nonparametric_theta,self.compute_helper_vars)
        dfre_nd.fn_obj.__init__(self,self.rfunc,self.drfunc,self.compute_helper_vars)
        if len(global_gain.shape) < 2:
            global_gain = global_gain[np.newaxis,:]
        self.helper_vars['global_gain'] = global_gain
        self.helper_vars['coupling_bds'] = coupling_bds
        
    def compute_gain_mult(self,theta):
        global_gain = self.helper_vars['global_gain'] # k x N
        nroi = self.helper_vars['nroi']
        nangle = self.helper_vars['nangle']
        nparam = self.helper_vars['nparam']
        kgain = global_gain.shape[0]
        coupling = np.zeros((nroi,kgain)) # n, k
        for i in range(kgain):
            coupling[:,i] = theta[nangle+1+i::nparam]
        gain_mult = np.exp((coupling[:,:,np.newaxis,np.newaxis]*global_gain[np.newaxis,:,np.newaxis,:]).sum(1)) # n x T x N
        return gain_mult
        
    def rfunc(self,theta,gain_mult=None): #r_step_T_nonparametric_theta(self,theta):
        # theta: one per stim condition, plus one baseline in the -1 position
        # timepart: T x 1
        qwhere = self.helper_vars['qwhere'] # n x N
        step_pulse = self.helper_vars['step_pulse'] # T
        nangle = self.helper_vars['nangle']
        nparam = self.helper_vars['nparam']
        baseline = theta[nangle::nparam] # n
        trialpart = theta[qwhere] - baseline[:,np.newaxis] # n x N
        if gain_mult is None:
            gain_mult = self.compute_gain_mult(theta)
        return (step_pulse[np.newaxis,:,np.newaxis]*trialpart[:,np.newaxis,:] + baseline[:,np.newaxis,np.newaxis])*gain_mult        # return n x T x N
        
    def drfunc(self,theta): #d_r_step_T_nonparametric_theta(self,theta):
        # d x n x T x N
        global_gain = self.helper_vars['global_gain'] # k x N
        nangle = self.helper_vars['nangle']
        nparam = self.helper_vars['nparam']
        kgain = global_gain.shape[0]
        gain_mult = self.compute_gain_mult(theta)
        r = self.rfunc(theta,gain_mult=gain_mult)
        deriv = self.helper_vars['deriv'].copy() # nd x n x T x N
        deriv = deriv*gain_mult
        for i in range(kgain):
            deriv[nangle+1+i::nparam] = global_gain[i][np.newaxis,np.newaxis,:]*r
        return deriv
    
    def compute_helper_vars(self,fit_obj):
        data_obj = fit_obj.data_obj
        angle = data_obj.stim_id # y x N
        nangle = np.unique(angle,axis=1).shape[1] # need to decide how to handle multiple dimensions with this naive fn. Easiest way would be row-wise, but could build in orientation dependence explicitly
        ntrial = angle.shape[1]
        nbefore = data_obj.data_params['nbefore']
        nafter = data_obj.data_params['nafter']
        stimlen = data_obj.data_params['stimlen']
        global_gain = self.helper_vars['global_gain']
        kgain = global_gain.shape[0]
        nroi = fit_obj.nroi
        p = fit_obj.p
        nttotal = nbefore+nafter+stimlen-p
        nparam = nangle+1+kgain
        #q = np.zeros((nroi,nangle+1+kgain,ntrial),dtype='bool') # n x d x N
        q = np.zeros((nparam,ntrial),dtype='bool') # d x N
        qwhere = np.zeros((nroi,ntrial),dtype='int') # n x N
        for i in range(nangle):
            q[i] = angle[0]==i # label trials with a particular stim
        for j in range(nroi):
#            for i in range(nangle):
#                q[j,i] = angle==i # label trials with a particular stim
            #qwhere[j] = np.where(q[j].T)[1]+j*(nangle+1+kgain) # index the relevant location on the optimized weight parameters; every nangle+1 parameters is a new ROI
            qwhere[j] = (np.where(q.T)[1]+j*nparam).astype('int') # index the relevant location on the optimized weight parameters; every nangle+1 parameters is a new ROI
        #q[:,-kgain-1] = -1 # n x d x N np.tile(np.identity(nroi),(1,1,ntrial))
        q = q.astype('int')
        q[nangle] = -1 # n x d x N np.tile(np.identity(nroi),(1,1,ntrial))
        #q = q.reshape((nroi*(nangle+1+kgain),ntrial))
        #qwhere = np.where(q.T)[1]
        step_pulse = np.concatenate((np.zeros((nbefore-p,)),np.ones((stimlen,)),np.zeros((nafter,))))
        deriv = np.zeros((nroi,nparam,nroi,nttotal,ntrial)) # n d n T N
        #for j in range(nroi):
        #    deriv[j,:,j,:,:] = q[j,:,np.newaxis,:]*step_pulse[np.newaxis,:,np.newaxis] # n x d x n x T x N
        #    deriv[j,nangle,j,:,:] = deriv[j,nangle,j,:,:] + 1 # time-independent bias
        idind = np.identity(nroi,dtype='bool')[:,np.newaxis,:]
        deriv[np.tile(idind,(1,nparam,1))] = q[:,np.newaxis,:]*step_pulse[np.newaxis,:,np.newaxis] # d T N
        deriv[:,nangle:nangle+1][idind] = deriv[:,nangle:nangle+1][idind] + 1
        deriv = np.reshape(deriv,(nroi*nparam,nroi,nttotal,ntrial))
        #deriv = q[:,:,np.newaxis,np.newaxis,:]*step_pulse[np.newaxis,np.newaxis,np.newaxis,:,np.newaxis] # n x d x n x T x N
        #deriv = np.tile(deriv,(1,nroi,1,1))
        #deriv[-nroi*(kgain+1):-kgain*nroi] = 1

        theta_guess = np.zeros((nroi,nparam))
        for i in range(nangle):
            theta_guess[:,i] = data_obj.S[:,nbefore:-nafter][:,:,angle[0]==i].mean(1).mean(1)       
        theta_guess[:,nangle] = data_obj.S[:,:nbefore].mean(1).mean(1)
        theta_guess[:,nangle+1:] = 0
        theta_guess = theta_guess.reshape((nroi*nparam,))

        bounds = [(5e-3,np.inf) for x in theta_guess] 
        for i in range(kgain):
            bounds[nangle+1+i::nparam] = [self.helper_vars['coupling_bds']]

        
        self.helper_vars['qwhere'] = qwhere
        self.helper_vars['step_pulse'] = step_pulse
        self.helper_vars['deriv'] = deriv
        self.helper_vars['theta_guess'] = theta_guess
        self.helper_vars['bounds'] = bounds
        #self.helper_vars['angle'] = nangle
        self.helper_vars['nangle'] = nangle
        self.helper_vars['nbefore'] = nbefore
        self.helper_vars['ntrial'] = ntrial
        self.helper_vars['nparam'] = nparam 
        self.helper_vars['nroi'] = nroi
        self.helper_vars['nparam'] = nparam
        self.helper_vars['nttotal'] = nttotal
		
#class model_from_expression:
#	def __init__(self,expr,symbols):
#		# expr a sympy expression; symbols a list of sympy symbols
#		# symsize = {s:s.size for s in symbols} # how many DOF in each model component
#		sizelist = np.array([s.size for s in symbols]) # array containing number of DOFs per symbol
#		bds = np.cumsum(np.concatenate(((0,),sizelist)))
#		self.symloc = {s:slice(bds[i]:bds[i+1]) for i,s in enumerate(symbols)} # for each symbol, value is the slice of theta subtended by that symbol's values
#		rfunc = lambda theta: expr.evalf(subs={s:theta[symloc[s]] for s in symbols})
#		drfunc = np.autograd(rfunc)
#		self.rfunc = lambda self, theta: rfunc(theta)
#		self.drfunc = lambda self, theta: drfunc(theta)
#		self.compute_helper_vars = lambda self, fit_obj: pass
#		
#	def gen_deriv(self,dexpr,symloc):
#		dexpr_list = [d.evalf(subs={s:theta[symloc[s] for s in symloc.keys()}) for d in dexpr]
#		return np.concatenate(dexpr_list,axis=0)

class model_from_rate_function:
    # strategy: pass in a function that takes N arguments, and a list of N argument shapes
    def __init__(self,ratefunc,sizedict,helper_vars):
    	# expr a sympy expression; symbols a list of sympy symbols
    	# symsize = {s:s.size for s in symbols} # how many DOF in each model component
    	# sizelist = np.array([s.size for s in symbols]) # array containing number of DOFs per symbol
        self.paramlist = list(sizedict.keys())
        sizelist = [sizedict[x] for x in self.paramlist]
        bds = np.cumsum(np.concatenate(((0,),sizelist)))
        self.nparams = bds[-1]
        self.argslice = [slice(bds[i],bds[i+1]) for i in range(len(sizelist))] # for each symbol, value is the slice of theta subtended by that symbol's values
        self.slicedict = {key:val for key,val in zip(self.paramlist,self.argslice)}
        self.ratefunc = lambda theta: ratefunc(*[theta[slc] for slc in self.argslice])
        self.dratefunc = jacobian(self.ratefunc)
        #rfunc = lambda theta: ratefunc(*[theta[slc] for slc in self.argslice])
        #drfunc = lambda theta: jacobian(rfunc)(theta).transpose((2,0,1))
        #self.rfunc = lambda self, theta: rfunc(theta)
        #self.drfunc = lambda self, theta: drfunc(theta)
        self.helper_vars = helper_vars
    	#self.compute_helper_vars = compute_helper_vars
    
    def rfunc(self,theta):
        return self.ratefunc(theta)

    def drfunc(self,theta):
        return self.dratefunc(theta).transpose((3,0,1,2))
    
    def compute_helper_vars(self,fit_obj):
        self.helper_vars['theta_guess'] = np.zeros((self.nparams,))
        self.helper_vars['bounds'] = [None]*self.nparams
        for key in self.helper_vars['bounds_dict']:
            if not isinstance(self.helper_vars['bounds_dict'][key],list):
                self.helper_vars['bounds_dict'][key] = [self.helper_vars['bounds_dict'][key]]
        for i in range(len(self.argslice)):
            self.helper_vars['theta_guess'][self.argslice[i]] = self.helper_vars['theta_guess_dict'][self.paramlist[i]]
            #if len(self.helper_vars['bounds'][self.argslice[i]])>1:
            self.helper_vars['bounds'][self.argslice[i]] = self.helper_vars['bounds_dict'][self.paramlist[i]]
            #else:
            #    self.helper_vars['bounds'][self.argslice[i]] = [self.helper_vars['bounds_dict'][self.paramlist[i]]]
