#!/usr/bin/env

import calnet.utils
import autograd.numpy as np

class Model(object):
    
    Wmx = None
    Wmy = None
    Wsx = None
    Wsy = None
    s02 = None
    k = None
    kappa = None
    XX = None
    XXp = None
    Eta = None
    Xi = None
    rate_f = None
    rate_fprime = None
    nP = 0
    nQ = 0
    nS = 0
    nT = 0
    
    def __init__(self, Wdict, rate_f=calnet.utils.f_miller_troyer, rate_fprime=calnet.utils.fprime_m_miller_troyer, u_fn=calnet.utils.u_fn_WW):
        # Wmx,Wmy,Wsx,Wsy,s02,k,kappa,XX,XXp,Eta,Xi
        for key in Wdict:
            setattr(self,key,Wdict[key])
            
        self.rate_f = rate_f
        self.rate_fprime = rate_fprime
        self.u_fn = u_fn
        
        self.nP = self.Wmx.shape[0]
        self.nQ = self.Wmy.shape[0]
        self.nS = int(np.round(self.Eta.shape[1]/self.nQ))
        self.nT = 1
        self.nN = self.Eta.shape[0]
        
        wws = ['WWmx','WWmy','WWsx','WWsy']
        ws = ['Wmx','Wmy','Wsx','Wsy']
        for w,ww in zip(ws,wws):
            W = getattr(self,w)
            WW = calnet.utils.gen_Weight_k_kappa(W,self.k,self.kappa)
            setattr(self,ww,WW)
        
        self.YY = self.compute_f_(self.Eta,self.Xi,self.s02)
        self.resEta = self.Eta - self.u_fn_m(self.XX,self.YY)
        self.resXi = self.Xi - self.u_fn_s(self.XX,self.YY)
        
    def compute_f_(self,Eta,Xi,s02):
        return self.rate_f(Eta,Xi**2+np.concatenate([s02 for ipixel in range(self.nS*self.nT)],axis=0))
        
    def u_fn_m(self,XX,YY):
        return self.u_fn(XX,YY,self.WWmx,self.WWmy)
        
    def u_fn_s(self,XX,YY):
        return self.u_fn(XX,YY,self.WWsx,self.WWsy)
        
    def fXY(self,XX,YY,istim=None,res_factor=1.):
        if istim is None:
            Eta = res_factor*self.resEta + self.u_fn_m(XX,YY)
            Xi = res_factor*self.resXi + self.u_fn_s(XX,YY)
        else:
            Eta = res_factor*self.resEta[istim] + self.u_fn_m(XX,YY)
            Xi = res_factor*self.resXi[istim] + self.u_fn_s(XX,YY)
        return self.compute_f_(Eta,Xi,self.s02)
     
    def fY(self,YY,istim=None,residuals_on=True):
        if residuals_on:
            return self.fXY(self.XX[istim],YY,istim=istim,res_factor=1.)
        else:
            return self.fXY(self.XX[istim],YY,istim=istim,res_factor=0.)    

class ModelOri(Model):
    def __init__(self, Wdict, rate_f=calnet.utils.f_miller_troyer, rate_fprime=calnet.utils.fprime_m_miller_troyer, u_fn=calnet.utils.u_fn_WW,nS=2,nT=2):
        # Wmx,Wmy,Wsx,Wsy,s02,k,kappa,T,XX,XXp,Eta,Xi
        for key in Wdict:
            setattr(self,key,Wdict[key])
            
        self.rate_f = rate_f
        self.rate_fprime = rate_fprime
        self.u_fn = u_fn
        
        self.nP = self.Wmx.shape[0]
        self.nQ = self.Wmy.shape[0]
        self.nS = nS
        self.nT = nT
        self.nN = self.Eta.shape[0]
        
        wws = ['WWmx','WWmy','WWsx','WWsy']
        ws = ['Wmx','Wmy','Wsx','Wsy']
        for w,ww in zip(ws,wws):
            W = getattr(self,w)
            WW = calnet.utils.gen_Weight_k_kappa_t(W,self.k,self.kappa,self.T)
            setattr(self,ww,WW)
        
        self.YY = self.compute_f_(self.Eta,self.Xi,self.s02)
        self.resEta = self.Eta - self.u_fn_m(self.XX,self.YY)
        self.resXi = self.Xi - self.u_fn_s(self.XX,self.YY)
        
class Dataset(object):
    def __init__(self,dsfiles=[],modal_uparams=[]):
        print('done')
# class Classifier(object):
#     def __init__(self):
        
