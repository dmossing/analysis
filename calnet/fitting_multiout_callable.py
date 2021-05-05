#!/usr/bin/env python
import autograd.numpy as np
import scipy.optimize as sop
from autograd import elementwise_grad as egrad
from autograd import grad
from autograd import jacobian
from autograd import hessian
import matplotlib.pyplot as plt
from calnet import utils
import scipy.signal as ssi

# Yhat is all measured tuning curves, Y is the averages of the model tuning curves
def parse_W1(W,opt):
    shapes1 = opt['shapes1']
    Ws = utils.parse_thing(W,shapes1)
    return Ws

def parse_W2(W,opt):
    shapes2 = opt['shapes2']
    Ws = utils.parse_thing(W,shapes2)
    return Ws

def sorted_r_eigs(w):
    drW,prW = np.linalg.eig(w)
    srtinds = np.argsort(drW)
    return drW[srtinds],prW[:,srtinds]

def u_fn(XX,YY,Wx,Wy,K,kappa,T,opt,power=True):
    WWx,WWy = [gen_Weight(W,K,kappa,T,opt,power=power) for W in [Wx,Wy]]
    return XX @ WWx + YY @ WWy

def u_fn_in_out(XX,YY,Wx,Wy,Kin,Kxout,Kyout,kappa,Tin,Txout,Tyout,opt,power=True):
    WWx,WWy = [gen_Weight_in_out(W,Kin,Kout,kappa,Tin,Tout,opt,power=power) for [W,Kout,Tout] in zip([Wx,Wy],[Kxout,Kyout],[Txout,Tyout])]
    return XX @ WWx + YY @ WWy

def u_fn_WW(XX,YY,WW):
    XXYY = np.concatenate((XX,YY),axis=1)
    return XXYY @ WW

def unparse_W(*Ws):
    return np.concatenate([ww.flatten() for ww in Ws])

def normalize(arr,opt):
    arrsum = arr.sum(1)
    well_behaved = (arrsum>0)[:,np.newaxis]
    arrnorm = well_behaved*arr/arrsum[:,np.newaxis] + (~well_behaved)*np.ones_like(arr)/arr.shape[1]
    return arrnorm

def gen_Weight(W,K,kappa,T,opt,power=True):
    nS,nT = opt['nS'],opt['nT']
    return utils.gen_Weight_k_kappa_t(W,K,kappa,T,nS=nS,nT=nT,power=power) 

def gen_Weight_in_out(W,Kin,Kout,kappa,Tin,Tout,opt,power=True):
    nS,nT = opt['nS'],opt['nT']
    return utils.gen_Weight_in_out_k_kappa_t(W,Kin,Kout,kappa,Tin,Tout,nS=nS,nT=nT,power=power) 

def gen_Weight_flex(*args,**kwargs):
    opt = args[-1]
    if 'axon' in opt and opt['axon']:
        Wx,Wy,Kin,Kxout,Kyout,kappa,Tin,Txout,Tyout,opt = args
        WWx,WWy = [gen_Weight_in_out(W,Kin,Kout,kappa,Tin,Tout,opt,**kwargs) for [W,Kout,Tout] in zip([Wx,Wy],[Kxout,Kyout],[Txout,Tyout])]
        return WWx,WWy
    else:
        W0x,W0y,K,kappa,T,opt = args
        WWx,WWy = [gen_Weight(W,K,kappa,T,**kwargs) for W in [Wx,Wy]]
        return WWx,WWy

def gen_Weight_c(*args,**kwargs):
    return np.concatenate(gen_Weight_flex(*args,**kwargs),axis=0) 
    
def compute_kl_divergence(stim_deriv,noise,mu_data,mu_model,pc_list,opt):
    nQ,nS,nT,foldT = opt['nQ'],opt['nS'],opt['nT'],opt['foldT']
    return utils.compute_kl_divergence(stim_deriv,noise,mu_data,mu_model,pc_list,nQ=nQ,nS=nS,nT=nT,foldT=foldT)

def compute_var(Xi,s02,opt):
    nS,nT,fudge = opt['nS'],opt['nT'],opt['fudge']
    return fudge+Xi**2+np.concatenate([s02 for ipixel in range(nS*nT)],axis=0)

def compute_fprime_(Eta,Xi,s02,opt):
    pop_deriv_fn = opt['pop_deriv_fn']
    return pop_deriv_fn(Eta,compute_var(Xi,s02,opt))*Xi

def compute_f_(Eta,Xi,s02,opt):
    pop_rate_fn = opt['pop_rate_fn']
    return pop_rate_fn(Eta,compute_var(Xi,s02,opt))

def W1_W2_from_list(Wlist):
    W0x,W0y,W1x,W1y,W2x,W2y,W3x,W3y,s02,K0,K1,K2,K3,kappa,T0,T1,T2,T3,XX,XXp,Eta,Xi,h1,h2,bl,amp = Wlist
    W1 = unparse_W(W0x,W0y,W1x,W1y,W2x,W2y,W3x,W3y,s02,K0,K1,K2,K3,kappa,T0,T1,T2,T3,h1,h2,bl,amp)
    W2 = unparse_W(XX,XXp,Eta,Xi)
    return W1,W2

def compute_WWs(W1,opt):
    if 'axon' in opt and opt['axon']:
        return compute_WWs_in_out(W1,opt)
    W0x,W0y,W1x,W1y,W2x,W2y,W3x,W3y,s02,K0,K1,K2,K3,kappa,T0,T1,T2,T3,h1,h2,bl,amp = parse_W1(W1,opt)
    nN = opt['nN']
    nondim = opt['nondim']
    if nondim:
        W1x,W1y,W2x,W2y,W3x,W3y,K1,K2,K3,T1,T2,T3 = [x*y for x,y in zip([W1x,W1y,W2x,W2y,W3x,W3y,K1,K2,K3,T1,T2,T3],[W0x,W0y,W0x,W0y,W0x,W0y,K0,K0,K0,T0,T0,T0])]
    power0 = True
    power1 = False
    WW0 = gen_Weight_c(W0x,W0y,K0,kappa,T0,opt,power=power0)
    WW1 = gen_Weight_c(W1x,W1y,K0,kappa,T0,opt,power=power0) + gen_Weight_c(W0x,W0y,K1,kappa,T0,opt,power=power1) + gen_Weight_c(W0x,W0y,K0,kappa,T1,opt,power=power1)
    WW2 = gen_Weight_c(W2x,W2y,K0,kappa,T0,opt,power=power0) + gen_Weight_c(W0x,W0y,K2,kappa,T0,opt,power=power1) + gen_Weight_c(W0x,W0y,K0,kappa,T2,opt,power=power1)
    WW3 = gen_Weight_c(W3x,W3y,K0,kappa,T0,opt,power=power0) + gen_Weight_c(W0x,W0y,K3,kappa,T0,opt,power=power1) + gen_Weight_c(W0x,W0y,K0,kappa,T3,opt,power=power1)
    return WW0,WW1,WW2,WW3

def compute_WWs_in_out(W1,opt):
    W0x,W0y,W1x,W1y,W2x,W2y,W3x,W3y,s02,Kin0,Kin1,Kxout0,Kyout0,Kxout1,Kyout1,kappa,Tin0,Tin1,Txout0,Tyout0,Txout1,Tyout1,h1,h2,bl,amp = parse_W1(W1,opt)
    nondim = opt['nondim']
    if nondim:
        W1x,W1y,W2x,W2y,W3x,W3y,Kin1,Kxout1,Kyout1,Tin1,Txout1,Tyout1 = [x*y for x,y in zip([W1x,W1y,W2x,W2y,W3x,W3y,Kin1,Kxout1,Kyout1,Tin1,Txout1,Tyout1],[W0x,W0y,W0x,W0y,W0x,W0y,Kin0,Kxout0,Kyout0,Tin0,Txout0,Tyout0])]
    power0 = True
    power1 = False

    WW0 = gen_Weight_c(W0x,W0y,Kin0,Kxout0,Kyout0,kappa,Tin0,Txout0,Tyout0,opt,power=power0)
    WW1 = gen_Weight_c(W1x,W1y,Kin0,Kxout0,Kyout0,kappa,Tin0,Txout0,Tyout0,opt,power=power0) + gen_Weight_c(W0x,W0y,Kin1,Kxout0,Kyout0,kappa,Tin0,Txout0,Tyout0,opt,power=power1) + gen_Weight_c(W0x,W0y,Kin0,Kxout0,Kyout0,kappa,Tin1,Txout0,Tyout0,opt,power=power1)
    WW2 = gen_Weight_c(W2x,W2y,Kin0,Kxout0,Kyout0,kappa,Tin0,Txout0,Tyout0,opt,power=power0) + gen_Weight_c(W0x,W0y,Kin0,Kxout1,Kyout1,kappa,Tin0,Txout0,Tyout0,opt,power=power1) + gen_Weight_c(W0x,W0y,Kin0,Kxout0,Kyout0,kappa,Tin0,Txout1,Tyout1,opt,power=power1)
    WW3 = gen_Weight_c(W3x,W3y,Kin0,Kxout0,Kyout0,kappa,Tin0,Txout0,Tyout0,opt,power=power0)
    return WW0,WW1,WW2,WW3

def compute_us(W1,W2,fval,fprimeval,opt):
    if 'axon' in opt and opt['axon']:
        return compute_us_in_out(W1,W2,fval,fprimeval,opt)
    #W0x,W0y,W1x,W1y,W2x,W2y,W3x,W3y,s02,K0,K1,K2,K3,kappa,T0,T1,T2,T3,h1,h2,bl,amp = parse_W1(W1,opt)
    XX,XXp,Eta,Xi = parse_W2(W2,opt)
    
    WWlist = compute_WWs(W1,opt)

    nN = opt['nN']
    #nondim = opt['nondim']
    #if nondim:
    #    W1x,W1y,W2x,W2y,W3x,W3y,K1,K2,K3,T1,T2,T3 = [x*y for x,y in zip([W1x,W1y,W2x,W2y,W3x,W3y,K1,K2,K3,T1,T2,T3],[W0x,W0y,W0x,W0y,W0x,W0y,K0,K0,K0,T0,T0,T0])]
    if fval.shape[0]==2*nN:
        XX = np.concatenate((XX,XX),axis=0)
        XXp = np.concatenate((XXp,XXp),axis=0)
    power0 = True
    power1 = False
    #WWlist = compute_WWs(W1,W2,opt)
    u0,u1,u2,u3 = [u_fn_WW(x,y,WW) for x,y,WW in zip([XX,XX,XXp,XXp],[fval,fval,fprimeval,fprimeval],WWlist)]
    #u0 = u_fn(XX,fval,W0x,W0y,K0,kappa,T0,opt,power=power0)
    #u1 = u_fn(XX,fval,W1x,W1y,K0,kappa,T0,opt,power=power0) + u_fn(XX,fval,W0x,W0y,K1,kappa,T0,opt,power=power1) + u_fn(XX,fval,W0x,W0y,K0,kappa,T1,opt,power=power1)
    #u2 = u_fn(XXp,fprimeval,W2x,W2y,K0,kappa,T0,opt,power=power0) + u_fn(XXp,fprimeval,W0x,W0y,K2,kappa,T0,opt,power=power1) + u_fn(XXp,fprimeval,W0x,W0y,K0,kappa,T2,opt,power=power1)
    #u3 = u_fn(XXp,fprimeval,W3x,W3y,K0,kappa,T0,opt,power=power0) + u_fn(XXp,fprimeval,W0x,W0y,K3,kappa,T0,opt,power=power1) + u_fn(XXp,fprimeval,W0x,W0y,K0,kappa,T3,opt,power=power1)
    return u0,u1,u2,u3

def compute_us_in_out(W1,W2,fval,fprimeval,opt):
    run_modulation = ('run_modulation' in opt) and opt['run_modulation']
    if run_modulation:
        W0x,W0y,W1x,W1y,W2x,W2y,W3x,W3y,W0xrun,W0yrun,s02,Kin0,Kin1,Kxout0,Kyout0,Kxout1,Kyout1,kappa,Tin0,Tin1,Txout0,Tyout0,Txout1,Tyout1,h1,h2,bl,amp = parse_W1(W1,opt)
    else:
        W0x,W0y,W1x,W1y,W2x,W2y,W3x,W3y,s02,Kin0,Kin1,Kxout0,Kyout0,Kxout1,Kyout1,kappa,Tin0,Tin1,Txout0,Tyout0,Txout1,Tyout1,h1,h2,bl,amp = parse_W1(W1,opt)
    XX,XXp,Eta,Xi = parse_W2(W2,opt)
    
    WWlist = compute_WWs(W1,opt)

    nN = opt['nN']
    nondim = opt['nondim']
    if nondim:
        #if run_modulation:
        #    W1x,W1y,W2x,W2y,W3x,W3y,W0xrun,W0yrun,Kin1,Kxout1,Kyout1,Tin1,Txout1,Tyout1 = [x*y for x,y in zip([W1x,W1y,W2x,W2y,W3x,W3y,W0xrun,W0yrun,Kin1,Kxout1,Kyout1,Tin1,Txout1,Tyout1],[W0x,W0y,W0x,W0y,W0x,W0y,W0x,W0y,Kin0,Kxout0,Kyout0,Tin0,Txout0,Tyout0])]
        #else:
        W1x,W1y,W2x,W2y,W3x,W3y,Kin1,Kxout1,Kyout1,Tin1,Txout1,Tyout1 = [x*y for x,y in zip([W1x,W1y,W2x,W2y,W3x,W3y,Kin1,Kxout1,Kyout1,Tin1,Txout1,Tyout1],[W0x,W0y,W0x,W0y,W0x,W0y,Kin0,Kxout0,Kyout0,Tin0,Txout0,Tyout0])]

    if fval.shape[0]==2*nN:
        XX = np.concatenate((XX,XX),axis=0)
        XXp = np.concatenate((XXp,XXp),axis=0)
    power0 = True
    power1 = False

    if run_modulation:
        u0,u1,u2,u3 = [np.zeros(fval.shape) for _ in range(4)]
        run_condition = XX[:,2]
        W0xs,W1xs,W2xs,W3xs = [[Wx,Wx*(1+W0xrun)] for Wx in [W0x,W1x,W2x,W3x]]
        W0ys,W1ys,W2ys,W3ys = [[Wy,Wy*(1+W0yrun)] for Wy in [W0y,W1y,W2y,W3y]]
        for irun,crit in enumerate([1-run_condition[:,np.newaxis],run_condition[:,np.newaxis]]):
            u0 = u0 + crit*(u_fn_in_out(XX,fval,W0xs[irun],W0ys[irun],Kin0,Kxout0,Kyout0,kappa,Tin0,Txout0,Tyout0,opt,power=power0))
            u1 = u1 + crit*(u_fn_in_out(XX,fval,W1xs[irun],W1ys[irun],Kin0,Kxout0,Kyout0,kappa,Tin0,Txout0,Tyout0,opt,power=power0) + u_fn_in_out(XX,fval,W0xs[irun],W0ys[irun],Kin1,Kxout0,Kyout0,kappa,Tin0,Txout0,Tyout0,opt,power=power1) + u_fn_in_out(XX,fval,W0xs[irun],W0ys[irun],Kin0,Kxout0,Kyout0,kappa,Tin1,Txout0,Tyout0,opt,power=power1))
            u2 = u2 + crit*(u_fn_in_out(XXp,fprimeval,W2xs[irun],W2ys[irun],Kin0,Kxout0,Kyout0,kappa,Tin0,Txout0,Tyout0,opt,power=power0) + u_fn_in_out(XXp,fprimeval,W0xs[irun],W0ys[irun],Kin0,Kxout1,Kyout1,kappa,Tin0,Txout0,Tyout0,opt,power=power1) + u_fn_in_out(XXp,fprimeval,W0xs[irun],W0ys[irun],Kin0,Kxout0,Kyout0,kappa,Tin0,Txout1,Tyout1,opt,power=power1))
            u3 = u3 + crit*(u_fn_in_out(XXp,fprimeval,W3xs[irun],W3ys[irun],Kin0,Kxout0,Kyout0,kappa,Tin0,Txout0,Tyout0,opt,power=power0))
    else:
        u0,u1,u2,u3 = [u_fn_WW(x,y,WW) for x,y,WW in zip([XX,XX,XXp,XXp],[fval,fval,fprimeval,fprimeval],WWlist)]
        #u0 = u_fn_in_out(XX,fval,W0x,W0y,Kin0,Kxout0,Kyout0,kappa,Tin0,Txout0,Tyout0,opt,power=power0)
        #u1 = u_fn_in_out(XX,fval,W1x,W1y,Kin0,Kxout0,Kyout0,kappa,Tin0,Txout0,Tyout0,opt,power=power0) + u_fn_in_out(XX,fval,W0x,W0y,Kin1,Kxout0,Kyout0,kappa,Tin0,Txout0,Tyout0,opt,power=power1) + u_fn_in_out(XX,fval,W0x,W0y,Kin0,Kxout0,Kyout0,kappa,Tin1,Txout0,Tyout0,opt,power=power1)
        #u2 = u_fn_in_out(XXp,fprimeval,W2x,W2y,Kin0,Kxout0,Kyout0,kappa,Tin0,Txout0,Tyout0,opt,power=power0) + u_fn_in_out(XXp,fprimeval,W0x,W0y,Kin0,Kxout1,Kyout1,kappa,Tin0,Txout0,Tyout0,opt,power=power1) + u_fn_in_out(XXp,fprimeval,W0x,W0y,Kin0,Kxout0,Kyout0,kappa,Tin0,Txout1,Tyout1,opt,power=power1)
        #u3 = u_fn_in_out(XXp,fprimeval,W3x,W3y,Kin0,Kxout0,Kyout0,kappa,Tin0,Txout0,Tyout0,opt,power=power0)
    return u0,u1,u2,u3

def compute_res(W1,W2,opt):
    W0x,W0y,W1x,W1y,W2x,W2y,W3x,W3y,s02,K0,K1,K2,K3,kappa,T0,T1,T2,T3,h1,h2,bl,amp = parse_W1(W1,opt)
    XX,XXp,Eta,Xi = parse_W2(W2,opt)
    fval = compute_f_(Eta,Xi,s02,opt)
    fprimeval = compute_fprime_(Eta,Xi,s02,opt)
    u0,u1,u2,u3 = compute_us(W1,W2,fval,fprimeval,opt)
    resEta = Eta - u0 - u2
    resXi  = Xi - u1 - u3
    return resEta,resXi

def compute_f_fprime_(W1,W2,opt):
    W0x,W0y,W1x,W1y,W2x,W2y,W3x,W3y,s02,K0,K1,K2,K3,kappa,T0,T1,T2,T3,h1,h2,bl,amp = parse_W1(W1,opt)
    XX,XXp,Eta,Xi = parse_W2(W2,opt)
    return compute_f_(Eta,Xi,s02,opt),compute_fprime_(Eta,Xi,s02,opt)

def compute_f_fprime_t_(W1,W2,perturbation,max_dist=1,opt=None): # max dist added 10/14/20
    dt,niter = opt['dt'],opt['niter']
    W0x,W0y,W1x,W1y,W2x,W2y,W3x,W3y,s02,K0,K1,K2,K3,kappa,T0,T1,T2,T3,h1,h2,bl,amp = parse_W1(W1,opt)
    XX,XXp,Eta,Xi = parse_W2(W2,opt)
    fval = compute_f_(Eta,Xi,s02,opt)
    fprimeval = compute_fprime_(Eta,Xi,s02,opt)
    u0,u1,u2,u3 = compute_us(W1,W2,fval,fprimeval,opt)
    resEta = Eta - u0 - u2
    resXi  = Xi - u1 - u3
    YY = fval + perturbation
    YYp = fprimeval
    def dYYdt(YY,Eta1,Xi1,opt):
        return -YY + compute_f_(Eta1,Xi1,s02,opt)
    def dYYpdt(YYp,Eta1,Xi1,opt):
        return -YYp +compute_fprime_(Eta1,Xi1,s02,opt)
    for t in range(niter):
        if np.mean(np.abs(YY-fval)) < max_dist:
            u0,u1,u2,u3 = compute_us(W1,W2,YY,YYp,opt)
            Eta1 = resEta + u0 + u2
            Xi1 = resXi + u1 + u3
            YY = YY + dt*dYYdt(YY,Eta1,Xi1,opt)
            YYp = YYp + dt*dYYpdt(YYp,Eta1,Xi1,opt)
        elif np.remainder(t,500)==0:
            print('unstable fixed point?')
    
    return YY,YYp

def compute_f_fprime_t_12_(W1,W2,perturbation,max_dist=1,opt=None): # max dist added 10/14/20
    dt,niter = opt['dt'],opt['niter']
    W0x,W0y,W1x,W1y,W2x,W2y,W3x,W3y,s02,K0,K1,K2,K3,kappa,T0,T1,T2,T3,h1,h2,bl,amp = parse_W1(W1,opt)
    XX,XXp,Eta,Xi = parse_W2(W2,opt)
    fval = compute_f_(Eta,Xi,s02,opt)
    fprimeval = compute_fprime_(Eta,Xi,s02,opt)
    if share_residuals:
        u0,u1,u2,u3 = compute_us(W1,W2,fval,fprimeval,opt)
        resEta = Eta - u0 - u2
        resXi  = Xi - u1 - u3
        resEta12 = np.concatenate((resEta,resEta),axis=0)
        resXi12 = np.concatenate((resXi,resXi),axis=0)
    else:
        resEta12 = 0
        resXi12 = 0
    nQ,nS,nT = [opt[key] for key in ['nQ','nS','nT']]
    dHH = np.zeros((nN,nQ*nS*nT))
    dHH[:,np.arange(2,nQ*nS*nT,nQ)] = 1
    dHH = np.concatenate((dHH*h1,dHH*h2),axis=0)
    YY = fval + perturbation
    YYp = fprimeval
    XX12 = np.concatenate((XX,XX),axis=0)
    YY12 = np.concatenate((YY,YY),axis=0)
    YYp12 = np.concatenate((YYp,YYp),axis=0)
    def dYYdt(YY,Eta1,Xi1,opt):
        return -YY + compute_f_(Eta1,Xi1,s02,opt)
    def dYYpdt(YYp,Eta1,Xi1,opt):
        return -YYp + compute_fprime_(Eta1,Xi1,s02,opt)
    for t in range(niter):
        if np.mean(np.abs(YY-fval)) < max_dist:
            u0,u1,u2,u3 = compute_us(W1,W2,YY12,YYp12,opt)
            Eta121 = resEta12 + u0 + u2 + dHH
            Xi121 = resXi12 + u1 + u3
            YY12 = YY12 + dt*dYYdt(YY12,Eta121,Xi121,opt)
            YYp12 = YYp12 + dt*dYYpdt(YYp12,Eta121,Xi121,opt)
        elif np.remainder(t,500)==0:
            print('unstable fixed point?')

    #YY12 = YY12 + np.tile(bl,nS*nT)[np.newaxis,:]
    
    return YY12,YYp12

def compute_f_fprime_t_avg_(W1,W2,perturbation,burn_in=0.5,max_dist=1,opt=None):
    dt,niter,axon = opt['dt'],opt['niter'],opt['axon']
    if not axon:
        W0x,W0y,W1x,W1y,W2x,W2y,W3x,W3y,s02,K0,K1,K2,K3,kappa,T0,T1,T2,T3,h1,h2,bl,amp = parse_W1(W1,opt)
    else:
        W0x,W0y,W1x,W1y,W2x,W2y,W3x,W3y,s02,Kin0,Kin1,Kxout0,Kyout0,Kxout1,Kyout1,kappa,Tin0,Tin1,Txout0,Tyout0,Txout1,Tyout1,h1,h2,bl,amp = parse_W1(W1,opt)
    XX,XXp,Eta,Xi = parse_W2(W2,opt)
    fval = compute_f_(Eta,Xi,s02,opt)
    fprimeval = compute_fprime_(Eta,Xi,s02,opt)
    u0,u1,u2,u3 = compute_us(W1,W2,fval,fprimeval,opt)
    resEta = Eta - u0 - u2
    resXi  = Xi - u1 - u3
    YY = fval + perturbation
    YYp = fprimeval
    YYmean = np.zeros_like(Eta)
    YYprimemean = np.zeros_like(Eta)
    def dYYdt(YY,Eta1,Xi1,opt):
        return -YY + compute_f_(Eta1,Xi1,s02,opt)
    def dYYpdt(YYp,Eta1,Xi1,opt):
        return -YYp + compute_fprime_(Eta1,Xi1,s02,opt)
    for t in range(niter):
        if np.mean(np.abs(YY-fval)) < max_dist:
            u0,u1,u2,u3 = compute_us(W1,W2,YY,YYp,opt)
            Eta1 = resEta + u0 + u2
            Xi1 = resXi + u1 + u3
            YY = YY + dt*dYYdt(YY,Eta1,Xi1,opt)
            YYp = YYp + dt*dYYpdt(YYp,Eta1,Xi1,opt)
        else:
            print('unstable fixed point?')
        if t>niter*burn_in:
            YYmean = YYmean + 1/niter/burn_in*YY
            YYprimemean = YYprimemean + 1/niter/burn_in*YYp

    return YYmean,YYprimemean

def compute_f_fprime_t_avg_12_(W1,W2,perturbation,max_dist=1,burn_in=0.5,opt=None): # max dist added 10/14/20
    dt,niter = opt['dt'],opt['niter']
    W0x,W0y,W1x,W1y,W2x,W2y,W3x,W3y,s02,K0,K1,K2,K3,kappa,T0,T1,T2,T3,h1,h2,bl,amp = parse_W1(W1,opt)
    XX,XXp,Eta,Xi = parse_W2(W2,opt)
    fval = compute_f_(Eta,Xi,s02,opt)
    fprimeval = compute_fprime_(Eta,Xi,s02,opt)
    u0,u1,u2,u3 = compute_us(W1,W2,fval,fprimeval,opt)
    if share_residuals:
        resEta = Eta - u0 - u2
        resXi  = Xi - u1 - u3
        resEta12 = np.concatenate((resEta,resEta),axis=0)
        resXi12 = np.concatenate((resXi,resXi),axis=0)
    else:
        resEta12 = 0
        resXi12 = 0
    nQ,nS,nT = [opt[key] for key in ['nQ','nS','nT']]
    dHH = np.zeros((nN,nQ*nS*nT))
    dHH[:,np.arange(2,nQ*nS*nT,nQ)] = 1
    dHH = np.concatenate((dHH*h1,dHH*h2),axis=0)
    YY = fval + perturbation
    YYp = fprimeval
    XX12 = np.concatenate((XX,XX),axis=0)
    YY12 = np.concatenate((YY,YY),axis=0)
    YYp12 = np.concatenate((YYp,YYp),axis=0)
    YYmean = np.zeros_like(YY12)
    YYprimemean = np.zeros_like(YY12)
    def dYYdt(YY,Eta1,Xi1,opt):
        return -YY + compute_f_(Eta1,Xi1,s02,opt)
    def dYYpdt(YYp,Eta1,Xi1,opt):
        return -YYp + compute_fprime_(Eta1,Xi1,s02,opt)
    for t in range(niter):
        if np.mean(np.abs(YY-fval)) < max_dist:
            u0,u1,u2,u3 = compute_us(W1,W2,YY12,YYp12,opt)
            Eta121 = resEta12 + u0 + u2 + dHH
            Xi121 = resXi12 + u1 + u3
            YY12 = YY12 + dt*dYYdt(YY12,Eta121,Xi121,opt)
            YYp12 = YYp12 + dt*dYYpdt(YYp12,Eta121,Xi121,opt)
        elif np.remainder(t,500)==0:
            print('unstable fixed point?')
        if t>niter*burn_in:
            YYmean = YYmean + 1/niter/burn_in*YY12
            YYprimemean = YYprimemean + 1/niter/burn_in*YYp12

    return YYmean,YYprimemean

def gen_opt(nN=36,nP=2,nQ=4,nS=2,nT=2,foldT=True,pop_rate_fn=utils.f_miller_troyer,pop_deriv_fn=utils.fprime_miller_troyer,fudge=1e-4,dt=1e-1,niter=100,nondim=False):

    shapes1 = [(nP,nQ),(nQ,nQ),(nP,nQ),(nQ,nQ),(nP,nQ),(nQ,nQ),(nP,nQ),(nQ,nQ),(nQ,),(nQ*(nS>1),),(nQ*(nS>1),),(nQ*(nS>1),),(nQ*(nS>1),),(1,),(nQ*(nT>1),),(nQ*(nT>1),),(nQ*(nT>1),),(nQ*(nT>1),),(1,),(1,),(nQ,),(nT*nS*nQ,)]
    shapes2 = [(nN,nT*nS*nP),(nN,nT*nS*nP),(nN,nT*nS*nQ),(nN,nT*nS*nQ)]

    opt = {}
    keys = ['nN','nP','nQ','nS','nT','shapes1','shapes2','foldT','pop_rate_fn','pop_deriv_fn','fudge','dt','niter','nondim','axon']
    vals = [nN,nP,nQ,nS,nT,shapes1,shapes2,foldT,pop_rate_fn,pop_deriv_fn,fudge,dt,niter,nondim,False]
    for key,val in zip(keys,vals):
        opt[key] = val

    return opt

def gen_opt_axon(nN=36,nP=2,nQ=4,nS=2,nT=2,foldT=True,pop_rate_fn=utils.f_miller_troyer,pop_deriv_fn=utils.fprime_miller_troyer,fudge=1e-4,dt=1e-1,niter=100,nondim=False,run_modulation=False,axon=True):

    if not run_modulation:
        shapes1 = [(nP,nQ),(nQ,nQ),(nP,nQ),(nQ,nQ),(nP,nQ),(nQ,nQ),(nP,nQ),(nQ,nQ),(nQ,),(nQ*(nS>1),),(nQ*(nS>1),),(nP*(nS>1),),(nQ*(nS>1),),(nP*(nS>1),),(nQ*(nS>1),),(1,),(nQ*(nT>1),),(nQ*(nT>1),),(nP*(nT>1),),(nQ*(nT>1),),(nP*(nT>1),),(nQ*(nT>1),),(1,),(1,),(nQ,),(nT*nS*nQ,)]
    else:
        shapes1 = [(nP,nQ),(nQ,nQ),(nP,nQ),(nQ,nQ),(nP,nQ),(nQ,nQ),(nP,nQ),(nQ,nQ),(nP,nQ),(nQ,nQ),(nQ,),(nQ*(nS>1),),(nQ*(nS>1),),(nP*(nS>1),),(nQ*(nS>1),),(nP*(nS>1),),(nQ*(nS>1),),(1,),(nQ*(nT>1),),(nQ*(nT>1),),(nP*(nT>1),),(nQ*(nT>1),),(nP*(nT>1),),(nQ*(nT>1),),(1,),(1,),(nQ,),(nT*nS*nQ,)]
    shapes2 = [(nN,nT*nS*nP),(nN,nT*nS*nP),(nN,nT*nS*nQ),(nN,nT*nS*nQ)]

    opt = {}
    keys = ['nN','nP','nQ','nS','nT','shapes1','shapes2','foldT','pop_rate_fn','pop_deriv_fn','fudge','dt','niter','nondim','run_modulation','axon']
    vals = [nN,nP,nQ,nS,nT,shapes1,shapes2,foldT,pop_rate_fn,pop_deriv_fn,fudge,dt,niter,nondim,run_modulation,axon]
    for key,val in zip(keys,vals):
        opt[key] = val

    return opt


def fit_W_sim(Xhat,Xpc_list,Yhat,Ypc_list,dYY,pop_rate_fn=None,pop_deriv_fn=None,neuron_rate_fn=None,W10list=None,W20list=None,bounds1=None,bounds2=None,dt=1e-1,perturbation_size=5e-2,niter=1,wt_dict=None,eta=0.1,compute_hessian=False,l2_penalty=1.0,constrain_isn=False,opto_mask=None,nsize=6,ncontrast=6,coupling_constraints=[(1,0,-1)],tv=False,topo_stims=np.arange(36),topo_shape=(6,6),use_opto_transforms=False,opto_transform1=None,opto_transform2=None,share_residuals=False,stimwise=False,simulate1=True,simulate2=False,verbose=True,foldT=True):
    # coupling constraints: (i,j,sgn) --> coupling term i->j is constrained to be > 0 (sgn=1) or < 0 (sgn=-1)

    fudge = 1e-4
    noise = 1
    big_val = 1e6
    
    fprime_m = pop_deriv_fn #utils.fprime_miller_troyer #egrad(pop_rate_fn,0)
    
    YYhat = utils.flatten_nested_list_of_2d_arrays(Yhat)
    XXhat = utils.flatten_nested_list_of_2d_arrays(Xhat)
    
    nS = len(Yhat)
    nT = len(Yhat[0])
    assert(nS==len(Xhat))
    assert(nT==len(Xhat[0]))
    nN,nP = Xhat[0][0].shape
    nQ = Yhat[0][0].shape[1]
    assert(nN==Yhat[0][0].shape[0])

    opt = gen_opt(nN=nN,nP=nP,nQ=nQ,nS=nS,nT=nT,foldT=foldT,pop_rate_fn=pop_rate_fn,pop_deriv_fn=pop_deriv_fn,fudge=fudge)
    
    def add_key_val(d,key,val):
        if not key in d:
            d[key] = val
    
    if wt_dict is None:
        wt_dict = {}
    add_key_val(wt_dict,'celltypes',np.ones((1,nT*nS*nQ)))
    add_key_val(wt_dict,'inputs',np.concatenate([np.array((1,0)) for i in range(nT*nS)],axis=0)[np.newaxis,:])
    add_key_val(wt_dict,'stims',np.ones((nN,1)))
    add_key_val(wt_dict,'X',1)
    add_key_val(wt_dict,'Y',1)
    add_key_val(wt_dict,'Eta',1)
    add_key_val(wt_dict,'Xi',1)
    add_key_val(wt_dict,'barrier',1)
    add_key_val(wt_dict,'opto',100.)
    add_key_val(wt_dict,'dYY',1.)
    add_key_val(wt_dict,'coupling',0)
    add_key_val(wt_dict,'isn',0.01)
    add_key_val(wt_dict,'tv',0.01)
    add_key_val(wt_dict,'celltypesOpto',np.ones((1,nT*nS*nQ)))
    add_key_val(wt_dict,'stimsOpto',np.ones((nN,1)))
    add_key_val(wt_dict,'dirOpto',np.ones((2,)))
    add_key_val(wt_dict,'smi',1)
    add_key_val(wt_dict,'smi_chrimson',0.5)
    
    wtCell = wt_dict['celltypes']
    wtInp = wt_dict['inputs']
    wtStim = wt_dict['stims']
    wtX = wt_dict['X']
    wtY = wt_dict['Y']
    wtEta = wt_dict['Eta']
    wtXi = wt_dict['Xi']
    barrier_wt = wt_dict['barrier']
    wtOpto = wt_dict['opto']
    wtISN = wt_dict['isn']
    wtdYY = wt_dict['dYY']
    #wtEta12 = wt_dict['Eta12']
    #wtEtaTV = wt_dict['EtaTV']
    wtTV = wt_dict['tv']
    wtCoupling = wt_dict['coupling']
    wtCellOpto = wt_dict['celltypesOpto']
    wtStimOpto = wt_dict['stimsOpto']
    wtDirOpto = wt_dict['dirOpto']
    wtSMI = wt_dict['smi']
    wtSMIchrimson = wt_dict['smi_chrimson']

    #if wtEtaTV > 0:
    #    assert(nsize*ncontrast==nN)

    if wtCoupling > 0:
        assert(not coupling_constraints is None)
        constrain_coupling = True
    else:
        constrain_coupling = False

    first = True
        
    # Yhat is all measured tuning curves, Y is the averages of the model tuning curves
    def parse_W1(W):
        Ws = utils.parse_thing(W,shapes1)
        return Ws

    def parse_W2(W):
        Ws = utils.parse_thing(W,shapes2)
        return Ws
    
    def unparse_W(*Ws):
        return np.concatenate([ww.flatten() for ww in Ws])
    
    def normalize(arr):
        arrsum = arr.sum(1)
        well_behaved = (arrsum>0)[:,np.newaxis]
        arrnorm = well_behaved*arr/arrsum[:,np.newaxis] + (~well_behaved)*np.ones_like(arr)/arr.shape[1]
        return arrnorm
    
    def gen_Weight(W,K,kappa,T,power=True):
        return utils.gen_Weight_k_kappa_t(W,K,kappa,T,nS=nS,nT=nT,power=power)
        
    def compute_kl_divergence(stim_deriv,noise,mu_data,mu_model,pc_list):
        return utils.compute_kl_divergence(stim_deriv,noise,mu_data,mu_model,pc_list,nQ=nQ,nS=nS,nT=nT,foldT=foldT)

    def compute_var(Xi,s02):
        return fudge+Xi**2+np.concatenate([s02 for ipixel in range(nS*nT)],axis=0)

    def compute_fprime_(Eta,Xi,s02):
        return fprime_m(Eta,compute_var(Xi,s02))*Xi

    def compute_f_(Eta,Xi,s02):
        return pop_rate_fn(Eta,compute_var(Xi,s02))

    def compute_us(W1,W2,fval,fprimeval):
        W0x,W0y,W1x,W1y,W2x,W2y,W3x,W3y,s02,K0,K1,K2,K3,kappa,T0,T1,T2,T3,h1,h2,bl,amp = parse_W1(W1)
        XX,XXp,Eta,Xi = parse_W2(W2)
        if fval.shape[0]==2*nN:
            XX = np.concatenate((XX,XX),axis=0)
            XXp = np.concatenate((XXp,XXp),axis=0)
        power0 = True
        power1 = False
        u0 = u_fn(XX,fval,W0x,W0y,K0,kappa,T0,power=power0)
        u1 = u_fn(XX,fval,W1x,W1y,K0,kappa,T0,power=power0) + u_fn(XX,fval,W0x,W0y,K1,kappa,T0,power=power1) + u_fn(XX,fval,W0x,W0y,K0,kappa,T1,power=power1)
        u2 = u_fn(XXp,fprimeval,W2x,W2y,K0,kappa,T0,power=power0) + u_fn(XXp,fprimeval,W0x,W0y,K2,kappa,T0,power=power1) + u_fn(XXp,fprimeval,W0x,W0y,K0,kappa,T2,power=power1)
        u3 = u_fn(XXp,fprimeval,W3x,W3y,K0,kappa,T0,power=power0) + u_fn(XXp,fprimeval,W0x,W0y,K3,kappa,T0,power=power1) + u_fn(XXp,fprimeval,W0x,W0y,K0,kappa,T3,power=power1)
        return u0,u1,u2,u3

    def compute_f_fprime_(W1,W2):
        #W0x,W0y,W1x,W1y,W2x,W2y,W3x,W3y,s02,K0,K1,K2,K3,kappa,T0,T1,T2,T3,h1,h2,bl,amp = parse_W1(W1)
        W0x,W0y,W1x,W1y,W2x,W2y,W3x,W3y,s02,K0,K1,K2,K3,kappa,T0,T1,T2,T3,h1,h2,bl,amp = parse_W1(W1)
        XX,XXp,Eta,Xi = parse_W2(W2)
        return compute_f_(Eta,Xi,s02),compute_fprime_(Eta,Xi,s02)

    def compute_f_fprime_t_(W1,W2,perturbation,max_dist=1): # max dist added 10/14/20
        #Wmx,Wmy,Wsx,Wsy,s02,K,kappa,T,XX,XXp,Eta,Xi,h1,h2 = parse_W(W)
        W0x,W0y,W1x,W1y,W2x,W2y,W3x,W3y,s02,K0,K1,K2,K3,kappa,T0,T1,T2,T3,h1,h2,bl,amp = parse_W1(W1)
        XX,XXp,Eta,Xi = parse_W2(W2)
        fval = compute_f_(Eta,Xi,s02)
        fprimeval = compute_fprime_(Eta,Xi,s02)
        u0,u1,u2,u3 = compute_us(W1,W2,fval,fprimeval)
        resEta = Eta - u0 - u2
        resXi  = Xi - u1 - u3
        YY = fval + perturbation
        YYp = fprimeval
        def dYYdt(YY,Eta1,Xi1):
            return -YY + compute_f_(Eta1,Xi1,s02)
        def dYYpdt(YYp,Eta1,Xi1):
            return -YYp +compute_fprime_(Eta1,Xi1,s02)
        for t in range(niter):
            if np.mean(np.abs(YY-fval)) < max_dist:
                u0,u1,u2,u3 = compute_us(W1,W2,YY,YYp)
                Eta1 = resEta + u0 + u2
                Xi1 = resXi + u1 + u3
                YY = YY + dt*dYYdt(YY,Eta1,Xi1)
                YYp = YYp + dt*dYYpdt(YYp,Eta1,Xi1)
            elif np.remainder(t,500)==0:
                print('unstable fixed point?')

        #YY = YY + np.tile(bl,nS*nT)[np.newaxis,:]
            
        #YYp = compute_fprime_(Eta1,Xi1,s02)
        
        return YY,YYp

    def compute_f_fprime_t_12_(W1,W2,perturbation,max_dist=1): # max dist added 10/14/20
        #Wmx,Wmy,Wsx,Wsy,s02,K,kappa,T,XX,XXp,Eta,Xi,h1,h2 = parse_W(W)
        W0x,W0y,W1x,W1y,W2x,W2y,W3x,W3y,s02,K0,K1,K2,K3,kappa,T0,T1,T2,T3,h1,h2,bl,amp = parse_W1(W1)
        XX,XXp,Eta,Xi = parse_W2(W2)
        fval = compute_f_(Eta,Xi,s02)
        fprimeval = compute_fprime_(Eta,Xi,s02)
        if share_residuals:
            u0,u1,u2,u3 = compute_us(W1,W2,fval,fprimeval)
            resEta = Eta - u0 - u2
            resXi  = Xi - u1 - u3
            resEta12 = np.concatenate((resEta,resEta),axis=0)
            resXi12 = np.concatenate((resXi,resXi),axis=0)
        else:
            resEta12 = 0
            resXi12 = 0
        dHH = np.zeros((nN,nQ*nS*nT))
        dHH[:,np.arange(2,nQ*nS*nT,nQ)] = 1
        dHH = np.concatenate((dHH*h1,dHH*h2),axis=0)
        YY = fval + perturbation
        YYp = fprimeval
        XX12 = np.concatenate((XX,XX),axis=0)
        YY12 = np.concatenate((YY,YY),axis=0)
        YYp12 = np.concatenate((YYp,YYp),axis=0)
        def dYYdt(YY,Eta1,Xi1):
            return -YY + compute_f_(Eta1,Xi1,s02)
        def dYYpdt(YYp,Eta1,Xi1):
            return -YYp + compute_fprime_(Eta1,Xi1,s02)
        for t in range(niter):
            if np.mean(np.abs(YY-fval)) < max_dist:
                u0,u1,u2,u3 = compute_us(W1,W2,YY12,YYp12)
                Eta121 = resEta12 + u0 + u2 + dHH
                Xi121 = resXi12 + u1 + u3
                YY12 = YY12 + dt*dYYdt(YY12,Eta121,Xi121)
                YYp12 = YYp12 + dt*dYYpdt(YYp12,Eta121,Xi121)
            elif np.remainder(t,500)==0:
                print('unstable fixed point?')

        #YY12 = YY12 + np.tile(bl,nS*nT)[np.newaxis,:]
        
        return YY12,YYp12

    # GOT TO THIS LINE IN ADAPTING CODE

    def compute_f_fprime_t_avg_(W1,W2,perturbation,burn_in=0.5,max_dist=1):
        #Wmx,Wmy,Wsx,Wsy,s02,K,kappa,T,XX,XXp,Eta,Xi,h1,h2 = parse_W(W)
        W0x,W0y,W1x,W1y,W2x,W2y,W3x,W3y,s02,K0,K1,K2,K3,kappa,T0,T1,T2,T3,h1,h2,bl,amp = parse_W1(W1)
        XX,XXp,Eta,Xi = parse_W2(W2)
        fval = compute_f_(Eta,Xi,s02)
        fprimeval = compute_fprime_(Eta,Xi,s02)
        u0,u1,u2,u3 = compute_us(W1,W2,fval,fprimeval)
        resEta = Eta - u0 - u2
        resXi  = Xi - u1 - u3
        YY = fval + perturbation
        YYp = fprimeval
        YYmean = np.zeros_like(Eta)
        YYprimemean = np.zeros_like(Eta)
        def dYYdt(YY,Eta1,Xi1):
            return -YY + compute_f_(Eta1,Xi1,s02)
        def dYYpdt(YYp,Eta1,Xi1):
            return -YYp + compute_fprime_(Eta1,Xi1,s02)
        for t in range(niter):
            if np.mean(np.abs(YY-fval)) < max_dist:
                u0,u1,u2,u3 = compute_us(W1,W2,YY,YYp)
                Eta1 = resEta + u0 + u2
                Xi1 = resXi + u1 + u3
                YY = YY + dt*dYYdt(YY,Eta1,Xi1)
                YYp = YYp + dt*dYYpdt(YYp,Eta1,Xi1)
            else:
                print('unstable fixed point?')
            #Eta1 = resEta + u_fn(XX,YY,Wmx,Wmy,K,kappa,T)
            #Xi1 = resXi + u_fn(XX,YY,Wsx,Wsy,K,kappa,T)
            #YY = YY + dt*dYYdt(YY,Eta1,Xi1)
            if t>niter*burn_in:
                #YYp = compute_fprime_(Eta1,Xi1,s02)
                YYmean = YYmean + 1/niter/burn_in*YY
                YYprimemean = YYprimemean + 1/niter/burn_in*YYp

        #YYmean = YYmean + np.tile(bl,nS*nT)[np.newaxis,:]
            
        return YYmean,YYprimemean

    def compute_f_fprime_t_avg_12_(W1,W2,perturbation,max_dist=1,burn_in=0.5): # max dist added 10/14/20
        #Wmx,Wmy,Wsx,Wsy,s02,K,kappa,T,XX,XXp,Eta,Xi,h1,h2 = parse_W(W)
        W0x,W0y,W1x,W1y,W2x,W2y,W3x,W3y,s02,K0,K1,K2,K3,kappa,T0,T1,T2,T3,h1,h2,bl,amp = parse_W1(W1)
        XX,XXp,Eta,Xi = parse_W2(W2)
        fval = compute_f_(Eta,Xi,s02)
        fprimeval = compute_fprime_(Eta,Xi,s02)
        u0,u1,u2,u3 = compute_us(W1,W2,fval,fprimeval)
        if share_residuals:
            resEta = Eta - u0 - u2
            resXi  = Xi - u1 - u3
            resEta12 = np.concatenate((resEta,resEta),axis=0)
            resXi12 = np.concatenate((resXi,resXi),axis=0)
        else:
            resEta12 = 0
            resXi12 = 0
        dHH = np.zeros((nN,nQ*nS*nT))
        dHH[:,np.arange(2,nQ*nS*nT,nQ)] = 1
        dHH = np.concatenate((dHH*h1,dHH*h2),axis=0)
        YY = fval + perturbation
        YYp = fprimeval
        XX12 = np.concatenate((XX,XX),axis=0)
        YY12 = np.concatenate((YY,YY),axis=0)
        YYp12 = np.concatenate((YYp,YYp),axis=0)
        YYmean = np.zeros_like(YY12)
        YYprimemean = np.zeros_like(YY12)
        def dYYdt(YY,Eta1,Xi1):
            return -YY + compute_f_(Eta1,Xi1,s02)
        def dYYpdt(YYp,Eta1,Xi1):
            return -YYp + compute_fprime_(Eta1,Xi1,s02)
        for t in range(niter):
            if np.mean(np.abs(YY-fval)) < max_dist:
                u0,u1,u2,u3 = compute_us(W1,W2,YY12,YYp12)
                Eta121 = resEta12 + u0 + u2 + dHH
                Xi121 = resXi12 + u1 + u3
                YY12 = YY12 + dt*dYYdt(YY12,Eta121,Xi121)
                YYp12 = YYp12 + dt*dYYpdt(YYp12,Eta121,Xi121)
            elif np.remainder(t,500)==0:
                print('unstable fixed point?')
            if t>niter*burn_in:
                YYmean = YYmean + 1/niter/burn_in*YY12
                YYprimemean = YYprimemean + 1/niter/burn_in*YYp12

        #YYmean = YYmean + np.tile(bl,nS*nT)[np.newaxis,:]
        
        return YYmean,YYprimemean

    def u_fn(XX,YY,Wx,Wy,K,kappa,T,power=True):
        WWx,WWy = [gen_Weight(W,K,kappa,T,power=power) for W in [Wx,Wy]]
        return XX @ WWx + YY @ WWy
                    
    def minusLW(W1,W2,simulate=True,verbose=True):
        
        def compute_sq_error(a,b,wt):
            return np.sum(wt*(a-b)**2)
        
        def compute_kl_error(mu_data,pc_list,mu_model,fprimeval,wt):
            # how to model variability in X?
            kl = compute_kl_divergence(fprimeval,noise,mu_data,mu_model,pc_list)#,foldT=foldT,nQ=nQ,nS=nS,nT=nT)
            return kl #wt*kl
            # principled way would be to use 1/wt for noise term. Should add later.

        def compute_opto_error_nonlinear(fval,fval12,wt=None):
            if wt is None:
                wt = np.ones((2*nN,nQ*nS*nT))
            fval_both = np.concatenate((np.concatenate((fval,fval),axis=0)[:,np.newaxis,:],\
                    fval12[:,np.newaxis,:]),axis=1)
            this_fval12 = opto_transform1.preprocess(fval_both)
            dYY12 = this_fval12[:,1,:] - this_fval12[:,0,:]
            dYYterm = np.sum(wt[opto_mask]*(dYY12[opto_mask] - dYY[opto_mask])**2)
            return dYYterm

        def compute_opto_error_nonlinear_transform(fval,fval12,wt=None):
            if wt is None:
                wt = np.ones((2*nN,nQ*nS*nT))
            fval_both = np.concatenate((np.concatenate((fval,fval),axis=0)[:,np.newaxis,:],\
                    fval12[:,np.newaxis,:]),axis=1)
            this_fval12 = opto_transform1.preprocess(fval_both)[:,1,:]
            fval12target = np.concatenate((opto_transform1.transform(fval),opto_transform2.transform(fval)),axis=0)
            dYYterm = np.sum(wt[opto_mask]*(this_fval12[opto_mask] - fval12target[opto_mask])**2)
            return dYYterm

        def compute_coupling(W1,W2):
            #Wmx,Wmy,Wsx,Wsy,s02,K,kappa,T,XX,XXp,Eta,Xi,h1,h2 = parse_W(W)
            W0x,W0y,W1x,W1y,W2x,W2y,W3x,W3y,s02,K0,K1,K2,K3,kappa,T0,T1,T2,T3,h1,h2,bl,amp = parse_W1(W1)
            XX,XXp,Eta,Xi = parse_W2(W2)
            WWy = gen_Weight(W0y,K0,kappa,T0,power=True)
            Phi = fprime_m(Eta,compute_var(Xi,s02))
            Phi = np.concatenate((Phi,Phi),axis=0)
            Phi1 = np.array([np.diag(phi) for phi in Phi])
            coupling = np.array([phi1 @ np.linalg.inv(np.eye(nQ*nS*nT) - WWy @ phi1) for phi1 in Phi1])
            return coupling

        def compute_coupling_error(W1,W2,i,j,sgn=-1):
            # constrain coupling term i,j to have a specified sign, 
            # -1 for negative or +1 for positive
            coupling = compute_coupling(W1,W2)
            log_arg = sgn*coupling[:,i,j]
            cost = utils.minus_sum_log_ceil(log_arg,big_val/nN)
            return cost

        #def compute_eta_tv(this_Eta):
        #    Etar = this_Eta.reshape((nsize,ncontrast,nQ*nS*nT))
        #    diff_size = np.sum(np.abs(np.diff(Etar,axis=0)))
        #    diff_contrast = np.sum(np.abs(np.diff(Etar,axis=1)))
        #    return diff_size + diff_contrast

        def compute_isn_error(W1,W2):
            #Wmx,Wmy,Wsx,Wsy,s02,K,kappa,T,XX,XXp,Eta,Xi,h1,h2 = parse_W(W)
            W0x,W0y,W1x,W1y,W2x,W2y,W3x,W3y,s02,K0,K1,K2,K3,kappa,T0,T1,T2,T3,h1,h2,bl,amp = parse_W1(W1)
            XX,XXp,Eta,Xi = parse_W2(W2)
            Phi = fprime_m(Eta,compute_var(Xi,s02))
            #print('min Eta: %f'%np.min(Eta[:,0]))
            #print('WEE: %f'%Wmy[0,0])
            #print('min phiE*WEE: %f'%np.min(Phi[:,0]*Wmy[0,0]))
            log_arg = Phi[:,0]*W0y[0,0]-1
            cost = utils.minus_sum_log_ceil(log_arg,big_val/nN)
            #print('ISN cost: %f'%cost)
            return cost
        
        def compute_tv_error(W1,W2):
            # sq l2 norm for tv error
            #Wmx,Wmy,Wsx,Wsy,s02,K,kappa,T,XX,XXp,Eta,Xi,h1,h2 = parse_W(W)
            W0x,W0y,W1x,W1y,W2x,W2y,W3x,W3y,s02,K0,K1,K2,K3,kappa,T0,T1,T2,T3,h1,h2,bl,amp = parse_W1(W1)
            XX,XXp,Eta,Xi = parse_W2(W2)
            topo_var_list = [arr.reshape(topo_shape+(-1,)) for arr in \
                    [XX,XXp,Eta,Xi]]
            sqdiffy = [np.sum(np.abs(np.diff(top,axis=0))**2) for top in topo_var_list]
            sqdiffx = [np.sum(np.abs(np.diff(top,axis=1))**2) for top in topo_var_list]
            cost = np.sum(sqdiffy+sqdiffx)
            return cost

        def compute_smi_error(fval,fval12,halo_frac=0.5):
            fval = compute_f_(Eta,Xi,s02)
            ipc = 0
            def compute_dsmi(fval):
                fpc = fval[:,ipc].reshape(topo_shape)
                smi = fpc[-1,:]/np.max(fpc,0)
                dsmi = smi[1] - smi[5]
                return dsmi
            dsmis = [compute_dsmi(f) for f in [fval,fval12[:nN],fval12[nN:]]]
            smi_halo_error = halo_frac*(dsmis[0] - dsmis[1])**2
            smi_chrimson_error = (1-halo_frac)*utils.minus_sum_log_ceil(dsmis[2] - dsmis[1],big_val)
            return smi_halo_error,smi_chrimson_error

        #Wmx,Wmy,Wsx,Wsy,s02,K,kappa,T,XX,XXp,Eta,Xi,h1,h2 = parse_W(W)
        W0x,W0y,W1x,W1y,W2x,W2y,W3x,W3y,s02,K0,K1,K2,K3,kappa,T0,T1,T2,T3,h1,h2,bl,amp = parse_W1(W1)
        XX,XXp,Eta,Xi = parse_W2(W2)

        #utils.print_labeled('T',T)
        #utils.print_labeled('K',K)
        #utils.print_labeled('Wmy',Wmy)
        
        perturbation = perturbation_size*np.random.randn(*Eta.shape)
        
#         fval,fprimeval = compute_f_fprime_t_(W1,W2,perturbation) # Eta the mean input per cell, Xi the stdev. input per cell, s02 the baseline variability in input
        #print('simulate: '+str(simulate))
        if simulate:
            fval,fprimeval = compute_f_fprime_t_avg_(W1,W2,perturbation) # Eta the mean input per cell, Xi the stdev. input per cell, s02 the baseline variability in input
        else:
            fval,fprimeval = compute_f_fprime_(W1,W2) # Eta the mean input per cell, Xi the stdev. input per cell, s02 the baseline variability in input
        fval12,fprimeval12 = compute_f_fprime_t_avg_12_(W1,W2,perturbation) # Eta the mean input per cell, Xi the stdev. input per cell, s02 the baseline variability in input
        #utils.print_labeled('fval',fval)

        bltile = np.tile(bl,nS*nT)[np.newaxis,:]
        
        Xterm = compute_kl_error(XXhat,Xpc_list,XX,XXp,wtStim*wtInp) # XX the modeled input layer (L4)
        Yterm = compute_kl_error(YYhat,Ypc_list,amp*fval+bltile,amp*fprimeval,wtStim*wtCell) # fval the modeled output layer (L2/3)

        u0,u1,u2,u3 = compute_us(W1,W2,fval,fprimeval)

        Etaterm = compute_sq_error(Eta,u0 + u2,wtStim*wtCell) # magnitude of fudge factor in mean input
        Xiterm = compute_sq_error(Xi,u1 + u3,wtStim*wtCell) # magnitude of fudge factor in input variability
        # returns value float
        #Optoterm = compute_opto_error_nonlinear(W) #testing out 8/20/20
        opto_wt = np.concatenate([wtStimOpto*wtCellOpto*w for w in wtDirOpto],axis=0)
        if use_opto_transforms:
            dYYterm = compute_opto_error_nonlinear_transform(amp*fval+bltile,amp*fval12+bltile,opto_wt)
        else:
            dYYterm = compute_opto_error_nonlinear(amp*fval+bltile,amp*fval12+bltile,opto_wt)
        if wtSMI != 0:
            SMIhaloterm,SMIchrimsonterm = compute_smi_error(fval,fval12,halo_frac=1-wtSMIchrimson)
        else:
            SMIhaloterm,SMIchrimsonterm = 0,0
        Optoterm = 0# wtdYY*dYYterm
        cost = wtX*Xterm + wtY*Yterm + wtEta*Etaterm + wtXi*Xiterm + wtOpto*Optoterm + wtSMI*SMIhaloterm + wtSMI*SMIchrimsonterm# + wtEtaTV*EtaTVterm 
        if constrain_isn:
            ISNterm = compute_isn_error(W1,W2)
            cost = cost + wtISN*ISNterm
        if constrain_coupling:
            Couplingterm = 0
            for el in coupling_constraints:
                i,j,sgn = el
                Couplingterm = Couplingterm + compute_coupling_error(W1,W2,i,j,sgn)
            cost = cost + wtCoupling*Couplingterm
        if tv:
            TVterm = compute_tv_error(W1,W2)
            cost = cost + wtTV*TVterm

        #print('Yterm as float: '+str(float(Yterm)))
        #print('Yterm as float: '+str(Yterm.astype('float')))
            
        if isinstance(Yterm,float) and verbose:
            print('X:%f'%(wtX*Xterm))
            print('Y:%f'%(wtY*Yterm.astype('float')))
            print('Eta:%f'%(wtEta*Etaterm))
            print('Xi:%f'%(wtXi*Xiterm))
            print('Opto dYY:%f'%(wtOpto*wtdYY*dYYterm))
            #print('Opto Eta:%f'%(wtOpto*wtEta12*Eta12term))
            #print('TV:%f'%(wtEtaTV*EtaTVterm))
            print('TV:%f'%(wtTV*TVterm))
            print('SMI halo:%f'%(wtSMI*SMIhaloterm))
            print('SMI chrimson:%f'%(wtSMI*SMIchrimsonterm))
            if constrain_isn:
                print('ISN:%f'%(wtISN*ISNterm))
            if constrain_coupling:
                print('coupling:%f'%(wtCoupling*Couplingterm))

        #lbls = ['Yterm']
        #vars = [Yterm]
        lbls = ['cost']
        vars = [cost]
        if verbose:
            for lbl,var in zip(lbls,vars):
                utils.print_labeled(lbl,var)
        return cost

    def minusdLdW1(W1,W2,simulate=True,verbose=True): 
        # returns value (R,)
        # sum in first dimension: (N,1) times (N,1) times (N,P)
#         return jacobian(minusLW)(W)
        return grad(lambda W1: minusLW(W1,W2,simulate=simulate,verbose=verbose))(W1)
        
    def minusdLdW2(W1,W2,simulate=True,verbose=True): 
        # returns value (R,)
        # sum in first dimension: (N,1) times (N,1) times (N,P)
#         return jacobian(minusLW)(W)
        return grad(lambda W2: minusLW(W1,W2,simulate=simulate,verbose=verbose))(W2)

    def fix_violations(w,bounds):
        lb = np.array([b[0] for b in bounds])
        ub = np.array([b[1] for b in bounds])
        lb_violation = w<lb
        ub_violation = w>ub
        w[lb_violation] = lb[lb_violation]
        w[ub_violation] = ub[ub_violation]
        return w,lb_violation,ub_violation
    
    def sorted_r_eigs(w):
        drW,prW = np.linalg.eig(w)
        srtinds = np.argsort(drW)
        return drW[srtinds],prW[:,srtinds]
    
    def compute_eig_penalty_(W0y,K0,kappa,T0):
        # still need to finish! Hopefully won't need
        # need to fix this to reflect addition of kappa argument
        Wsquig = gen_Weight(W0y,K0,kappa,T0,power=True)
        drW,prW = sorted_r_eigs(Wsquig - np.eye(nQ*nS*nT))
        plW = np.linalg.inv(prW)
        eig_outer_all = [np.real(np.outer(plW[:,k],prW[k,:])) for k in range(nS*nQ*nT)]
        eig_penalty_size_all = [barrier_wt/np.abs(np.real(drW[k])) for k in range(nS*nQ*nT)]
        eig_penalty_dir_w = [eig_penalty_size*((eig_outer[:nQ,:nQ] + eig_outer[nQ:,nQ:]) + K0[np.newaxis,:]*(eig_outer[:nQ,nQ:] + kappa*eig_outer[nQ:,:nQ])) for eig_outer,eig_penalty_size in zip(eig_outer_all,eig_penalty_size_all)]
        eig_penalty_dir_k = [eig_penalty_size*((eig_outer[:nQ,nQ:] + eig_outer[nQ:,:nQ]*kappa)*W0y).sum(0) for eig_outer,eig_penalty_size in zip(eig_outer_all,eig_penalty_size_all)]
        eig_penalty_dir_kappa = [eig_penalty_size*(eig_outer[nQ:,:nQ]*k0[np.newaxis,:]*W0y).sum().reshape((1,)) for eig_outer,eig_penalty_size in zip(eig_outer_all,eig_penalty_size_all)]
        eig_penalty_dir_w = np.array(eig_penalty_dir_w).sum(0)
        eig_penalty_dir_k = np.array(eig_penalty_dir_k).sum(0)
        eig_penalty_dir_kappa = np.array(eig_penalty_dir_kappa).sum(0)
        return eig_penalty_dir_w,eig_penalty_dir_k,eig_penalty_dir_kappa
    
    def compute_eig_penalty(W):
        # still need to finish! Hopefully won't need
        W0mx,W0my,W0sx,W0sy,s020,K0,kappa0,T0,XX0,XXp0,Eta0,Xi0,h10,h20,Eta10,Eta20 = parse_W(W)
        eig_penalty_dir_w,eig_penalty_dir_k,eig_penalty_dir_kappa = compute_eig_penalty_(W0my,k0,kappa0)
        eig_penalty_W = unparse_W(np.zeros_like(W0mx),eig_penalty_dir_w,np.zeros_like(W0sx),np.zeros_like(W0sy),np.zeros_like(s020),eig_penalty_dir_k,eig_penalty_dir_kappa,np.zeros_like(XX0),np.zeros_like(XXp0),np.zeros_like(Eta0),np.zeros_like(Xi0))
#         assert(True==False)
        return eig_penalty_W
    
    def optimize1(W10,W20,compute_hessian=False,simulate=True,verbose=True):

        allhot = np.zeros(W10.shape)
        allhot[:nP*nQ+nQ**2] = 1
        W_l2_reg = lambda W: np.sum((W*allhot)**2)
        f = lambda W: minusLW(W,W20,simulate=simulate,verbose=verbose) + l2_penalty*W_l2_reg(W)
        fprime = lambda W: minusdLdW1(W,W20,simulate=simulate,verbose=verbose) + 2*l2_penalty*W*allhot

        fix_violations(W10,bounds1)
        
        #W11,loss,result = sop.fmin_l_bfgs_b(f,W10,fprime=fprime,bounds=bounds1,factr=1e2,maxiter=int(1e3),maxls=40)
        options = {}
        options['factr']=1e2
        options['maxiter']=int(1e3)
        options['maxls']=40
        result = sop.minimize(f,W10,jac=fprime,bounds=bounds1,options=options,method='L-BFGS-B')
        W11 = result.x
        loss = result.fun
        if compute_hessian:
            gr = grad(lambda W1: minusLW(W1,W2,simulate=simulate,verbose=verbose))(W1)
            hess = hessian(lambda W1: minusLW(W1,W2,simulate=simulate,verbose=verbose))(W1)
        else:
            gr = None
            hess = None
        
#         W0mx,W0my,W0sx,W0sy,s020,k0,kappa0,XX0,XXp0,Eta0,Xi0 = parse_W(W1)
        
        return W11,loss,gr,hess,result
    
    def optimize2(W10,W20,compute_hessian=False,simulate=False,verbose=True): 
        to_zero = np.array([(b[0]==0)&(b[1]==0) for b in bounds2])

        def f(w):
            W = np.zeros_like(W20)
            W[~to_zero] = w
            return minusLW(W10,W,simulate=simulate,verbose=verbose)

        def fprime(w):
            W = np.zeros_like(W20)
            W[~to_zero] = w
            return minusdLdW2(W10,W,simulate=simulate,verbose=verbose)[~to_zero]

        w20 = W20[~to_zero]
        #w21,loss,result = sop.fmin_cg(f,w20,fprime=fprime)
        options = {}
        options['gtol'] = 1e-1
        result = sop.minimize(f,w20,jac=fprime,options=options,method='CG')
        w21 = result.x
        loss = result.fun
        W21 = np.zeros_like(W20)
        W21[~to_zero] = w21

        if compute_hessian:
            gr = grad(lambda W2: minusLW(W1,W2,simulate=simulate,verbose=verbose))(W2)
            hess = hessian(lambda W2: minusLW(W1,W2,simulate=simulate,verbose=verbose))(W2)
        else:
            gr = None
            hess = None
        
        return W21,loss,gr,hess,result

    def optimize2_stimwise(W10,W20,compute_hessian=False,simulate=False,verbose=True): 
        to_zero = np.array([(b[0]==0)&(b[1]==0) for b in bounds2])

        W21 = W20.copy()#np.zeros_like(W20)
        for istim in range(nN):
            print('on stimulus #%d'%istim)
            in_this_stim_list = [np.zeros(shp,dtype='bool') for shp in shapes2]
            for ivar in range(len(shapes2)):
                in_this_stim_list[ivar][istim] = True
            in_this_stim = unparse_W(*in_this_stim_list)
            relevant = ~to_zero & in_this_stim

            def f(w):
                W = np.zeros_like(W20)
                W[~relevant] = W21[~relevant]
                W[relevant] = w
                return minusLW(W10,W,simulate=simulate,verbose=verbose)

            def fprime(w):
                W = np.zeros_like(W20)
                W[~relevant] = W21[~relevant]
                W[relevant] = w
                return minusdLdW2(W10,W,simulate=simulate,verbose=verbose)[relevant]

            w20 = W20[relevant]
            #w21,loss,result = sop.fmin_cg(f,w20,fprime=fprime)
            options = {}
            options['gtol'] = 1e-2
            result = sop.minimize(f,w20,jac=fprime,options=options,method='CG')
            w21 = result.x
            loss = result.fun
            print('sum of relevant: '+str(relevant.sum()))
            W21[relevant] = w21

        if compute_hessian:
            gr = grad(lambda W2: minusLW(W1,W2,simulate=simulate,verbose=verbose))(W21)
            hess = hessian(lambda W2: minusLW(W1,W2,simulate=simulate,verbose=verbose))(W21)
        else:
            gr = None
            hess = None
        
        return W21,loss,gr,hess,result
    
    W10 = unparse_W(*W10list)
    W20 = unparse_W(*W20list)

    #simulate1,simulate2 = True,False
    verbose1,verbose2 = verbose,verbose

    old_loss = np.inf
    if stimwise:
        W21,loss,gr,hess,result = optimize2_stimwise(W10,W20,compute_hessian=compute_hessian,simulate=simulate2,verbose=verbose2)
    else:
        W21,loss,gr,hess,result = optimize2(W10,W20,compute_hessian=compute_hessian,simulate=simulate2,verbose=verbose2)
    W11,loss,gr,hess,result = optimize1(W10,W21,compute_hessian=compute_hessian,simulate=simulate1,verbose=verbose1)

    delta = old_loss - loss
    while delta > 0.1:
        old_loss = loss
        #W11,loss,gr,hess,result = optimize1(W10,W20,compute_hessian=compute_hessian,simulate=True)
        #W21,loss,gr,hess,result = optimize2(W11,W20,compute_hessian=compute_hessian,simulate=False)
        if stimwise:
            W21,loss,gr,hess,result = optimize2_stimwise(W11,W21,compute_hessian=compute_hessian,simulate=simulate2,verbose=verbose2)
        else:
            W21,loss,gr,hess,result = optimize2(W11,W21,compute_hessian=compute_hessian,simulate=simulate2,verbose=verbose2)
        W11,loss,gr,hess,result = optimize1(W11,W21,compute_hessian=compute_hessian,simulate=simulate1,verbose=verbose1)
        delta = old_loss - loss

        
    W1t = parse_W1(W11) #[Wmx,Wmy,Wsx,Wsy,s02,K,kappa,T,h1,h2,bl,amp]
    W2t = parse_W2(W21) #[XX,XXp,Eta,Xi]
    
    return W1t,W2t,loss,gr,hess,result

def load_Rs_mean_cov(ca_data_file=None,fit_running=True,fit_non_running=True,fit_sc=True,fit_fg=True,fit_ret=False,foldT=False,pdim=0,nT=2):

    def get_Rs_slices(Rs_mean,Rs_cov,slcs):
        for iR in range(len(Rs_mean)):
            for ialign in range(len(Rs_mean[iR])):
                Rs_mean[iR][ialign] = np.concatenate([Rs_mean[iR][ialign][slc] for slc in slcs])
                for idim in range(len(Rs_cov[iR][ialign])):
                    #print(Rs_cov[iR][ialign][idim][1].shape)
                    #print(slcs)
                    #print(len([Rs_cov[iR][ialign][idim][1][slc] for slc in slcs]))
                    #Rs_cov[iR][ialign][idim][1] = np.concatenate([Rs_cov[iR][ialign][idim][1][slc] for slc in slcs])
                    Rs_cov[iR][ialign][idim] = (Rs_cov[iR][ialign][idim][0],np.concatenate([Rs_cov[iR][ialign][idim][1][slc] for slc in slcs]))
        return Rs_mean,Rs_cov

    def get_Rs_slice(Rs_mean,Rs_cov,slc):
        return get_Rs_slices(Rs_mean,Rs_cov,[slc])

    def get_pc_dim(Ypc_list,nN=36,nPQ=4,nS=2,nT=2,idim=0,foldT=True):
        if foldT:
            Yp = np.zeros((nN*nT,nPQ*nS))
            for iS in range(nS):
                for iQ in range(nPQ):
                    #print('Ypc shape %d, %d: '%(iS,iQ)+str((Ypc_list[iS][iQ][idim][0]*Ypc_list[iS][iQ][idim][1]).shape))
                    Yp[:,iS*nPQ+iQ] = Ypc_list[iS][iQ][idim][0]*Ypc_list[iS][iQ][idim][1]
            Yp = utils.unfold_T_(Yp,nS=nS,nT=nT,nPQ=nPQ)
        else:
            Yp = np.zeros((nN,nPQ*nS*nT))
            for iS in range(nS):
                for iT in range(nT):
                    for iQ in range(nPQ):
                        #print('Ypc shape %d, %d: '%(iS,iQ)+str((Ypc_list[iS][iQ][idim][0]*Ypc_list[iS][iQ][idim][1]).shape))
                        Yp[:,iS*nT*nPQ+iT*nPQ+iQ] = Ypc_list[iS][iT][iQ][idim][0]*Ypc_list[iS][iT][iQ][idim][1]
        return Yp

    nrun = 2
    nsize,ncontrast,ndir = 6,6,8
    nstim_fg = 5
    nstim_ret = 5

    fit_both_running = (fit_non_running and fit_running)
    fit_all_stims = (fit_sc and fit_fg and fit_ret)

    if not fit_both_running:
        nrun = 1
        if fit_non_running:
            irun = 0
        elif fit_running:
            irun = 1

    nsc = nrun*nsize*ncontrast*ndir
    nfg = nrun*nstim_fg*ndir
    nret = nrun*nstim_ret

    npfile = np.load(ca_data_file,allow_pickle=True)[()]#,{'rs':rs},allow_pickle=True) # ,'rs_denoise':rs_denoise
    if fit_both_running:
        Rs_mean = npfile['Rs_mean_run']
        Rs_cov = npfile['Rs_cov_run']
    else:
        Rs_mean = npfile['Rs_mean'][irun]
        Rs_cov = npfile['Rs_cov'][irun]

    if not fit_all_stims:
        slcs = []
        if fit_sc:
            slcs = slcs + [slice(None,nsc)]
        if fit_fg:
            slcs = slcs + [slice(nsc,nsc+nfg)]
        if fit_ret:
            slcs = slcs + [slice(nsc+nfg,None)]
        Rs_mean,Rs_cov = get_Rs_slices(Rs_mean,Rs_cov,slcs)
    
    if nT==3:
        ori_dirs = [[0,4],[1,3,5,7],[2,6]]
    elif nT==2:
        ori_dirs = [[0,4],[2,6]] #[[0,4],[1,3,5,7],[2,6]]
    else:
        nT = 1
        ori_dirs = [[0,1,2,3,4,5,6,7]]

    ndims = 5
    nS = len(Rs_mean[0])

    fg_start = nsc*fit_sc
    ret_start = fg_start + nfg*fit_fg

    def sum_to_1(r):
        R = r.reshape((r.shape[0],-1))
        R = R/np.nansum(R[:,~np.isnan(R.sum(0))],axis=1)[:,np.newaxis] # changed 21/4/10
        return R

    def norm_to_mean(r):
        R = r.reshape((r.shape[0],-1))
        R = R/np.nanmean(R[:,~np.isnan(R.sum(0))],axis=1)[:,np.newaxis]
        return R

    def ori_avg(Rs,these_ori_dirs):
        if fit_sc:
            rs_sc = np.nanmean(Rs[:nsc].reshape((nrun,nsize,ncontrast,ndir))[:,:,:,these_ori_dirs],-1)
            rs_sc[:,1:,1:] = ssi.convolve(rs_sc,kernel,'valid')
            rs_sc = rs_sc.reshape((nrun*nsize*ncontrast))
        else:
            rs_sc = np.zeros((0,))
        if fit_fg:
            rs_fg = np.nanmean(Rs[fg_start:fg_start+nfg].reshape((nrun,nstim_fg,ndir))[:,:,these_ori_dirs],-1)
            rs_fg = rs_fg.reshape((nrun*nstim_fg))
        else:
            rs_fg = np.zeros((0,))
        if fit_ret:
            rs_ret = Rs[ret_start:ret_start+nret]
        else:
            rs_ret = np.zeros((0,))
        Rso = np.concatenate((rs_sc,rs_fg,rs_ret))
        return Rso

    #def ori_avg(Rs,these_ori_dirs):
    #    if fit_sc:
    #        rs_sc = np.nanmean(Rs[:nsc].reshape((nrun,nsize,ncontrast,ndir))[:,:,:,these_ori_dirs],-1)
    #        rs_sc[:,1:,1:] = ssi.convolve(rs_sc,kernel,'valid')
    #        rs_sc = rs_sc.reshape((nrun*nsize*ncontrast))
    #        if fit_fg:
    #            rs_fg = np.nanmean(Rs[nsc:].reshape((nrun,nstim_fg,ndir))[:,:,these_ori_dirs],-1)
    #            rs_fg = rs_fg.reshape((nrun*nstim_fg))
    #        else:
    #            rs_fg = np.zeros((0,))
    #    elif fit_fg:
    #        rs_sc = np.zeros((0,))
    #        rs_fg = np.nanmean(Rs.reshape((nrun,nstim_fg,ndir))[:,:,these_ori_dirs],-1)
    #        rs_fg = rs_fg.reshape((nrun*nstim_fg))
    #    Rso = np.concatenate((rs_sc,rs_fg))
    #    return Rso

    Rso_mean = [[[None for iT in range(nT)] for iS in range(nS)] for icelltype in range(len(Rs_mean))]
    Rso_cov = [[[[[None,None] for idim in range(ndims)] for iT in range(nT)] for iS in range(nS)] for icelltype in range(len(Rs_mean))]

    kernel = np.ones((1,2,2))
    kernel = kernel/kernel.sum()

    for iR,r in enumerate(Rs_mean):
        for ialign in range(nS):
            for iori in range(nT):
                Rso_mean[iR][ialign][iori] = ori_avg(Rs_mean[iR][ialign],ori_dirs[iori])
                for idim in range(ndims):
                    Rso_cov[iR][ialign][iori][idim][0] = Rs_cov[iR][ialign][idim][0]
                    Rso_cov[iR][ialign][iori][idim][1] = ori_avg(Rs_cov[iR][ialign][idim][1],ori_dirs[iori])

    def set_bound(bd,code,val=0):
        # set bounds to 0 where 0s occur in 'code'
        for iitem in range(len(bd)):
            bd[iitem][code[iitem]] = val

    nN = (36*fit_sc + 5*fit_fg + 5*fit_ret)*(1 + fit_both_running)
    nS = 2
    nP = 2 + fit_both_running
    #nT = 2
    nQ = 4

    ndims = 5
    ncelltypes = 5
    #print('foldT: %d'%foldT)
    if foldT:
        Yhat = [None for iS in range(nS)]
        Xhat = [None for iS in range(nS)]
        Ypc_list = [None for iS in range(nS)]
        Xpc_list = [None for iS in range(nS)]
        print('have not written this yet')
        assert(True==False)
    else:
        Yhat = [[None for iT in range(nT)] for iS in range(nS)]
        Xhat = [[None for iT in range(nT)] for iS in range(nS)]
        Ypc_list = [[None for iT in range(nT)] for iS in range(nS)]
        Xpc_list = [[None for iT in range(nT)] for iS in range(nS)]
        for iS in range(nS):
            mx = np.zeros((ncelltypes,))
            yy = [None for icelltype in range(ncelltypes)]
            for icelltype in range(ncelltypes):
                yy[icelltype] = np.concatenate(Rso_mean[icelltype][iS])
                mx[icelltype] = np.nanmax(yy[icelltype])
            for iT in range(nT):
                y = [Rso_mean[icelltype][iS][iT][:,np.newaxis]/mx[icelltype] for icelltype in range(1,ncelltypes)]
                Yhat[iS][iT] = np.concatenate(y,axis=1)
                Ypc_list[iS][iT] = [None for icelltype in range(1,ncelltypes)]
                for icelltype in range(1,ncelltypes):
                    Ypc_list[iS][iT][icelltype-1] = [(this_dim[0]/mx[icelltype],this_dim[1]) for this_dim in Rso_cov[icelltype][iS][iT]]
                icelltype = 0
                x = Rso_mean[icelltype][iS][iT][:,np.newaxis]/mx[icelltype]
                if fit_both_running:
                    run_vector = np.zeros_like(x)
                    if fit_all_stims:
                        run_vector[int(np.round(nsc/2)):nsc] = 1
                        run_vector[-int(np.round(nfg/2)):] = 1
                    else:
                        run_vector[int(np.round(run_vector.shape[0]/2)):,:] = 1
                else:
                    run_vector = np.zeros((x.shape[0],0))
                Xhat[iS][iT] = np.concatenate((x,np.ones_like(x),run_vector),axis=1)
                Xpc_list[iS][iT] = [None for iinput in range(2+fit_both_running)]
                Xpc_list[iS][iT][0] = [(this_dim[0]/mx[icelltype],this_dim[1]) for this_dim in Rso_cov[icelltype][iS][iT]]
                Xpc_list[iS][iT][1] = [(0,np.zeros((Xhat[0][0].shape[0],))) for idim in range(ndims)]
                if fit_both_running:
                    Xpc_list[iS][iT][2] = [(0,np.zeros((Xhat[0][0].shape[0],))) for idim in range(ndims)]

    YYhat = utils.flatten_nested_list_of_2d_arrays(Yhat)
    XXhat = utils.flatten_nested_list_of_2d_arrays(Xhat)
    XXphat = get_pc_dim(Xpc_list,nN=nN,nPQ=nP,nS=nS,nT=nT,idim=pdim,foldT=foldT)
    YYphat = get_pc_dim(Ypc_list,nN=nN,nPQ=nQ,nS=nS,nT=nT,idim=pdim,foldT=foldT)
    return XXhat,YYhat,XXphat,YYphat

def initialize_params(XXhat,YYhat,opt,wpcpc=4,wpvpv=-6):

    keys = ['nP','nQ','nN','nS','nT','allow_s02','allow_A','allow_B','pop_rate_fn','pop_deriv_fn']

    nP,nQ,nN,nS,nT,allow_s02,allow_A,allow_B,rate_f,rate_fprime = [opt[key] for key in keys]

    n = 1
    celltype_wt = np.tile(np.array((n,1,1,1)),nS*nT)[np.newaxis]/n

    shapes = [(nP,nQ),(nQ,nQ),(1,nQ),(1,nQ),(1,nQ),(1,nQ)]

    W0x_bounds = 3*np.ones((nP,nQ),dtype=int)
    W0x_bounds[0,:] = 2 # L4 PCs are excitatory
    W0x_bounds[0,1] = 0 # SSTs don't receive L4 input

    W0y_bounds = 3*np.ones((nQ,nQ),dtype=int)
    W0y_bounds[0,:] = 2 # PCs are excitatory
    W0y_bounds[1:,:] = -2 # all the cell types except PCs are inhibitory
    W0y_bounds[1,1] = 0 # SSTs don't inhibit themselves
    # W0y_bounds[3,1] = 0 # PVs are allowed to inhibit SSTs, consistent with Hillel's unpublished results, but not consistent with Pfeffer et al.
    W0y_bounds[2,0] = 0 # VIPs don't inhibit L2/3 PCs. According to Pfeffer et al., only L5 PCs were found to get VIP inhibition
    np.fill_diagonal(W0y_bounds,0)

    K0_bounds = 1.5*np.ones((1,nQ))

    # opt_param[is02,iA,iB] = np.zeros((nP+nQ+4,nQ))
    # opt_cost[is02,iA,iB] = np.zeros((nQ,))

    opt_param = np.zeros((nP+nQ+4,nQ))
#     opt_cost = np.zeros((nQ,))

    YYmodeled = np.zeros(YYhat.shape)
    YYpmodeled = np.zeros(YYhat.shape)

    if allow_s02:
        S02_bounds = 2*np.ones((1,nQ)) # permitting noise as a free parameter #1*np.ones((1,nQ))#
    else:
        S02_bounds = 1*np.ones((1,nQ))

    if allow_A:
        A_bounds = 2*np.ones((1,nQ)) # 1*np.ones((1,nQ))#
    else:
        A_bounds = 1*np.ones((1,nQ))

    if allow_B:
        B_bounds = 3*np.ones((1,nQ))
    else:
        B_bounds = 0*np.ones((1,nQ))

    big_val = 1e4

    bdlist = [W0x_bounds,W0y_bounds,K0_bounds,S02_bounds,A_bounds,B_bounds]

    lb,ub = [[sgn*np.inf*np.ones(shp) for shp in shapes] for sgn in [-1,1]]
    lb,ub = utils.set_bounds_by_code(lb,ub,bdlist)

    lb[0][0,0] = 1
    ub[0][0,0] = np.inf
    lb[0][0,3] = 0
    ub[0][0,3] = np.inf

    lb[1][0,0] = 0
    ub[1][0,0] = np.inf
    lb[1][3,3] = -np.inf
    ub[1][3,3] = -2
    lb[1][0,3] = 0
    ub[1][0,3] = np.inf
    lb[1][3,0] = -np.inf
    ub[1][3,0] = 0

#     lb[1][0,0] = wpcpc
#     ub[1][0,0] = wpcpc
#     lb[1][3,3] = wpvpv
#     ub[1][3,3] = wpvpv
#     lb[1][0,3] = wpcpc
#     ub[1][0,3] = wpcpc
#     lb[1][3,0] = wpvpv
#     ub[1][3,0] = wpvpv

    XXstack = np.concatenate((XXhat,XXhat[:,list(nP+np.arange(nP))+list(np.arange(nP))]),axis=0)
    YYstack = np.concatenate((YYhat,YYhat[:,list(nQ+np.arange(nQ))+list(np.arange(nQ))]),axis=0)
    l2_penalty = 0#1e-4
    eig_penalty = 1e-2
    for itype in [0,1,2,3]:
        this_lb,this_ub = [np.concatenate([llb[:,itype].flatten() for llb in bb]) for bb in [lb,ub]]
        these_bounds = list(zip(this_lb,this_ub))
        shps = [(nP,),(nQ,),(1,),(1,),(1,),(1,)]
    #     others = np.arange(nS*nQ)
    #     others = [x for x in others if not x%nQ==itype]
        def cost(wx,wy,k,s02,a,b):
            y = compute_y(wx,wy,k,s02,a,b)
            l2_term = np.sum(wx**2)+np.sum(wy**2)
            if itype == 0:
                yprime = compute_yprime(wx,wy,k,s02,a,b)
                eig_term = utils.minus_sum_log_slope(yprime*wy[0]*(1+k) - 1,big_val)
            else:
                eig_term = 0
            return np.sum(celltype_wt[0,itype]*(YYstack[:,itype] - y)**2) + l2_penalty*l2_term + eig_penalty*eig_term
        def compute_y(wx,wy,k,s02,a,b):
            y = b + a*rate_f(XXstack @ np.concatenate((wx,k*wx)) + YYstack @ np.concatenate((wy,k*wy)),s02) #[:,others]
            return y
        def compute_yprime(wx,wy,k,s02,a,b):
            y = a*rate_fprime(XXstack @ np.concatenate((wx,k*wx)) + YYstack @ np.concatenate((wy,k*wy)),s02) #[:,others]
            return y
        def compute_y_(x):
            y = compute_y(*utils.parse_thing(x,shps))
            return y
        def compute_yprime_(x):
            y = compute_yprime(*utils.parse_thing(x,shps))
            return y
        def cost_(x):
            args = utils.parse_thing(x,shps)
            return cost(*args)
        def cost_prime_(x):
            return grad(cost_)(x)
        def unparse(*args):
            return np.concatenate([x.flatten() for x in args])
        wx0 = np.zeros((nP,))
        wy0 = np.zeros((nQ,))
        if itype == 0 or itype==3:
            wx0[0] = 1
            wy0[0] = wpcpc
            wy0[3] = wpvpv
#         elif itype == 3:
#             wy0[0] = wpcpc
        k0 = np.array((1,))
        s020 = np.array((1,))
        a0 = np.array((1,))
        b0 = np.array((0,))
        x0 = unparse(wx0,wy0,k0,s020,a0,b0)
        result = sop.minimize(cost_,x0,jac=cost_prime_,method='L-BFGS-B',bounds=these_bounds)
#         print(result)
        x1 = result.x
        opt_param[:,itype] = x1

        YYmodeled[:,itype] = compute_y_(x1)[:nN]
        YYmodeled[:,nQ+itype] = compute_y_(x1)[nN:]
        YYpmodeled[:,itype] = compute_yprime_(x1)[:nN]
        YYpmodeled[:,nQ+itype] = compute_yprime_(x1)[nN:]
#         opt_cost[itype] = result.fun

    eig_penalty = 1e-2#1e-2
    pc_eig_penalty = 1e-2#1e-2
    l2_penalty = 0#1e-4

    kappa = 1
    T = np.array(())

    lb[1][0,0] = 0
    ub[1][0,0] = np.inf
    lb[1][3,3] = -np.inf
    ub[1][3,3] = 0
    lb[1][0,3] = 0
    ub[1][0,3] = np.inf
    lb[1][3,0] = -np.inf
    ub[1][3,0] = 0

    lb[3] = 0*np.ones_like(lb[3])
    ub[3] = np.inf*np.ones_like(ub[3])

    def compute_WW(W,K):
        WWy = utils.gen_Weight_k_kappa_t(W,K[0],kappa,T,nS=nS,nT=nT)
        return WWy

    def compute_eigs(Wmy0,K0,YYp=YYpmodeled):

#         Wmy0 = opt_param[nP:nP+nQ]
#         K0 = opt_param[nP+nQ]
        WWy = compute_WW(Wmy0,K0)

        Phi = [np.diag(YYp[istim,:]) for istim in range(nN)]

        w = np.zeros(YYp.shape)

        wlist = [None for _ in range(nN)]

        for istim in range(nN):
            this_w,_ = sorted_r_eigs(WWy @ Phi[istim] - np.eye(WWy.shape[0]))
            wlist[istim] = np.real(this_w)[np.newaxis]
            #w[istim,:]
        w = np.concatenate(wlist,axis=0)

        this_w,_ = sorted_r_eigs(WWy - np.eye(WWy.shape[0]))
        w = np.concatenate((w,np.real(this_w)[np.newaxis]),axis=0)

        return w

    def compute_YY(Wx,Wy,K,S02,A,B):
        WWx,WWy = [compute_WW(W,K) for W in [Wx,Wy]]
#         print((WWx.shape,WWy.shape))
        AA,BB,SS02 = [np.concatenate((x,x),axis=1) for x in [A,B,S02]]
        YY = BB + AA*rate_f(XXhat @ WWx + YYhat @ WWy,SS02) #[:,others]
        YYp = AA*rate_fprime(XXhat @ WWx + YYhat @ WWy,SS02)
        return YY,YYp

    def Cost(Wx,Wy,K,S02,A,B):
        YY,YYp = compute_YY(Wx,Wy,K,S02,A,B)
        l2_term = np.sum(Wx**2)+np.sum(Wy**2)
        pc_eig_term = utils.minus_sum_log_slope(YYp[:,0]*Wy[0,0]*(1+K[0,0]) - 1,big_val)
#         pc_eig_term = utils.minus_sum_log_slope(compute_eigs(Wy[0:1,0:1],K[:,0:1],YYp=YYp[:,0::nQ])[:,-1],big_val)
        eig_term = utils.minus_sum_log_slope(-compute_eigs(Wy,K,YYp=YYp)[:,-1],big_val)
        this_cost = np.sum(celltype_wt*(YYhat - YY)**2) + l2_penalty*l2_term + eig_penalty*eig_term + pc_eig_penalty*pc_eig_term
        if np.isnan(this_cost):
            print((pc_eig_term,eig_term))
        return this_cost

    def Cost_(x):
        args = utils.parse_thing(x,shapes)
        return Cost(*args)

    def Cost_prime_(x):
        this_grad = grad(Cost_)(x)
        if np.any(np.isnan(this_grad)):
            print(x)
        return this_grad

    def callback(x):
        print(Cost_prime_(x))

    this_lb,this_ub = [np.concatenate([llb.flatten() for llb in bb]) for bb in [lb,ub]]
    these_bounds = list(zip(this_lb,this_ub))

    x0 = opt_param.flatten()

    Wmx0,Wmy0,K0,s020,amplitude0,baseline0 = utils.parse_thing(x0,shapes)
    w0 = compute_eigs(Wmy0,K0,YYp=YYpmodeled)
#     w0pc = compute_eigs(Wmy0[0:1,0:1],K0[:,0:1],YYp=YYpmodeled[:,0::nQ])
    w0pc = (YYpmodeled[:,0]*Wmy0[0,0] - 1)[:,np.newaxis]

    result = sop.minimize(Cost_,x0,method='L-BFGS-B',bounds=these_bounds,jac=Cost_prime_)#,callback=callback)
    x1 = result.x
    opt_param = x1

    Wmx1,Wmy1,K1,s021,amplitude1,baseline1 = utils.parse_thing(opt_param,shapes)
    YY,YYp = compute_YY(Wmx1,Wmy1,K1,s021,amplitude1,baseline1)
    w1 = compute_eigs(Wmy1,K1,YYp=YYp)
#     w1pc = compute_eigs(Wmy1[0:1,0:1],K1[:,0:1],YYp=YYp[:,0::nQ])
    w1pc = (YYp[:,0]*Wmy1[0,0] - 1)[:,np.newaxis]

    return opt_param,result
