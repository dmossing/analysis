#!/usr/bin/env python

import calnet.utils
import numpy as np

# helper fns

# def compute_f_(Eta,Xi,s02):
#     return utils.f_miller_troyer(Eta,Xi**2+np.concatenate((s02,s02),axis=0))

# def compute_fprime_m_(Eta,Xi,s02):
#     return utils.fprime_miller_troyer(Eta,Xi**2+np.concatenate((s02,s02),axis=0))*Xi

# def compute_fprime_s_(Eta,Xi,s02):
#     s2 = Xi**2+np.concatenate((s02,s02),axis=0)
#     return utils.fprime_s_miller_troyer(Eta,s2)*(Xi/s2)

# def sorted_r_eigs(w):
#     drW,prW = np.linalg.eig(w)
#     srtinds = np.argsort(drW)
#     return drW[srtinds],prW[:,srtinds]

# def gen_Weight(W,K,kappa):
#     WW0 = np.concatenate((W,W*K[np.newaxis,:]),axis=1)
#     WW1 = np.concatenate((W*K[np.newaxis,:]*kappa,W),axis=1)
#     WW = np.concatenate((WW0,WW1),axis=0)
#     return WW

# def u_fn(XX,YY,Wx,Wy,k,kappa):
#     WWx,WWy = [gen_Weight(W,k,kappa) for W in [Wx,Wy]]
#     return u_fn_WW(XX,YY,WWx,WWy)# XX @ WWx + YY @ WWy

# def u_fn_WW(XX,YY,WWx,WWy):
#     return XX @ WWx + YY @ WWy

# def load_weights(npyfile):
#     datafile = np.load(npyfile,allow_pickle=True)[()]
#     return datafile['as_list']

# dynamics fns

# def compute_steady_state(Wlist,Niter=int(3e3),max_val=2.5,Ny=50,fix_dim=None,stim_vals=None):
#     Wmx,Wmy,Wsx,Wsy,s02,k,kappa,XX,XXp,Eta,Xi = Wlist
#     nQ = Wmy.shape[0]
#     nN = Eta.shape[0]
#     nS = int(Eta.shape[1]/nQ)
    
#     if fix_dim is None:
#         fix_dim = [None]
#         yvals = (0,)
#         Ny = 1
#         max_val = 0
#     else:
#         yvals = np.linspace(0,max_val,Ny)
#     Nfix = len(fix_dim)
    
#     if stim_vals is None:
#         stim_vals = np.arange(nN)
#     Nstim = len(stim_vals)
    
#     yvals = np.linspace(0,max_val,Ny)
    
#     fval = compute_f_(Eta,Xi,s02)
#     resEta = Eta - u_fn(XX,fval,Wmx,Wmy,k,kappa)
#     resXi = Xi - u_fn(XX,fval,Wsx,Wsy,k,kappa)
    
#     def fY(XX,YY):
#         return compute_f_(resEta[stim_val]+u_fn(XX,YY,Wmx,Wmy,k,kappa),resXi[stim_val]+u_fn(XX,YY,Wsx,Wsy,k,kappa),s02)
#     def predict_YY_fix_dim(XX,YY0,eta=1e-2,fix_dim=None,run_backward=False):
#         def dYYdt(YY):
#             return -YY + fY(XX,YY)
#         YY = np.zeros((Niter+1,nS*nQ))
#         YY[0] = YY0.copy() #np.zeros((nN,nS*nQ))
#         dYY = np.zeros_like(YY[0])
#         iiter = 0
#         while iiter < Niter: #np.abs(dYY).sum()>1e-8*np.abs(YY).sum():
#             dYY = eta*dYYdt(YY[iiter])
#             if not fix_dim is None:
#                 dYY[fix_dim] = 0
#             if run_backward:
#                 dYY = -dYY
#             YY[iiter+1] = YY[iiter] + dYY
#             iiter = iiter+1
#         return YY
    
#     YY_ss = np.zeros((Nfix,Nstim,Ny,Niter+1,nS*nQ))
#     YY0 = compute_f_(Eta,Xi,s02)
#     for istim,stim_val in enumerate(stim_vals): #range(nN):
#         print(istim)
#         for iy,yval in enumerate(yvals):
#             for ifix in range(Nfix):
#                 yy0 = YY0[stim_val] #+np.random.randn(yy0.shape)
#                 yy0[fix_dim[ifix]] = yval
#                 YY_ss[ifix,istim,iy] = predict_YY_fix_dim(XX[stim_val],yy0,fix_dim=fix_dim[ifix])
#     if fix_dim[0] is None:
#         YY_ss = YY_ss[0,:,0,:,:]
#     return YY_ss

def compute_steady_state_Model(Model,Niter=int(3e3),max_val=2.5,Ny=50,fix_dim=None,stim_vals=None,dt=1e-1,sim_type='fix',inj_mag=np.array((0,))):
    desired = ['Wmx','Wmy','Wsx','Wsy','s02','K','kappa','XX','XXp','Eta','Xi']
    Wmx,Wmy,Wsx,Wsy,s02,k,kappa,XX,XXp,Eta,Xi = [getattr(Model,key) for key in desired]
    
    if fix_dim is None:
        fix_dim = [None]
        Nfix = 1
        yvals = (0,)
        Ny = 1
        max_val = 0
    else:
        if sim_type is 'fix':
            Nfix = len(fix_dim)
            yvals = np.linspace(0,max_val,Ny)
        elif sim_type is 'inj':
            Nfix = len(inj_mag)
    
    if stim_vals is None:
        stim_vals = np.arange(XX.shape[0])
    Nstim = len(stim_vals)
    
    yvals = np.linspace(0,max_val,Ny)
    
    def fY(XX,YY,stim_val,current_inj=None):
        return Model.fXY(XX,YY,istim=stim_val,current_inj=current_inj)
    
    def predict_YY_fix_dim(XX,YY0,stim_val,dt=dt,fix_dim=0,run_backward=False):
        def dYYdt(YY):
            return -YY + fY(XX,YY,stim_val)
        YY = np.zeros((Niter+1,YY0.shape[0]))
        YY[0] = YY0.copy() #np.zeros((nN,nS*nQ))
        dYY = np.zeros_like(YY[0])
        iiter = 0
        while iiter < Niter: #np.abs(dYY).sum()>1e-8*np.abs(YY).sum():
            dYY = dt*dYYdt(YY[iiter])
            if not fix_dim is None:
                dYY[fix_dim] = 0
            if run_backward:
                dYY = -dYY
            YY[iiter+1] = YY[iiter] + dYY
            iiter = iiter+1
        return YY
    
    def predict_YY_current_injection(XX,YY0,stim_val,dt=dt,inj_dim=0,run_backward=False,inj_mag=0):
        def dYYdt(YY,current_inj=None):
            return -YY + fY(XX,YY,stim_val,current_inj=current_inj)
        YY = np.zeros((Niter+1,YY0.shape[0]))
        YY[0] = YY0.copy() #np.zeros((nN,nS*nQ))
        dYY = np.zeros_like(YY[0])
        iiter = 0
        current_inj = np.zeros((YY0.shape[0],))
        if not inj_dim is None:
            current_inj[inj_dim] = inj_mag
        while iiter < Niter: #np.abs(dYY).sum()>1e-8*np.abs(YY).sum():
            dYY = dt*dYYdt(YY[iiter],current_inj=current_inj)
            if run_backward:
                dYY = -dYY
            YY[iiter+1] = YY[iiter] + dYY
            iiter = iiter+1
        return YY
    
    if sim_type is 'fix':
        YY_ss = np.zeros((Nfix,Nstim,Ny,Niter+1,Model.YY.shape[1]))
    elif sim_type is 'inj':
        YY_ss = np.zeros((Nfix,Nstim,Niter+1,Model.YY.shape[1]))
    YY0 = Model.YY #compute_f_(Eta,Xi,s02)
    for istim,stim_val in enumerate(stim_vals): #range(nN):
        print(istim)
        if sim_type is 'fix':
            for iy,yval in enumerate(yvals):
                for ifix in range(Nfix):
                    yy0 = YY0[stim_val] #+np.random.randn(yy0.shape)
                    yy0[fix_dim[ifix]] = yval
                    YY_ss[ifix,istim,iy] = predict_YY_fix_dim(XX[stim_val],yy0,stim_val,fix_dim=fix_dim[ifix],dt=dt)
        elif sim_type is 'inj':
            for ifix in range(Nfix):
                yy0 = YY0[stim_val] #+np.random.randn(yy0.shape)
                YY_ss[ifix,istim] = predict_YY_current_injection(XX[stim_val],yy0,stim_val,inj_dim=fix_dim,dt=dt,inj_mag=inj_mag[ifix])

    if sim_type is 'fix':
        if fix_dim[0] is None:
            YY_ss = YY_ss[0,:,0,:,:]
    elif sim_type is 'inj':
        if fix_dim is None:
            YY_ss = YY_ss[0,:,:,:]

    return YY_ss
