#!/usr/bin/env python

import calnet.utils
import numpy as np

# dynamics fns

def compute_steady_state_Model(Model,Niter=int(3e3),max_val=2.5,Ny=50,fix_dim=None,stim_vals=None,dt=1e-1,sim_type='fix',inj_mag=np.array((0,)),residuals_on=True):
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

    if residuals_on:
        res_factor = 1.
    else:
        res_factor = 0.
    
    if stim_vals is None:
        stim_vals = np.arange(XX.shape[0])
    Nstim = len(stim_vals)
    
    yvals = np.linspace(0,max_val,Ny)
    
    def fY(XX,YY,stim_val,current_inj=None):
        return Model.fXY(XX,YY,istim=stim_val,current_inj=current_inj,res_factor=res_factor)
    
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
        #print(istim)
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
