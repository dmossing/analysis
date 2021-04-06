#!/usr/bin/env python

# in this notebook, I will try to fit a model relating the mean behavior of L4, L2/3, SST and VIP cells

import pyute as ut
import autograd.numpy as np
import matplotlib.pyplot as plt
import sklearn
import h5py
import pdb
import scipy.optimize as sop
from mpl_toolkits.mplot3d import Axes3D
import sklearn.discriminant_analysis as skd
import autograd.scipy.special as ssp
from autograd import elementwise_grad as egrad
from autograd import grad
from autograd import jacobian
from autograd import hessian
import size_contrast_analysis as sca
import scipy.stats as sst
import sim_utils
from importlib import reload
reload(sim_utils)
import calnet.utils
import calnet.fitting_2step_spatial_feature_opto_multiout_nonlinear
import calnet.fitting_multiout_callable as fmc
import opto_utils
import scipy.signal as ssi
import scipy.optimize as sop
import warnings

warnings.filterwarnings('error',message='',category=RuntimeWarning,module='scipy.optimize')

def invert_f_mt(y):
    xstar = np.zeros_like(y)
    for iy,yy in enumerate(y):
        if not isinstance(yy,np.ndarray):
            to_invert = lambda x: sim_utils.f_miller_troyer(x,1)-yy
            xstar[iy] = sop.root_scalar(to_invert,x0=yy,x1=0).root
        else:
            xstar[iy] = invert_f_mt(yy)
    return xstar

def compute_fprime_m__(Eta,Xi,s02,nS=2,nT=2):
    return sim_utils.fprime_miller_troyer(Eta,Xi**2+np.concatenate([s02 for ipixel in range(nS*nT)]))*Xi

def invert_fprime_mt(Ypc_list,Eta0,nN=36,nQ=4,nS=2,nT=2,fudge=1e-2):
    Yp = get_pc_dim(Ypc_list,nN=nN,nPQ=nQ,nS=nS,nT=nT,idim=0)
    fp = compute_fprime_m__(Eta0,np.zeros_like(Eta0),np.ones((nQ,)),nS=nS,nT=nT)
    #Ypc_list[iS][icelltype-1] = [(s[idim],v[idim]) for idim in range(ndims)]
    Xi0 = Yp/(fp + fudge)
    return Xi0

def get_pc_dim(Ypc_list,nN=36,nPQ=4,nS=2,nT=2,idim=0):
    Yp = np.zeros((nN*nT,nPQ*nS))
    for iS in range(nS):
        for iQ in range(nPQ):
            #print('Ypc shape %d, %d: '%(iS,iQ)+str((Ypc_list[iS][iQ][idim][0]*Ypc_list[iS][iQ][idim][1]).shape))
            Yp[:,iS*nPQ+iQ] = Ypc_list[iS][iQ][idim][0]*Ypc_list[iS][iQ][idim][1] 
    Yp = calnet.utils.unfold_T_(Yp,nS=nS,nT=nT,nPQ=nPQ)
    return Yp

def initialize_W(Xhat,Yhat,scale_by=0.2):
    nP = Xhat[0][0].shape[1]
    nQ = Yhat[0][0].shape[1]
    nN = Yhat[0][0].shape[0]
    nS = len(Yhat)
    nT = int(Ypc_list[0][0][0][1].shape[0]/nN)
    YYhat = calnet.utils.flatten_nested_list_of_2d_arrays(Yhat)
    XXhat = calnet.utils.flatten_nested_list_of_2d_arrays(Xhat)
    Wmy0 = np.zeros((nQ,nQ))
    Wmx0 = np.zeros((nP,nQ))
    Ymatrix = np.zeros((nN,nQ))
    for itype in range(nQ):
        Ymatrix[:,itype] = invert_f_mt(YYhat[:,itype])
        others = np.setdiff1d(np.arange(nQ),itype)
        Xmatrix = np.concatenate((XXhat[:,:nP],YYhat[:,others]),axis=1)
        Bmatrix = np.linalg.pinv(Xmatrix) @ Ymatrix[:,itype]
        Wmx0[:,itype] = Bmatrix[:nP]
        Wmy0[others,itype] = Bmatrix[nP:]
        #Ymatrix_pred[:,itype] = Xmatrix @ Bmatrix
    return scale_by*Wmx0,scale_by*Wmy0

def compute_W_lsq(EtaXi,XXhat,YYhat,nP=2,nQ=4,freeze_vals=None,lam=0):
    # EtaXi the input currents
    # XXhat the input layer activity
    # YYhat the recurrent layer activity providing synaptic inputs
    Ymatrix = np.zeros(EtaXi.shape)
    Ymatrix_pred = np.zeros(EtaXi.shape)
    Wx = np.zeros((nP,nQ))
    Wy = np.zeros((nQ,nQ))
    if freeze_vals is None:
        resEtaXi = EtaXi - 0
    else:
        zeroed = [np.isnan(fv) for fv in freeze_vals]
        #print(zeroed)
        freeze_vals[0][zeroed[0]] = 0
        freeze_vals[1][zeroed[1]] = 0
        resEtaXi = EtaXi - XXhat @ freeze_vals[0] - YYhat @ freeze_vals[1]
        Wx[~zeroed[0]] = freeze_vals[0][~zeroed[0]]
        Wy[~zeroed[1]] = freeze_vals[1][~zeroed[1]]
    for itype in range(nQ):
        Ymatrix[:,itype] = resEtaXi[:,itype]
        #others = np.setdiff1d(np.arange(nQ),itype)
        xothers = np.ones((nP,),dtype='bool')
        others = ~np.in1d(np.arange(nQ),itype)
        if not freeze_vals is None:
            xothers = xothers & (freeze_vals[0][:,itype] == 0) & ~zeroed[0][:,itype]
            others = others & (freeze_vals[1][:,itype] == 0) & ~zeroed[1][:,itype]
        Xmatrix = np.concatenate((XXhat[:,xothers],YYhat[:,others]),axis=1)
        #print('X shape: '+str(Xmatrix.shape))
        #print('Y shape: '+str(Ymatrix.shape))
        if not np.all(Xmatrix==0):
            #Bmatrix = np.linalg.pinv(Xmatrix) @ Ymatrix[:,itype]
            Bmatrix = pinv_tik(Xmatrix,lam=lam) @ Ymatrix[:,itype]
            this_nP = int(xothers.sum())
            Wx[xothers,itype] = Bmatrix[:this_nP]
            Wy[others,itype] = Bmatrix[this_nP:]
            Ymatrix_pred[:,itype] = Xmatrix @ Bmatrix
    #print((Wx,Wy))
    resEtaXi = resEtaXi - Ymatrix_pred
    return Wx,Wy,resEtaXi

def pinv_tik(X,lam=0):
    #print('XTX: '+str(X.T @ X))
    if lam==0:
        return np.linalg.pinv(X)
    else:
        return np.linalg.inv(X.T @ X + lam*np.eye(X.shape[1])) @ X.T

def compute_KT_lsq(EtaXi,XXhat,YYhat,Wx,Wy,nP=2,nQ=4,lam=0):
    KT = np.zeros((nQ,))
    Ymatrix = np.zeros(EtaXi.shape)
    Ymatrix_pred = np.zeros(EtaXi.shape)
    for itype in range(nQ):
        Ymatrix[:,itype] = EtaXi[:,itype]
        #others = np.setdiff1d(np.arange(nQ),itype)
        #Xmatrix = XXhat @ Wx[:,itype:itype+1] + YYhat[:,others] @ Wy[others,itype:itype+1]
        Xmatrix = XXhat @ Wx[:,itype:itype+1] + YYhat @ Wy[:,itype:itype+1]
        #Bmatrix = np.linalg.pinv(Xmatrix) @ Ymatrix[:,itype]
        #print('KT X shape: '+str(Xmatrix.shape))
        #print('KT Y shape: '+str(Ymatrix.shape))
        if not np.all(Xmatrix==0):
            Bmatrix = pinv_tik(Xmatrix,lam=lam) @ Ymatrix[:,itype]
            KT[itype] = Bmatrix[0]
            Ymatrix_pred[:,itype] = Xmatrix @ Bmatrix
    resEtaXi = EtaXi - Ymatrix_pred
    return KT,resEtaXi

def pixelwise_flatten(XY,nN=36,nPQ=2,nS=2,nT=2,sameS=True,sameT=True):
    XYr = XY.reshape((nN,nS,nT,nPQ))
    if not sameS:
        XYr = XYr[:,::-1,:,:]
    if not sameT:
        XYr = XYr[:,:,::-1,:]
    XYr = XYr.transpose((1,2,0,3)).reshape((nS*nT*nN,nPQ))
    return XYr

def norm_to_W0(W1,W0):
    to_norm = (W0 != 0)
    W1[to_norm] = W1[to_norm]/W0[to_norm]
    return W1

def initialize_Ws(Xhat,Yhat,Xpc_list,Ypc_list,scale_by=1,freeze_vals=[None for _ in range(4)],lams=np.zeros((8,)),nondim=False):
    nP = Xhat[0][0].shape[1]
    nQ = Yhat[0][0].shape[1]
    nN = Yhat[0][0].shape[0]
    nS = len(Yhat)
    nT = len(Yhat[0])

    XXhat = calnet.utils.flatten_nested_list_of_2d_arrays(Xhat)
    YYhat = calnet.utils.flatten_nested_list_of_2d_arrays(Yhat)
    XXphat = get_pc_dim(Xpc_list,nN=nN,nPQ=nP,nS=nS,nT=nT,idim=0)
    YYphat = get_pc_dim(Ypc_list,nN=nN,nPQ=nQ,nS=nS,nT=nT,idim=0)

    Eta0 = invert_f_mt(YYhat)
    Xi0 = invert_fprime_mt(Ypc_list,Eta0,nN=nN,nQ=nQ,nS=nS,nT=nT)

    sameSs = [True,False,True]
    sameTs = [True,True,False]

    datas = [XXhat,YYhat,XXphat,YYphat,Eta0,Xi0]
    nPQs = [nP,nQ,nP,nQ,nQ,nQ]

    XYs = [None for data in datas]

    for idata in range(len(datas)):
        XYs[idata] = [None for sameS in sameSs]
        for isame,(sameS,sameT) in enumerate(zip(sameSs,sameTs)):
            XYs[idata][isame] = pixelwise_flatten(datas[idata],nN=nN,nPQ=nPQs[idata],nS=nS,nT=nT,sameS=sameS,sameT=sameT)
    # XYs: [XXhat,YYhat,XXphat,YYphat], each flattened into a list of: [original values, swapped S, swapped T]
    thisEta0 = XYs[4][0]
    thisXi0 = XYs[5][0]


    # derivations in ipad notes from 21/4/1
    idata1,idata2 = 0,1 # X,Y
    idiff1,idiff2 = 0,0 # same,same
    W0x,W0y,resEta0 = compute_W_lsq(thisEta0,XYs[idata1][idiff1],XYs[idata2][idiff2],nP=nP,nQ=nQ,freeze_vals=freeze_vals[0],lam=lams[0])
    idiff1,idiff2 = 1,1 # diffS,diffS
    K0,resEta0 = compute_KT_lsq(resEta0,XYs[idata1][idiff1],XYs[idata2][idiff2],W0x,W0y,nP=nP,nQ=nQ,lam=lams[4])
    idiff1,idiff2 = 2,2 # diffT,diffT
    T0,resEta0 = compute_KT_lsq(resEta0,XYs[idata1][idiff1],XYs[idata2][idiff2],W0x,W0y,nP=nP,nQ=nQ,lam=lams[6])

    idata1,idata2 = 0,1 # X,Y
    idiff1,idiff2 = 0,0 # same,same
    W1x,W1y,resXi0 = compute_W_lsq(thisXi0,XYs[idata1][idiff1],XYs[idata2][idiff2],nP=nP,nQ=nQ,freeze_vals=freeze_vals[1],lam=lams[1])
    idiff1,idiff2 = 1,1 # diffS,diffS
    K1,resXi0 = compute_KT_lsq(resXi0,XYs[idata1][idiff1],XYs[idata2][idiff2],W0x,W0y,nP=nP,nQ=nQ,lam=lams[5])
    idiff1,idiff2 = 2,2 # diffT,diffT
    T1,resXi0 = compute_KT_lsq(resXi0,XYs[idata1][idiff1],XYs[idata2][idiff2],W0x,W0y,nP=nP,nQ=nQ,lam=lams[7])

    idata1,idata2 = 2,3 # Xp,Yp
    idiff1,idiff2 = 0,0 # same,same
    W2x,W2y,resEta0 = compute_W_lsq(resEta0,XYs[idata1][idiff1],XYs[idata2][idiff2],nP=nP,nQ=nQ,freeze_vals=freeze_vals[2],lam=lams[2])

    idata1,idata2 = 2,3 # Xp,Yp
    idiff1,idiff2 = 0,0 # same,same
    W3x,W3y,resXi0 = compute_W_lsq(resXi0,XYs[idata1][idiff1],XYs[idata2][idiff2],nP=nP,nQ=nQ,freeze_vals=freeze_vals[3],lam=lams[3])

    W1x,W1y,K1,T1,W2x,W2y,W3x,W3y = [norm_to_W0(x,y) for x,y in zip([W1x,W1y,K1,T1,W2x,W2y,W3x,W3y],[W0x,W0y,K0,T0,W0x,W0y,W0x,W0y])]
    
    Wlist = [W0x,W0y,W1x,W1y,W2x,W2y,W3x,W3y,K0,K1,T0,T1]
    Wlist = [scale_by*w for w in Wlist]
    #print(Wlist)

    return Wlist

def fit_weights_and_save(weights_file,ca_data_file='rs_vm_denoise_200605.npy',opto_silencing_data_file='vip_halo_data_for_sim.npy',opto_activation_data_file='vip_chrimson_data_for_sim.npy',constrain_wts=None,allow_var=True,multiout=True,multiout2=False,fit_s02=True,constrain_isn=True,tv=False,l2_penalty=0.01,init_noise=0.1,init_W_from_lsq=False,scale_init_by=1,init_W_from_file=False,init_file=None,foldT=False,free_amplitude=False,correct_Eta=False,init_Eta_with_s02=False,no_halo_res=False,ignore_halo_vip=False,use_opto_transforms=False,norm_opto_transforms=False,nondim=False):
    
    
    nsize,ncontrast = 6,6
    
    npfile = np.load(ca_data_file,allow_pickle=True)[()]#,{'rs':rs},allow_pickle=True) # ,'rs_denoise':rs_denoise
    rs = npfile['rs']
    
    nsize,ncontrast,ndir = 6,6,8
    ori_dirs = [[0,4],[2,6]] #[[0,4],[1,3,5,7],[2,6]]
    nT = len(ori_dirs)
    nS = len(rs[0])
    
    def sum_to_1(r):
        R = r.reshape((r.shape[0],-1))
        R = R/np.nansum(R,axis=1)[:,np.newaxis] # changed 8/28
        return R
    
    def norm_to_mean(r):
        R = r.reshape((r.shape[0],-1))
        R = R/np.nanmean(R[:,~np.isnan(R.sum(0))],axis=1)[:,np.newaxis]
        return R
    
    Rs = [[None,None] for i in range(len(rs))]
    Rso = [[[None for iT in range(nT)] for iS in range(nS)] for icelltype in range(len(rs))]
    rso = [[[None for iT in range(nT)] for iS in range(nS)] for icelltype in range(len(rs))]
    
    for iR,r in enumerate(rs):#rs_denoise):
        print(iR)
        for ialign in range(nS):
            Rs[iR][ialign] = sum_to_1(r[ialign][:,:nsize,:])
    
    kernel = np.ones((1,2,2))
    kernel = kernel/kernel.sum()
    
    for iR,r in enumerate(rs):
        for ialign in range(nS):
            for iori in range(nT):
                Rso[iR][ialign][iori] = np.nanmean(Rs[iR][ialign].reshape((-1,nsize,ncontrast,ndir))[:,:,:,ori_dirs[iori]],-1)
                Rso[iR][ialign][iori][:,:,0] = np.nanmean(Rso[iR][ialign][iori][:,:,0],1)[:,np.newaxis]

                Rso[iR][ialign][iori][:,1:,1:] = ssi.convolve(Rso[iR][ialign][iori],kernel,'valid')
                Rso[iR][ialign][iori] = Rso[iR][ialign][iori].reshape(Rso[iR][ialign][iori].shape[0],-1) #nroi x nstim

    def set_bound(bd,code,val=0):
        # set bounds to 0 where 0s occur in 'code'
        for iitem in range(len(bd)):
            bd[iitem][code[iitem]] = val
    
    nN = 36
    nS = 2
    nP = 2
    nT = 2
    nQ = 4
    
    # code for bounds: 0 , constrained to 0
    # +/-1 , constrained to +/-1
    # 1.5, constrained to [0,1]
    # -1.5, constrained to [-1,1]
    # 2 , constrained to [0,inf)
    # -2 , constrained to (-inf,0]
    # 3 , unconstrained
    
    W0x_bounds = 3*np.ones((nP,nQ),dtype=int)
    W0x_bounds[0,:] = 2 # L4 PCs are excitatory
    W0x_bounds[0,1] = 0 # SSTs don't receive L4 input
    
    if allow_var:
        W1x_bounds = 3*np.ones(W0x_bounds.shape) #W0x_bounds.copy()*0 #np.zeros_like(W0x_bounds)
        W1x_bounds[0,1] = 0
    else:
        W1x_bounds = np.zeros(W0x_bounds.shape) #W0x_bounds.copy()*0 #np.zeros_like(W0x_bounds)
    
    W0y_bounds = 3*np.ones((nQ,nQ),dtype=int)
    W0y_bounds[0,:] = 2 # PCs are excitatory
    W0y_bounds[1:,:] = -2 # all the cell types except PCs are inhibitory
    W0y_bounds[1,1] = 0 # SSTs don't inhibit themselves
    # W0y_bounds[3,1] = 0 # PVs are allowed to inhibit SSTs, consistent with Hillel's unpublished results, but not consistent with Pfeffer et al.
    W0y_bounds[2,0] = 0 # VIPs don't inhibit L2/3 PCs. According to Pfeffer et al., only L5 PCs were found to get VIP inhibition


    if not constrain_wts is None:
        for wt in constrain_wts:
            W0y_bounds[wt[0],wt[1]] = 0
            Wsy_bounds[wt[0],wt[1]] = 0
    
    def tile_nS_nT_nN(kernel):
        row = np.concatenate([kernel for idim in range(nS*nT)],axis=0)[np.newaxis,:]
        tiled = np.concatenate([row for irow in range(nN)],axis=0)
        return tiled
    
    if fit_s02:
        s02_bounds = 2*np.ones((nQ,)) # permitting noise as a free parameter
    else:
        s02_bounds = np.ones((nQ,))
    
    k0_bounds = 1.5*np.ones((nQ,))
    
    kappa_bounds = np.ones((1,))
    # kappa_bounds = 2*np.ones((1,))
    
    T0_bounds = 1.5*np.ones((nQ,))
    #T_bounds[2:4] = 1 # PV and VIP are constrained to have flat ori tuning
    #T0_bounds[1:4] = 1 # SST,VIP, and PV are constrained to have flat ori tuning

    if allow_var:
        if nondim:
            W1y_bounds = -1.5*np.ones(W0y_bounds.shape) #W0y_bounds.copy()*0 #np.zeros_like(W0y_bounds)
            k1_bounds = -1.5*np.ones(k0_bounds.shape) #W0y_bounds.copy()*0 #np.zeros_like(W0y_bounds)
            T1_bounds = -1.5*np.ones(T0_bounds.shape) #W0y_bounds.copy()*0 #np.zeros_like(W0y_bounds)
        else:
            W1y_bounds = 3*np.ones(W0y_bounds.shape) #W0y_bounds.copy()*0 #np.zeros_like(W0y_bounds)
            k1_bounds = 3*np.ones(k0_bounds.shape) #W0y_bounds.copy()*0 #np.zeros_like(W0y_bounds)
            T1_bounds = 3*np.ones(T0_bounds.shape) #W0y_bounds.copy()*0 #np.zeros_like(W0y_bounds)
        W1y_bounds[1,1] = 0
        #W1y_bounds[3,1] = 0 
        W1y_bounds[2,0] = 0
        W1y_bounds[2,2] = 0 # newly added: no VIP-VIP inhibition
    else:
        W1y_bounds = np.zeros(W0y_bounds.shape) #W0y_bounds.copy()*0 #np.zeros_like(W0y_bounds)
        k1_bounds = 0*np.ones(k0_bounds.shape) #W0y_bounds.copy()*0 #np.zeros_like(W0y_bounds)
        T1_bounds = 0*np.ones(T0_bounds.shape) #W0y_bounds.copy()*0 #np.zeros_like(W0y_bounds)

    if multiout:
        W2x_bounds = W1x_bounds.copy()
        W2y_bounds = W1y_bounds.copy()
        if multiout2:
            W3x_bounds = W1x_bounds.copy()
            W3y_bounds = W1y_bounds.copy()
        else:
            W3x_bounds = W1x_bounds.copy()*0
            W3y_bounds = W1y_bounds.copy()*0
    else:
        W2x_bounds = W1x_bounds.copy()*0
        W2y_bounds = W1y_bounds.copy()*0
        W3x_bounds = W1x_bounds.copy()*0
        W3y_bounds = W1y_bounds.copy()*0
    k2_bounds = k1_bounds.copy()*0
    T2_bounds = T1_bounds.copy()*0
    k3_bounds = k1_bounds.copy()*0
    T3_bounds = T1_bounds.copy()*0
    
    X_bounds = tile_nS_nT_nN(np.array([2,1]))
    # X_bounds = np.array([np.array([2,1,2,1])]*nN)
    
    Xp_bounds = tile_nS_nT_nN(np.array([3,0])) # edited to set XXp to 0 for spont. term
    # Xp_bounds = np.array([np.array([3,1,3,1])]*nN)
    
    # Y_bounds = tile_nS_nT_nN(2*np.ones((nQ,)))
    # # Y_bounds = 2*np.ones((nN,nT*nS*nQ))
    
    Eta_bounds = tile_nS_nT_nN(3*np.ones((nQ,)))
    # Eta_bounds = 3*np.ones((nN,nT*nS*nQ))
    
    #if allow_var:
    #    Xi_bounds = tile_nS_nT_nN(3*np.ones((nQ,)))
    #else:
    #    Xi_bounds = tile_nS_nT_nN(np.zeros((nQ,)))
    Xi_bounds = tile_nS_nT_nN(3*np.ones((nQ,))) # temporarily allowing Xi even if W1 is not allowed

    # Xi_bounds = 3*np.ones((nN,nT*nS*nQ))
    
    h1_bounds = -2*np.ones((1,))

    h2_bounds = 2*np.ones((1,))

    bl_bounds = 3*np.ones((nQ,))

    if free_amplitude:
        amp_bounds = 2*np.ones((nT*nS*nQ,))
    else:
        amp_bounds = 1*np.ones((nT*nS*nQ,))
    
    # shapes = [(nP,nQ),(nQ,nQ),(nP,nQ),(nQ,nQ),(nQ,),(nQ,),(1,),(nN,nS*nP),(nN,nS*nQ),(nN,nS*nQ),(nN,nS*nQ)]
    #shapes = [(nP,nQ),(nQ,nQ),(nP,nQ),(nQ,nQ),(nP,nQ),(nQ,nQ),(nP,nQ),(nQ,nQ),(nQ,),(nQ*(nS-1),),(nQ*(nS-1),),(nQ*(nS-1),),(nQ*(nS-1),),(1,),(nQ*(nT-1),),(nQ*(nT-1),),(nQ*(nT-1),),(nQ*(nT-1),),(nN,nT*nS*nP),(nN,nT*nS*nP),(nN,nT*nS*nQ),(nN,nT*nS*nQ),(1,)]
    shapes1 = [(nP,nQ),(nQ,nQ),(nP,nQ),(nQ,nQ),(nP,nQ),(nQ,nQ),(nP,nQ),(nQ,nQ),(nQ,),(nQ*(nS-1),),(nQ*(nS-1),),(nQ*(nS-1),),(nQ*(nS-1),),(1,),(nQ*(nT-1),),(nQ*(nT-1),),(nQ*(nT-1),),(nQ*(nT-1),),(1,),(1,),(nQ,),(nT*nS*nQ,)]
    shapes2 = [(nN,nT*nS*nP),(nN,nT*nS*nP),(nN,nT*nS*nQ),(nN,nT*nS*nQ)]
    #         W0x,    W0y,    W1x,    W1y,    W2x,    W2y,    W3x,    W3y,    s02,  k,    kappa,T,   XX,            XXp,          Eta,          Xi
    
    #lb = [-np.inf*np.ones(shp) for shp in shapes]
    #ub = [np.inf*np.ones(shp) for shp in shapes]
    #bdlist = [W0x_bounds,W0y_bounds,W1x_bounds,W1y_bounds,W2x_bounds,W2y_bounds,W3x_bounds,W3y_bounds,s02_bounds,k0_bounds,k1_bounds,k2_bounds,k3_bounds,kappa_bounds,T0_bounds,T1_bounds,T2_bounds,T3_bounds,X_bounds,Xp_bounds,Eta_bounds,Xi_bounds,h_bounds]
    bd1list = [W0x_bounds,W0y_bounds,W1x_bounds,W1y_bounds,W2x_bounds,W2y_bounds,W3x_bounds,W3y_bounds,s02_bounds,k0_bounds,k1_bounds,k2_bounds,k3_bounds,kappa_bounds,T0_bounds,T1_bounds,T2_bounds,T3_bounds,h1_bounds,h2_bounds,bl_bounds,amp_bounds]
    bd2list = [X_bounds,Xp_bounds,Eta_bounds,Xi_bounds]

    lb1,ub1 = [[sgn*np.inf*np.ones(shp) for shp in shapes1] for sgn in [-1,1]]
    lb1,ub1 = calnet.utils.set_bounds_by_code(lb1,ub1,bd1list)
    lb2,ub2 = [[sgn*np.inf*np.ones(shp) for shp in shapes2] for sgn in [-1,1]]
    lb2,ub2 = calnet.utils.set_bounds_by_code(lb2,ub2,bd2list)

    lb1 = np.concatenate([a.flatten() for a in lb1])
    ub1 = np.concatenate([b.flatten() for b in ub1])
    lb2 = np.concatenate([a.flatten() for a in lb2])
    ub2 = np.concatenate([b.flatten() for b in ub2])
    bounds1 = [(a,b) for a,b in zip(lb1,ub1)]
    bounds2 = [(a,b) for a,b in zip(lb2,ub2)]
    
    nS = 2
    ndims = 5
    ncelltypes = 5
    #print('foldT: %d'%foldT)
    if foldT:
        Yhat = [None for iS in range(nS)]
        Xhat = [None for iS in range(nS)]
        Ypc_list = [None for iS in range(nS)]
        Xpc_list = [None for iS in range(nS)]
        for iS in range(nS):
            Yhat[iS] = [None for iT in range(nT)]
            Xhat[iS] = [None for iT in range(nT)]
            mx = np.zeros((ncelltypes,))
            yy = [None for icelltype in range(ncelltypes)]
            for icelltype in range(ncelltypes):
                yy[icelltype] = np.nanmean(Rso[icelltype][iS][0],0)
                mx[icelltype] = np.nanmax(yy[icelltype])
            for iT in range(nT):
                y = [np.nanmean(Rso[icelltype][iS][iT],axis=0)[:,np.newaxis]/mx[icelltype] for icelltype in range(1,ncelltypes)]
                Yhat[iS][iT] = np.concatenate(y,axis=1)
                icelltype = 0
                x = np.nanmean(Rso[icelltype][iS][iT],0)[:,np.newaxis]/mx[icelltype]
                Xhat[iS][iT] = np.concatenate((x,np.ones_like(x)),axis=1)

            Ypc_list[iS] = [None for icelltype in range(1,ncelltypes)]
            for icelltype in range(1,ncelltypes):
                # Rso: nroi x nN 
                rss = np.concatenate([Rso[icelltype][iS][iT].copy() for iT in range(nT)],axis=1)#.reshape(Rs[icelltype][ialign].shape[0],-1)
                rss = rss[np.isnan(rss).sum(1)==0]
                #print('rss shape: '+str(rss.shape))
                try:
                    u,s,v = np.linalg.svd(rss-np.mean(rss,0)[np.newaxis])
                    #print('v shape: '+str(v.shape))
                    Ypc_list[iS][icelltype-1] = [(s[idim],v[idim]) for idim in range(ndims)]
                except:
                    print('nope on Y')

            icelltype = 0
            rss = np.concatenate([Rso[icelltype][iS][iT].copy() for iT in range(nT)],axis=1)#.reshape(Rs[icelltype][ialign].shape[0],-1)
            rss = rss[np.isnan(rss).sum(1)==0]
            u,s,v = np.linalg.svd(rss-rss.mean(0)[np.newaxis])
            Xpc_list[iS] = [None for iinput in range(2)]
            Xpc_list[iS][0] = [(s[idim],v[idim]) for idim in range(ndims)]
            Xpc_list[iS][1] = [(0,np.zeros((Xpc_list[0][0][0][1].shape[0],))) for idim in range(ndims)]

    else:
        Yhat = [[None for iT in range(nT)] for iS in range(nS)]
        Xhat = [[None for iT in range(nT)] for iS in range(nS)]
        Ypc_list = [[None for iT in range(nT)] for iS in range(nS)]
        Xpc_list = [[None for iT in range(nT)] for iS in range(nS)]
        for iS in range(nS):
            mx = np.zeros((ncelltypes,))
            yy = [None for icelltype in range(ncelltypes)]
            for icelltype in range(ncelltypes):
                yy[icelltype] = np.nanmean(Rso[icelltype][iS][0],0)
                mx[icelltype] = np.nanmax(yy[icelltype])
            for iT in range(nT):
                y = [np.nanmean(Rso[icelltype][iS][iT],axis=0)[:,np.newaxis]/mx[icelltype] for icelltype in range(1,ncelltypes)]
                Ypc_list[iS][iT] = [None for icelltype in range(1,ncelltypes)]
                for icelltype in range(1,ncelltypes):
                    rss = Rso[icelltype][iS][iT].copy() #.reshape(Rs[icelltype][ialign].shape[0],-1)
                    rss = rss[np.isnan(rss).sum(1)==0]
                    try:
                        u,s,v = np.linalg.svd(rss-np.mean(rss,0)[np.newaxis])
                        Ypc_list[iS][iT][icelltype-1] = [(s[idim],v[idim]) for idim in range(ndims)]
                    except:
                        print('nope on Y')
                Yhat[iS][iT] = np.concatenate(y,axis=1)
                icelltype = 0
                x = np.nanmean(Rso[icelltype][iS][iT],0)[:,np.newaxis]/mx[icelltype]
                Xhat[iS][iT] = np.concatenate((x,np.ones_like(x)),axis=1)
                icelltype = 0
                rss = Rso[icelltype][iS][iT].copy()
                rss = rss[np.isnan(rss).sum(1)==0]
                u,s,v = np.linalg.svd(rss-rss.mean(0)[np.newaxis])
                Xpc_list[iS][iT] = [None for iinput in range(2)]
                Xpc_list[iS][iT][0] = [(s[idim],v[idim]) for idim in range(ndims)]
                Xpc_list[iS][iT][1] = [(0,np.zeros((Xhat[0][0].shape[0],))) for idim in range(ndims)]
    nN,nP = Xhat[0][0].shape
    nQ = Yhat[0][0].shape[1]
    
    def compute_f_(Eta,Xi,s02):
        return sim_utils.f_miller_troyer(Eta,Xi**2+np.concatenate([s02 for ipixel in range(nS*nT)]))
    def compute_fprime_m_(Eta,Xi,s02):
        return sim_utils.fprime_miller_troyer(Eta,Xi**2+np.concatenate([s02 for ipixel in range(nS*nT)]))*Xi
    def compute_fprime_s_(Eta,Xi,s02):
        s2 = Xi**2+np.concatenate((s02,s02),axis=0)
        return sim_utils.fprime_s_miller_troyer(Eta,s2)*(Xi/s2)
    def sorted_r_eigs(w):
        drW,prW = np.linalg.eig(w)
        srtinds = np.argsort(drW)
        return drW[srtinds],prW[:,srtinds]
    
    #0.W0x,1.W0y,2.W1x,3.W1y,4.W2x,5.W2y,6.W3x,7.W3y,8.s02,9.K0,10.K1,11.K2,12.K3,13.kappa,14.T0,15.T1,16.T2,17.T3,18.h1,19.h2,20.bl,21.amp
    #0.XX,1.XXp,2.Eta,3.Xi
    
    #shapes = [(nP,nQ),(nQ,nQ),(nP,nQ),(nQ,nQ),(nP,nQ),(nQ,nQ),(nP,nQ),(nQ,nQ),(nQ,),(nQ*(nS-1),),(nQ*(nS-1),),(nQ*(nS-1),),(nQ*(nS-1),),(1,),(nQ*(nT-1),),(nQ*(nT-1),),(nQ*(nT-1),),(nQ*(nT-1),),(nN,nT*nS*nP),(nN,nT*nS*nP),(nN,nT*nS*nQ),(nN,nT*nS*nQ),(1,)]
    shapes1 = [(nP,nQ),(nQ,nQ),(nP,nQ),(nQ,nQ),(nP,nQ),(nQ,nQ),(nP,nQ),(nQ,nQ),(nQ,),(nQ*(nS-1),),(nQ*(nS-1),),(nQ*(nS-1),),(nQ*(nS-1),),(1,),(nQ*(nT-1),),(nQ*(nT-1),),(nQ*(nT-1),),(nQ*(nT-1),),(1,),(1,),(nQ,),(nT*nS*nQ,)]
    shapes2 = [(nN,nT*nS*nP),(nN,nT*nS*nP),(nN,nT*nS*nQ),(nN,nT*nS*nQ)]
    
    import sim_utils

    YYhat = calnet.utils.flatten_nested_list_of_2d_arrays(Yhat)
    
    opto_dict = np.load(opto_silencing_data_file,allow_pickle=True)[()]

    Yhat_opto = opto_dict['Yhat_opto']
    Yhat_opto = Yhat_opto.reshape((nN*2,-1))
    Yhat_opto[0::12] = np.nanmean(Yhat_opto[0::12],axis=0)[np.newaxis]
    Yhat_opto[1::12] = np.nanmean(Yhat_opto[1::12],axis=0)[np.newaxis]
    Yhat_opto = Yhat_opto/np.nanmax(Yhat_opto[0::2],0)[np.newaxis,:]
    #print(Yhat_opto.shape)
    h_opto = opto_dict['h_opto']
    #dYY1 = Yhat_opto[1::2]-Yhat_opto[0::2]

    YYhat_halo = Yhat_opto.reshape((nN,2,-1))
    opto_transform1 = calnet.utils.fit_opto_transform(YYhat_halo,norm01=norm_opto_transforms)

    if no_halo_res:
        opto_transform1.res[:,[0,2,3,4,6,7]] = 0

    dYY1 = opto_transform1.transform(YYhat) - opto_transform1.preprocess(YYhat)
    #print('delta bias: %f'%dXX1[:,1].mean())
    #YYhat_halo_sim = calnet.utils.simulate_opto_effect(YYhat,YYhat_halo)
    #dYY1 = YYhat_halo_sim[:,1,:] - YYhat_halo_sim[:,0,:]

    def overwrite_plus_n(arr,to_overwrite,n):
        arr[:,to_overwrite] = arr[:,int(to_overwrite+n)]
        return arr

    for to_overwrite in [1,2]:
        n = 4
        dYY1,opto_transform1.slope,opto_transform1.intercept,opto_transform1.res \
                = [overwrite_plus_n(x,to_overwrite,n) for x in \
                        [dYY1,opto_transform1.slope,opto_transform1.intercept,opto_transform1.res]]
    for to_overwrite in [7]:
        n = -4
        dYY1,opto_transform1.slope,opto_transform1.intercept,opto_transform1.res \
                = [overwrite_plus_n(x,to_overwrite,n) for x in \
                        [dYY1,opto_transform1.slope,opto_transform1.intercept,opto_transform1.res]]

    opto_dict = np.load(opto_activation_data_file,allow_pickle=True)[()]

    Yhat_opto = opto_dict['Yhat_opto']
    Yhat_opto = Yhat_opto.reshape((nN*2,-1))
    Yhat_opto[0::12] = np.nanmean(Yhat_opto[0::12],axis=0)[np.newaxis]
    Yhat_opto[1::12] = np.nanmean(Yhat_opto[1::12],axis=0)[np.newaxis]
    Yhat_opto = Yhat_opto/Yhat_opto[0::2].max(0)[np.newaxis,:]
    #print(Yhat_opto.shape)
    h_opto = opto_dict['h_opto']
    #dYY2 = Yhat_opto[1::2]-Yhat_opto[0::2]

    YYhat_chrimson = Yhat_opto.reshape((nN,2,-1))
    opto_transform2 = calnet.utils.fit_opto_transform(YYhat_chrimson,norm01=norm_opto_transforms)

    dYY2 = opto_transform2.transform(YYhat) - opto_transform2.preprocess(YYhat)

    dYY = np.concatenate((dYY1,dYY2),axis=0)

    if ignore_halo_vip:
        dYY1[:,2::nQ] = np.nan
    
    from importlib import reload
    reload(calnet)
    reload(calnet.fitting_2step_spatial_feature_opto_multiout_nonlinear)
    reload(sim_utils)
    wt_dict = {}
    wt_dict['X'] = 1
    wt_dict['Y'] = 3
    wt_dict['Eta'] = 1# 10
    wt_dict['Xi'] = 0.1
    wt_dict['stims'] = np.ones((nN,1)) #(np.arange(30)/30)[:,np.newaxis]**1 #
    wt_dict['barrier'] = 0. #30.0 #0.1
    wt_dict['opto'] = 0#1e0#1e-1#1e1
    wt_dict['smi'] = 0
    wt_dict['isn'] = 0.1
    wt_dict['tv'] = 1


    YYhat = calnet.utils.flatten_nested_list_of_2d_arrays(Yhat)
    XXhat = calnet.utils.flatten_nested_list_of_2d_arrays(Xhat)
    Eta0 = invert_f_mt(YYhat)
    Xi0 = invert_fprime_mt(Ypc_list,Eta0,nN=nN,nQ=nQ,nS=nS,nT=nT)

    ntries = 1
    nhyper = 1
    dt = 1e-1
    niter = int(np.round(10/dt)) #int(1e4)
    perturbation_size = 5e-2
    W1t = [[None for itry in range(ntries)] for ihyper in range(nhyper)]
    W2t = [[None for itry in range(ntries)] for ihyper in range(nhyper)]
    loss = np.zeros((nhyper,ntries))
    is_neg = np.array([b[1] for b in bounds1])==0
    counter = 0
    negatize = [np.zeros(shp,dtype='bool') for shp in shapes1]
    for ishp,shp in enumerate(shapes1):
        nel = np.prod(shp)
        negatize[ishp][:][is_neg[counter:counter+nel].reshape(shp)] = True
        counter = counter + nel
    for ihyper in range(nhyper):
        for itry in range(ntries):
            print((ihyper,itry))
            W10list = [init_noise*(ihyper+1)*np.random.rand(*shp) for shp in shapes1]
            W20list = [init_noise*(ihyper+1)*np.random.rand(*shp) for shp in shapes2]
            counter = 0
            for ishp,shp in enumerate(shapes1):
                W10list[ishp][negatize[ishp]] = -W10list[ishp][negatize[ishp]]
            nextraW = 4
            nextraK = nextraW + 3
            nextraT = nextraK + 3
            W10list[nextraW+4] = np.ones(shapes1[nextraW+4]) # s02
            W10list[nextraW+5] = np.ones(shapes1[nextraW+5]) # K
            W10list[nextraW+6] = np.ones(shapes1[nextraW+6]) # K
            W10list[nextraW+7] = np.zeros(shapes1[nextraW+7]) # K
            W10list[nextraW+8] = np.zeros(shapes1[nextraW+8]) # K
            W10list[nextraK+6] = np.ones(shapes1[nextraK+6]) # kappa
            W10list[nextraK+7] = np.ones(shapes1[nextraK+7]) # T
            W10list[nextraK+8] = np.ones(shapes1[nextraK+8]) # T
            W10list[nextraK+9] = np.zeros(shapes1[nextraK+9]) # T
            W10list[nextraK+10] = np.zeros(shapes1[nextraK+10]) # T
            W20list[0] = XXhat #np.concatenate(Xhat,axis=1) #XX
            W20list[1] = get_pc_dim(Xpc_list,nN=nN,nPQ=nP,nS=nS,nT=nT,idim=0) #XXp
            W20list[2] = Eta0 #np.zeros(shapes[nextraT+10]) #Eta
            W20list[3] = Xi0 #Xi
            isn_init = np.array(((3,5),(-5,-5)))
            if init_W_from_lsq:
                # shapes1
                #0.W0x,1.W0y,2.W1x,3.W1y,4.W2x,5.W2y,6.W3x,7.W3y,8.s02,9.K0,10.K1,11.K2,12.K3,13.kappa,14.T0,15.T1,16.T2,17.T3,18.h1,19.h2,20.bl,21.amp
                # shapes2
                #0.XX,1.XXp,2.Eta,3.Xi
                #W0x,W0y,W1x,W1y,W2x,W2y,W3x,W3y,K0,K1,T0,T1 = initialize_Ws(Xhat,Yhat,Xpc_list,Ypc_list,scale_by=1)
                nvar,nxy = 4,2
                freeze_vals = [[None for _ in range(nxy)] for _ in range(nvar)]
                lams = 100*np.array((0,1,1,1,0,1,0,1))
                for ivar in range(nvar):
                    for ixy in range(nxy):
                        iflat = np.ravel_multi_index((ivar,ixy),(nvar,nxy))
                        freeze_vals[ivar][ixy] = np.zeros(bd1list[iflat].shape)
                        freeze_vals[ivar][ixy][bd1list[iflat]==0] = np.nan
                if constrain_isn:
                    freeze_vals[0][1][slice(0,None,3)][:,slice(0,None,3)] = isn_init
                thisWlist = initialize_Ws(Xhat,Yhat,Xpc_list,Ypc_list,scale_by=1,freeze_vals=freeze_vals,lams=lams)
                Winds = [0,1,2,3,4,5,6,7,9,10,14,15]
                for ivar,Wind in enumerate(Winds):
                    W10list[Wind] = thisWlist[ivar]
                #W10list[0],W10list[1] = initialize_W(Xhat,Yhat,scale_by=scale_init_by)
                for Wind in Winds:
                    W10list[Wind] = W10list[Wind] + init_noise*np.random.randn(*W10list[Wind].shape)
            else:
                if constrain_isn:
                    W10list[1][slice(0,None,3)][:,slice(0,None,3)] = isn_init
                    #W10list[1][0,0] = 3 
                    #W10list[1][0,3] = 5 
                    #W10list[1][3,0] = -5
                    #W10list[1][3,3] = -5
            np.save('/home/dan/calnet_data/W0list.npy',{'W10list':W10list,'W20list':W20list,'bd1list':bd1list,'bd2list':bd2list,'freeze_vals':freeze_vals,'bounds1':bounds1,'bounds2':bounds2},allow_pickle=True)

            if init_W_from_file:
                npyfile = np.load(init_file,allow_pickle=True)[()]
                print(len(npyfile['as_list']))
                print([w.shape for w in npyfile['as_list']])
                W10list = [npyfile['as_list'][ivar] for ivar in [0,1,2,3,4,5,6,7,12]]
                W20list = [npyfile['as_list'][ivar] for ivar in [8,9,10,11]]
                if correct_Eta:
                    #assert(True==False)
                    W20list[2] = Eta0.copy()
                if len(W10list) < len(shapes1):
                    #assert(True==False)
                    W10list = W10list + [np.array(1),np.zeros((nQ,)),np.zeros((nT*nS*nQ,))] # add bl, amp #np.array(1), #h2, 
                #W10 = unparse_W(W10list)
                #W20 = unparse_W(W20list)
                opt = fmc.gen_opt()
                #resEta0,resXi0 = fmc.compute_res(W10,W20,opt)
                if init_W1xy_with_res:
                    W1x0,W1y0,k10,T10 = optimize_W1xy(W10list,W20list,opt)
                    W0list[2] = W1x0
                    W0list[3] = W1y0
                    W0list[10] = k10
                    W0list[15] = T10
                if init_W2xy_with_res:
                    W2x0,W2y0 = optimize_W2xy(W10list,W20list,opt)
                    W0list[4] = W2x0
                    W0list[5] = W2y0
                if init_Eta_with_s02:
                    #assert(True==False)
                    s02 = W10list[4].copy()
                    Eta0 = invert_f_mt_with_s02(YYhat,s02,nS=nS,nT=nT)
                    W20list[2] = Eta0.copy()
                for ivar in [0,1,4,5]: # Wmx, Wmy, s02, k
                    print(init_noise)
                    W10list[ivar] = W10list[ivar] + init_noise*np.random.randn(*W10list[ivar].shape)
                #W0list = npyfile['as_list']

                extra_Ws = [np.zeros_like(W10list[ivar]) for ivar in range(2)]
                extra_ks = [np.zeros_like(W10list[5]) for ivar in range(3)]
                extra_Ts = [np.zeros_like(W10list[7]) for ivar in range(3)]
                W10list = W10list[:4] + extra_Ws*2 + W10list[4:6] + extra_ks + W10list[6:8] + extra_Ts + W10list[8:]

            W1t[ihyper][itry],W2t[ihyper][itry],loss[ihyper][itry],gr,hess,result = calnet.fitting_2step_spatial_feature_opto_multiout_nonlinear.fit_W_sim(Xhat,Xpc_list,Yhat,Ypc_list,pop_rate_fn=sim_utils.f_miller_troyer,pop_deriv_fn=sim_utils.fprime_miller_troyer,neuron_rate_fn=sim_utils.evaluate_f_mt,W10list=W10list.copy(),W20list=W20list.copy(),bounds1=bounds1,bounds2=bounds2,niter=niter,wt_dict=wt_dict,l2_penalty=l2_penalty,compute_hessian=False,dt=dt,perturbation_size=perturbation_size,dYY=dYY,constrain_isn=constrain_isn,tv=tv,foldT=foldT,use_opto_transforms=use_opto_transforms,opto_transform1=opto_transform1,opto_transform2=opto_transform2,nondim=nondim)
    
    #def parse_W(W):
    #    W0x,W0y,W1x,W1y,W2x,W2y,W3x,W3y,s02,k0,k1,k2,k3,kappa,T0,T1,T2,T3,XX,XXp,Eta,Xi,h = W
    #    return W0x,W0y,W1x,W1y,W2x,W2y,W3x,W3y,s02,k0,k1,k2,k3,kappa,T0,T1,T2,T3,XX,XXp,Eta,Xi,h
    def parse_W1(W):
        W0x,W0y,W1x,W1y,W2x,W2y,W3x,W3y,s02,k0,k1,k2,k3,kappa,T0,T1,T2,T3,h1,h2,bl,amp = W #h2,
        return W0x,W0y,W1x,W1y,W2x,W2y,W3x,W3y,s02,k0,k1,k2,k3,kappa,T0,T1,T2,T3,h1,h2,bl,amp #h2,
    def parse_W2(W):
        XX,XXp,Eta,Xi = W
        return XX,XXp,Eta,Xi    

    def unparse_W(Ws):
        return np.concatenate([ww.flatten() for ww in Ws])
    
    itry = 0
    W0x,W0y,W1x,W1y,W2x,W2y,W3x,W3y,s02,k0,k1,k2,k3,kappa,T0,T1,T2,T3,h1,h2,bl,amp = parse_W1(W1t[0][0])#h2,
    XX,XXp,Eta,Xi = parse_W2(W2t[0][0])
    
    #labels = ['W0x','W0y','W1x','W1y','W2x','W2y','W3x','W3y','s02','K0','K1','K2','K3','kappa','T0','T1','T2','T3','XX','XXp','Eta','Xi','h']
    labels1 = ['W0x','W0y','W1x','W1y','W2x','W2y','W3x','W3y','s02','K0','K1','K2','K3','kappa','T0','T1','T2','T3','h1','h2','bl','amp']#,'h2'
    labels2 = ['XX','XXp','Eta','Xi']
    Wstar_dict = {}
    for i,label in enumerate(labels1):
        Wstar_dict[label] = W1t[0][0][i]
    for i,label in enumerate(labels2):
        Wstar_dict[label] = W2t[0][0][i]
    #Wstar_dict = {}
    #for i,label in enumerate(labels):
    #    Wstar_dict[label] = W1t[0][0][i]
    Wstar_dict['as_list'] = [W0x,W0y,W1x,W1y,W2x,W2y,W3x,W3y,s02,k0,k1,k2,k3,kappa,T0,T1,T2,T3,XX,XXp,Eta,Xi,h1,h2,bl,amp]#,h2
    Wstar_dict['loss'] = loss[0][0]
    Wstar_dict['wt_dict'] = wt_dict
    np.save(weights_file,Wstar_dict,allow_pickle=True)
    
