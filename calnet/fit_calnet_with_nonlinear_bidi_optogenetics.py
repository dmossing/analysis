#!/usr/bin/env python
# coding: utf-8

# in this notebook, I will try to fit a model relating the mean behavior of L4, L2/3, SST and VIP cells

# load the data

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
import calnet.fitting_spatial_feature_opto_nonlinear
import opto_utils
import scipy.signal as ssi
import scipy.optimize as sop

def invert_f_mt(y,s02=1,floor=-5):
    xstar = np.zeros_like(y)
    for iy,yy in enumerate(y):
        if not isinstance(yy,np.ndarray):
            to_invert = lambda x: sim_utils.f_miller_troyer(x,s02)-yy
            #xstar[iy] = sop.root_scalar(to_invert,x0=yy,x1=0).root
            xstar[iy] = sop.root_scalar(to_invert,x0=1,x1=0).root
        else:
            xstar[iy] = invert_f_mt(yy,s02=s02,floor=floor)
    return np.maximum(xstar,floor)

def invert_f_mt_with_s02(YYhat,s02,nS=2,nT=1):
    Eta0 = np.zeros_like(YYhat)
    nQ = int(YYhat.shape[1]/nS/nT)
    for iS in range(nS):
        for iT in range(nT):
            for itype in range(nQ):
                this_col = iS*nT*nQ + iT*nQ + itype
                Eta0[:,this_col] = invert_f_mt(YYhat[:,this_col],s02=s02[itype])
    return Eta0


def initialize_W(Xhat,Yhat,scale_by=0.2):
    nP = Xhat[0][0].shape[1]
    nQ = Yhat[0][0].shape[1]
    nN = Yhat[0][0].shape[0]
    YYhat = calnet.utils.flatten_nested_list_of_2d_arrays(Yhat)
    XXhat = calnet.utils.flatten_nested_list_of_2d_arrays(Xhat)
    Wmy0 = np.zeros((nQ,nQ))
    Wmx0 = np.zeros((nP,nQ))
    #Ymatrix_pred = np.zeros((nN,nQ))
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

def fit_weights_and_save(weights_file,ca_data_file='rs_vm_denoise_200605.npy',opto_silencing_data_file='vip_halo_data_for_sim.npy',opto_activation_data_file='vip_chrimson_data_for_sim.npy',constrain_wts=None,allow_var=True,fit_s02=True,constrain_isn=True,tv=False,l2_penalty=0.01,init_noise=0.1,init_W_from_lsq=False,scale_init_by=1,init_W_from_file=False,init_file=None,correct_Eta=False,init_Eta_with_s02=False,init_Eta12_with_dYY=False,use_opto_transforms=False):
    
    nsize,ncontrast = 6,6
    
    npfile = np.load(ca_data_file,allow_pickle=True)[()]#,{'rs':rs,'rs_denoise':rs_denoise},allow_pickle=True)
    rs = npfile['rs']
    #rs_denoise = npfile['rs_denoise']
    
    nsize,ncontrast,ndir = 6,6,8
    #ori_dirs = [[0,4],[2,6]] #[[0,4],[1,3,5,7],[2,6]]
    ori_dirs = [[0,1,2,3,4,5,6,7]]
    nT = len(ori_dirs)
    nS = len(rs[0])
    
    def sum_to_1(r):
        R = r.reshape((r.shape[0],-1))
        #R = R/np.nansum(R[:,~np.isnan(R.sum(0))],axis=1)[:,np.newaxis]
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
            #Rs[iR][ialign] = r[ialign][:,:nsize,:]
            #sm = np.nanmean(np.nansum(np.nansum(Rs[iR][ialign],1),1))
            #Rs[iR][ialign] = Rs[iR][ialign]/sm
            Rs[iR][ialign] = sum_to_1(r[ialign][:,:nsize,:])
    #         Rs[iR][ialign] = von_mises_denoise(Rs[iR][ialign].reshape((-1,nsize,ncontrast,ndir)))
    
    kernel = np.ones((1,2,2))
    kernel = kernel/kernel.sum()
    
    for iR,r in enumerate(rs):
        for ialign in range(nS):
            for iori in range(nT):
                Rso[iR][ialign][iori] = np.nanmean(Rs[iR][ialign].reshape((-1,nsize,ncontrast,ndir))[:,:,:,ori_dirs[iori]],-1)
                Rso[iR][ialign][iori][:,:,0] = np.nanmean(Rso[iR][ialign][iori][:,:,0],1)[:,np.newaxis] # average 0 contrast values
                Rso[iR][ialign][iori][:,1:,1:] = ssi.convolve(Rso[iR][ialign][iori],kernel,'valid')
                Rso[iR][ialign][iori] = Rso[iR][ialign][iori].reshape(Rso[iR][ialign][iori].shape[0],-1)
                #Rso[iR][ialign][iori] = Rso[iR][ialign][iori]/np.nanmean(Rso[iR][ialign][iori],-1)[:,np.newaxis]
    
    def set_bound(bd,code,val=0):
        # set bounds to 0 where 0s occur in 'code'
        for iitem in range(len(bd)):
            bd[iitem][code[iitem]] = val
    
    nN = 36
    nS = 2
    nP = 2
    nT = 1
    nQ = 4
    
    # code for bounds: 0 , constrained to 0
    # +/-1 , constrained to +/-1
    # 1.5, constrained to [0,1]
    # 2 , constrained to [0,inf)
    # -2 , constrained to (-inf,0]
    # 3 , unconstrained
    
    Wmx_bounds = 3*np.ones((nP,nQ),dtype=int)
    Wmx_bounds[0,1] = 0 # SSTs don't receive L4 input
    
    if allow_var:
        Wsx_bounds = 3*np.ones(Wmx_bounds.shape) #Wmx_bounds.copy()*0 #np.zeros_like(Wmx_bounds)
        Wsx_bounds[0,1] = 0
    else:
        Wsx_bounds = np.zeros(Wmx_bounds.shape) #Wmx_bounds.copy()*0 #np.zeros_like(Wmx_bounds)
    
    Wmy_bounds = 3*np.ones((nQ,nQ),dtype=int)
    Wmy_bounds[0,:] = 2 # PCs are excitatory
    Wmy_bounds[1:,:] = -2 # all the cell types except PCs are inhibitory
    Wmy_bounds[1,1] = 0 # SSTs don't inhibit themselves
    # Wmy_bounds[3,1] = 0 # PVs are allowed to inhibit SSTs, consistent with Hillel's unpublished results, but not consistent with Pfeffer et al.
    Wmy_bounds[2,0] = 0 # VIPs don't inhibit L2/3 PCs. According to Pfeffer et al., only L5 PCs were found to get VIP inhibition

    if allow_var:
        Wsy_bounds = 3*np.ones(Wmy_bounds.shape) #Wmy_bounds.copy()*0 #np.zeros_like(Wmy_bounds)
        Wsy_bounds[1,1] = 0
        Wsy_bounds[3,1] = 0 
        Wsy_bounds[2,0] = 0
    else:
        Wsy_bounds = np.zeros(Wmy_bounds.shape) #Wmy_bounds.copy()*0 #np.zeros_like(Wmy_bounds)

    if not constrain_wts is None:
        for wt in constrain_wts:
            Wmy_bounds[wt[0],wt[1]] = 0
            Wsy_bounds[wt[0],wt[1]] = 0
    
    def tile_nS_nT_nN(kernel):
        row = np.concatenate([kernel for idim in range(nS*nT)],axis=0)[np.newaxis,:]
        tiled = np.concatenate([row for irow in range(nN)],axis=0)
        return tiled
    
    if fit_s02:
        s02_bounds = 2*np.ones((nQ,)) # permitting noise as a free parameter
    else:
        s02_bounds = np.ones((nQ,))
    
    k_bounds = 1.5*np.ones((nQ*(nS-1),))
    
    kappa_bounds = np.ones((1,))
    # kappa_bounds = 2*np.ones((1,))
    
    T_bounds = 1.5*np.ones((nQ*(nT-1),))
    
    X_bounds = tile_nS_nT_nN(np.array([2,1]))
    # X_bounds = np.array([np.array([2,1,2,1])]*nN)
    
    Xp_bounds = tile_nS_nT_nN(np.array([3,1]))
    # Xp_bounds = np.array([np.array([3,1,3,1])]*nN)
    
    # Y_bounds = tile_nS_nT_nN(2*np.ones((nQ,)))
    # # Y_bounds = 2*np.ones((nN,nT*nS*nQ))
    
    Eta_bounds = tile_nS_nT_nN(3*np.ones((nQ,)))
    # Eta_bounds = 3*np.ones((nN,nT*nS*nQ))
    
    if allow_var:
        Xi_bounds = tile_nS_nT_nN(3*np.ones((nQ,)))
    else:
        Xi_bounds = tile_nS_nT_nN(np.zeros((nQ,)))

    # Xi_bounds = 3*np.ones((nN,nT*nS*nQ))
    
    h1_bounds = -2*np.ones((1,))
    
    h2_bounds = 2*np.ones((1,))
    
    
    # In[8]:
    
    
    # shapes = [(nP,nQ),(nQ,nQ),(nP,nQ),(nQ,nQ),(nQ,),(nQ,),(1,),(nN,nS*nP),(nN,nS*nQ),(nN,nS*nQ),(nN,nS*nQ)]
    shapes = [(nP,nQ),(nQ,nQ),(nP,nQ),(nQ,nQ),(nQ,),(nQ*(nS-1),),(1,),(nQ*(nT-1),),(nN,nT*nS*nP),(nN,nT*nS*nP),(nN,nT*nS*nQ),(nN,nT*nS*nQ),(1,),(1,),(nN,nT*nS*nQ),(nN,nT*nS*nQ)]
    print('size of shapes: '+str(np.sum([np.prod(shp) for shp in shapes])))
    #         Wmx,    Wmy,    Wsx,    Wsy,    s02,  k,    kappa,T,   XX,            XXp,          Eta,          Xi, h1, h2, Eta1,   Eta2
    
    lb = [-np.inf*np.ones(shp) for shp in shapes]
    ub = [np.inf*np.ones(shp) for shp in shapes]
    bdlist = [Wmx_bounds,Wmy_bounds,Wsx_bounds,Wsy_bounds,s02_bounds,k_bounds,kappa_bounds,T_bounds,X_bounds,Xp_bounds,Eta_bounds,Xi_bounds,h1_bounds,h2_bounds,Eta_bounds,Eta_bounds]
    
    set_bound(lb,[bd==0 for bd in bdlist],val=0)
    set_bound(ub,[bd==0 for bd in bdlist],val=0)
    
    set_bound(lb,[bd==2 for bd in bdlist],val=0)
    
    set_bound(ub,[bd==-2 for bd in bdlist],val=0)
    
    set_bound(lb,[bd==1 for bd in bdlist],val=1)
    set_bound(ub,[bd==1 for bd in bdlist],val=1)
    
    set_bound(lb,[bd==1.5 for bd in bdlist],val=0)
    set_bound(ub,[bd==1.5 for bd in bdlist],val=1)
    
    set_bound(lb,[bd==-1 for bd in bdlist],val=-1)
    set_bound(ub,[bd==-1 for bd in bdlist],val=-1)
    
    # for bd in [lb,ub]:
    #     for ind in [2,3]:
    #         bd[ind][:,1] = 0
    
    # temporary for no variation expt.
    # lb[2] = np.zeros_like(lb[2])
    # lb[3] = np.zeros_like(lb[3])
    # lb[4] = np.ones_like(lb[4])
    # lb[5] = np.zeros_like(lb[5])
    # ub[2] = np.zeros_like(ub[2])
    # ub[3] = np.zeros_like(ub[3])
    # ub[4] = np.ones_like(ub[4])
    # ub[5] = np.ones_like(ub[5])
    # temporary for no variation expt.
    lb = np.concatenate([a.flatten() for a in lb])
    ub = np.concatenate([b.flatten() for b in ub])
    bounds = [(a,b) for a,b in zip(lb,ub)]
    
    
    # In[10]:
    
    
    nS = 2
    print('nT: '+str(nT))
    ndims = 5
    ncelltypes = 5
    Yhat = [[None for iT in range(nT)] for iS in range(nS)]
    Xhat = [[None for iT in range(nT)] for iS in range(nS)]
    Ypc_list = [[None for iT in range(nT)] for iS in range(nS)]
    Xpc_list = [[None for iT in range(nT)] for iS in range(nS)]
    mx = [None for iS in range(nS)]
    for iS in range(nS):
        mx[iS] = np.zeros((ncelltypes,))
        yy = [None for icelltype in range(ncelltypes)]
        for icelltype in range(ncelltypes):
            yy[icelltype] = np.nanmean(Rso[icelltype][iS][0],0)
            mx[iS][icelltype] = np.nanmax(yy[icelltype])
        for iT in range(nT):
            y = [np.nanmean(Rso[icelltype][iS][iT],axis=0)[:,np.newaxis]/mx[iS][icelltype] for icelltype in range(1,ncelltypes)]
            Ypc_list[iS][iT] = [None for icelltype in range(1,ncelltypes)]
            for icelltype in range(1,ncelltypes):
                rss = Rso[icelltype][iS][iT].copy()#/mx[iS][icelltype] #.reshape(Rs[icelltype][ialign].shape[0],-1)
                #rss = Rso[icelltype][iS][iT].copy() #.reshape(Rs[icelltype][ialign].shape[0],-1)
                rss = rss[np.isnan(rss).sum(1)==0]
        #         print(rss.max())
        #         rss[rss<0] = 0
        #         rss = rss[np.random.randn(rss.shape[0])>0]
                try:
                    u,s,v = np.linalg.svd(rss-np.mean(rss,0)[np.newaxis])
                    Ypc_list[iS][iT][icelltype-1] = [(s[idim],v[idim]) for idim in range(ndims)]
    #                 print('yep on Y')
    #                 print(np.min(np.sum(rs[icelltype][iS][iT],axis=1)))
                except:
    #                 print('nope on Y')
                    print(np.mean(np.isnan(rss)))
                    print(np.min(np.sum(rs[icelltype][iS][iT],axis=1)))
            Yhat[iS][iT] = np.concatenate(y,axis=1)
    #         x = sim_utils.columnize(Rso[0][iS][iT])[:,np.newaxis]
            icelltype = 0
            #x = np.nanmean(Rso[icelltype][iS][iT],0)[:,np.newaxis]#/mx[iS][icelltype]
            x = np.nanmean(Rso[icelltype][iS][iT],0)[:,np.newaxis]/mx[iS][icelltype]
    #         opto_column = np.concatenate((np.zeros((nN,)),np.zeros((nNO/2,)),np.ones((nNO/2,))),axis=0)[:,np.newaxis]
            Xhat[iS][iT] = np.concatenate((x,np.ones_like(x)),axis=1)
    #         Xhat[iS][iT] = np.concatenate((x,np.ones_like(x),opto_column),axis=1)
            icelltype = 0
            #rss = Rso[icelltype][iS][iT].copy()/mx[iS][icelltype]
            rss = Rso[icelltype][iS][iT].copy()
            rss = rss[np.isnan(rss).sum(1)==0]
    #         try:
            u,s,v = np.linalg.svd(rss-rss.mean(0)[np.newaxis])
            Xpc_list[iS][iT] = [None for iinput in range(2)]
            Xpc_list[iS][iT][0] = [(s[idim],v[idim]) for idim in range(ndims)]
            Xpc_list[iS][iT][1] = [(0,np.zeros((Xhat[0][0].shape[0],))) for idim in range(ndims)]
    #         except:
    #             print('nope on X')
    #             print(np.mean(np.isnan(rss)))
    #             print(np.min(np.sum(Rso[icelltype][iS][iT],axis=1)))
    nN,nP = Xhat[0][0].shape
    print('nP: '+str(nP))
    nQ = Yhat[0][0].shape[1]
    
    
    # In[11]:
    
    
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
    
    
    # In[12]:
    
    
    #         0.Wmx,  1.Wmy,  2.Wsx,  3.Wsy,  4.s02,5.K,  6.kappa,7.T,8.XX,        9.XXp,        10.Eta,       11.Xi,   12.h1,  13.h2,  14.Eta1,    15.Eta2
    
    shapes = [(nP,nQ),(nQ,nQ),(nP,nQ),(nQ,nQ),(nQ,),(nQ*(nS-1),),(1,),(nQ*(nT-1),),(nN,nT*nS*nP),(nN,nT*nS*nP),(nN,nT*nS*nQ),(nN,nT*nS*nQ),(1,),(1,),(nN,nT*nS*nQ),(nN,nT*nS*nQ)]
    print('size of shapes: '+str(np.sum([np.prod(shp) for shp in shapes])))
    
    
    import calnet.fitting_spatial_feature
    import sim_utils

    YYhat = calnet.utils.flatten_nested_list_of_2d_arrays(Yhat)
    XXhat = calnet.utils.flatten_nested_list_of_2d_arrays(Xhat)
    
    opto_dict = np.load(opto_silencing_data_file,allow_pickle=True)[()]

    Yhat_opto = opto_dict['Yhat_opto']
    Yhat_opto = np.nanmean(np.reshape(Yhat_opto,(nN,2,nS,2,nQ)),3).reshape((nN*2,-1))
    Yhat_opto = Yhat_opto/Yhat_opto[0::2].max(0)[np.newaxis,:]
    print(Yhat_opto.shape)
    h_opto = opto_dict['h_opto']
    #dYY1 = Yhat_opto[1::2]-Yhat_opto[0::2]
        
    YYhat_halo = Yhat_opto.reshape((nN,2,-1))
    opto_transform1 = calnet.utils.fit_opto_transform(YYhat_halo)

    opto_transform1.res[:,[0,2,3,4,6,7]] = 0

    dYY1 = opto_transform1.transform(YYhat) - YYhat
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
    #for to_overwrite in [1,2]:
    #    dYY1[:,to_overwrite] = dYY1[:,to_overwrite+4]
    #for to_overwrite in [7]:
    #    dYY1[:,to_overwrite] = dYY1[:,to_overwrite-4]
    
    #Yhat_opto = opto_dict['Yhat_opto']
    #for iS in range(nS):
    #    mx = np.zeros((nQ,))
    #    for iQ in range(nQ):
    #        slicer = slice(nQ*nT*iS+iQ,nQ*nT*(1+iS),nQ)
    #        mx[iQ] = np.nanmax(Yhat_opto[0::2][:,slicer])
    #        Yhat_opto[:,slicer] = Yhat_opto[:,slicer]/mx[iQ]
    ##Yhat_opto = Yhat_opto/Yhat_opto[0::2].max(0)[np.newaxis,:]
    #print(Yhat_opto.shape)
    #h_opto = opto_dict['h_opto']
    #dYY1 = Yhat_opto[1::2]-Yhat_opto[0::2]
    #for to_overwrite in [1,2,5,6]: # overwrite sst and vip with off-centered values
    #    dYY1[:,to_overwrite] = dYY1[:,to_overwrite+8]
    #for to_overwrite in [11,15]:
    #    dYY1[:,to_overwrite] = np.nan #dYY1[:,to_overwrite-8]


    opto_dict = np.load(opto_activation_data_file,allow_pickle=True)[()]

    Yhat_opto = opto_dict['Yhat_opto']
    Yhat_opto = np.nanmean(np.reshape(Yhat_opto,(nN,2,nS,2,nQ)),3).reshape((nN*2,-1))
    Yhat_opto = Yhat_opto/Yhat_opto[0::2].max(0)[np.newaxis,:]
    print(Yhat_opto.shape)
    h_opto = opto_dict['h_opto']
    #dYY2 = Yhat_opto[1::2]-Yhat_opto[0::2]

    YYhat_chrimson = Yhat_opto.reshape((nN,2,-1))
    opto_transform2 = calnet.utils.fit_opto_transform(YYhat_chrimson)
    dYY2 = opto_transform2.transform(YYhat) - YYhat
    #YYhat_chrimson_sim = calnet.utils.simulate_opto_effect(YYhat,YYhat_chrimson)
    #dYY2 = YYhat_chrimson_sim[:,1,:] - YYhat_chrimson_sim[:,0,:]

    #Yhat_opto = opto_dict['Yhat_opto']
    #for iS in range(nS):
    #    mx = np.zeros((nQ,))
    #    for iQ in range(nQ):
    #        slicer = slice(nQ*nT*iS+iQ,nQ*nT*(1+iS),nQ)
    #        mx[iQ] = np.nanmax(Yhat_opto[0::2][:,slicer])
    #        Yhat_opto[:,slicer] = Yhat_opto[:,slicer]/mx[iQ]
    ##Yhat_opto = Yhat_opto/Yhat_opto[0::2].max(0)[np.newaxis,:]
    #print(Yhat_opto.shape)
    #h_opto = opto_dict['h_opto']
    #dYY2 = Yhat_opto[1::2]-Yhat_opto[0::2]
    
    print('dYY1 mean: %03f'%np.nanmean(np.abs(dYY1)))
    print('dYY2 mean: %03f'%np.nanmean(np.abs(dYY2)))

    dYY = np.concatenate((dYY1,dYY2),axis=0)
    
    titles = ['VIP silencing','VIP activation']
    for itype in [0,1,2,3]:
        plt.figure(figsize=(5,2.5))
        for iyy,dyy in enumerate([dYY1,dYY2]):
            plt.subplot(1,2,iyy+1)
            if np.sum(np.isnan(dyy[:,itype]))==0:
                sca.scatter_size_contrast(YYhat[:,itype],YYhat[:,itype]+dyy[:,itype],nsize=6,ncontrast=6)#,mn=0)
            plt.title(titles[iyy])
            plt.xlabel('cell type %d event rate, \n light off'%itype)
            plt.ylabel('cell type %d event rate, \n light on'%itype)
            ut.erase_top_right()
        plt.tight_layout()
        ut.mkdir('figures')
        plt.savefig('figures/scatter_light_on_light_off_target_celltype_%d.eps'%itype)
    
    opto_mask = ~np.isnan(dYY)
    #dYY[nN:][~opto_mask[nN:]] = -dYY[:nN][~opto_mask[nN:]]

    print('mean of opto_mask: '+str(opto_mask.mean()))
    
    #dYY[~opto_mask] = 0
    def zero_nans(arr):
        arr[np.isnan(arr)] = 0
        return arr
    #dYY,opto_transform1.slope,opto_transform1.intercept,opto_transform1.res,\
    #        opto_transform2.slope,opto_transform2.intercept,opto_transform2.res\
    #        = [zero_nans(x) for x in \
    #                [dYY,opto_transform1.slope,opto_transform1.intercept,opto_transform1.res,\
    #                opto_transform2.slope,opto_transform2.intercept,opto_transform2.res]]
    dYY = zero_nans(dYY)

    to_adjust = np.logical_or(np.isnan(opto_transform2.slope[0]),np.isnan(opto_transform2.intercept[0]))

    opto_transform2.slope[:,to_adjust] = 1/opto_transform1.slope[:,to_adjust]
    opto_transform2.intercept[:,to_adjust] = -opto_transform1.intercept[:,to_adjust]/opto_transform1.slope[:,to_adjust]
    opto_transform2.res[:,to_adjust] = -opto_transform1.res[:,to_adjust]/opto_transform1.slope[:,to_adjust]
    
    np.save('/Users/dan/Documents/notebooks/mossing-PC/shared_data/calnet_data/dYY.npy',dYY)
    
    from importlib import reload
    reload(calnet)
    #reload(calnet.fitting_spatial_feature_opto_nonlinear)
    reload(sim_utils)
    # reload(calnet.fitting_spatial_feature)
    # W0list = [np.ones(shp) for shp in shapes]
    wt_dict = {}
    wt_dict['X'] = 1
    wt_dict['Y'] = 15
    wt_dict['Eta'] = 10 # 1 # 
    wt_dict['Xi'] = 0.1
    wt_dict['stims'] = np.ones((nN,1)) #(np.arange(30)/30)[:,np.newaxis]**1 #
    wt_dict['barrier'] = 0. #30.0 #0.1
    wt_dict['opto'] = 1e-1#1e1
    wt_dict['isn'] = 3
    wt_dict['tv'] = 1
    wt_dict['stimsOpto'] = 0.6*np.ones((nN,1))
    wt_dict['stimsOpto'][0::6] = 3
    wt_dict['celltypesOpto'] = 0.67*np.ones((1,nQ*nS*nT))
    wt_dict['celltypesOpto'][0,0::nQ] = 2
    wt_dict['dirOpto'] = np.array((1,0.5))
    wt_dict['dYY'] = 1#1000
    wt_dict['Eta12'] = 1
    wt_dict['coupling'] = 1

    np.save('XXYYhat.npy',{'YYhat':YYhat,'XXhat':XXhat,'rs':rs,'Rs':Rs,'Rso':Rso,'Ypc_list':Ypc_list,'Xpc_list':Xpc_list})
    Eta0 = invert_f_mt(YYhat)
    Eta10 = invert_f_mt(YYhat + dYY[:nN])
    Eta20 = invert_f_mt(YYhat + dYY[nN:])
    print('mean Eta1 diff: '+str(np.mean(np.abs(Eta0-Eta10))))
    print('mean Eta2 diff: '+str(np.mean(np.abs(Eta0-Eta20))))


    ntries = 1
    nhyper = 1
    dt = 1e-1
    niter = int(np.round(10/dt)) #int(1e4)
    perturbation_size = 5e-2
    # learning_rate = 1e-4 # 1e-5 #np.linspace(3e-4,1e-3,niter+1) # 1e-5
    #l2_penalty = 0.1
    Wt = [[None for itry in range(ntries)] for ihyper in range(nhyper)]
    loss = np.zeros((nhyper,ntries))
    is_neg = np.array([b[1] for b in bounds])==0
    counter = 0
    negatize = [np.zeros(shp,dtype='bool') for shp in shapes]
    print(shapes)
    for ishp,shp in enumerate(shapes):
        nel = np.prod(shp)
        negatize[ishp][:][is_neg[counter:counter+nel].reshape(shp)] = True
        counter = counter + nel
    for ihyper in range(nhyper):
        for itry in range(ntries):
            print((ihyper,itry))
            W0list = [init_noise*(ihyper+1)*np.random.rand(*shp) for shp in shapes]
            print('size of shapes: '+str(np.sum([np.prod(shp) for shp in shapes])))
            print('size of w0: '+str(np.sum([np.size(x) for x in W0list])))
            print('len(W0list) : '+str(len(W0list)))
            counter = 0
            for ishp,shp in enumerate(shapes):
                W0list[ishp][negatize[ishp]] = -W0list[ishp][negatize[ishp]]
            W0list[4] = np.ones(shapes[5]) # s02
            W0list[5] = np.ones(shapes[5]) # K
            W0list[6] = np.ones(shapes[6]) # kappa
            W0list[7] = np.ones(shapes[7]) # T
            W0list[8] = np.concatenate(Xhat,axis=1) #XX
            W0list[9] = np.zeros_like(W0list[8]) #XXp
            W0list[10] = Eta0.copy() #np.zeros(shapes[10]) #Eta
            W0list[11] = np.zeros(shapes[11]) #Xi
            W0list[14] = Eta10.copy() # Eta1
            W0list[15] = Eta20.copy() # Eta2
            #[Wmx,Wmy,Wsx,Wsy,s02,k,kappa,T,XX,XXp,Eta,Xi]
    #         W0list = Wstar_dict['as_list'].copy()
    #         W0list[1][1,0] = -1.5
    #         W0list[1][3,0] = -1.5
            if init_W_from_lsq:
                W0list[0],W0list[1] = initialize_W(Xhat,Yhat,scale_by=scale_init_by)
                for ivar in range(0,2):
                    W0list[ivar] = W0list[ivar] + init_noise*np.random.randn(*W0list[ivar].shape)
            if constrain_isn:
                W0list[1][0,0] = 3 
                W0list[1][0,3] = 5 
                W0list[1][3,0] = -5
                W0list[1][3,3] = -5

            #if constrain_isn:
            #    W0list[1][0,0] = 2
            #    W0list[1][0,3] = 2
            #    W0list[1][3,0] = -2
            #    W0list[1][3,3] = -2

            #if wt_dict['coupling'] > 0:
            #    W0list[1][1,0] = -1

            if init_W_from_file:
                npyfile = np.load(init_file,allow_pickle=True)[()]
                W0list = npyfile['as_list']
                if W0list[8].size == nN*nS*2*nP:
                    W0list[7] = np.array(())
                    W0list[1][1,0] = W0list[1][1,0]
                    W0list[8] = np.nanmean(W0list[8].reshape((nN,nS,2,nP)),2).flatten() #XX
                    W0list[9] = np.nanmean(W0list[9].reshape((nN,nS,2,nP)),2).flatten() #XXp
                    W0list[10] = np.nanmean(W0list[10].reshape((nN,nS,2,nQ)),2).flatten() #Eta
                    W0list[11] = np.nanmean(W0list[11].reshape((nN,nS,2,nQ)),2).flatten() #Xi
                if correct_Eta:
                    W0list[10] = Eta0.copy()
                if len(W0list) < len(shapes):
                    W0list = W0list[:-1] + [np.array(-0.5),np.array(1),Eta10.copy(),Eta20.copy()] # add h1,h2,Eta1,Eta2
                if init_Eta_with_s02:
                    s02 = W0list[4].copy()
                    Eta0 = invert_f_mt_with_s02(YYhat,s02,nS=nS,nT=nT)
                    Eta10 = invert_f_mt_with_s02(YYhat+dYY[:nN],s02,nS=nS,nT=nT)
                    Eta20 = invert_f_mt_with_s02(YYhat+dYY[nN:],s02,nS=nS,nT=nT)
                    W0list[10] = Eta0.copy()
                    W0list[14] = Eta10.copy()
                    W0list[15] = Eta20.copy()
                if init_Eta12_with_dYY:
                    Eta0 = W0list[10].copy().reshape((nN,nQ*nS*nT))
                    Xi0 = W0list[11].copy().reshape((nN,nQ*nS*nT))
                    s020 = W0list[4].copy()
                    YY0s = compute_f_(Eta0,Xi0,s020)
                    this_YY1 = opto_transform1.transform(YY0s)
                    this_YY2 = opto_transform2.transform(YY0s)
                    Eta10 = invert_f_mt_with_s02(this_YY1,s020,nS=nS,nT=nT)
                    Eta20 = invert_f_mt_with_s02(this_YY2,s020,nS=nS,nT=nT)
                    W0list[14] = Eta10.copy()
                    W0list[15] = Eta20.copy()

                    YY10s = compute_f_(Eta10,Xi0,s020)
                    YY20s = compute_f_(Eta20,Xi0,s020)
                    titles = ['VIP silencing','VIP activation']
                    for itype in [0,1,2,3]:
                        plt.figure(figsize=(5,2.5))
                        for iyy,yy in enumerate([YY10s,YY20s]):
                            plt.subplot(1,2,iyy+1)
                            if np.sum(np.isnan(yy[:,itype]))==0:
                                sca.scatter_size_contrast(YY0s[:,itype],yy[:,itype],nsize=6,ncontrast=6)#,mn=0)
                            plt.title(titles[iyy])
                            plt.xlabel('cell type %d event rate, \n light off'%itype)
                            plt.ylabel('cell type %d event rate, \n light on'%itype)
                            ut.erase_top_right()
                        plt.tight_layout()
                        ut.mkdir('figures')
                        plt.savefig('figures/scatter_light_on_light_off_init_celltype_%d.eps'%itype)
                #if wt_dict['coupling'] > 0:
                #    W0list[1][1,0] = W0list[1][1,0] - 1
                for ivar in [0,1,4,5]: # Wmx, Wmy, s02, k
                    W0list[ivar] = W0list[ivar] + init_noise*np.random.randn(*W0list[ivar].shape)

            # wt_dict['Xi'] = 10
            # wt_dict['Eta'] = 10
            print('size of bounds: '+str(np.sum([np.size(x) for x in bdlist])))
            print('size of w0: '+str(np.sum([np.size(x) for x in W0list])))
            print('size of shapes: '+str(np.sum([np.prod(shp) for shp in shapes])))
            Wt[ihyper][itry],loss[ihyper][itry],gr,hess,result = calnet.fitting_spatial_feature_opto_nonlinear.fit_W_sim(Xhat,Xpc_list,Yhat,Ypc_list,pop_rate_fn=sim_utils.f_miller_troyer,pop_deriv_fn=sim_utils.fprime_miller_troyer,neuron_rate_fn=sim_utils.evaluate_f_mt,W0list=W0list.copy(),bounds=bounds,niter=niter,wt_dict=wt_dict,l2_penalty=l2_penalty,compute_hessian=False,dt=dt,perturbation_size=perturbation_size,dYY=dYY,constrain_isn=constrain_isn,tv=tv,opto_mask=opto_mask,use_opto_transforms=use_opto_transforms,opto_transform1=opto_transform1,opto_transform2=opto_transform2)
    #         Wt[ihyper][itry] = [w[-1] for w in Wt_temp]
    #         loss[ihyper,itry] = loss_temp[-1]
    
    
    def parse_W(W):
        Wmx,Wmy,Wsx,Wsy,s02,K,kappa,T,XX,XXp,Eta,Xi,h1,h2,Eta1,Eta2 = W
        return Wmx,Wmy,Wsx,Wsy,s02,K,kappa,T,XX,XXp,Eta,Xi,h1,h2,Eta1,Eta2
    
    
    itry = 0
    Wmx,Wmy,Wsx,Wsy,s02,K,kappa,T,XX,XXp,Eta,Xi,h1,h2,Eta1,Eta2 = parse_W(Wt[0][0])
    
    labels = ['Wmx','Wmy','Wsx','Wsy','s02','K','kappa','T','XX','XXp','Eta','Xi','h1','h2','Eta1','Eta2']
    Wstar_dict = {}
    for i,label in enumerate(labels):
        Wstar_dict[label] = Wt[0][0][i]
    Wstar_dict['as_list'] = [Wmx,Wmy,Wsx,Wsy,s02,K,kappa,T,XX,XXp,Eta,Xi,h1,h2,Eta1,Eta2]
    Wstar_dict['loss'] = loss[0][0]
    Wstar_dict['wt_dict'] = wt_dict
    np.save(weights_file,Wstar_dict,allow_pickle=True)
