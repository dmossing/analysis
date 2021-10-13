#!/usr/bin/env python

import sys
import calnet.calnet as cc
import calnet.dynamics as dyn
import numpy as np
import glob
import multiprocessing as mp
import opto_utils
import pyute as ut
import scipy.stats as sst

calnet_base = '/home/dan/calnet_data/'
Niter = int(1e3)
opto_levels = 1*np.linspace(0,-1,21)
#opto_levels = np.array((0,0.5))
dt = 1e-1
sim_options = {}
for lbl,val in zip(['Niter','opto_levels','dt'],[Niter,opto_levels,dt]):
    sim_options[lbl] = val

vip_halo_l4_matfile = calnet_base+'VIPHaloSizeContNoInterpLayer.mat'

def compute_opto_line(scall_halo_l4):
    xdata = scall_halo_l4[:,:,:,0]
    ydata = scall_halo_l4[:,:,:,1]
    xdata = xdata.flatten()
    ydata = ydata.flatten()
    lkat = ~np.isnan(xdata) & ~np.isnan(ydata)
    opto_slope,opto_intercept,_,_,_ = sst.linregress(xdata[lkat],ydata[lkat])
    return opto_slope,opto_intercept

def adjust_XX(mdl):
    mdl.XX[:,0::2] = mdl.XX[:,0::2]*opto_slope + opto_intercept*mdl.XX[:,0::2].mean(0,keepdims=True)
    return mdl

def build_models(weights_files):
    nwt = len(weights_files)

    mdls = [None for iwt in range(nwt)]
    for iwt in range(nwt):
        wtdict = np.load(weights_files[iwt],allow_pickle=True)[()]
        mdls[iwt] = cc.ModelOri(wtdict,nT=1)
        mdls[iwt] = adjust_XX(mdls[iwt])

    mdls_no_pcpc = [None for iwt in range(nwt)]
    for iwt in range(nwt):
        wtdict = np.load(weights_files[iwt],allow_pickle=True)[()]
        mdls_no_pcpc[iwt] = cc.ModelOri(wtdict,nT=1)
        mdls_no_pcpc[iwt].Wmy[0,0] = 0
        mdls_no_pcpc[iwt].set_WW()
        mdls_no_pcpc[iwt] = adjust_XX(mdls_no_pcpc[iwt])

    mdls_no_pcpv = [None for iwt in range(nwt)]
    for iwt in range(nwt):
        wtdict = np.load(weights_files[iwt],allow_pickle=True)[()]
        mdls_no_pcpv[iwt] = cc.ModelOri(wtdict,nT=1)
        mdls_no_pcpv[iwt].Wmy[[0,0,3,3],[0,3,0,3]] = 0
        mdls_no_pcpv[iwt].set_WW()
        mdls_no_pcpv[iwt] = adjust_XX(mdls_no_pcpv[iwt])

    mdls_no_vipbias = [None for iwt in range(nwt)]
    for iwt in range(nwt):
        wtdict = np.load(weights_files[iwt],allow_pickle=True)[()]
        mdls_no_vipbias[iwt] = cc.ModelOri(wtdict,nT=1)
        mdls_no_vipbias[iwt].Wmx[1,2] = mdls_no_vipbias[iwt].Wmx[1,2] - 1
        mdls_no_vipbias[iwt].set_WW()
        mdls_no_vipbias[iwt] = adjust_XX(mdls_no_vipbias[iwt])

    mdls_no_pcvip = [None for iwt in range(nwt)]
    for iwt in range(nwt):
        wtdict = np.load(weights_files[iwt],allow_pickle=True)[()]
        mdls_no_pcvip[iwt] = cc.ModelOri(wtdict,nT=1)
        mdls_no_pcvip[iwt].Wmx[0,2] = 0
        mdls_no_pcvip[iwt].Wmy[0,2] = 0
        mdls_no_pcvip[iwt].set_WW()
        mdls_no_pcvip[iwt] = adjust_XX(mdls_no_pcvip[iwt])

    mdls_no_pcsst = [None for iwt in range(nwt)]
    for iwt in range(nwt):
        wtdict = np.load(weights_files[iwt],allow_pickle=True)[()]
        mdls_no_pcsst[iwt] = cc.ModelOri(wtdict,nT=1)
        mdls_no_pcsst[iwt].Wmx[0,1] = 0
        mdls_no_pcsst[iwt].Wmy[0,1] = 0
        mdls_no_pcsst[iwt].set_WW()
        mdls_no_pcsst[iwt] = adjust_XX(mdls_no_pcsst[iwt])
        
    return mdls,mdls_no_pcpc,mdls_no_pcpv,mdls_no_pcvip,mdls_no_pcsst,mdls_no_vipbias

def run_on_mdl(mdl,sim_options):
    Niter,opto_levels,dt = [sim_options[key] for key in ['Niter','opto_levels','dt']]
    average_last = int(np.floor(Niter/5))
    fix_dim = [2,6]
    this_YY_opto = dyn.compute_steady_state_Model(mdl,Niter=Niter,fix_dim=fix_dim,inj_mag=opto_levels,sim_type='inj',dt=dt)
    to_return = np.nanmean(this_YY_opto[:,:,-average_last:],2)
    return to_return

def simulate_opto_effects(mdls,sim_options=sim_options,pool_size=1):
    nwt = len(mdls)
    Niter,opto_levels,dt = [sim_options[key] for key in ['Niter','opto_levels','dt']]
    nopto = sim_options['opto_levels'].size
    if pool_size==1:
        YY_opto_tavg = np.zeros((nwt,nopto)+mdls[0].Eta.shape)
        for iwt in range(nwt):
            print('model #%d'%iwt)
            YY_opto_tavg[iwt] = run_on_mdl(mdls[iwt],sim_options)
    else:
        with mp.Pool(pool_size) as p:
            YY_opto_tavg = p.starmap(run_on_mdl,[(m,sim_options) for m in mdls])
        YY_opto_tavg = np.array(YY_opto_tavg)

    return YY_opto_tavg

def build_models_and_simulate_opto_effects(weights_files,target_file,sim_options=sim_options,pool_size=1):
    all_mdls = build_models(weights_files)
    YYs = [simulate_opto_effects(mdls,sim_options=sim_options,pool_size=pool_size) for mdls in all_mdls]
    mdls = all_mdls[0]
    nwt = len(mdls)
    iwt = 0
    nQ,nS,nT = mdls[iwt].nQ,mdls[iwt].nS,mdls[iwt].nT
    bltiles = np.zeros((nwt,nQ*nS*nT))
    amps = np.ones((nwt,nQ*nS*nT))
    for iwt in range(nwt):
        wtdict = np.load(weights_files[iwt],allow_pickle=True)[()]
        if 'bl' in wtdict:
            bl = wtdict['bl']
            bltiles[iwt] = np.tile(bl,nS*nT)
        if 'amp' in wtdict:
            amp = wtdict['amp']
            amps[iwt] = amp
    def transform(YY_opto_tavg):
        YY_opto_tavg = amps[:,np.newaxis,np.newaxis,:]*YY_opto_tavg + bltiles[:,np.newaxis,np.newaxis,:]
        return YY_opto_tavg
    YYs = [transform(YY) for YY in YYs]
    keys = ['YY_opto','YY_opto_no_pcpc','YY_opto_no_pcpv','YY_opto_tavg_no_pcvip','YY_opto_tavg_no_pcsst','YY_opto_tavg_no_vipbias']
    to_save = dict(zip(keys,YYs))
    XX_opto = np.concatenate([mdl.XX[np.newaxis] for mdl in mdls],axis=0)
    to_save['XX_opto'] = XX_opto
    np.save(target_file,to_save)

def run(fit_lbl,calnet_base=calnet_base,sim_options=sim_options,pool_size=1,lcutoff=10):
    weights_fold = calnet_base + 'weights/weights_%s/'%fit_lbl
    weights_files = glob.glob(weights_fold+'*.npy')
    weights_files.sort()
    losses = np.zeros((len(weights_files),))
    for iwt in range(len(weights_files)):
        Wstar_dict = np.load(weights_files[iwt],allow_pickle=True)[()]
        losses[iwt] = Wstar_dict['loss']
    weights_files = [wf for wf,l in zip(weights_files,losses) if l < np.nanpercentile(losses,lcutoff)]
    target_file = calnet_base + 'dynamics/vip_halo_l4_opto_tavg_connection_deletion_%s.npy'%fit_lbl
    build_models_and_simulate_opto_effects(weights_files,target_file,sim_options=sim_options,pool_size=pool_size)

if __name__=='__main__':

    rrs_orig_layer,rfs_orig_layer = ut.loadmat(vip_halo_l4_matfile,['rrs','rfs'])
    scall_halo_l4 = opto_utils.norm_to_mean_light_off(rrs_orig_layer[0,0])
    opto_slope,opto_intercept = compute_opto_line(scall_halo_l4)

    fit_lbl = sys.argv[1]
    if len(sys.argv)==3:
        pool_size = int(sys.argv[2])
        run(fit_lbl,pool_size=pool_size)
    elif len(sys.argv)>3:
        pool_size = int(sys.argv[2])
        calnet_base = sys.argv[3]
        run(fit_lbl,pool_size=pool_size,calnet_base=calnet_base)
    else:
        run(fit_lbl)
