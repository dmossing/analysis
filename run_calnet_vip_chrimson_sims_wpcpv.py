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
opto_levels = np.array((1.1,))
#opto_levels = np.array((0,0.5))
dt = 1e-1
sim_options = {}
for lbl,val in zip(['Niter','opto_levels','dt'],[Niter,opto_levels,dt]):
    sim_options[lbl] = val

vip_chrimson_l4_npyfile = calnet_base+'vip_chrimson_data_for_sim.npy'

def compute_opto_line(scall_chrimson_l4):
    this_data = np.nanmean(scall_chrimson_l4,3)
    xdata = this_data[:,:,:,0]
    ydata = this_data[:,:,:,1]
    xdata = xdata.flatten()
    ydata = ydata.flatten()
    lkat = ~np.isnan(xdata) & ~np.isnan(ydata)
    opto_slope,opto_intercept,_,_,_ = sst.linregress(xdata[lkat],ydata[lkat])
    return opto_slope,opto_intercept

def adjust_XX(mdl):
    mdl.XX[:,0::2] = mdl.XX[:,0::2]*opto_slope + opto_intercept*mdl.XX[:,0::2].mean(0,keepdims=True)
    return mdl

def build_models(weights_files):
    #weights_files = np.load(weights_files_list_file,allow_pickle=True)
    nwt = len(weights_files)

    mdls = [None for iwt in range(nwt)]
    for iwt in range(nwt):
        wtdict = np.load(weights_files[iwt],allow_pickle=True)[()]
        mdls[iwt] = cc.ModelOri(wtdict,nT=1)
        mdls[iwt] = adjust_XX(mdls[iwt])

    mdls_no_pcpc = [None for iwt in range(nwt)]
    for iwt in range(nwt):
        wtdict = np.load(weights_files[iwt],allow_pickle=True)[()]
        wtdict['Wmy'][0,0] = 0
        mdls_no_pcpc[iwt] = cc.ModelOri(wtdict,nT=1)
        mdls_no_pcpc[iwt] = adjust_XX(mdls_no_pcpc[iwt])

    mdls_no_pcpv = [None for iwt in range(nwt)]
    for iwt in range(nwt):
        wtdict = np.load(weights_files[iwt],allow_pickle=True)[()]
        wtdict['Wmy'][[0,0,3,3],[0,3,0,3]] = 0
        mdls_no_pcpv[iwt] = cc.ModelOri(wtdict,nT=1)
        mdls_no_pcpv[iwt] = adjust_XX(mdls_no_pcpv[iwt])

    mdls_no_vipbias = [None for iwt in range(nwt)]
    for iwt in range(nwt):
        wtdict = np.load(weights_files[iwt],allow_pickle=True)[()]
        wtdict['Wmy'][1,2] = wtdict['Wmy'][1,2] - 1
        mdls_no_vipbias[iwt] = cc.ModelOri(wtdict,nT=1)
        mdls_no_vipbias[iwt] = adjust_XX(mdls_no_vipbias[iwt])

    mdls_no_pcvip = [None for iwt in range(nwt)]
    for iwt in range(nwt):
        wtdict = np.load(weights_files[iwt],allow_pickle=True)[()]
        wtdict['Wmy'][0,2] = 0
        wtdict['Wmx'][0,2] = 0
        mdls_no_pcvip[iwt] = cc.ModelOri(wtdict,nT=1)
        mdls_no_pcvip[iwt] = adjust_XX(mdls_no_pcvip[iwt])

    mdls_no_pcsst = [None for iwt in range(nwt)]
    for iwt in range(nwt):
        wtdict = np.load(weights_files[iwt],allow_pickle=True)[()]
        wtdict['Wmy'][0,1] = 0
        wtdict['Wmx'][0,1] = 0
        mdls_no_pcsst[iwt] = cc.ModelOri(wtdict,nT=1)
        mdls_no_pcsst[iwt] = adjust_XX(mdls_no_pcsst[iwt])
        
    return mdls,mdls_no_pcpc,mdls_no_pcpv,mdls_no_pcvip,mdls_no_pcsst,mdls_no_vipbias

def run_on_mdl(mdl,sim_options):
    Niter,opto_levels,dt = [sim_options[key] for key in ['Niter','opto_levels','dt']]
    average_last = int(np.floor(Niter/5))
    this_YY_opto = dyn.compute_steady_state_Model(mdl,Niter=Niter,fix_dim=2,inj_mag=opto_levels,sim_type='inj',dt=dt)
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
    #return mdls,mdls_no_pcpc,mdls_no_pcpv,mdls_no_pcvip,mdls_no_pcsst,mdls_no_vipbias
    all_mdls = build_models(weights_files)
    mdls = all_mdls[0]
    XX_opto = np.concatenate([mdl.XX[np.newaxis] for mdl in mdls],axis=0)
    nwt = len(mdls)
    all_YYs = [simulate_opto_effects(mdls,sim_options=sim_options,pool_size=pool_size) for mdls in all_mdls]
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
    all_YYs =  [amps[:,np.newaxis,np.newaxis,:]*YY + bltiles[:,np.newaxis,np.newaxis,:] for YY in all_YYs]
    #YY_opto,YY_opto_no_pcpc,YY_opto_no_pcpv = all_YYs
    #np.save(target_file,{'YY_opto':YY_opto,'YY_opto_no_pcpc':YY_opto_no_pcpc,'YY_opto_no_pcpv':YY_opto_no_pcpv,'XX_opto':XX_opto})
    YY_opto,YY_opto_no_pcpc,YY_opto_no_pcpv,YY_opto_no_pcvip,YY_opto_no_pcsst,YY_opto_no_vipbias = all_YYs
    np.save(target_file,{'YY_opto':YY_opto,'YY_opto_no_pcpc':YY_opto_no_pcpc,'YY_opto_no_pcpv':YY_opto_no_pcpv,\
            'YY_opto_no_pcvip':YY_opto_no_pcvip,'YY_opto_no_pcsst':YY_opto_no_pcsst,'YY_opto_no_vipbias':YY_opto_no_vipbias,'XX_opto':XX_opto})

def run(fit_lbl,calnet_base=calnet_base,sim_options=sim_options,pool_size=1):
    weights_fold = calnet_base + 'weights/weights_%s/'%fit_lbl
    weights_files = glob.glob(weights_fold+'*.npy')
    weights_files.sort()
    target_file = calnet_base + 'dynamics/vip_chrimson_l23_as_l4_opto_tavg_wpcpv_%s.npy'%fit_lbl
    build_models_and_simulate_opto_effects(weights_files,target_file,sim_options=sim_options,pool_size=pool_size)

if __name__=='__main__':

    scallanimal_aligned = np.load(vip_chrimson_l4_npyfile,allow_pickle=True)[()]['scallanimal_aligned']
    scall_chrimson_l4 = opto_utils.norm_to_mean_light_off(scallanimal_aligned)
    opto_slope,opto_intercept = compute_opto_line(scall_chrimson_l4)

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
