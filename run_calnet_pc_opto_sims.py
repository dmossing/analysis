#!/usr/bin/env python

import sys
import calnet.calnet as cc
import calnet.dynamics as dyn
import numpy as np
import glob
import multiprocessing as mp

calnet_base = '/home/dan/calnet_data/'
Niter = int(2e3)
opto_levels = 1*np.linspace(-1.5,1.5,31)
#opto_levels = 1*np.linspace(-0.15,0.15,31)
dt = 1e-1
fix_dim = 0
avg_last_factor = 0.5
sim_options = {}
for lbl,val in zip(['Niter','opto_levels','dt','fix_dim','avg_last_factor'],[Niter,opto_levels,dt,fix_dim,avg_last_factor]):
    sim_options[lbl] = val

def build_models(weights_files):
    #weights_files = np.load(weights_files_list_file,allow_pickle=True)
    nwt = len(weights_files)

    mdls = [None for iwt in range(nwt)]
    for iwt in range(nwt):
        wtdict = np.load(weights_files[iwt],allow_pickle=True)[()]
        mdls[iwt] = cc.ModelOri(wtdict,nT=1)

    mdls_no_pcpc = [None for iwt in range(nwt)]
    for iwt in range(nwt):
        wtdict = np.load(weights_files[iwt],allow_pickle=True)[()]
        mdls_no_pcpc[iwt] = cc.ModelOri(wtdict,nT=1)
        mdls_no_pcpc[iwt].Wmy[0,0] = 0
        mdls_no_pcpc[iwt].set_WW()

    mdls_no_pcpv = [None for iwt in range(nwt)]
    for iwt in range(nwt):
        wtdict = np.load(weights_files[iwt],allow_pickle=True)[()]
        mdls_no_pcpv[iwt] = cc.ModelOri(wtdict,nT=1)
        mdls_no_pcpv[iwt].Wmy[[0,0,3,3],[0,3,0,3]] = 0
        mdls_no_pcpv[iwt].set_WW()

    mdls_no_vipbias = [None for iwt in range(nwt)]
    for iwt in range(nwt):
        wtdict = np.load(weights_files[iwt],allow_pickle=True)[()]
        mdls_no_vipbias[iwt] = cc.ModelOri(wtdict,nT=1)
        mdls_no_vipbias[iwt].Wmx[1,2] = mdls_no_vipbias[iwt].Wmx[1,2] - 1
        mdls_no_vipbias[iwt].set_WW()

    mdls_no_pcvip = [None for iwt in range(nwt)]
    for iwt in range(nwt):
        wtdict = np.load(weights_files[iwt],allow_pickle=True)[()]
        mdls_no_pcvip[iwt] = cc.ModelOri(wtdict,nT=1)
        mdls_no_pcvip[iwt].Wmx[0,2] = 0
        mdls_no_pcvip[iwt].Wmy[0,2] = 0
        mdls_no_pcvip[iwt].set_WW()

    mdls_no_pcsst = [None for iwt in range(nwt)]
    for iwt in range(nwt):
        wtdict = np.load(weights_files[iwt],allow_pickle=True)[()]
        mdls_no_pcsst[iwt] = cc.ModelOri(wtdict,nT=1)
        mdls_no_pcsst[iwt].Wmx[0,1] = 0
        mdls_no_pcsst[iwt].Wmy[0,1] = 0
        mdls_no_pcsst[iwt].set_WW()
        
    mdls_no_sstvip = [None for iwt in range(nwt)]
    for iwt in range(nwt):
        wtdict = np.load(weights_files[iwt],allow_pickle=True)[()]
        mdls_no_sstvip[iwt] = cc.ModelOri(wtdict,nT=1)
        mdls_no_sstvip[iwt].Wmy[1,2] = 0
        mdls_no_sstvip[iwt].set_WW()
        
    return mdls,mdls_no_pcpc,mdls_no_pcpv,mdls_no_pcvip,mdls_no_pcsst,mdls_no_vipbias,mdls_no_sstvip

def run_on_mdl(mdl,sim_options):
    Niter,opto_levels,dt,fix_dim,avg_last_factor = [sim_options[key] for key in ['Niter','opto_levels','dt','fix_dim','avg_last_factor']]
    average_last = int(np.floor(Niter*avg_last_factor)) #/5
    this_YY_opto = dyn.compute_steady_state_Model(mdl,Niter=Niter,fix_dim=fix_dim,inj_mag=opto_levels,sim_type='inj',dt=dt)
    to_return = np.nanmean(this_YY_opto[:,:,-average_last:],2)
    return to_return

def simulate_opto_effects(mdls,sim_options=sim_options,pool_size=1):
    nwt = len(mdls)
    Niter,opto_levels,dt,fix_dim,avg_last_factor = [sim_options[key] for key in ['Niter','opto_levels','dt','fix_dim','avg_last_factor']]
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
    print(len(all_mdls))
    mdls = all_mdls[0]
    #mdls,mdls_no_pcpc,mdls_no_pcpv = build_models(weights_files)
    nwt = len(mdls)
    #YY_opto_tavg = simulate_opto_effects(mdls,sim_options=sim_options,pool_size=pool_size)
    #YY_opto_tavg_no_pcpc = simulate_opto_effects(mdls_no_pcpc,sim_options=sim_options,pool_size=pool_size)
    #YY_opto_tavg_no_pcpv = simulate_opto_effects(mdls_no_pcpv,sim_options=sim_options,pool_size=pool_size)
    print(nwt)
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
    keys = ['YY_opto','YY_opto_no_pcpc','YY_opto_no_pcpv','YY_opto_tavg_no_pcvip','YY_opto_tavg_no_pcsst','YY_opto_tavg_no_vipbias','YY_opto_tavg_no_sstvip']
    to_save = dict(zip(keys,YYs))
    np.save(target_file,to_save)
    #YY_opto_tavg = transform(YY_opto_tavg)
    #YY_opto_tavg_no_pcpc = transform(YY_opto_tavg_no_pcpc)
    #YY_opto_tavg_no_pcpv = transform(YY_opto_tavg_no_pcpv)
    #np.save(target_file,{'YY_opto':YY_opto_tavg,'YY_opto_no_pcpc':YY_opto_tavg_no_pcpc,'YY_opto_no_pcpv':YY_opto_tavg_no_pcpv})

def run(fit_lbl,calnet_base=calnet_base,sim_options=sim_options,pool_size=1):
    weights_fold = calnet_base + 'weights/weights_%s/'%fit_lbl
    weights_files = glob.glob(weights_fold+'*.npy')
    weights_files.sort()
    target_file = calnet_base + 'dynamics/pc_opto_tavg_%s.npy'%fit_lbl
    #target_file = calnet_base + 'dynamics/pc_small_opto_tavg_%s.npy'%fit_lbl
    build_models_and_simulate_opto_effects(weights_files,target_file,sim_options=sim_options,pool_size=pool_size)

if __name__=='__main__':
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
