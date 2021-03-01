#!/usr/bin/env python

import sys
import calnet.calnet as cc
import calnet.dynamics as dyn
import numpy as np
import glob
import multiprocessing as mp

calnet_base = '/home/dan/calnet_data/'
Niter = int(2e3)
#opto_levels = 1*np.linspace(-1,1,21)
opto_levels = np.array((-0.3,0,0.3))
#opto_levels = 1*np.linspace(0,1.5,16)
#opto_levels = 1*np.linspace(-0.15,0.15,31)
dt = 1e-1
fix_dim = 0
avg_last_factor = 0.5
sim_type = 'layer4'
sim_options = {}
for lbl,val in zip(['Niter','opto_levels','dt','fix_dim','avg_last_factor','sim_type'],[Niter,opto_levels,dt,fix_dim,avg_last_factor,sim_type]):
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
        wtdict['Wmy'][0,0] = 0
        mdls_no_pcpc[iwt] = cc.ModelOri(wtdict,nT=1)

    mdls_no_pcpv = [None for iwt in range(nwt)]
    for iwt in range(nwt):
        wtdict = np.load(weights_files[iwt],allow_pickle=True)[()]
        wtdict['Wmy'][[0,0,3,3],[0,3,0,3]] = 0
        mdls_no_pcpv[iwt] = cc.ModelOri(wtdict,nT=1)

    mdls_no_pcvip = [None for iwt in range(nwt)]
    for iwt in range(nwt):
        wtdict = np.load(weights_files[iwt],allow_pickle=True)[()]
        wtdict['Wmy'][0,2] = 0
        wtdict['Wmx'][0,2] = 0
        mdls_no_pcvip[iwt] = cc.ModelOri(wtdict,nT=1)
        
    return mdls,mdls_no_pcpc,mdls_no_pcpv,mdls_no_pcvip

def run_on_mdl(mdl,sim_options):
    Niter,opto_levels,dt,fix_dim,avg_last_factor,sim_type = [sim_options[key] for key in ['Niter','opto_levels','dt','fix_dim','avg_last_factor','sim_type']]
    average_last = int(np.floor(Niter*avg_last_factor)) #/5
    #print('sim type: %s'%sim_type)
    this_YY_opto = dyn.compute_steady_state_Model(mdl,Niter=Niter,fix_dim=fix_dim,inj_mag=opto_levels,sim_type=sim_type,dt=dt)
    to_return = np.nanmean(this_YY_opto[:,:,-average_last:],2)
    return to_return

def simulate_opto_effects(mdls,sim_options=sim_options,pool_size=1):
    nwt = len(mdls)
    Niter,opto_levels,dt,fix_dim,avg_last_factor,sim_type = [sim_options[key] for key in ['Niter','opto_levels','dt','fix_dim','avg_last_factor','sim_type']]
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
    mdls,mdls_no_pcpc,mdls_no_pcpv,mdls_no_pcvip = build_models(weights_files)
    nwt = len(mdls)
    YY_opto_tavg = simulate_opto_effects(mdls,sim_options=sim_options,pool_size=pool_size)
    YY_opto_tavg_no_pcpc = simulate_opto_effects(mdls_no_pcpc,sim_options=sim_options,pool_size=pool_size)
    YY_opto_tavg_no_pcpv = simulate_opto_effects(mdls_no_pcpv,sim_options=sim_options,pool_size=pool_size)
    YY_opto_tavg_no_pcvip = simulate_opto_effects(mdls_no_pcvip,sim_options=sim_options,pool_size=pool_size)
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
    YY_opto_tavg = transform(YY_opto_tavg)
    YY_opto_tavg_no_pcpc = transform(YY_opto_tavg_no_pcpc)
    YY_opto_tavg_no_pcpv = transform(YY_opto_tavg_no_pcpv)
    YY_opto_tavg_no_pcvip = transform(YY_opto_tavg_no_pcvip)
    np.save(target_file,{'YY_opto':YY_opto_tavg,'YY_opto_no_pcpc':YY_opto_tavg_no_pcpc,'YY_opto_no_pcpv':YY_opto_tavg_no_pcpv,'YY_opto_no_pcvip':YY_opto_tavg_no_pcvip})

def run(fit_lbl,calnet_base=calnet_base,sim_options=sim_options,pool_size=1):
    weights_fold = calnet_base + 'weights/weights_%s/'%fit_lbl
    weights_files = glob.glob(weights_fold+'*.npy')
    weights_files.sort()
    #weights_files = weights_files[:1]
    target_file = calnet_base + 'dynamics/l4_opto_tavg_w_and_wo_wpcvip_%s.npy'%fit_lbl
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
