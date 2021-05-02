#!/usr/bin/env python

import calnet.fit_calnet_2step_ori_multiout_axon_with_fg_run_modulation as fcowo
import calnet.utils
import sys
import pyute as ut
import numpy as np
import multiprocessing as mp
import glob


calnet_data_fold = '/home/dan/calnet_data/'

#ca_data_file = calnet_data_fold+'rs_vm_denoise_200605.npy'
#ca_data_file = calnet_data_fold+'rs_sc_fg_pval_0_05_210410.npy'
ca_data_file = calnet_data_fold+'rs_sc_fg_ret_pval_0_05_210425.npy'
opto_silencing_data_file = calnet_data_fold+'vip_halo_data_for_sim_vip_full_info.npy'
opto_activation_data_file = calnet_data_fold+'vip_chrimson_data_for_sim.npy'

init_noise = 0.3
allow_var = False#True
multiout = False#True
multiout2 = False#True
tv = True
correct_Eta = False
init_Eta_with_s02 = False
init_Eta12_with_dYY = False
use_opto_transforms = True
norm_opto_transforms = True
share_residuals = True
stimwise = False
simulate1 = True
simulate2 = True
verbose = True
free_amplitude = False
no_halo_res = False
ignore_halo_vip = False
foldT = False
nondim = True
fit_running = False
fit_non_running = True
fit_sc = True
fit_fg = False
fit_ret = False#True
l2_penalty = 0.1
l1_penalty = 10.0
nT = 1#1
axon = False
run_modulation = False
init_W_from_lsq = False
init_W_from_file = True

parallel = True

fit_options = {'ca_data_file':ca_data_file,\
'opto_silencing_data_file':opto_silencing_data_file,\
'opto_activation_data_file':opto_activation_data_file,\
'allow_var':allow_var,\
'multiout':multiout,\
'multiout2':multiout2,\
'fit_s02':True,\
'constrain_isn':True,\
'tv':tv,\
'l2_penalty':l2_penalty,\
'l1_penalty':l1_penalty,\
'init_noise':init_noise,\
'init_W_from_lsq':init_W_from_lsq,\
'scale_init_by':1,\
'init_W_from_file':init_W_from_file,\
'correct_Eta':correct_Eta,\
'init_Eta_with_s02':init_Eta_with_s02,\
'no_halo_res':no_halo_res,\
'ignore_halo_vip':ignore_halo_vip,\
'use_opto_transforms':use_opto_transforms,\
'norm_opto_transforms':norm_opto_transforms,\
'foldT':foldT,\
'nondim':nondim,\
'fit_running':fit_running,\
'fit_non_running':fit_non_running,\
'fit_sc':fit_sc,\
'fit_fg':fit_fg,\
'fit_ret':fit_ret,\
'nT':nT,\
'axon':axon,\
'run_modulation':run_modulation}

zero_extra_weights = [np.zeros((2,4),dtype='bool'),np.zeros((4,4),dtype='bool')]
#zero_extra_weights[1][2,1] = True # temporarily constraining VIP->SST weight to be 0
#zero_extra_weights[1][1,2] = True # temporarily constraining SST->VIP weight to be 0

#def run_fitting(init_file,target_name,seed=None):
#    print('running %s -> %s'%(init_file,target_name))
#    if not seed is None:
#        np.random.seed(seed=seed)
#    fcowo.fit_weights_and_save(target_name,ca_data_file=ca_data_file,opto_silencing_data_file=opto_silencing_data_file,opto_activation_data_file=opto_activation_data_file,allow_var=False,fit_s02=True,constrain_isn=True,tv=tv,l2_penalty=0.1,init_noise=init_noise,init_W_from_lsq=True,scale_init_by=1,init_W_from_file=True,init_file=init_file,correct_Eta=correct_Eta,init_Eta_with_s02=init_Eta_with_s02,init_Eta12_with_dYY=init_Eta12_with_dYY,use_opto_transforms=use_opto_transforms,share_residuals=share_residuals,stimwise=stimwise,simulate1=simulate1,simulate2=simulate2,verbose=verbose,free_amplitude=free_amplitude,norm_opto_transforms=norm_opto_transforms,zero_extra_weights=zero_extra_weights)
#
#def run_fitting_one_arg(inp):
#    init_file,target_name,seed = inp
#    run_fitting(init_file,target_name,seed=seed)

if __name__=="__main__":
    weight_base = sys.argv[1]
    
    if len(sys.argv)>2:
        nreps = int(sys.argv[2])
    else:
        nreps = 1

    if len(sys.argv)>3:
        nprocesses = int(sys.argv[3])
    else:
        nprocesses = 3

    if len(sys.argv)>4:
        weights_files_base = str(sys.argv[4])
    else:
        weights_files_base = None

    if len(sys.argv)>5:
        pctile_cutoff = int(sys.argv[5])
    else:
        pctile_cutoff = 100

    if len(sys.argv)>6:
        offset = int(sys.argv[6])
    else:
        offset = 0

    if not weights_files_base is None:
        print(calnet_data_fold+'weights/'+weights_files_base+'/*.npy')
        weights_files = glob.glob(calnet_data_fold+'weights/'+weights_files_base+'/*.npy')
        weights_files.sort()

    losses = np.zeros((len(weights_files),))
    for ifile,weights_file in enumerate(weights_files):
        npyfile = np.load(weights_file,allow_pickle=True)[()]
        losses[ifile] = npyfile['loss']
    print(np.nanpercentile(losses,pctile_cutoff))

    weights_files = [wf for (wf,loss) in zip(weights_files,losses) if loss<np.nanpercentile(losses,pctile_cutoff)]
    print(weights_files)

    init_files = list(weights_files)
    ntries = len(init_files)

    fws_fn = fcowo.fit_weights_and_save

    calnet.utils.run_all_fitting(fws_fn=fws_fn,calnet_data_fold=calnet_data_fold,\
            weight_base=weight_base,offset=offset,nreps=nreps,fit_options=fit_options,parallel=parallel,nprocesses=nprocesses,init_files=init_files)
