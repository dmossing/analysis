#!/usr/bin/env python

import calnet.fit_calnet_2step_ori_multiout_axon_with_fg_run_modulation as fcowo
import sys
import pyute as ut
import autograd.numpy as np
import calnet.utils
import glob

calnet_data_fold = '/home/dan/calnet_data/'

#ca_data_file = calnet_data_fold+'rs_vm_denoise_200605.npy'
#ca_data_file = calnet_data_fold+'rs_sc_fg_pval_0_05_210410.npy'
ca_data_file = calnet_data_fold+'rs_sc_fg_ret_pval_0_05_210423.npy'
opto_silencing_data_file = calnet_data_fold+'vip_halo_data_for_sim_vip_full_info.npy'
opto_activation_data_file = calnet_data_fold+'vip_chrimson_data_for_sim.npy'

init_noise = 0.3
allow_var = True
multiout = True
multiout2 = True
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
ignore_halo_vip=False
foldT = False
nondim = True
fit_running = True#False
fit_non_running = True
fit_sc = True
fit_fg = True
fit_ret = True
l2_penalty = 0.1
l1_penalty = 10.0

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
'init_W_from_lsq':True,\
'scale_init_by':1,\
'init_W_from_file':False,\
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
'fit_ret':fit_ret}

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

    offset = 0

    fws_fn = fcowo.fit_weights_and_save

    calnet.utils.run_all_fitting(fws_fn=fws_fn,calnet_data_fold=calnet_data_fold,\
            weight_base=weight_base,offset=offset,nreps=nreps,fit_options=fit_options,parallel=parallel,nprocesses=nprocesses)

