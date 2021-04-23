#!/usr/bin/env python

import calnet.fit_calnet_2step_ori_multiout as fcowo
import sys
import pyute as ut
import numpy as np
import calnet.utils
import glob

calnet_data_fold = '/home/dan/calnet_data/'

#ca_data_file = calnet_data_fold+'rs_vm_denoise_200605.npy'
ca_data_file = calnet_data_fold+'rs_200828.npy'
opto_silencing_data_file = calnet_data_fold+'vip_halo_data_for_sim_vip_full_info.npy'
opto_activation_data_file = calnet_data_fold+'vip_chrimson_data_for_sim.npy'

init_noise = 0.1
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
foldT = True
nondim = True

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
'l2_penalty':0.1,\
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
'nondim':nondim}

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

