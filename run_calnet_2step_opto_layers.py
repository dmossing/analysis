#!/usr/bin/env python

import calnet.fit_calnet_2step_with_tight_nonlinear_bidi_optogenetics as fcowo
import sys
import pyute as ut
import numpy as np

calnet_data_fold = '/home/dan/calnet_data/'

#ca_data_file = calnet_data_fold+'rs_vm_denoise_200605.npy'
ca_data_file = calnet_data_fold+'rs_200828.npy'
#opto_silencing_data_file = calnet_data_fold+'vip_halo_data_for_sim.npy'
opto_silencing_data_file = calnet_data_fold+'vip_halo_data_for_sim_vip_full_info.npy'
opto_activation_data_file = calnet_data_fold+'vip_chrimson_data_for_sim.npy'

ntries = 1

init_noise = 0.1
tv = True
correct_Eta = False
init_Eta_with_s02 = False
init_Eta12_with_dYY = True
use_opto_transforms = True
share_residuals = True
stimwise = False
simulate1 = True
simulate2 = True
help_constrain_isn = True
ignore_halo_vip = False

if __name__=="__main__":
    weight_base = sys.argv[1]
    
    if len(sys.argv)>2:
        nreps = int(sys.argv[2])
    else:
        nreps = 1

    ut.mkdir(calnet_data_fold+'weights/'+weight_base)

    weights_files = []
    for irep in range(nreps):
        weights_files = weights_files + [calnet_data_fold+'weights/'+weight_base+'/%03d.npy'%itry for itry in range(ntries*irep,ntries*(irep+1))]

    for irep in range(nreps):
        for itry in range(ntries):
            fcowo.fit_weights_and_save(weights_files[irep*ntries+itry],ca_data_file=ca_data_file,opto_silencing_data_file=opto_silencing_data_file,opto_activation_data_file=opto_activation_data_file,allow_var=False,fit_s02=True,constrain_isn=True,tv=tv,l2_penalty=0.1,init_noise=init_noise,init_W_from_lsq=True,scale_init_by=1,correct_Eta=correct_Eta,init_Eta_with_s02=init_Eta_with_s02,init_Eta12_with_dYY=init_Eta12_with_dYY,use_opto_transforms=use_opto_transforms,share_residuals=share_residuals,stimwise=stimwise,simulate1=simulate1,simulate2=simulate2,help_constrain_isn=help_constrain_isn,ignore_halo_vip=ignore_halo_vip)
