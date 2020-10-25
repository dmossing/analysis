#!/usr/bin/env python

import calnet.fit_calnet_ori_with_optogenetics as fcowo
import sys
import pyute as ut

calnet_data_fold = '/Users/dan/Documents/notebooks/mossing-PC/shared_data/calnet_data/'

ca_data_file = calnet_data_fold+'rs_vm_denoise_200605.npy'
opto_data_file = calnet_data_fold+'vip_halo_data_for_sim.npy'


if __name__=="__main__":
    weight_base = sys.argv[1]
    ntries = int(sys.argv[2])

    ut.mkdir(calnet_data_fold+'weights/'+weight_base)

    weights_files = [calnet_data_fold+'weights/'+weight_base+'/%03d.npy'%itry for itry in range(ntries)]

    for itry in range(ntries):
        fcowo.fit_weights_and_save(weights_files[itry],ca_data_file=ca_data_file,opto_data_file=opto_data_file,allow_var=False,fit_s02=True,constrain_isn=True,l2_penalty=0.1,init_noise=0.1,init_W_from_lsq=True,scale_init_by=1)
