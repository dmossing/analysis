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

    for i1 in range(4):
        for i2 in range(4):

            weights_files = [calnet_data_fold+'weights/'+weight_base+'/constrain_%d_%d_%03d.npy'%(i1,i2,itry) for itry in range(ntries)]

            for itry in range(ntries):
                fcowo.fit_weights_and_save(weights_files[itry],ca_data_file=ca_data_file,opto_data_file=opto_data_file)
