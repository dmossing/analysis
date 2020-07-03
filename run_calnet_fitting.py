#!/usr/bin/env python

import fit_calnet_ori_with_optogenetics as fcowo
import sys

ca_data_file = '/Users/dan/Documents/notebooks/mossing-PC/simulation/rs_vm_denoise_200605.npy'
opto_data_file = '/Users/dan/Documents/notebooks/mossing-PC/shared_data/vip_halo_data_for_sim.npy'

if __name__=="__main__":
    ntries = int(sys.argv[1])

    weights_files = ['/Users/dan/Documents/notebooks/mossing-PC/shared_data/weights_200703_%d.npy'%itry for itry in range(ntries)]

    for itry in range(ntries):
        fcowo.fit_weights_and_save(weights_files[itry],ca_data_file=ca_data_file,opto_data_file=opto_data_file)
