#!/usr/bin/env python

import calnet.fit_calnet_ori_with_optogenetics as fcowo
import sys

with open('~/adesnal/path_locations.txt','r') as f:
    path_locs = f.read().splitlines()
for loc in path_locs:    
    print(loc)
    sys.path.insert(0,loc)

calnet_data_fold = '/global/scratch/mossing/calnet_data/'

ca_data_file = calnet_data_fold+'rs_vm_denoise_200605.npy'
opto_data_file = calnet_data_fold+'vip_halo_data_for_sim.npy'

if __name__=="__main__":
    weight_base = sys.argv[1]
    ntries = int(sys.argv[2])

    weights_files = [calnet_data_fold+'weights/'+weight_base+'_%03d.npy'%itry for itry in range(ntries)]

    for itry in range(ntries):
        fcowo.fit_weights_and_save(weights_files[itry],ca_data_file=ca_data_file,opto_data_file=opto_data_file)
