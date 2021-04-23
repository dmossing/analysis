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

weights_base = calnet_data_fold + 'weights/weights_200721e/'
good_fits = ['006','010','012','014','018','019','020','021','023','025','030',\
             '035','042','045','049','058','060','064','066','076','078','083']
weights_files = [weights_base+fit+'.npy' for fit in good_fits]

weights_base = calnet_data_fold + 'weights/weights_200722a/'
good_fits = ['003','012','014','025','028','029','031','048','059','060',\
             '066','072','079','081','088','089','090','094']
weights_files = weights_files + [weights_base+fit+'.npy' for fit in good_fits]

weights_base = calnet_data_fold + 'weights/weights_200723a/'
good_fits = ['000','001','002','004','011','025','026','027','029','032',\
             '043','045','064','065','069','070','072','076','080','086',\
            '087','089','093','096']
weights_files = weights_files + [weights_base+fit+'.npy' for fit in good_fits]


weights_base = calnet_data_fold + 'weights/weights_200724a/'
good_fits = ['000','003','004','017','018','028','029','034','037','044',\
             '046','050','054','058','060','062','066','069','072','079',\
            '086','087','091','092','097']
weights_files = weights_files + [weights_base+fit+'.npy' for fit in good_fits]


weights_base = calnet_data_fold + 'weights/weights_200725a/'
good_fits = ['002','003','005','010','015','034','042','070','072','073',\
             '074','076','079','080','085','087','089','092','094','102',\
            '103','105','118','120','123','132','134','137','138','141',\
            '143','145','147','161','164','166','171','174','180','182',\
            '185','192','193','199']
weights_files = weights_files + [weights_base+fit+'.npy' for fit in good_fits]

#weights_files = np.load('/Users/dan/Documents/notebooks/mossing-PC/simulation/weights_files_list.npy',allow_pickle=True)[()]

init_files = weights_files
ntries = len(init_files)

init_noise = 0.1
allow_var = True
multiout = False #True
multiout2 = False #True
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
'init_W_from_file':True,\
'correct_Eta':correct_Eta,\
'init_Eta_with_s02':init_Eta_with_s02,\
'no_halo_res':no_halo_res,\
'ignore_halo_vip':ignore_halo_vip,\
'use_opto_transforms':use_opto_transforms,\
'norm_opto_transforms':norm_opto_transforms,\
'foldT':foldT}

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
        init_base = sys.argv[4]
    else:
        init_base = None
        
    if not init_base is None:
        init_files = glob.glob(calnet_data_fold + 'weights/' + init_base + '/*.npy')
        init_files.sort()
        #print(calnet_data_fold + 'weights/' + init_base + '/*.npy')
        #print(init_files)

    offset = 0

    fws_fn = fcowo.fit_weights_and_save

    calnet.utils.run_all_fitting(fws_fn=fws_fn,init_files=init_files,calnet_data_fold=calnet_data_fold,\
            weight_base=weight_base,offset=offset,nreps=nreps,fit_options=fit_options,parallel=parallel,nprocesses=nprocesses)

