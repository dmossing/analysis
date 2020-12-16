#!/usr/bin/env python

import calnet.fit_calnet_2step_with_tight_nonlinear_bidi_optogenetics_baseline as fcowo
import sys
import pyute as ut
import numpy as np
import multiprocessing as mp
import glob

calnet_data_fold = '/home/dan/calnet_data/'

#ca_data_file = calnet_data_fold+'rs_vm_denoise_200605.npy'
ca_data_file = calnet_data_fold+'rs_200828.npy'
#opto_silencing_data_file = calnet_data_fold+'vip_halo_data_for_sim.npy'
opto_silencing_data_file = calnet_data_fold+'vip_halo_data_for_sim_vip_full_info.npy'
opto_activation_data_file = calnet_data_fold+'vip_chrimson_data_for_sim.npy'

weights_base = calnet_data_fold+'weights/weights_200721e/'
good_fits = ['006','010','012','014','018','019','020','021','023','025','030',\
             '035','042','045','049','058','060','064','066','076','078','083']
weights_files = [weights_base+fit+'.npy' for fit in good_fits]

weights_base = calnet_data_fold+'weights/weights_200722a/'
good_fits = ['003','012','014','025','028','029','031','048','059','060',\
             '066','072','079','081','088','089','090','094']
weights_files = weights_files + [weights_base+fit+'.npy' for fit in good_fits]

weights_base = calnet_data_fold+'weights/weights_200723a/'
good_fits = ['000','001','002','004','011','025','026','027','029','032',\
             '043','045','064','065','069','070','072','076','080','086',\
            '087','089','093','096']
weights_files = weights_files + [weights_base+fit+'.npy' for fit in good_fits]


weights_base = calnet_data_fold+'weights/weights_200724a/'
good_fits = ['000','003','004','017','018','028','029','034','037','044',\
             '046','050','054','058','060','062','066','069','072','079',\
            '086','087','091','092','097']
weights_files = weights_files + [weights_base+fit+'.npy' for fit in good_fits]


weights_base = calnet_data_fold+'weights/weights_200725a/'
good_fits = ['002','003','005','010','015','034','042','070','072','073',\
             '074','076','079','080','085','087','089','092','094','102',\
            '103','105','118','120','123','132','134','137','138','141',\
            '143','145','147','161','164','166','171','174','180','182',\
            '185','192','193','199']
weights_files = weights_files + [weights_base+fit+'.npy' for fit in good_fits]

#weights_files = np.load('/Users/dan/Documents/notebooks/mossing-PC/simulation/weights_files_list.npy',allow_pickle=True)[()]

init_files = weights_files
ntries = len(init_files)

init_noise = 0#0.1
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

zero_extra_weights = [np.zeros((2,4),dtype='bool'),np.zeros((4,4),dtype='bool')]
#zero_extra_weights[1][2,1] = True # temporarily constraining VIP->SST weight to be 0
zero_extra_weights[1][1,2] = True # temporarily constraining SST->VIP weight to be 0

def run_fitting(init_file,target_name,seed=None):
    print('running %s -> %s'%(init_file,target_name))
    if not seed is None:
        np.random.seed(seed=seed)
    fcowo.fit_weights_and_save(target_name,ca_data_file=ca_data_file,opto_silencing_data_file=opto_silencing_data_file,opto_activation_data_file=opto_activation_data_file,allow_var=False,fit_s02=True,constrain_isn=True,tv=tv,l2_penalty=0.1,init_noise=init_noise,init_W_from_lsq=True,scale_init_by=1,init_W_from_file=True,init_file=init_file,correct_Eta=correct_Eta,init_Eta_with_s02=init_Eta_with_s02,init_Eta12_with_dYY=init_Eta12_with_dYY,use_opto_transforms=use_opto_transforms,share_residuals=share_residuals,stimwise=stimwise,simulate1=simulate1,simulate2=simulate2,verbose=verbose,free_amplitude=free_amplitude,norm_opto_transforms=norm_opto_transforms,zero_extra_weights=zero_extra_weights)

def run_fitting_one_arg(inp):
    init_file,target_name,seed = inp
    run_fitting(init_file,target_name,seed=seed)

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
        weights_files = glob.glob(weights_files_base+'/*.npy')
        weights_files.sort()

    losses = np.zeros((len(weights_files),))
    for ifile,weights_file in enumerate(weights_files):
        npyfile = np.load(weights_file,allow_pickle=True)[()]
        losses[ifile] = npyfile['loss']
    print(np.nanpercentile(losses,pctile_cutoff))

    weights_files = [wf for (wf,loss) in zip(weights_files,losses) if loss<np.nanpercentile(losses,pctile_cutoff)]
    print(weights_files)

    #if not weights_files_base is None:
    #    weights_files = np.load(weights_files_file,allow_pickle=True)[()]
    init_files = list(weights_files)
    ntries = len(init_files)

    #print(init_files)
        
    ut.mkdir(calnet_data_fold+'weights/'+weight_base)

   # weights_files = []
   # for irep in range(nreps):
   #     weights_files = weights_files + [calnet_data_fold+'weights/'+weight_base+'/%03d.npy'%itr for itr in range(ntries*irep,ntries*(irep+1))]

    #for irep in range(nreps):
    #    for itry in range(ntries):
    #        #run_fitting(init_files[itry],weights_files[irep*ntries+itry])
    #        print('starting job %d'%(irep*ntries+itry))
    #        p = mp.Process(target=run_fitting,args=(weights_files[irep*ntries+itry],))
    #        p.start()
    #        p.join()
    
    weights_files = []
    all_init_files = []
    seeds = []
    for irep in range(nreps):
        these_fit_nos = offset + np.arange(ntries*irep,ntries*(irep+1))
        weights_files = weights_files + [calnet_data_fold+'weights/'+weight_base+'/%04d.npy'%itr for itr in these_fit_nos]
        all_init_files = all_init_files + init_files
        seeds = seeds + list(these_fit_nos)

    inp = zip(all_init_files,weights_files,seeds)

    #for this_inp in inp:
    #    run_fitting_one_arg(this_inp)
    with mp.Pool(processes=nprocesses) as p:
        p.map(run_fitting_one_arg,inp)
