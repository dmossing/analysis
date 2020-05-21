#!/usr/bin/env python

import os
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import h5py
from oasis.functions import deconvolve
from oasis import oasisAR1, oasisAR2
import pyute as ut
import analysis_template as at
from importlib import reload
reload(ut)
reload(at)
import scipy.ndimage.filters as sfi
import scipy.stats as sst
import scipy.ndimage.measurements as snm
from mpl_toolkits.mplot3d import Axes3D
import scipy.optimize as sop
import pdb

blcutoff = 20
ds = 10
blspan = 3000
nbefore = 4
nafter = 4

def add_data_struct_h5(filename, cell_type='PyrL23', keylist=None, frame_rate_dict=None, proc=None, nbefore=8, nafter=8):
    #with h5py.File(filename,mode='w+') as data_struct:
    with ut.hdf5edit(filename) as data_struct:
    #data_struct = {}
        ret_vars = {}
        for key in keylist:
            if len(proc[key])>0:
                gdind = 0
            else:
                gdind = 1
            dfof = proc[key]['dtrialwise'][:]
            decon = proc[key]['strialwise'][:] 
            t_offset = proc[key]['trialwise_t_offset'][:] 
            ret_vars[key] = proc['/'.join([key,'ret_vars'])]
            #calcium_responses_au = np.nanmean(proc[key]['trialwise'][:,:,nbefore:-nafter],-1)
            running_speed_cm_s = 4*np.pi/180*proc[key]['trialrun'][:] # 4 cm from disk ctr to estimated mouse location
            paramdict = proc['/'.join([key,'ret_vars','paramdict_normal'])]
            #print(paramdict['xo'].shape)
            rf_ctr = np.concatenate((paramdict['xo'][:][np.newaxis,:],-paramdict['yo'][:][np.newaxis,:]),axis=0)
            stim_offset = ret_vars[key]['position'][:] - paramdict['ctr'][:]
            rf_distance_deg = np.sqrt(((rf_ctr-stim_offset[:,np.newaxis])**2).sum(0))
            rf_displacement_deg = rf_ctr-stim_offset[:,np.newaxis]
            cell_id = np.arange(dfof.shape[0])
            unubs,inubs = np.unique(proc[key]['nubs'][:],return_inverse=True)
            usize,isize = np.unique(proc[key]['size'][:],return_inverse=True)
            uangle,iangle = np.unique(proc[key]['angle'][:],return_inverse=True)
            stimulus_id = np.concatenate((inubs,isize[np.newaxis],iangle[np.newaxis]),axis=0)
            stimulus_nubs = unubs
            stimulus_size_deg = usize
            stimulus_direction = uangle
            session_id = key
            mouse_id = key.split('_')[1]
            
            if not session_id in data_struct.keys():
                this_session = data_struct.create_group(session_id)
                this_session['mouse_id'] = mouse_id
                this_session['cell_type'] = cell_type
                this_session.create_dataset('cell_id',data=cell_id)
            else:
                this_session = data_struct[session_id]

            exptno = 0
            while 'nub_'+str(exptno) in this_session.keys():
                exptno = exptno+1
            this_expt = this_session.create_group('nub_'+str(exptno))
            this_expt.create_dataset('stimulus_id',data=stimulus_id)
            this_expt.create_dataset('stimulus_nubs_active',data=stimulus_size_deg)
            this_expt.create_dataset('stimulus_size_deg',data=stimulus_size_deg)
            this_expt.create_dataset('stimulus_direction',data=stimulus_direction)
            this_expt['rf_mapping_pval'] = ret_vars[key]['pval_ret'][:]
            this_expt['rf_distance_deg'] = rf_distance_deg
            this_expt['rf_displacement_deg'] = rf_displacement_deg
            this_expt['rf_ctr'] = rf_ctr
            this_expt['stim_offset_deg'] = stim_offset
            this_expt.create_dataset('running_speed_cm_s',data=running_speed_cm_s)
            this_expt.create_dataset('F',data=dfof)
            this_expt.create_dataset('raw_trialwise',data=proc[key]['raw_trialwise'][:])
            this_expt.create_dataset('neuropil_trialwise',data=proc[key]['neuropil_trialwise'][:])
            this_expt.create_dataset('decon',data=decon)
            this_expt.create_dataset('t_offset',data=t_offset)
            this_expt['nbefore'] = nbefore
            this_expt['nafter'] = nafter

def analyze_simply(folds=None,files=None,rets=None,adjust_fns=None,rgs=None,datafoldbase='/home/mossing/scratch/2Pdata/',stimfoldbase='/home/mossing/scratch/visual_stim/',procname='nub_proc.hdf5'):
    if isinstance(datafoldbase,str):
        datafoldbase = [datafoldbase]*len(folds)
    if isinstance(stimfoldbase,str):
        stimfoldbase = [stimfoldbase]*len(folds)
    if os.path.exists(procname):
        os.remove(procname)
    #stim_params = nub_params()
    stim_params = nub_params_binary()
    session_ids = []
    for thisfold,thisfile,frame_adjust,rg,thisdatafoldbase,thisstimfoldbase,retnumber in zip(folds,files,adjust_fns,rgs,datafoldbase,stimfoldbase,rets):

        session_id = at.gen_session_id(thisfold)
        datafiles = at.gen_datafiles(thisdatafoldbase,thisfold,thisfile,nplanes=4)
        stimfile = at.gen_stimfile(thisstimfoldbase,thisfold,thisfile)
        retfile = thisdatafoldbase+thisfold+'retinotopy_'+retnumber+'.mat'

        nbefore = 8
        nafter = 8

        proc = at.analyze(datafiles,stimfile,frame_adjust=frame_adjust,rg=rg,nbefore=nbefore,nafter=nafter,stim_params=stim_params)
        
        proc['position'] = sio.loadmat(stimfile,squeeze_me=True)['result'][()]['position']

        ut.dict_to_hdf5(procname,session_id,proc)
        session_ids.append(session_id)
    return session_ids

def nub_params():
    params_and_fns = [None for i in range(7)]
    for i in range(5):
        params_and_fns[i] = ('nub' + str(i),lambda result: result['gratingInfo'][()]['Nubs'][()][i])
    params_and_fns[5] = ('size',lambda result: result['gratingInfo'][()]['Size'][()])
    params_and_fns[6] = ('angle',lambda result: result['gratingInfo'][()]['Orientation'][()])
    return params_and_fns

def nub_params_binary():
    params_and_fns = [None for i in range(3)]
    def gen_binary_nub(arr):
        mult_by = np.array([2**n for n in range(arr.shape[0])])[::-1,np.newaxis]
        return (arr*mult_by).sum(0)
    params_and_fns[0] = ('nubs',lambda result: gen_binary_nub(result['gratingInfo'][()]['Nubs'][()]))
    params_and_fns[1] = ('size',lambda result: result['gratingInfo'][()]['Size'][()])
    params_and_fns[2] = ('angle',lambda result: result['gratingInfo'][()]['Orientation'][()])
    return params_and_fns

def add_data_struct_h5_simply(filename, cell_type='PyrL23', keylist=None, frame_rate_dict=None, proc=None, nbefore=8, nafter=8):
    groupname = 'nub'
    #featurenames= ['nub' + str(i) for i in range(5)] + ['size','angle']
    #datasetnames = ['stimulus_nub_' + str(i) + '_on' for i in range(5)] + ['stimulus_size','stimulus_direction_deg']
    featurenames= ['nubs','size','angle']
    datasetnames = ['stimulus_nubs_active','stimulus_size','stimulus_direction_deg']
    grouplist = at.add_data_struct_h5(filename,cell_type=cell_type,keylist=keylist,frame_rate_dict=frame_rate_dict,proc=proc,nbefore=nbefore,nafter=nafter,featurenames=featurenames,datasetnames=datasetnames,groupname=groupname)

    at.add_ret_to_data_struct(filename,keylist=keylist,proc=proc,grouplist=grouplist)
    return grouplist

def add_evan_data_struct_h5_simply(filename, cell_type='PyrL23', keylist=None, frame_rate_dict=None, proc=None, nbefore=8, nafter=8):
    groupname = 'nub'
    featurenames= ['nubs']
    datasetnames = ['stimulus_nubs_active']
    grouplist = at.add_evan_data_struct_h5(filename,cell_type=cell_type,keylist=keylist,frame_rate_dict=frame_rate_dict,proc=proc,nbefore=nbefore,nafter=nafter,featurenames=featurenames,datasetnames=datasetnames,groupname=groupname)

    return grouplist

