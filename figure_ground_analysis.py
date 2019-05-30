#!/usr/bin/env python

import os
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import h5py
from oasis.functions import deconvolve
from oasis import oasisAR1, oasisAR2
import pyute as ut

from importlib import reload
import scipy.ndimage.filters as sfi
import scipy.stats as sst
import scipy.ndimage.measurements as snm
from mpl_toolkits.mplot3d import Axes3D


def analyze_figure_ground(datafiles,stimfile,retfile=None,frame_adjust=None,rg=None,nbefore=4,nafter=4):
    nbydepth = get_nbydepth(datafiles)
    #trialwise,ctrialwise,strialwise,dfof,straces = ut.gen_precise_trialwise(datafiles,frame_adjust=frame_adjust)
    trialwise,ctrialwise,strialwise,dfof,straces,dtrialwise,trialwise_t_offset = ut.gen_precise_trialwise(datafiles,rg=rg,frame_adjust=frame_adjust,nbefore=nbefore,nafter=nafter)
    print(strialwise.shape)
    zstrialwise = sst.zscore(strialwise.reshape((strialwise.shape[0],-1)).T).T.reshape(strialwise.shape)
    
    result = sio.loadmat(stimfile,squeeze_me=True)['result'][()]
    
    infofile = sio.loadmat(datafiles[0][:-12]+'.mat',squeeze_me=True)
    frame = infofile['info'][()]['frame'][()]
    if frame_adjust:
        print('adjusted')
        frame = frame_adjust(frame)
    if np.remainder(frame.shape[0],2):
        print('deleted one')
        frame = frame[:-1]
    
    data = strialwise #[:,:,nbefore:-nafter]
    
    try:
        dxdt = sio.loadmat(datafiles[1],squeeze_me=True)['dxdt']
    except:
        with h5py.File(datafiles[1],mode='r') as f:
            dxdt = f['dxdt'][:].T
            
    trialrun = np.zeros(frame[0::2].shape)
    for i in range(len(trialrun)):
        trialrun[i] = dxdt[frame[0::2][i]:frame[1::2][i]].mean()
    runtrial = trialrun>100

    pval = np.zeros(strialwise.shape[0])
    for i in range(strialwise.shape[0]):
        _,pval[i] = sst.ttest_rel(strialwise[i,:,nbefore-1],strialwise[i,:,nbefore+1])
        
    stimparams = result['stimParams']

    order = ['ctrl','fig','grnd','iso','cross']
    norder = len(order)
    ori = stimparams[0]
    sz = stimparams[1]
    figContrast = stimparams[-2]
    grndContrast = stimparams[-1]

    paramdict = {}
    paramdict['ctrl'] = np.logical_and(figContrast==0,grndContrast==0)
    paramdict['fig'] = np.logical_and(figContrast==1,grndContrast==0)
    paramdict['grnd'] = np.logical_and(np.logical_and(figContrast==0,grndContrast==1),sz>0)
    paramdict['iso'] = sz==0
    paramdict['cross'] = np.logical_and(figContrast==1,grndContrast==1)
        
    indexlut,stimp = np.unique(stimparams,axis=1,return_inverse=True)

    angle = stimparams[0]
    size = stimparams[1]
    contrast = stimparams[4]

    ucontrast = np.unique(contrast)
    uangle = np.unique(angle)
    usize = np.unique(size)
    ncontrast = len(ucontrast)
    nangle = len(uangle)
    nsize = len(usize)

    angle180 = np.remainder(angle,180)
    uangle180 = np.unique(angle180)
    nangle180 = len(uangle180)

    Smean = np.zeros((strialwise.shape[0],norder,nangle180,strialwise.shape[2]))
    Stavg = np.zeros((strialwise.shape[0],norder,nangle180,int(strialwise.shape[1]/nangle/norder)))

    Strials = {}
    Sspont = {}
    #print(runtrial.shape)
    #for i,name in enumerate(order):
    #    for j,theta in enumerate(uangle180):
    #        lkat = np.logical_and(runtrial,np.logical_and(angle180==theta,paramdict[name]))
    #        if lkat.sum()==1:
    #            print('problem')
    #        Smean[:,i,j,:] = data[:,lkat,:].mean(1)
    #        Strials[(i,j)] = data[:,lkat,nbefore:-nafter].mean(2)
    #        Sspont[(i,j)] = data[:,lkat,:nbefore].mean(2)

    lb = np.zeros((strialwise.shape[0],norder,nangle180))
    ub = np.zeros((strialwise.shape[0],norder,nangle180))

    #for i in range(norder):
    #    print(i)
    #    for j in range(nangle180):
    #        lb[:,i,j],ub[:,i,j] = ut.bootstrap(Strials[(i,j)],np.mean,axis=1,pct=(16,84))
            # mn[:,i,j,k] = np.nanmean(Strials[(i,j,k)],axis=1)

    pval_fig = np.zeros((strialwise.shape[0],nangle180))
    #for j,theta in enumerate(uangle180):
    #    print(theta)
    #    figind = int(np.where(np.array([x=='fig' for x in order]))[0])
    #    _,pval_fig[:,j] = sst.ttest_rel(Sspont[(figind,j)],Strials[(figind,j)],axis=1)
    #    
    pval_grnd = np.zeros((strialwise.shape[0],nangle180))
    #for j,theta in enumerate(uangle180):
    #    print(theta)
    #    grndind = int(np.where(np.array([x=='grnd' for x in order]))[0])
    #    _,pval_grnd[:,j] = sst.ttest_rel(Sspont[(grndind,j)],Strials[(grndind,j)],axis=1)
                
    Savg = np.nanmean(np.nanmean(Smean[:,:,:,nbefore:-nafter],axis=-1),axis=2)

    Storiavg = Stavg.mean(1)
    # _,pval = sst.ttest_ind(Storiavg[:,0,-1].T,Storiavg[:,0,0].T)

    #suppressed = np.logical_and(pval<0.05,Savg[:,0,-1]<Savg[:,0,0])
    #facilitated = np.logical_and(pval<0.05,Savg[:,0,-1]>Savg[:,0,0])
    proc = {}
    proc['Smean'] = Smean
    proc['lb'] = lb
    proc['ub'] = ub
    proc['pval_fig'] = pval_fig
    proc['pval_grnd'] = pval_grnd
    proc['trialrun'] = trialrun
    proc['strialwise'] = strialwise
    proc['dtrialwise'] = strialwise
    proc['trialwise'] = strialwise
    proc['dfof'] = strialwise
    proc['trialwise_t_offset'] = strialwise
    proc['order'] = order
    proc['angle'] = angle
    proc['paramdict'] = paramdict
    proc['Sspont'] = Sspont

    #return Savg,Smean,lb,ub,pval_fig,pval_grnd,trialrun
    return Savg,proc

def get_nbydepth(datafiles):
    nbydepth = np.zeros((len(datafiles),))
    for i,datafile in enumerate(datafiles):
        with h5py.File(datafile,mode='r') as f:
            nbydepth[i] = (f['corrected'][:].T.shape[0])
    return nbydepth

def analyze_everything(folds,files,rets,adjust_fns):
    soriavg = {}
    strialavg = {}
    lb = {}
    ub = {}
    pval_fig = {}
    pval_grnd = {}
    nbydepth = {}
    ret_vars = {}
    trialrun = {}
    datafoldbase = '/home/mossing/scratch/2Pdata/'
    datafoldbase_old = '/home/mossing/excitation/2P/'
    stimfoldbase = '/home/mossing/scratch/visual_stim/'
    for thisfold,thisfile,retnumber,frame_adjust in zip(folds,files,rets,adjust_fns):
        if thisfold[:2]=='18':
             datafold = datafoldbase+thisfold+'ot/'
        else:
             datafold = datafoldbase_old+thisfold+'ot/'
        datafiles = [thisfile+'_ot_'+number+'.rois' for number in ['000','001','002','003']]

        stimfold = stimfoldbase+thisfold
        stimfile = thisfile+'.mat'

        datafiles = [datafold+file for file in datafiles]
        stimfile = stimfold+stimfile
        retfile = datafoldbase+thisfold+'retinotopy_{}.mat'.format(retnumber)

        nbefore = 4
        nafter = 4
        soriavg[thisfold],strialavg[thisfold],lb[thisfold],ub[thisfold],pval_fig[thisfold],pval_grnd[thisfold],trialrun[thisfold] = analyze_figure_ground(datafiles,stimfile,retfile,frame_adjust)
        try: 
            ret_vars[thisfold] = sio.loadmat(retfile,squeeze_me=True)#['ret'][:]
            print('loaded retinotopy file')
            ret_vars[thisfold]['position'] = sio.loadmat(stimfile,squeeze_me=True)['result'][()]['position']
        except:
            print('retinotopy not saved for '+thisfile)
            ret_vars[thisfold] = None
        nbydepth[thisfold] = get_nbydepth(datafiles)
    return soriavg,strialavg,lb,ub,pval_fig,pval_grnd,nbydepth,ret_vars,trialrun

def analyze_everything_by_criterion(folds=None,files=None,rets=None,adjust_fns=None,rgs=None,criteria=None,datafoldbase='/home/mossing/scratch/2Pdata/',stimfoldbase='/home/mossing/scratch/visual_stim/',criterion_cutoff=0.2):
    if isinstance(datafoldbase,str):
        datafoldbase = [datafoldbase]*len(folds)
    if isinstance(stimfoldbase,str):
        stimfoldbase = [stimfoldbase]*len(folds)
    soriavg = {}
    strialavg = {}
    lb = {}
    ub = {}
    pval_fig = {}
    pval_grnd = {}
    nbydepth = {}
    ret_vars = {}
    trialrun = {}
    proc = {}
    for thisfold,thisfile,retnumber,frame_adjust,thisdatafoldbase,thisstimfoldbase,rg in zip(folds,files,rets,adjust_fns,datafoldbase,stimfoldbase,rgs):

        #soriavg[thisfold] = [None]*2
        #strialavg[thisfold] = [None]*2
        #lb[thisfold] = [None]*2
        #ub[thisfold] = [None]*2
        #pval[thisfold] = [None]*2
        #nbydepth[thisfold] = [None]*2
        #spont[thisfold] = [None]*2
        #ret_vars[thisfold] = [None]*2
        #Smean_stat[thisfold] = [None]*2
        #proc[thisfold] = [None]*2

        datafiles = [thisfile+'_ot_'+number+'.rois' for number in ['000','001','002','003']]

        stimfold = thisstimfoldbase+thisfold
        stimfile = thisfile+'.mat'

        datafiles = [thisdatafoldbase+thisfold+'ot/'+file for file in datafiles]
        #datafiles = [x for x in datafiles if os.path.exists(x)]
        stimfile = stimfold+stimfile
        retfile = thisdatafoldbase+thisfold+'retinotopy_{}.mat'.format(retnumber)

        nbefore = 8
        nafter = 8
        soriavg[thisfold],proc[thisfold] = analyze_figure_ground(datafiles,stimfile,retfile,frame_adjust,rg=rg)
        try: 
            ret_vars[thisfold] = sio.loadmat(retfile,squeeze_me=True)#['ret'][:]
            print('loaded retinotopy file')
            ret_vars[thisfold]['position'] = sio.loadmat(stimfile,squeeze_me=True)['result'][()]['position']
        except:
            print('retinotopy not saved for '+thisfile)
            ret_vars[thisfold] = None
        nbydepth[thisfold] = get_nbydepth(datafiles)
        proc[thisfold]['ret_vars'] = ret_vars[thisfold]
        proc[thisfold]['nbydepth'] = nbydepth
    return soriavg,proc

def gen_full_data_struct(cell_type='PyrL23', keylist=None, frame_rate_dict=None, proc=None, nbefore=8, nafter=8):
    data_struct = {}
    ret_vars = {}
    for key in keylist:
        ret_vars[key] = proc[key]['ret_vars']
        dfof = proc[key]['dtrialwise']
        #calcium_responses_au = np.nanmean(proc[key]['trialwise'][:,:,nbefore:-nafter],-1)
        running_speed_cm_s = 4*np.pi/180*proc[key]['trialrun'] # 4 cm from disk ctr to estimated mouse location
        rf_ctr = np.concatenate((ret_vars[key]['paramdict_normal'][()]['xo'][np.newaxis,:],-ret_vars[key]['paramdict_normal'][()]['yo'][np.newaxis,:]),axis=0)
        stim_offset = ret_vars[key]['position'] - ret_vars[key]['paramdict_normal'][()]['ctr']
        rf_distance_deg = np.sqrt(((rf_ctr-stim_offset[:,np.newaxis])**2).sum(0))
        rf_displacement_deg = rf_ctr-stim_offset[:,np.newaxis]
        cell_id = np.arange(dfof.shape[0])
        stimulus_id = np.zeros((2,) + proc[key]['paramdict']['iso'].shape)
        order = proc[key]['order']
        for i,stimtype in enumerate(order):
            stimulus_id[0][proc[key]['paramdict'][stimtype]] = i
        stimulus_id[1] = proc[key]['angle']
        session_id = 'session_'+key[:-1].replace('/','_')
        mouse_id = key.split('/')[1]
        
        data_struct[session_id] = {}
        data_struct[session_id]['mouse_id'] = mouse_id
        data_struct[session_id]['stimulus_id'] = stimulus_id
        data_struct[session_id]['order'] = order
        data_struct[session_id]['cell_id'] = cell_id
        data_struct[session_id]['cell_type'] = cell_type
        data_struct[session_id]['mouse_id'] = mouse_id
        data_struct[session_id]['F'] = dfof
        data_struct[session_id]['nbefore'] = nbefore
        data_struct[session_id]['nafter'] = nafter
        data_struct[session_id]['rf_mapping_pval'] = ret_vars[key]['pval_ret']
        data_struct[session_id]['rf_distance_deg'] = rf_distance_deg
        data_struct[session_id]['rf_displacement_deg'] = rf_displacement_deg
        data_struct[session_id]['rf_ctr'] = rf_ctr
        data_struct[session_id]['running_speed_cm_s'] = running_speed_cm_s
    return data_struct

def add_data_struct_h5(filename, cell_type='PyrL23', keylist=None, frame_rate_dict=None, proc=None, ret_vars=None, nbefore=8, nafter=8):
    #with h5py.File(filename,mode='w+') as data_struct:
    with hdf5edit(filename) as data_struct:
        for key in keylist:
            if len(proc[key][0])>0:
                gdind = 0
            else:
                gdind = 1
            dfof = proc[key][gdind]['dtrialwise']
            decon = proc[key][gdind]['strialwise'] 
            running_speed_cm_s = 4*np.pi/180*proc[key][gdind]['trialrun'] # 4 cm from disk ctr to estimated mouse location
            rf_ctr = np.concatenate((ret_vars[key]['paramdict_normal'][()]['xo'][np.newaxis,:],-ret_vars[key]['paramdict_normal'][()]['yo'][np.newaxis,:]),axis=0)
            stim_offset = ret_vars[key]['position'] - ret_vars[key]['paramdict_normal'][()]['ctr']
            rf_distance_deg = np.sqrt(((rf_ctr-stim_offset[:,np.newaxis])**2).sum(0))
            rf_displacement_deg = rf_ctr-stim_offset[:,np.newaxis]
            cell_id = np.arange(dfof.shape[0])
            uangle,iangle = np.unique(proc[key]['angle'],return_inverse=True)
            stimulus_id = np.zeros((2,) + proc[key]['paramdict']['iso'].shape)
            order = proc[key]['order']
            for i,stimtype in enumerate(order):
                stimulus_id[0][proc[key]['paramdict'][stimtype]] = i
            stimulus_id[1] = proc[key]['angle']
            stimulus_direction = uangle
            session_id = 'session_'+key[:-1].replace('/','_')
            mouse_id = key.split('/')[1]
            
            if not session_id in data_struct.keys():
                this_session = data_struct.create_group(session_id)
                this_session['mouse_id'] = mouse_id
                this_session['cell_type'] = cell_type
                this_session.create_dataset('cell_id',data=cell_id)
            else:
                this_session = data_struct[session_id]

            exptno = 0
            while 'figure_ground_'+str(exptno) in this_session.keys():
                exptno = exptno+1
            this_expt = this_session.create_group('figure_ground_'+str(exptno))
            this_expt.create_dataset('stimulus_id',data=stimulus_id)
            this_expt['order'] = order
            this_expt.create_dataset('stimulus_direction',data=stimulus_direction)
            this_expt['stim_offset_deg'] = stim_offset
            this_expt.create_dataset('running_speed_cm_s',data=running_speed_cm_s)
            this_expt.create_dataset('F',data=dfof)
            this_expt.create_dataset('raw_trialwise',data=proc[key][gdind]['raw_trialwise'])
            this_expt.create_dataset('neuropil_trialwise',data=proc[key][gdind]['neuropil_trialwise'])
            this_expt.create_dataset('decon',data=decon)
            this_expt['nbefore'] = nbefore
            this_expt['nafter'] = nafter
