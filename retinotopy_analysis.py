#!/usr/bin/env python

import os
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import h5py
from oasis.functions import deconvolve
from oasis import oasisAR1, oasisAR2
import pyute as ut
import scipy.stats as sst
import scipy.ndimage.filters as sfi

blcutoff = 1
ds = 10
blspan = 3000
nbefore = 0
nafter = 0

def analyze_precise_retinotopy(datafiles,stimfile,retfile,criterion=lambda x: x>100,rg=(2,-10),nbefore=nbefore,nafter=nafter,gridsize=10):
    nbydepth = np.zeros((len(datafiles),))
    for i,datafile in enumerate(datafiles):
        corrected = ut.loadmat(datafile,'corrected')
        nbydepth[i] = corrected.shape[0]
#         with h5py.File(datafile,mode='r') as f:
#             nbydepth[i] = (f['corrected'][:].T.shape[0])
    trialwise,ctrialwise,strialwise,dfof,straces,dtrialwise,trialwise_t_offset = ut.gen_precise_trialwise(datafiles,rg=rg,nbefore=nbefore,nafter=nafter)
    zstrialwise = sst.zscore(strialwise.reshape((strialwise.shape[0],-1)).T).T.reshape(strialwise.shape)

    result = sio.loadmat(stimfile,squeeze_me=True)['result'][()]

    infofile = sio.loadmat(datafiles[0][:-12]+'.mat',squeeze_me=True)
    #retfile = sio.loadmat(retfile,squeeze_me=True)

    locinds = result['locinds'] #retfile['locinds']

    has_inverse = False
    try:
#        inverted = result['inverted'][()]
        inverted = np.tile(result['inverted'],(result['repetitions'],))
        has_inverse = True
    except:
        has_inverse = False

    frame = infofile['info'][()]['frame'][()]
    frame = np.unique(frame[rg[0]:frame.size+rg[1]]) # this format for all the retinotopic mapping through 12/12

    data = strialwise[:,:,nbefore:strialwise.shape[-1]-nafter]

    Ny = locinds[:,0].max()
    Nx = locinds[:,1].max()
    
    try:
        try:
            dxdt = sio.loadmat(datafiles[1],squeeze_me=True)['dxdt']
        except:
            with h5py.File(datafiles[1],mode='r') as f:
                dxdt = f['dxdt'][:].T
    except:
        print('no running data saved; assuming all running')
        dxdt = 101*np.ones((frame.max(),))

    trialrun = np.zeros(frame[0::2].shape)
    for i in range(len(trialrun)):
        trialrun[i] = dxdt[frame[0::2][i]:frame[1::2][i]].mean()
    runtrial = criterion(np.abs(trialrun)) #>0

    if has_inverse:
        ret = np.zeros((data.shape[0],Ny,Nx,2))
        for j in range(Ny):
            for k in range(Nx):
                lkat = np.logical_and(np.logical_and(np.logical_and(locinds[:,0]==j+1,locinds[:,1]==k+1),runtrial),np.nanmax(np.nanmax(data,0),-1))
                lkat_reg = np.logical_and(lkat,np.logical_not(inverted))
                lkat_inv = np.logical_and(lkat,inverted)
                n_reg = lkat_reg.sum()
                n_inv = lkat_inv.sum()
                print((n_reg,n_inv))
                for idx in np.where(lkat_reg)[0]:
                    ret[:,j,k,0] = ret[:,j,k,0] + data[:,idx].mean(1)/n_reg
                for idx in np.where(lkat_inv)[0]:
                    ret[:,j,k,1] = ret[:,j,k,1] + data[:,idx].mean(1)/n_inv
                assert(~np.isnan(np.nanmax(ret[:,j,k])))
    else:
        ret = np.zeros((data.shape[0],Ny,Nx))
        for j in range(Ny):
            for k in range(Nx):
                lkat_reg = np.logical_and(np.logical_and(locinds[:,0]==j+1,locinds[:,1]==k+1),runtrial)
                n_reg = lkat_reg.sum()
                print((n_reg,))#n_inv))
                for idx in np.where(lkat_reg)[0]:
                    ret[:,j,k] = ret[:,j,k] + data[:,idx].mean(1)/n_reg
                assert(~np.isnan(np.nanmax(ret[:,j,k])))

    if 'range' in result.dtype.names:
        gridsize = 5
        ctr = np.array((result['range'][0:2].mean(),-result['range'][2:].mean())) # ctr: x center of range, y center of range # fixed 18/10/30; for expts. after 18/10/30, this will have to be switched!
    else: 
        gridsize = 10
        ctr = np.array((0,0))
    
    # flipping for expts. after 18/10/30
    toflip = int(datafiles[0].split('/')[-4])>181030
    if toflip:
        ctr = ctr*np.array((1,-1))

    xrg = np.arange(-(Nx-1)*gridsize/2,(Nx+1)*gridsize/2,gridsize)
    yrg = np.arange(-(Ny-1)*gridsize/2,(Ny+1)*gridsize/2,gridsize)

    # inverting for expts. before 18/12/09
    notquitefixed = int(datafiles[0].split('/')[-4])<181209
    if toflip and notquitefixed:
        yrg = -yrg

    if has_inverse:
        paramdict = [ut.fit_2d_gaussian((xrg,yrg),ret[:,:,:,0]),ut.fit_2d_gaussian((xrg,yrg),ret[:,:,:,1])]
        paramdict[0]['gridsize'] = gridsize
        paramdict[1]['gridsize'] = gridsize
        paramdict[0]['ctr'] = ctr
        paramdict[1]['ctr'] = ctr
    else:
        paramdict = ut.fit_2d_gaussian((xrg,yrg),ret) #,ut.fit_2d_gaussian((xrg,yrg),ret[:,:,:,1])]
        paramdict['gridsize'] = gridsize
        paramdict['ctr'] = ctr

    pval_ret = np.zeros(strialwise.shape[0])
    for i in range(strialwise.shape[0]):
        _,pval_ret[i] = sst.ttest_rel(strialwise[i,:,nbefore-1],strialwise[i,:,nbefore+1])
    
    return ret,paramdict,pval_ret,trialrun,has_inverse,locinds


def analyze_retinotopy(datafiles,stimfile,retfile,criterion=lambda x: x>100,rg=(2,-10),nbefore=nbefore,nafter=nafter):
    nbydepth = np.zeros((len(datafiles),))
    for i,datafile in enumerate(datafiles):
        with h5py.File(datafile,mode='r') as f:
            nbydepth[i] = (f['corrected'][:].T.shape[0])
    trialwise,ctrialwise,strialwise,dfof = ut.gen_trialwise(datafiles,rg=rg,nbefore=nbefore,nafter=nafter)
    zstrialwise = sst.zscore(strialwise.reshape((strialwise.shape[0],-1)).T).T.reshape(strialwise.shape)

    result = sio.loadmat(stimfile,squeeze_me=True)['result'][()]

    infofile = sio.loadmat(datafiles[0][:-5]+'.mat',squeeze_me=True)
    retfile = sio.loadmat(retfile,squeeze_me=True)

    locinds = result['locinds'] #retfile['locinds']

    frame = infofile['info'][()]['frame'][()]
    frame = np.unique(frame[rg[0]:frame.size+rg[1]]) # this format for all the retinotopic mapping through 12/12

    data = strialwise[:,:,nbefore:strialwise.shape[-1]-nafter]

    Ny = locinds[:,0].max()
    Nx = locinds[:,1].max()
    
    try:
        try:
            dxdt = sio.loadmat(datafiles[1],squeeze_me=True)['dxdt']
        except:
            with h5py.File(datafiles[1],mode='r') as f:
                dxdt = f['dxdt'][:].T
    except:
        print('no running data saved; assuming all running')
        dxdt = 101*np.ones((frame.max(),))

    trialrun = np.zeros(frame[0::2].shape)
    for i in range(len(trialrun)):
        trialrun[i] = dxdt[frame[0::2][i]:frame[1::2][i]].mean()
    runtrial = criterion(np.abs(trialrun)) #>0

    ret = np.zeros((data.shape[0],Ny,Nx)) #,2))
    for j in range(Ny):
        for k in range(Nx):
            lkat_reg = np.logical_and(np.logical_and(locinds[:,0]==j+1,locinds[:,1]==k+1),runtrial)
            # lkat_reg = np.logical_and(lkat,np.logical_not(inverted))
            # lkat_inv = np.logical_and(lkat,inverted)
            n_reg = lkat_reg.sum()
            # n_inv = lkat_inv.sum()
            print((n_reg,))#n_inv))
            for idx in np.where(lkat_reg)[0]:
                ret[:,j,k] = ret[:,j,k] + data[:,idx].mean(1)/n_reg
            assert(~np.isnan(np.nanmax(ret[:,j,k])))
            # for idx in np.where(lkat_inv)[0]:
                # ret[:,j,k,1] = ret[:,j,k,1] + data[:,idx].mean(1)/n_inv

    xrg = np.arange(-(Nx-1)*gridsize/2,(Nx+1)*gridsize/2,gridsize)
    yrg = np.arange(-(Ny-1)*gridsize/2,(Ny+1)*gridsize/2,gridsize)

    paramdict = ut.fit_2d_gaussian((xrg,yrg),ret) #,ut.fit_2d_gaussian((xrg,yrg),ret[:,:,:,1])]

    pval = np.zeros(strialwise.shape[0])
    for i in range(strialwise.shape[0]):
        _,pval[i] = sst.ttest_rel(strialwise[i,:,nbefore-1],strialwise[i,:,nbefore+1])
    
    return ret,paramdict,pval,trialrun

def get_nbydepth(datafiles):
    nbydepth = np.zeros((len(datafiles),))
    for i,datafile in enumerate(datafiles):
        with h5py.File(datafile,mode='r') as f:
            nbydepth[i] = (f['corrected'][:].T.shape[0])
    return nbydepth

def gen_traces(datafiles,blcutoff=blcutoff,blspan=blspan): #nbefore=nbefore,nafter=nafter
    trialwise = np.array(())
    ctrialwise = np.array(())
    strialwise = np.array(())
    dfofall = np.array(())
    baselineall = np.array(())
    for datafile in datafiles:
        frm = sio.loadmat(datafile.replace('.rois','.mat'),squeeze_me=True)['info']['frame'][()][2:-10]
        with h5py.File(datafile,mode='r') as f:
            to_add = f['corrected'][:].T
            to_add[np.isnan(to_add)] = np.nanmin(to_add)
#             baseline = np.percentile(to_add,blcutoff,axis=1)
            baseline = sfi.percentile_filter(to_add[:,::ds],blcutoff,(1,int(blspan/ds)))
            baseline = np.repeat(baseline,ds,axis=1)
            for i in range(baseline.shape[0]):
                baseline[i] = sfi.gaussian_filter1d(baseline[i],blspan/2)
#             if baseline.shape[1]<to_add.shape[1]:
#                 baseline = np.hstack((baseline,np.repeat(baseline[:,-1],to_add.shape[1]-baseline.shape[1])))
            if baseline.shape[1]>to_add.shape[1]:
                baseline = baseline[:,:to_add.shape[1]]
            c = np.zeros_like(to_add)
            s = np.zeros_like(to_add)
            dfof = np.zeros_like(to_add)
            for i in range(c.shape[0]):
#                 dfof = (to_add[i]-baseline[i,np.newaxis])/baseline[i,np.newaxis]
                dfof[i] = (to_add[i]-baseline[i,:])/baseline[i,:]
                dfof[i][np.isnan(dfof[i])] = 0
                #try:
                c[i],s[i],_,_,_  = deconvolve(dfof[i].astype(np.float64),penalty=1,sn=5e-3)
#                except:
#                    throwaway = 0
#                    print("in "+datafile+" couldn't do "+str(i))
            try:
                trialwise = np.concatenate((trialwise,to_add),axis=0)
                ctrialwise = np.concatenate((ctrialwise,c),axis=0)
                strialwise = np.concatenate((strialwise,s),axis=0)
                dfofall = np.concatenate((dfofall,dfof),axis=0)
                baselineall = np.concatenate((baselineall,baseline),axis=0)
            except:
                trialwise = to_add.copy()
                ctrialwise = c.copy()
                strialwise = s.copy()
                dfofall = dfof.copy()
                baselineall = baseline.copy()
    return trialwise,ctrialwise,strialwise,dfofall,baselineall

def ontarget_by_retinotopy(ret_vars,ctr=None,rg=5,pcutoff=1e-2):
    # ctr is the location of the center in the expt of interest, relative to the retinotopy center, in degrees.
    # as given in the MATLAB output, positive y-values are upward on the monitor
    if ctr is None:
        ctr = np.array((0,0))
#         ctr = (ret[0].shape-np.array((1,1)))/2
#     com = np.zeros((ret.shape[0],2))
#     for i in range(com.shape[0]):
#         com[i] = snm.center_of_mass(ret[i]-ret[i].min())
#     return ((com-ctr[np.newaxis,:])**2).sum(1)<rg**2,com
    try: 
        xo = ret_vars['paramdict_normal']['xo'][()]
        yo = ret_vars['paramdict_normal']['yo'][()]
        ctr_ret = ret_vars['paramdict_normal']['ctr'][()]
    except:
        xo = ret_vars['paramdict']['xo'][()]
        yo = ret_vars['paramdict']['yo'][()]
        ctr_ret = ret_vars['paramdict']['ctr'][()]
    try:
        pval_ret = ret_vars['pval_ret']
    except:
        pval_ret = ret_vars['pval']
    return np.logical_and((xo+ctr_ret[0]-ctr[0])**2+(-yo+ctr_ret[1]-ctr[1])**2<rg**2,pval_ret<pcutoff),np.hstack((xo,yo)) # ctr_ret as calculated assumes upside down retinotopy center; true through 18/10/30. Grid used in fitting gaussians goes from negative to positive degrees with increasing y index, hence -yo.

def do_process(thisfold,thisfile,rg=(2,-10),nbefore=8,nafter=8,criterion=lambda x:x>100,datafoldbase='/home/mossing/scratch/2Pdata/',stimfoldbase='/home/mossing/scratch/visual_stim/'):

    #datafoldbase = '/home/mossing/scratch/2Pdata/'
    datafold = datafoldbase+thisfold+'ot/'
    datafiles = [thisfile+'_ot_'+number+'.rois' for number in ['000','001','002','003']]

    #stimfoldbase = '/home/mossing/scratch/visual_stim/'
    stimfold = stimfoldbase+thisfold
    stimfile = thisfile+'.mat'

    datafiles = [datafold+file for file in datafiles]
    datafiles = [x for x in datafiles if os.path.exists(x)]
    stimfile = stimfold+stimfile
    retfile = datafoldbase+thisfold+'retinotopy_'+thisfile[-3:]+'.mat'

#     nbefore = 4
#     nafter = 4

    ret,paramdict,pval,trialrun,has_inverse,locinds = analyze_precise_retinotopy(datafiles,stimfile,retfile,criterion=criterion,rg=rg,nbefore=nbefore,nafter=nafter)
    nbydepth = get_nbydepth(datafiles)
    trialwise,ctrialwise,strialwise,dfof,_,dtrialwise,trialwise_t_offset = ut.gen_precise_trialwise(datafiles,rg=rg,nbefore=nbefore,nafter=nafter)
#     traces,ctraces,straces,dfof,baseline = rt.gen_traces(datafiles)
    spont = strialwise[:,trialrun>100,:nbefore].mean(-1).mean(-1)
    proc = {}
    proc['trialwise'] = trialwise
    proc['dtrialwise'] = dtrialwise
    proc['strialwise'] = strialwise
    proc['trialwise_t_offset'] = trialwise_t_offset
    proc['spont'] = spont
    proc['locinds'] = locinds
    proc['dfof'] = dfof
    

    try:
        retfile_load = sio.loadmat(retfile)
    except:
        print('retinotopy file not accessible')
        retfile_load = {}

    if has_inverse:
        retfile_load['paramdict_normal'] = paramdict[0]
        retfile_load['paramdict_inv'] = paramdict[1]
    else:
        retfile_load['paramdict_normal'] = paramdict
    retfile_load['pval_ret'] = pval
    retfile_load['has_inverse'] = has_inverse
    retfile_load['ret'] = ret
    sio.savemat(retfile,retfile_load)
    print('saving here '+retfile)
    return ret,paramdict,pval,trialrun,has_inverse,nbydepth,proc

def analyze_everything(folds=None,files=None,rgs=None,criteria=None,nbefore=8,nafter=8,datafoldbase='/media/mossing/backup_1/data/2P/',stimfoldbase='/home/mossing/modulation/visual_stim/'):
    if isinstance(datafoldbase,str):
        datafoldbase = [datafoldbase]*len(folds)
    if isinstance(stimfoldbase,str):
        stimfoldbase = [stimfoldbase]*len(folds)
    ret = {}
    paramdict = {}
    pval = {}
    trialrun = {}
    has_inverse = {}
    nbydepth = {}
    proc = {}
    for thisfold,thisfile,rg,criterion,thisdatafoldbase,thisstimfoldbase in zip(folds,files,rgs,criteria,datafoldbase,stimfoldbase):
        ret[thisfold],paramdict[thisfold],pval[thisfold],trialrun[thisfold],has_inverse[thisfold],nbydepth[thisfold],proc[thisfold] = do_process(thisfold,thisfile,rg=rg,nbefore=nbefore,nafter=nafter,criterion=criterion,datafoldbase=thisdatafoldbase,stimfoldbase=thisstimfoldbase)
    return ret,paramdict,pval,trialrun,has_inverse,nbydepth,proc

def plot_peak_aligned(retx,xo,gridsize=10,alpha=1e-2,c='b'):
    Nx = retx.shape[1]
    xrg = np.arange(-(Nx-1)*gridsize/2,(Nx+1)*gridsize/2,gridsize)
    for i in range(retx.shape[0]):
        plt.plot(xrg-xo[i],retx[i]/retx[i].max(),alpha=alpha,c=c)
    plt.xlim(xrg.min()-gridsize,xrg.max()+gridsize)

def gen_full_data_struct(cell_type='PyrL23', keylist=None, frame_rate_dict=None, proc=None, nbefore=8, nafter=8,ops=None,roi_properties=None):
    data_struct = {}
    for key in keylist:
        dfof = proc[key]['dtrialwise']
        #calcium_responses_au = np.nanmean(proc[key]['trialwise'][:,:,nbefore:-nafter],-1)
        running_speed_cm_s = 4*np.pi/180*proc[key]['trialrun'] # 4 cm from disk ctr to estimated mouse location
        rf_ctr = np.concatenate((proc[key]['paramdict']['xo'][np.newaxis,:],-proc[key]['paramdict']['yo'][np.newaxis,:]),axis=0)
        cell_id = np.arange(dfof.shape[0])
        stimulus_id = proc[key]['locinds'].T - 1
        session_id = 'session_'+key[:-1].replace('/','_')
        mouse_id = key.split('/')[1]
        
        data_struct[session_id] = {}
        data_struct[session_id]['mouse_id'] = mouse_id
        data_struct[session_id]['stimulus_id'] = stimulus_id
        data_struct[session_id]['cell_id'] = cell_id
        data_struct[session_id]['cell_type'] = cell_type
        data_struct[session_id]['F'] = dfof
        data_struct[session_id]['nbefore'] = nbefore
        data_struct[session_id]['nafter'] = nafter
        data_struct[session_id]['rf_mapping_pval'] = proc[key]['pval']
        data_struct[session_id]['decon'] = proc[key]['strialwise']
        data_struct[session_id]['running_speed_cm_s'] = running_speed_cm_s
    #    if not ops is None:
    #        data_struct[session_id]['ops'] = ops[key]
    #    if not roi_properties is None:
    #        data_struct[session_id]['roi_properties'] = roi_properties[key]
    return data_struct

def add_data_struct_h5(filename, cell_type='PyrL23', keylist=None, frame_rate_dict=None, proc=None, ret_vars=None, nbefore=8, nafter=8):
    #with h5py.File(filename,mode='w+') as data_struct:
    #with h5py.File(filename,mode='r') as data_struct:
    with ut.hdf5edit(filename) as data_struct:
    #data_struct = {}
        for key in keylist:
            if len(proc[key][0])>0:
                gdind = 0
            else:
                gdind = 1
            dfof = proc[key]['dtrialwise']
            decon = proc[key]['strialwise']
            running_speed_cm_s = 4*np.pi/180*proc[key]['trialrun'] # 4 cm from disk ctr to estimated mouse location
            rf_ctr = np.concatenate((proc[key]['paramdict']['xo'][np.newaxis,:],-proc[key]['paramdict']['yo'][np.newaxis,:]),axis=0)
            cell_id = np.arange(dfof.shape[0])
            stimulus_id = proc[key]['locinds'].T - 1
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
            while 'retinotopy_'+str(exptno) in this_session.keys():
                exptno = exptno+1
            this_expt = this_session.create_group('retinotopy_'+str(exptno))
            this_expt.create_dataset('stimulus_id',data=stimulus_id)
            this_expt['rf_mapping_pval'] = proc[key]['pval']
            this_expt['rf_ctr'] = rf_ctr
            this_expt.create_dataset('running_speed_cm_s',data=running_speed_cm_s)
            this_expt.create_dataset('F',data=dfof)
            this_expt.create_dataset('raw_trialwise',data=proc[key][gdind]['raw_trialwise'])
            this_expt.create_dataset('neuropil_trialwise',data=proc[key][gdind]['neuropil_trialwise'])
            this_expt.create_dataset('decon',data=decon)
            this_expt['nbefore'] = nbefore
            this_expt['nafter'] = nafter
