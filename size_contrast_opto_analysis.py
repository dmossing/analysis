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
reload(ut)
import scipy.ndimage.filters as sfi
import scipy.stats as sst
import scipy.ndimage.measurements as snm
from mpl_toolkits.mplot3d import Axes3D
import scipy.optimize as sop
import pdb
import analysis_template as at

blcutoff = 1
ds = 10
blspan = 3000
nbefore = 4
nafter = 4

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
        frm = sio.loadmat(datafile.replace('.rois','.mat'),squeeze_me=True)['info']['frame'][()][1:]
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
                dfof[i] = (to_add[i]-baseline[i,:])/(baseline[i,:])
                dfof[i][np.isnan(dfof[i])] = 0
                #try:
                c[i],s[i],_,_,_  = deconvolve(dfof[i].astype(np.float64),penalty=1,sn=5e-3)
                #except:
                #    print("in "+datafile+" couldn't do "+str(i))
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

def analyze_size_contrast(datafiles,stimfile,retfile=None,frame_adjust=None,rg=(1,0),nbefore=nbefore,nafter=nafter,criterion=None,criterion_cutoff=None):
    if criterion is None:
        criterion = lambda x: np.abs(x)>100
    nbydepth = get_nbydepth(datafiles)
#     trialwise,ctrialwise,strialwise = gen_trialwise(datafiles,frame_adjust=frame_adjust)
    trialwise,ctrialwise,strialwise,dfof,straces,dtrialwise,trialwise_t_offset = ut.gen_precise_trialwise(datafiles,rg=rg,frame_adjust=frame_adjust,nbefore=nbefore,nafter=nafter)
    zstrialwise = sst.zscore(strialwise.reshape((strialwise.shape[0],-1)).T).T.reshape(strialwise.shape)
    
    result = sio.loadmat(stimfile,squeeze_me=True)['result'][()]
    
    infofile = sio.loadmat(datafiles[0][:-12]+'.mat',squeeze_me=True) # original .mat file
    frame = infofile['info'][()]['frame'][()]
    frame = frame[rg[0]:frame.size+rg[1]]
    if frame_adjust:
        frame = frame_adjust(frame)
    
    data = strialwise #[:,:,nbefore:-nafter]
    
    try:
        dxdt = sio.loadmat(datafiles[0],squeeze_me=True)['dxdt']
    except:
        with h5py.File(datafiles[0],mode='r') as f:
            dxdt = f['dxdt'][:].T
            
    trialrun = np.zeros(frame[0::2].shape)
    for i in range(len(trialrun)):
        trialrun[i] = np.abs(dxdt[frame[0::2][i]:frame[1::2][i]]).mean()
    runtrial = criterion(trialrun)
    print(runtrial.sum()/runtrial.size)
    if criterion_cutoff:
        if runtrial.sum()/runtrial.size < criterion_cutoff:
            #return Savg,Smean,lb,ub,pval,spont,Smean_stat,proc
            return [np.array(())]*8
    
    #pval = np.zeros(strialwise.shape[0])
    #for i in range(strialwise.shape[0]):
    #    _,pval[i] = sst.ttest_rel(data[i,:,nbefore-1],data[i,:,nbefore+1])
        
    stimparams = result['stimParams']
    gratingInfo = result['gratingInfo']
        
    indexlut,stimp = np.unique(stimparams,axis=1,return_inverse=True)
    
    #angle = stimparams[0]
    #size = stimparams[1]
    #contrast = stimparams[4]
    #light = stimparams[5]

    angle = gratingInfo['Orientation'][()]
    size = gratingInfo['Size'][()]
    contrast = gratingInfo['Contrast'][()]
    light = gratingInfo['lightsOn'][()]
    
    ucontrast = np.unique(contrast)
    uangle = np.unique(angle[~np.isnan(angle)])
    usize = np.unique(size)
    ulight = np.unique(light)
    ncontrast = len(ucontrast)
    nangle = len(uangle)
    nsize = len(usize)
    nlight = len(ulight)
    print(nlight)
    
    angle180 = np.remainder(angle,180)
    uangle180 = np.unique(angle180[~np.isnan(angle180)])
    nangle180 = len(uangle180)
    
    Smean = np.zeros((data.shape[0],nangle180,nsize,ncontrast,nlight,data.shape[2]))
    Smean_stat = np.zeros((data.shape[0],nangle180,nsize,ncontrast,nlight,data.shape[2]))
    Stavg = np.zeros((data.shape[0],nangle180,nsize,ncontrast,nlight,int(data.shape[1]/nangle/nsize/ncontrast/nlight)))
    
    Strials = {}
    Sspont = {}
    for i in range(nangle180):
        for j in range(nsize):
            for k in range(ncontrast):
                for ii in range(nlight):
                    lkat = np.logical_and(np.logical_and(runtrial,np.logical_and(angle180==uangle180[i],np.logical_and(size==usize[j],contrast==ucontrast[k]))),light==ulight[ii])
                    Smean[:,i,j,k,ii,:] = data[:,lkat,:].mean(1)
                    Strials[(i,j,k,ii)] = data[:,lkat,nbefore:-nafter].mean(2)
                    Sspont[(i,j,k,ii)] = data[:,lkat,:nbefore].mean(2)
                    stat = np.logical_and(np.logical_and(np.logical_not(runtrial),np.logical_and(angle180==uangle180[i],np.logical_and(size==usize[j],contrast==ucontrast[k]))),light==ulight[ii])
                    Smean_stat[:,i,j,k,ii] = data[:,stat].mean(1)
    
    lb = np.zeros((strialwise.shape[0],nangle180,nsize,ncontrast,nlight))
    ub = np.zeros((strialwise.shape[0],nangle180,nsize,ncontrast,nlight))
    # mn = np.zeros((strialwise.shape[0],nangle180,nsize,ncontrast))
    
    for i in range(nangle180):
        for j in range(nsize):
            for k in range(ncontrast):
                for ii in range(nlight):
                    if Strials[(i,j,k,ii)].size:
                        lb[:,i,j,k,ii],ub[:,i,j,k,ii] = ut.bootstrap(Strials[(i,j,k,ii)],np.mean,axis=1,pct=(16,84))
                    else:
                        lb[:,i,j,k,ii] = np.nan
                        ub[:,i,j,k,ii] = np.nan
                # mn[:,i,j,k] = np.nanmean(Strials[(i,j,k)],axis=1)
    
    pval = np.zeros((strialwise.shape[0],nangle180))
#     for i in range(pval.shape[0]):
#         print(i)
    for j,theta in enumerate(uangle180):
        _,pval[:,j] = sst.ttest_rel(Sspont[(j,0,ncontrast-1,0)],Strials[(j,0,ncontrast-1,0)],axis=1)
                
    Savg = np.nanmean(np.nanmean(Smean[:,:,:,:,:,nbefore:-nafter],axis=-1),axis=1)
    
    Storiavg = Stavg.mean(1)
    # _,pval = sst.ttest_ind(Storiavg[:,0,-1].T,Storiavg[:,0,0].T)
    
    #suppressed = np.logical_and(pval<0.05,Savg[:,0,-1]<Savg[:,0,0])
    #facilitated = np.logical_and(pval<0.05,Savg[:,0,-1]>Savg[:,0,0])
    spont = np.zeros((Savg.shape[0],))
    keylist = list(Sspont.keys())
    nkeys = len(keylist)
    for key in Sspont.keys():
        spont = spont + Sspont[key].mean(1)/nkeys

    proc = {}
    proc['runtrial'] = runtrial
    proc['trialrun'] = trialrun
    proc['angle'] = angle
    proc['size'] = size
    proc['contrast'] = contrast
    proc['light'] = light
    proc['trialwise'] = trialwise
    proc['strialwise'] = strialwise
    #proc['ctrialwise'] = ctrialwise
    proc['dtrialwise'] = dtrialwise
    proc['trialwise_t_offset'] = trialwise_t_offset
    proc['dfof'] = dfof
    proc['frame'] = frame
    
    return Savg,Smean,lb,ub,pval,spont,Smean_stat,proc


def analyze_everything(folds=None,files=None,rets=None,adjust_fns=None,rgs=None,criteria=None,datafoldbase=None,stimfoldbase=None):
    soriavg = {}
    strialavg = {}
    lb = {}
    ub = {}
    pval = {}
    nbydepth = {}
    spont = {}
    ret_vars = {}
    Smean_stat = {}
    proc = {}
    #datafoldbase = '/home/mossing/scratch/2Pdata/'
    #stimfoldbase = '/home/mossing/scratch/visual_stim/'
    for thisfold,thisfile,retnumber,frame_adjust,rg,criterion in zip(folds,files,rets,adjust_fns,rgs,criteria):
        datafold = datafoldbase+thisfold+'ot/'
        datafiles = [thisfile+'_ot_'+number+'.rois' for number in ['000','001','002','003']]

        stimfold = stimfoldbase+thisfold
        stimfile = thisfile+'.mat'

        datafiles = [datafold+file for file in datafiles]
        datafiles = [x for x in datafiles if os.path.exists(x)]
        stimfile = stimfold+stimfile
        retfile = datafoldbase+thisfold+'retinotopy_'+retnumber+'.mat'
        print(retfile)

        nbefore = 4
        nafter = 4

        soriavg[thisfold],strialavg[thisfold],lb[thisfold],ub[thisfold],pval[thisfold],spont[thisfold],Smean_stat[thisfold],proc[thisfold] = analyze_size_contrast(datafiles,stimfile,retfile,frame_adjust=frame_adjust,rg=rg,nbefore=nbefore,nafter=nafter,criterion=criterion)
        nbydepth[thisfold] = get_nbydepth(datafiles)
        try: 
            ret_vars[thisfold] = sio.loadmat(retfile,squeeze_me=True)#['ret'][:]
            print('loaded retinotopy file')
            ret_vars[thisfold]['position'] = sio.loadmat(stimfile,squeeze_me=True)['result'][()]['position']
        except:
            print('retinotopy not saved for '+thisfile)
            ret_vars[thisfold] = None
    return soriavg,strialavg,lb,ub,pval,nbydepth,spont,ret_vars,Smean_stat,proc

def analyze_everything_by_criterion(folds=None,files=None,rets=None,adjust_fns=None,rgs=None,criteria=None,criterion_cutoff=0.2,datafoldbase=None,stimfoldbase=None,procname='size_contrast_opto_proc.hdf5'):

    soriavg = {}
    strialavg = {}
    lb = {}
    ub = {}
    pval = {}
    nbydepth = {}
    spont = {}
    ret_vars = {}
    Smean_stat = {}
    #proc = {}
    proc_hdf5_name = procname
    #datafoldbase = '/home/mossing/scratch/2Pdata/'
    #stimfoldbase = '/home/mossing/scratch/visual_stim/'
    for thisfold,thisfile,retnumber,frame_adjust,rg,criterion,thisdfb,thissfb in zip(folds,files,rets,adjust_fns,rgs,criteria,datafoldbase,stimfoldbase):

        session_id = 'session_'+thisfold[:-1].replace('/','_')

        soriavg[thisfold] = [None]*2
        strialavg[thisfold] = [None]*2
        lb[thisfold] = [None]*2
        ub[thisfold] = [None]*2
        pval[thisfold] = [None]*2
        nbydepth[thisfold] = [None]*2
        spont[thisfold] = [None]*2
        ret_vars[thisfold] = [None]*2
        Smean_stat[thisfold] = [None]*2
        #proc[thisfold] = [None]*2
        proc = [None]*2

        datafold = thisdfb+thisfold+'ot/'
        datafiles = [thisfile+'_ot_'+number+'.rois' for number in ['000','001','002','003']]

        stimfold = thissfb+thisfold
        stimfile = thisfile+'.mat'

        datafiles = [datafold+file for file in datafiles]
        datafiles = [x for x in datafiles if os.path.exists(x)]
        stimfile = stimfold+stimfile
        retfile = thisdfb+thisfold+'retinotopy_'+retnumber+'.mat'
        print(retfile)

        nbefore = 4
        nafter = 4

        thesecriteria = [criterion,lambda x: np.logical_not(criterion(x))]

        for k in range(2):
            soriavg[thisfold][k],strialavg[thisfold][k],lb[thisfold][k],ub[thisfold][k],pval[thisfold][k],spont[thisfold][k],Smean_stat[thisfold][k],proc[k] = analyze_size_contrast(datafiles,stimfile,retfile,frame_adjust=frame_adjust,rg=rg,nbefore=nbefore,nafter=nafter,criterion=thesecriteria[k],criterion_cutoff=criterion_cutoff)

        nbydepth[thisfold] = get_nbydepth(datafiles)
        try: 
            ret_vars[thisfold] = sio.loadmat(retfile,squeeze_me=True)#['ret'][:]
            print('loaded retinotopy file')
            ret_vars[thisfold]['position'] = sio.loadmat(stimfile,squeeze_me=True)['result'][()]['position']
        except:
            print('retinotopy not saved for '+thisfile)
            ret_vars[thisfold] = None
        if len(proc[0]):
            gdind = 0
        else:
            gdind = 1
        needed_ret_vars = ['position','pval_ret','ret']
        ret_dicti = {varname:ret_vars[thisfold][varname] for varname in needed_ret_vars}
        ret_dicti['paramdict_normal'] = ut.matfile_to_dict(ret_vars[thisfold]['paramdict_normal'])
        proc[gdind]['ret_vars'] = ret_dicti
        ut.dict_to_hdf5(proc_hdf5_name,session_id,proc[gdind])
    return soriavg,strialavg,lb,ub,pval,nbydepth,spont,ret_vars,Smean_stat #,proc

def get_norm_curves(soriavg,lkat=None,sizes=None,contrasts=None,append_gray=False):
    # lkat a dict with keys corresponding to the expts. you'll summarize. lkat is a binary vector for each expt.
    # sizes are the desired rows of each ROI's 2d array. contrasts are the desired columns. append_gray says whether
    # to append the lowest contrast result to the left side of the resulting 2d arrays
    
    def parse_desired(shape,desired):
        if not desired is None:
            bins = np.in1d(np.arange(shape),desired)
        else:
            bins = np.ones((shape,),dtype='bool')
        return bins
    
    def norm_by_roi(arr):
        return arr/np.nanmax(np.nanmax(arr,axis=1),axis=1)[:,np.newaxis,np.newaxis]
    
    if lkat is None:
        lkat = {}
        for key in soriavg.keys():
            lkat[key] = np.ones((soriavg[key].shape[0],),dtype='bool')
    keylist = list(lkat.keys())

    snorm = np.array(())
    for key in keylist:
        biny = parse_desired(soriavg[key].shape[1],sizes)
        binx = parse_desired(soriavg[key].shape[2],contrasts)
        soriavgnorm = norm_by_roi(soriavg[key][lkat[key]])
        to_add = soriavgnorm[:,:,binx][:,biny]
        if append_gray:
            gray = np.tile(soriavgnorm[:,:,0].mean(1)[:,np.newaxis],(1,binx.sum()))[:,np.newaxis,:]
            to_add = np.concatenate((gray,to_add),axis=1)
            
        if snorm.size:
            snorm = np.vstack((snorm,to_add))
        else:
            snorm = to_add
            
    return snorm

def gen_full_data_struct(cell_type='PyrL23', keylist=None, frame_rate_dict=None, proc=None, ret_vars=None, nbefore=8, nafter=8):
    data_struct = {}
    for key in keylist:
        if len(proc[key][0])>0:
            gdind = 0
        else:
            gdind = 1
        dfof = proc[key][gdind]['dtrialwise']
        decon = np.nanmean(proc[key][gdind]['strialwise'][:,:,nbefore:-nafter],-1)
        #calcium_responses_au = np.nanmean(proc[key][gdind]['trialwise'][:,:,nbefore:-nafter],-1)
        running_speed_cm_s = 4*np.pi/180*proc[key][gdind]['trialrun'] # 4 cm from disk ctr to estimated mouse location
        rf_ctr = np.concatenate((ret_vars[key]['paramdict_normal'][()]['xo'][np.newaxis,:],-ret_vars[key]['paramdict_normal'][()]['yo'][np.newaxis,:]),axis=0)
        stim_offset = ret_vars[key]['position'] - ret_vars[key]['paramdict_normal'][()]['ctr']
        rf_distance_deg = np.sqrt(((rf_ctr-stim_offset[:,np.newaxis])**2).sum(0))
        rf_displacement_deg = rf_ctr-stim_offset[:,np.newaxis]
        cell_id = np.arange(dfof.shape[0])
        ucontrast,icontrast = np.unique(proc[key][gdind]['contrast'],return_inverse=True)
        usize,isize = np.unique(proc[key][gdind]['size'],return_inverse=True)
        uangle,iangle = np.unique(proc[key][gdind]['angle'],return_inverse=True)
        ulight,ilight = np.unique(proc[key][gdind]['light'],return_inverse=True)
        stimulus_id = np.concatenate((isize[np.newaxis],icontrast[np.newaxis],iangle[np.newaxis],ilight[np.newaxis]),axis=0)
        stimulus_size_deg = usize
        stimulus_contrast = ucontrast
        stimulus_direction = uangle
        stimulus_light = ulight
        session_id = 'session_'+key[:-1].replace('/','_')
        mouse_id = key.split('/')[1]
        
        data_struct[session_id] = {}
        data_struct[session_id]['mouse_id'] = mouse_id
        data_struct[session_id]['stimulus_id'] = stimulus_id
        data_struct[session_id]['stimulus_size_deg'] = stimulus_size_deg
        data_struct[session_id]['stimulus_contrast'] = stimulus_contrast
        data_struct[session_id]['stimulus_direction'] = stimulus_direction
        data_struct[session_id]['stimulus_light'] = stimulus_light
        data_struct[session_id]['cell_id'] = cell_id
        data_struct[session_id]['cell_type'] = cell_type
        data_struct[session_id]['mouse_id'] = mouse_id
        data_struct[session_id]['F'] = dfof
        data_struct[session_id]['decon'] = decon
        data_struct[session_id]['nbefore'] = nbefore
        data_struct[session_id]['nafter'] = nafter
        data_struct[session_id]['rf_mapping_pval'] = ret_vars[key]['pval_ret']
        data_struct[session_id]['rf_distance_deg'] = rf_distance_deg
        data_struct[session_id]['rf_displacement_deg'] = rf_displacement_deg
        data_struct[session_id]['running_speed_cm_s'] = running_speed_cm_s
    return data_struct

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
            print([el for el in proc[key]['trialwise_t_offset']])
            t_offset = proc[key]['trialwise_t_offset']['trialwise_t_offset'][:] 
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
            ucontrast,icontrast = np.unique(proc[key]['contrast'][:],return_inverse=True)
            usize,isize = np.unique(proc[key]['size'][:],return_inverse=True)
            uangle,iangle = np.unique(proc[key]['angle'][:],return_inverse=True)
            ulight,ilight = np.unique(proc[key]['light'],return_inverse=True)
            stimulus_id = np.concatenate((isize[np.newaxis],icontrast[np.newaxis],iangle[np.newaxis],ilight[np.newaxis]),axis=0)
            stimulus_size_deg = usize
            stimulus_contrast = ucontrast
            stimulus_direction = uangle
            stimulus_light = ulight
            #session_id = 'session_'+key[:-1].replace('/','_')
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
            while 'size_contrast_opto_'+str(exptno) in this_session.keys():
                exptno = exptno+1
            this_expt = this_session.create_group('size_contrast_'+str(exptno))
            this_expt.create_dataset('stimulus_id',data=stimulus_id)
            this_expt.create_dataset('stimulus_size_deg',data=stimulus_size_deg)
            this_expt.create_dataset('stimulus_contrast',data=stimulus_contrast)
            this_expt.create_dataset('stimulus_direction',data=stimulus_direction)
            this_expt.create_dataset('stimulus_light',data=stimulus_light)
            this_expt['rf_mapping_pval'] = ret_vars[key]['pval_ret'][:]
            this_expt['rf_distance_deg'] = rf_distance_deg
            this_expt['rf_displacement_deg'] = rf_displacement_deg
            this_expt['rf_ctr'] = rf_ctr
            this_expt['stim_offset_deg'] = stim_offset
            this_expt.create_dataset('running_speed_cm_s',data=running_speed_cm_s)
            this_expt.create_dataset('F',data=dfof)
            this_expt.create_dataset('decon',data=decon)
            this_expt.create_dataset('t_offset',data=t_offset)
            this_expt['nbefore'] = nbefore
            this_expt['nafter'] = nafter

def analyze_simply(folds=None,files=None,rets=None,adjust_fns=None,rgs=None,datafoldbase='/home/mossing/scratch/2Pdata/',stimfoldbase='/home/mossing/scratch/visual_stim/',procname='size_contrast_proc.hdf5'):
    if isinstance(datafoldbase,str):
        datafoldbase = [datafoldbase]*len(folds)
    if isinstance(stimfoldbase,str):
        stimfoldbase = [stimfoldbase]*len(folds)
    stim_params = size_contrast_opto_params()
    for thisfold,thisfile,frame_adjust,rg,thisdatafoldbase,thisstimfoldbase,retnumber in zip(folds,files,adjust_fns,rgs,datafoldbase,stimfoldbase,rets):

        session_id = at.gen_session_id(thisfold)
        datafiles = at.gen_datafiles(thisdatafoldbase,thisfold,thisfile,nplanes=4)
        stimfile = at.gen_stimfile(thisstimfoldbase,thisfold,thisfile)
        retfile = thisdatafoldbase+thisfold+'retinotopy_'+retnumber+'.mat'

        nbefore = 8
        nafter = 8

        proc = at.analyze(datafiles,stimfile,frame_adjust=frame_adjust,rg=rg,nbefore=nbefore,nafter=nafter,stim_params=stim_params)
        
        proc['ret_vars'] = at.gen_ret_vars(retfile,stimfile)

        ut.dict_to_hdf5(procname,session_id,proc)

def size_contrast_opto_params():
    paramlist = [('angle','Orientation'),('size','Size'),('contrast','Contrast'),('light','lightsOn')]
    params_and_fns = [None]*len(paramlist)
    for i,pair in enumerate(paramlist):
        param = pair[0]
        function = lambda result: result['gratingInfo'][()][pair[1]][()]
        params_and_fns[i] = (param,function)
    return params_and_fns

def add_data_struct_h5_simply(filename, cell_type='PyrL23', keylist=None, frame_rate_dict=None, proc=None, nbefore=8, nafter=8):
    groupname = 'size_contrast_opto'
    featurenames=['size','contrast','angle','light']
    datasetnames = ['stimulus_size_deg','stimulus_contrast','stimulus_direction_deg','stimulus_light']
    at.add_data_struct_h5(filename,cell_type=cell_type,keylist=keylist,frame_rate_dict=frame_rate_dict,proc=proc,nbefore=nbefore,nafter=nafter,featurenames=featurenames,datasetnames=datasetnames,groupname=groupname)
