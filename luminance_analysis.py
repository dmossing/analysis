#!/usr/bin/env python

import os
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import h5py
from oasis.functions import deconvolve
from oasis import oasisAR1, oasisAR2
import pyute as ut
import ca_processing as cap
from importlib import reload
reload(ut)
import scipy.ndimage.filters as sfi
import scipy.stats as sst
import scipy.ndimage.measurements as snm
from mpl_toolkits.mplot3d import Axes3D
import scipy.optimize as sop
import pdb

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

def analyze_luminance(datafiles,stimfile,retfile=None,frame_adjust=None,rg=(1,0),nbefore=nbefore,nafter=nafter,criterion=None,criterion_cutoff=None):
    if criterion is None:
        criterion = lambda x: np.abs(x)>100
    nbydepth = get_nbydepth(datafiles)
#     trialwise,ctrialwise,strialwise = gen_trialwise(datafiles,frame_adjust=frame_adjust)
    trialwise,ctrialwise,strialwise,dfof,straces,dtrialwise,trialwise_t_offset = cap.gen_precise_trialwise(datafiles,rg=rg,frame_adjust=frame_adjust,nbefore=nbefore,nafter=nafter,blcutoff=blcutoff)
    zstrialwise = sst.zscore(strialwise.reshape((strialwise.shape[0],-1)).T).T.reshape(strialwise.shape)
    
    result = sio.loadmat(stimfile,squeeze_me=True)['result'][()]
    
    infofile = sio.loadmat(datafiles[0][:-12]+'.mat',squeeze_me=True) # original .mat file
    frame = infofile['info'][()]['frame'][()]
    frame = frame[rg[0]:frame.size+rg[1]]
    #assert(True==False)
    if frame_adjust:
        frame = frame_adjust(frame)
    
    data = strialwise #[:,:,nbefore:-nafter]
    data_dfof = trialwise #[:,:,nbefore:-nafter]
    print(data.shape)
    
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
             
    stimparams = result['stimParams']
    gratingInfo = result['gratingInfo']
        
    indexlut,stimp = np.unique(stimparams,axis=0,return_inverse=True)
    
    intensity = gratingInfo['Intensity'][()]
    
    uintensity = np.unique(intensity)
    nintensity = len(uintensity)
 
#    Smean = np.zeros((data.shape[0],nangle180,nsize,ncontrast,data.shape[2]))
#    Fmean = np.zeros((data.shape[0],nangle180,nsize,ncontrast,data.shape[2]))
#    Smean_stat = np.zeros((data.shape[0],nangle180,nsize,ncontrast,data.shape[2]))
#    Stavg = np.zeros((data.shape[0],nangle180,nsize,ncontrast,int(data.shape[1]/nangle/nsize/ncontrast)))
#    
#    Strials = {}
#    Sspont = {}
#    for i in range(nangle180):
#        for j in range(nsize):
#            for k in range(ncontrast):
#                lkat = np.logical_and(runtrial,np.logical_and(angle180==uangle180[i],np.logical_and(size==usize[j],contrast==ucontrast[k])))
#                Smean[:,i,j,k,:] = np.nanmean(data[:,lkat,:],1)
#                Fmean[:,i,j,k,:] = np.nanmean(data_dfof[:,lkat,:],1)
#                Strials[(i,j,k)] = np.nanmean(data[:,lkat,nbefore:-nafter],2)
#                Sspont[(i,j,k)] = np.nanmean(data[:,lkat,:nbefore],2)
#                stat = np.logical_and(np.logical_not(runtrial),np.logical_and(angle180==uangle180[i],np.logical_and(size==usize[j],contrast==ucontrast[k])))
#                Smean_stat[:,i,j,k] = np.nanmean(data[:,stat],1)
# 
#    lb = np.zeros((strialwise.shape[0],nangle180,nsize,ncontrast))
#    ub = np.zeros((strialwise.shape[0],nangle180,nsize,ncontrast))
#    
#    for i in range(nangle180):
#        print(i)
#        for j in range(nsize):
#            for k in range(ncontrast):
#                if Strials[(i,j,k)].size:
#                    lb[:,i,j,k],ub[:,i,j,k] = ut.bootstrap(Strials[(i,j,k)],np.mean,axis=1,pct=(16,84))
#                else:
#                    lb[:,i,j,k] = np.nan
#                    ub[:,i,j,k] = np.nan
#    
#    pval = np.zeros((strialwise.shape[0],nangle180))
#    for j,theta in enumerate(uangle180):
#        print(theta)
#        _,pval[:,j] = sst.ttest_ind(Strials[(j,0,0)],Strials[(j,0,ncontrast-1)],axis=1)
#    Savg = np.nanmean(np.nanmean(Smean[:,:,:,:,nbefore:-nafter],axis=-1),axis=1)
#    Favg = np.nanmean(np.nanmean(Fmean[:,:,:,:,nbefore:-nafter],axis=-1),axis=1)
#    
#    spont = np.zeros((Savg.shape[0],))
#    keylist = list(Sspont.keys())
#    nkeys = len(keylist)
#    for key in Sspont.keys():
#        spont = spont + Sspont[key].mean(1)/nkeys

    proc = {}
    proc['runtrial'] = runtrial
    proc['trialrun'] = trialrun
    proc['intensity'] = intensity
    proc['trialwise'] = trialwise
    proc['strialwise'] = strialwise
    #proc['ctrialwise'] = ctrialwise
    proc['dtrialwise'] = dtrialwise
    proc['dfof'] = dfof
    proc['trialwise_t_offset'] = trialwise_t_offset
    #proc['straces'] = straces
    #proc['oriavg_dfof'] = Favg
    
    return proc


def analyze_everything(folds=None,files=None,rets=None,adjust_fns=None,rgs=None,criteria=None,datafoldbase='/home/mossing/scratch/2Pdata/',stimfoldbase='/home/mossing/scratch/visual_stim/'):
    if isinstance(datafoldbase,str):
        datafoldbase = [datafoldbase]*len(folds)
    if isinstance(stimfoldbase,str):
        stimfoldbase = [stimfoldbase]*len(folds)
    #soriavg = {}
    #strialavg = {}
    #lb = {}
    #ub = {}
    #pval = {}
    #nbydepth = {}
    #spont = {}
    #ret_vars = {}
    #Smean_stat = {}
    proc = {}
    for thisfold,thisfile,retnumber,frame_adjust,rg,criterion,thisdatafoldbase,thisstimfoldbase in zip(folds,files,rets,adjust_fns,rgs,criteria,datafoldbase,stimfoldbase):
        datafold = thisdatafoldbase+thisfold+'ot/'
        datafiles = [thisfile+'_ot_'+number+'.rois' for number in ['000','001','002','003']]

        stimfold = thisstimfoldbase+thisfold
        stimfile = thisfile+'.mat'

        datafiles = [datafold+file for file in datafiles]
        stimfile = stimfold+stimfile
        retfile = thisdatafoldbase+thisfold+'retinotopy_'+retnumber+'.mat'
        print(retfile)

        nbefore = 8
        nafter = 8

        proc[thisfold] = analyze_luminance(datafiles,stimfile,retfile,frame_adjust=frame_adjust,rg=rg,nbefore=nbefore,nafter=nafter,criterion=criterion)
        proc[thisfold]['nbydepth'] = get_nbydepth(datafiles)
        try: 
            proc[thisfold]['ret_vars'] = sio.loadmat(retfile,squeeze_me=True)#['ret'][:]
            print('loaded retinotopy file')
        except:
            print('retinotopy not saved for '+thisfile)
            proc[thisfold]['ret_vars'] = None
    return proc

def get_norm_curves(soriavg,lkat=None,sizes=None,contrasts=None,append_gray=False):
    # lkat a dict with keys corresponding to the expts. you'll summarize. lkat is a binary vector for each expt.
    # sizes are the desired rows of each ROI's 2d array. contrasts are the desired columns. append_gray says whether
    # to append the lowest contrast result to the left side of the resulting 2d arrays
    
    keylist = list(soriavg.keys())
    sizes_dict = {}
    contrasts_dict = {}
    for key in keylist:
        sizes_dict[key] = sizes
        contrasts_dict[key] = contrasts
    return get_norm_curves_row_col(soriavg,lkat=lkat,sizes=sizes_dict,contrasts=contrasts_dict,append_gray=append_gray)
    
    
    #def parse_desired(shape,desired):
    #    if not desired is None:
    #        bins = np.in1d(np.arange(shape),desired)
    #    else:
    #        bins = np.ones((shape,),dtype='bool')
    #    return bins
    #
    #def norm_by_roi(arr):
    #    return arr/np.nanmax(np.nanmax(arr,axis=1),axis=1)[:,np.newaxis,np.newaxis]
    #
    #if lkat is None:
    #    lkat = {}
    #    for key in soriavg.keys():
    #        lkat[key] = np.ones((soriavg[key].shape[0],),dtype='bool')
    #keylist = list(lkat.keys())

    #snorm = np.array(())
    #for key in keylist:
    #    biny = parse_desired(soriavg[key].shape[1],sizes)
    #    binx = parse_desired(soriavg[key].shape[2],contrasts)
    #    soriavgnorm = norm_by_roi(soriavg[key][lkat[key]])
    #    to_add = soriavgnorm[:,:,binx][:,biny]
    #    if append_gray:
    #        gray = np.tile(soriavgnorm[:,:,0].mean(1)[:,np.newaxis],(1,binx.sum()))[:,np.newaxis,:]
    #        to_add = np.concatenate((gray,to_add),axis=1)
    #        
    #    if snorm.size:
    #        snorm = np.vstack((snorm,to_add))
    #    else:
    #        snorm = to_add
    #        
    #return snorm

def get_norm_curves_row_col(soriavg,lkat=None,sizes=None,contrasts=None,append_gray=False):
    # lkat a dict with keys corresponding to the expts. you'll summarize. lkat is a binary vector for each expt.
    # sizes are the desired rows of each ROI's 2d array. contrasts are the desired columns. append_gray says whether
    # to append the lowest contrast result to the left side of the resulting 2d arrays
    keylist = list(soriavg.keys())
    listy = isinstance(soriavg[keylist[0]],list)
    for key in keylist:
        assert listy == isinstance(soriavg[key],list)
    
    if not listy:
        for key in keylist:
            soriavg[key] = [soriavg[key]]

    for key in keylist:
        assert len(soriavg[key]) == len(soriavg[keylist[0]])
    snorm = get_norm_curves_row_col_listy(soriavg,lkat,sizes,contrasts,append_gray)
    if not listy:
        return snorm[0]
    else:
        return snorm
    
def get_norm_curves_row_col_listy(soriavg,lkat=None,sizes=None,contrasts=None,append_gray=False):
    # same as get_norm_curves_row_col, but assumes soriavg is a dict of lists
    def parse_desired(shape,desired):
        if not desired is None:
            bins = np.in1d(np.arange(shape),desired)
        else:
            bins = np.ones((shape,),dtype='bool')
        return bins
    
    def norm_by_roi(arr):
        return arr/np.nanmax(np.nanmax(arr,axis=1),axis=1)[:,np.newaxis,np.newaxis]
    
    keylist = list(soriavg.keys())
    
    if lkat is None:
        lkat = {}
        for key in keylist:
            lkat[key] = np.ones((soriavg[key][0].shape[0],),dtype='bool')

    snorm = [np.array(())]*len(soriavg[keylist[0]])
    soriavgnorm = snorm.copy()
    for key in keylist:
        sz = [el.size for el in soriavg[key]]
        gdind = np.where(sz)[0][:1][0]
        biny = parse_desired(soriavg[key][gdind].shape[1],sizes[key])
        binx = parse_desired(soriavg[key][gdind].shape[2],contrasts[key])
        assert len(sizes[key])==biny.sum()
        for i,arr in enumerate(soriavg[key]):
            if sz[i]:
                soriavgnorm[i] = norm_by_roi(arr[lkat[key]])
                to_add = soriavgnorm[i][:,:,binx][:,biny]
                if append_gray:
                    shp = soriavgnorm[i].shape
                    if len(shp)==3:
                        gray = np.tile(soriavgnorm[i][:,:,0].mean(1)[:,np.newaxis],(1,binx.sum()))[:,np.newaxis,:]
                    else:
                        gray = np.tile(soriavgnorm[i][:,:,0,:].mean(1)[:,np.newaxis,np.newaxis],(1,1,binx.sum(),1))
                    to_add = np.concatenate((gray,to_add),axis=1)
                    
                if snorm[i].size:
                    snorm[i] = np.vstack((snorm[i],to_add))
                else:
                    snorm[i] = to_add
            
    return snorm

def append_gray(arr):
    # take the mean of the 0th nth-index, and append it as the 0th (n-1)th index
    shp = arr.shape
    if len(shp)==2:
        gray = np.tile(arr[:,0].mean(0)[np.newaxis],(1,arr.shape[1]))
        return np.concatenate((gray,arr),axis=0)

def analyze_everything_by_criterion(folds=None,files=None,rets=None,adjust_fns=None,rgs=None,criteria=None,datafoldbase='/home/mossing/scratch/2Pdata/',stimfoldbase='/home/mossing/scratch/visual_stim/',criterion_cutoff=0.2):
    if isinstance(datafoldbase,str):
        datafoldbase = [datafoldbase]*len(folds)
    if isinstance(stimfoldbase,str):
        stimfoldbase = [stimfoldbase]*len(folds)
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
    for thisfold,thisfile,retnumber,frame_adjust,rg,criterion,thisdatafoldbase,thisstimfoldbase in zip(folds,files,rets,adjust_fns,rgs,criteria,datafoldbase,stimfoldbase):

        soriavg[thisfold] = [None]*2
        strialavg[thisfold] = [None]*2
        lb[thisfold] = [None]*2
        ub[thisfold] = [None]*2
        pval[thisfold] = [None]*2
        nbydepth[thisfold] = [None]*2
        spont[thisfold] = [None]*2
        ret_vars[thisfold] = [None]*2
        Smean_stat[thisfold] = [None]*2
        proc[thisfold] = [None]*2

        datafold = thisdatafoldbase+thisfold+'ot/'
        datafiles = [thisfile+'_ot_'+number+'.rois' for number in ['000','001','002','003']]

        stimfold = thisstimfoldbase+thisfold
        stimfile = thisfile+'.mat'

        datafiles = [datafold+file for file in datafiles]
        datafiles = [x for x in datafiles if os.path.exists(x)]
        stimfile = stimfold+stimfile
        retfile = thisdatafoldbase+thisfold+'retinotopy_'+retnumber+'.mat'
        print(retfile)

        nbefore = 8
        nafter = 8
        
        thesecriteria = [criterion,lambda x: np.logical_not(criterion(x))]

        for k in range(2):
            #soriavg[thisfold][k],strialavg[thisfold][k],lb[thisfold][k],ub[thisfold][k],pval[thisfold][k],spont[thisfold][k],Smean_stat[thisfold][k],proc[thisfold][k] = analyze_size_contrast(datafiles,stimfile,retfile,frame_adjust=frame_adjust,rg=rg,nbefore=nbefore,nafter=nafter,criterion=thesecriteria[k],criterion_cutoff=criterion_cutoff)
            soriavg[thisfold][k],_,_,_,_,_,_,proc[thisfold][k] = analyze_size_contrast(datafiles,stimfile,retfile,frame_adjust=frame_adjust,rg=rg,nbefore=nbefore,nafter=nafter,criterion=thesecriteria[k],criterion_cutoff=criterion_cutoff) # changed 2/12/19
        nbydepth[thisfold] = get_nbydepth(datafiles)
        try: 
            ret_vars[thisfold] = sio.loadmat(retfile,squeeze_me=True)#['ret'][:]
            print('loaded retinotopy file')
            ret_vars[thisfold]['position'] = sio.loadmat(stimfile,squeeze_me=True)['result'][()]['position']
        except:
            print('retinotopy not saved for '+thisfile)
            ret_vars[thisfold] = None
    #return soriavg,strialavg,lb,ub,pval,nbydepth,spont,ret_vars,Smean_stat,proc
    return soriavg,ret_vars,proc # changed 2/12/19

def find_gdind(arrlist):
    sz = [el.size for el in arrlist]
    return np.where(sz)[0][:1][0]

def gen_data_struct(cell_type='PyrL23', keylist=None, frame_rate_dict=None, proc=None, ret_vars=None, nbefore=8, nafter=8):
    data_struct = {}
    for key in keylist:
        if len(proc[key][0])>0:
            gdind = 0
        else:
            gdind = 1
        decon = np.nanmean(proc[key][gdind]['strialwise'][:,:,nbefore:-nafter],-1)
        calcium_responses_au = np.nanmean(proc[key][gdind]['trialwise'][:,:,nbefore:-nafter],-1)
        running_speed_cm_s = 4*np.pi/180*proc[key][gdind]['trialrun'] # 4 cm from disk ctr to estimated mouse location
        rf_ctr = np.concatenate((ret_vars[key]['paramdict_normal'][()]['xo'][np.newaxis,:],-ret_vars[key]['paramdict_normal'][()]['yo'][np.newaxis,:]),axis=0)
        stim_offset = ret_vars[key]['position'] - ret_vars[key]['paramdict_normal'][()]['ctr']
        rf_distance_deg = np.sqrt(((rf_ctr-stim_offset[:,np.newaxis])**2).sum(0))
        cell_id = np.arange(decon.shape[0])
        ucontrast,icontrast = np.unique(proc[key][gdind]['contrast'],return_inverse=True)
        usize,isize = np.unique(proc[key][gdind]['size'],return_inverse=True)
        uangle,iangle = np.unique(proc[key][gdind]['angle'],return_inverse=True)
        stimulus_id = np.concatenate((isize[np.newaxis],icontrast[np.newaxis],iangle[np.newaxis]),axis=0)
        stimulus_size_deg = usize
        stimulus_contrast = ucontrast
        stimulus_direction = uangle
        session_id = 'session_'+key[:-1].replace('/','_')
        mouse_id = key.split('/')[1]
        
        data_struct[session_id] = {}
        data_struct[session_id]['mouse_id'] = mouse_id
        data_struct[session_id]['stimulus_id'] = stimulus_id
        data_struct[session_id]['stimulus_size_deg'] = stimulus_size_deg
        data_struct[session_id]['stimulus_contrast'] = stimulus_contrast
        data_struct[session_id]['stimulus_direction'] = stimulus_direction
        data_struct[session_id]['cell_id'] = cell_id
        data_struct[session_id]['cell_type'] = cell_type
        data_struct[session_id]['mouse_id'] = mouse_id
        data_struct[session_id]['calcium_responses_au'] = calcium_responses_au
        data_struct[session_id]['decon'] = decon
        data_struct[session_id]['rf_mapping_pval'] = ret_vars[key]['pval_ret']
        data_struct[session_id]['rf_distance_deg'] = rf_distance_deg
        data_struct[session_id]['rf_ctr'] = rf_ctr
        data_struct[session_id]['running_speed_cm_s'] = running_speed_cm_s
    return data_struct


def gen_full_data_struct(cell_type='PyrL23', keylist=None, frame_rate_dict=None, proc=None, ret_vars=None, nbefore=8, nafter=8):
    data_struct = {}
    if ret_vars is None:
        ret_vars = {}
        for key in keylist:
            ret_vars[key] = proc[key]['ret_vars']
    for key in keylist:
        dfof = proc[key]['dtrialwise']
        #calcium_responses_au = np.nanmean(proc[key]['trialwise'][:,:,nbefore:-nafter],-1)
        running_speed_cm_s = 4*np.pi/180*proc[key]['trialrun'] # 4 cm from disk ctr to estimated mouse location
        rf_ctr = np.concatenate((ret_vars[key]['paramdict_normal'][()]['xo'][np.newaxis,:],-ret_vars[key]['paramdict_normal'][()]['yo'][np.newaxis,:]),axis=0)
        stim_offset = np.array((0,0)) - ret_vars[key]['paramdict_normal'][()]['ctr']
        rf_distance_deg = np.sqrt(((rf_ctr-stim_offset[:,np.newaxis])**2).sum(0))
        cell_id = np.arange(dfof.shape[0])
        uintensity,iintensity = np.unique(proc[key]['intensity'],return_inverse=True)
        stimulus_id = np.concatenate((iintensity[np.newaxis],),axis=0)
        stimulus_intensity = uintensity
        session_id = 'session_'+key[:-1].replace('/','_')
        mouse_id = key.split('/')[1]
        
        data_struct[session_id] = {}
        data_struct[session_id]['mouse_id'] = mouse_id
        data_struct[session_id]['stimulus_id'] = stimulus_id
        data_struct[session_id]['stimulus_intensity'] = stimulus_intensity
        data_struct[session_id]['cell_id'] = cell_id
        data_struct[session_id]['cell_type'] = cell_type
        data_struct[session_id]['mouse_id'] = mouse_id
        data_struct[session_id]['rf_mapping_pval'] = ret_vars[key]['pval_ret']
        data_struct[session_id]['rf_distance_deg'] = rf_distance_deg
        data_struct[session_id]['rf_ctr'] = rf_ctr
        data_struct[session_id]['running_speed_cm_s'] = running_speed_cm_s
        data_struct[session_id]['F'] = dfof
        data_struct[session_id]['nbefore'] = nbefore
        data_struct[session_id]['nafter'] = nafter
    return data_struct
