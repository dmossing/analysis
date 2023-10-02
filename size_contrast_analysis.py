#!/usr/bin/env python

import os
import scipy.io as sio
import matplotlib.pyplot as plt
import autograd.numpy as np
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
import sklearn
from autograd import elementwise_grad as egrad
#import naka_rushton_analysis as nra
import sim_utils

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
    proc = {}
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

def analyze_size_contrast(datafiles,stimfile,retfile=None,frame_adjust=None,rg=(1,0),nbefore=nbefore,nafter=nafter,criterion=None,criterion_cutoff=None):
    if criterion is None:
        criterion = lambda x: np.abs(x)>100
    nbydepth = get_nbydepth(datafiles)
#     trialwise,ctrialwise,strialwise = gen_trialwise(datafiles,frame_adjust=frame_adjust)
    trialwise,ctrialwise,strialwise,dfof,straces,dtrialwise,proc1 = ut.gen_precise_trialwise(datafiles,rg=rg,frame_adjust=frame_adjust,nbefore=nbefore,nafter=nafter,blcutoff=blcutoff) # , trialwise_t_offset
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
        
    indexlut,stimp = np.unique(stimparams,axis=1,return_inverse=True)
    
    #angle = stimparams[0]
    #size = stimparams[1]
    #contrast = stimparams[4]
    angle = gratingInfo['Orientation'][()]
    size = gratingInfo['Size'][()]
    contrast = gratingInfo['Contrast'][()]
    
    ucontrast = np.unique(contrast)
    uangle = np.unique(angle)
    usize = np.unique(size)
    ncontrast = len(ucontrast)
    nangle = len(uangle)
    nsize = len(usize)
    
    angle180 = np.remainder(angle,180)
    uangle180 = np.unique(angle180)
    nangle180 = len(uangle180)
 
    Smean = np.zeros((data.shape[0],nangle180,nsize,ncontrast,data.shape[2]))
    Fmean = np.zeros((data.shape[0],nangle180,nsize,ncontrast,data.shape[2]))
    Smean_stat = np.zeros((data.shape[0],nangle180,nsize,ncontrast,data.shape[2]))
    Stavg = np.zeros((data.shape[0],nangle180,nsize,ncontrast,int(data.shape[1]/nangle/nsize/ncontrast)))
    
    Strials = {}
    Sspont = {}
    for i in range(nangle180):
        for j in range(nsize):
            for k in range(ncontrast):
                lkat = np.logical_and(runtrial,np.logical_and(angle180==uangle180[i],np.logical_and(size==usize[j],contrast==ucontrast[k])))
                Smean[:,i,j,k,:] = np.nanmean(data[:,lkat,:],1)
                Fmean[:,i,j,k,:] = np.nanmean(data_dfof[:,lkat,:],1)
                Strials[(i,j,k)] = np.nanmean(data[:,lkat,nbefore:-nafter],2)
                Sspont[(i,j,k)] = np.nanmean(data[:,lkat,:nbefore],2)
                stat = np.logical_and(np.logical_not(runtrial),np.logical_and(angle180==uangle180[i],np.logical_and(size==usize[j],contrast==ucontrast[k])))
                Smean_stat[:,i,j,k] = np.nanmean(data[:,stat],1)
 
    lb = np.zeros((strialwise.shape[0],nangle180,nsize,ncontrast))
    ub = np.zeros((strialwise.shape[0],nangle180,nsize,ncontrast))
    # mn = np.zeros((strialwise.shape[0],nangle180,nsize,ncontrast))
    
    for i in range(nangle180):
        print(i)
        for j in range(nsize):
            for k in range(ncontrast):
                if Strials[(i,j,k)].size:
                    lb[:,i,j,k],ub[:,i,j,k] = ut.bootstrap(Strials[(i,j,k)],np.mean,axis=1,pct=(16,84))
                else:
                    lb[:,i,j,k] = np.nan
                    ub[:,i,j,k] = np.nan
                # mn[:,i,j,k] = np.nanmean(Strials[(i,j,k)],axis=1)
    
    pval = np.zeros((strialwise.shape[0],nangle180))
#     for i in range(pval.shape[0]):
#         print(i)
    for j,theta in enumerate(uangle180):
        print(theta)
        #_,pval[:,j] = sst.ttest_rel(Sspont[(j,0,ncontrast-1)],Strials[(j,0,ncontrast-1)],axis=1)
        _,pval[:,j] = sst.ttest_ind(Strials[(j,0,0)],Strials[(j,0,ncontrast-1)],axis=1)
    #assert(np.logical_not(np.isnan(pval).sum()))            
    Savg = np.nanmean(np.nanmean(Smean[:,:,:,:,nbefore:-nafter],axis=-1),axis=1)
    Favg = np.nanmean(np.nanmean(Fmean[:,:,:,:,nbefore:-nafter],axis=-1),axis=1)
    
   # Storiavg = Stavg.mean(1)
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
    proc['trialwise'] = trialwise
    proc['strialwise'] = strialwise
    #proc['ctrialwise'] = ctrialwise
    proc['dtrialwise'] = dtrialwise
    proc['dfof'] = dfof
    proc['trialwise_t_offset'] = proc1['trialwise_t_offset']
    proc['raw_trialwise'] = proc1['raw_trialwise']
    proc['neuropil_trialwise'] = proc1['neuropil_trialwise']
    #proc['trialwise_t_offset'] = trialwise_t_offset
    #proc['straces'] = straces
    #proc['oriavg_dfof'] = Favg
    
    return Savg,Smean,lb,ub,pval,spont,Smean_stat,proc


def analyze_everything(folds=None,files=None,rets=None,adjust_fns=None,rgs=None,criteria=None,datafoldbase='/home/mossing/scratch/2Pdata/',stimfoldbase='/home/mossing/scratch/visual_stim/'):
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

        soriavg[thisfold],strialavg[thisfold],lb[thisfold],ub[thisfold],pval[thisfold],spont[thisfold],Smean_stat[thisfold],proc[thisfold] = analyze_size_contrast(datafiles,stimfile,retfile,frame_adjust=frame_adjust,rg=rg,nbefore=nbefore,nafter=nafter,criterion=criterion)
        nbydepth[thisfold] = get_nbydepth(datafiles)
        try: 
            ret_vars[thisfold] = sio.loadmat(retfile,squeeze_me=True)#['ret'][:]
            print('loaded retinotopy file')
            ret_vars[thisfold]['position'] = sio.loadmat(stimfile,squeeze_me=True)['result'][()]['position']
            proc[thisfold]['position'] = sio.loadmat(stimfile,squeeze_me=True)['result'][()]['position']
        except:
            print('retinotopy not saved for '+thisfile)
            ret_vars[thisfold] = None
            proc[thisfold]['position'] = sio.loadmat(stimfile,squeeze_me=True)['result'][()]['position']
    return soriavg,strialavg,lb,ub,pval,nbydepth,spont,ret_vars,Smean_stat,proc

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

def analyze_everything_by_criterion(folds=None,files=None,rets=None,adjust_fns=None,rgs=None,criteria=None,datafoldbase='/home/mossing/scratch/2Pdata/',stimfoldbase='/home/mossing/scratch/visual_stim/',criterion_cutoff=0.2,procname='size_contrast_proc.hdf5'):
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
    #proc = {}
    proc_hdf5_name = procname
    for thisfold,thisfile,retnumber,frame_adjust,rg,criterion,thisdatafoldbase,thisstimfoldbase in zip(folds,files,rets,adjust_fns,rgs,criteria,datafoldbase,stimfoldbase):

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
        proc = [None]*2

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
            soriavg[thisfold][k],_,_,_,_,_,_,proc[k] = analyze_size_contrast(datafiles,stimfile,retfile,frame_adjust=frame_adjust,rg=rg,nbefore=nbefore,nafter=nafter,criterion=thesecriteria[k],criterion_cutoff=criterion_cutoff) # changed 2/12/19
        nbydepth[thisfold] = get_nbydepth(datafiles)
        try: 
            ret_vars[thisfold] = sio.loadmat(retfile,squeeze_me=True)#['ret'][:]
            print('loaded retinotopy file')
            ret_vars[thisfold]['position'] = sio.loadmat(stimfile,squeeze_me=True)['result'][()]['position']
        except:
            print('retinotopy not saved for '+thisfile)
            ret_vars[thisfold] = None
    #return soriavg,strialavg,lb,ub,pval,nbydepth,spont,ret_vars,Smean_stat,proc
        if len(proc[0]):
            gdind = 0
        else:
            gdind = 1
        needed_ret_vars = ['position','pval_ret','ret']
        ret_dicti = {varname:ret_vars[thisfold][varname] for varname in needed_ret_vars}
        ret_dicti['paramdict_normal'] = ut.matfile_to_dict(ret_vars[thisfold]['paramdict_normal'])
        proc[gdind]['ret_vars'] = ret_dicti
        ut.dict_to_hdf5(proc_hdf5_name,session_id,proc[gdind])
    return soriavg,ret_vars # changed 2/12/19
    

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
        rf_sq_error = ret_vars[key]['paramdict_normal'][()]['sqerror']
        rf_distance_deg = np.sqrt(((rf_ctr-stim_offset[:,np.newaxis])**2).sum(0))
        rf_displacement_deg = rf_ctr-stim_offset[:,np.newaxis]
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
        data_struct[session_id]['rf_mapping_pval'] = ret_vars[key]['pval_ret'][:]
        data_struct[session_id]['rf_distance_deg'] = rf_distance_deg
        data_struct[session_id]['rf_displacement_deg'] = rf_displacement_deg
        data_struct[session_id]['rf_ctr'] = rf_ctr
        data_struct[session_id]['rf_sq_error'] = rf_sq_error
        data_struct[session_id]['running_speed_cm_s'] = running_speed_cm_s
    return data_struct


def gen_full_data_struct(cell_type='PyrL23', keylist=None, frame_rate_dict=None, proc=None, ret_vars=None, nbefore=8, nafter=8):
    data_struct = {}
    for key in keylist:
        if len(proc[key][0])>0:
            gdind = 0
        else:
            gdind = 1
        dfof = proc[key][gdind]['dtrialwise']
        decon = proc[key][gdind]['strialwise'] 
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
        data_struct[session_id]['rf_mapping_pval'] = ret_vars[key]['pval_ret']
        data_struct[session_id]['rf_distance_deg'] = rf_distance_deg
        data_struct[session_id]['rf_displacement_deg'] = rf_displacement_deg
        data_struct[session_id]['rf_ctr'] = rf_ctr
        data_struct[session_id]['running_speed_cm_s'] = running_speed_cm_s
        data_struct[session_id]['F'] = dfof
        data_struct[session_id]['raw_trialwise'] = proc[key][gdind]['raw_trialwise']
        data_struct[session_id]['neuropil_trialwise'] = proc[key][gdind]['neuropil_trialwise']
        data_struct[session_id]['decon'] = decon
        data_struct[session_id]['nbefore'] = nbefore
        data_struct[session_id]['nafter'] = nafter
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
            ucontrast,icontrast = np.unique(proc[key]['contrast'][:],return_inverse=True)
            usize,isize = np.unique(proc[key]['size'][:],return_inverse=True)
            uangle,iangle = np.unique(proc[key]['angle'][:],return_inverse=True)
            stimulus_id = np.concatenate((isize[np.newaxis],icontrast[np.newaxis],iangle[np.newaxis]),axis=0)
            stimulus_size_deg = usize
            stimulus_contrast = ucontrast
            stimulus_direction = uangle
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
            while 'size_contrast_'+str(exptno) in this_session.keys():
                exptno = exptno+1
            this_expt = this_session.create_group('size_contrast_'+str(exptno))
            this_expt.create_dataset('stimulus_id',data=stimulus_id)
            this_expt.create_dataset('stimulus_size_deg',data=stimulus_size_deg)
            this_expt.create_dataset('stimulus_contrast',data=stimulus_contrast)
            this_expt.create_dataset('stimulus_direction',data=stimulus_direction)
            this_expt['rf_mapping_pval'] = ret_vars[key]['pval_ret'][:]
            this_expt['rf_sq_error'] = ret_vars[key]['paramdict_normal']['sqerror'][:]
            sx = ret_vars[key]['paramdict_normal']['sigma_x'][:]
            sy = ret_vars[key]['paramdict_normal']['sigma_y'][:]
            this_expt['rf_sigma'] = np.sqrt(sx**2+sy**2)
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

def analyze_simply(folds=None,files=None,rets=None,adjust_fns=None,rgs=None,datafoldbase='/home/mossing/scratch/2Pdata/',stimfoldbase='/home/mossing/scratch/visual_stim/',procname='size_contrast_proc.hdf5'):
    if isinstance(datafoldbase,str):
        datafoldbase = [datafoldbase]*len(folds)
    if isinstance(stimfoldbase,str):
        stimfoldbase = [stimfoldbase]*len(folds)
    if os.path.exists(procname):
        os.remove(procname)
    stim_params = size_contrast_params_kludge()
    session_ids = []
    for thisfold,thisfile,frame_adjust,rg,thisdatafoldbase,thisstimfoldbase,retnumber in zip(folds,files,adjust_fns,rgs,datafoldbase,stimfoldbase,rets):

        session_id = at.gen_session_id(thisfold)
        datafiles = at.gen_datafiles(thisdatafoldbase,thisfold,thisfile,nplanes=4)
        stimfile = at.gen_stimfile(thisstimfoldbase,thisfold,thisfile)
        retfile = thisdatafoldbase+thisfold+'retinotopy_'+retnumber+'.mat'

        nbefore = 8
        nafter = 8

        proc = at.analyze(datafiles,stimfile,frame_adjust=frame_adjust,rg=rg,nbefore=nbefore,nafter=nafter,stim_params=stim_params)
        
    # now will get retinotopy info from retinotopy_0 data struct fields i/o from procfile
#        try:
#            proc['ret_vars'] = at.gen_ret_vars(retfile,stimfile)
#        except:
#            print('retinotopy not saved for ' + session_id)
        proc['position'] = sio.loadmat(stimfile,squeeze_me=True)['result'][()]['position']

        ut.dict_to_hdf5(procname,session_id,proc)
        session_ids.append(session_id)
    return session_ids

def size_contrast_params():
    paramlist = [('angle','Orientation'),('size','Size'),('contrast','Contrast')]
    params_and_fns = [None]*len(paramlist)
    for i,pair in enumerate(paramlist):
        print(pair[1])
        param = pair[0]
        function = lambda result: result['gratingInfo'][()][pair[1]][()]
        params_and_fns[i] = (param,function)
    return params_and_fns

def size_contrast_params_kludge():
    params_and_fns = [None]*3
    params_and_fns[0] = ('angle',lambda result: result['gratingInfo'][()]['Orientation'][()])
    params_and_fns[1] = ('size',lambda result: result['gratingInfo'][()]['Size'][()])
    params_and_fns[2] = ('contrast',lambda result: result['gratingInfo'][()]['Contrast'][()])
    return params_and_fns

def add_data_struct_h5_simply(filename, cell_type='PyrL23', keylist=None, frame_rate_dict=None, proc=None, nbefore=8, nafter=8):
    groupname = 'size_contrast'
    featurenames=['size','contrast','angle']
    datasetnames = ['stimulus_size_deg','stimulus_contrast','stimulus_direction_deg']
    grouplist = at.add_data_struct_h5(filename,cell_type=cell_type,keylist=keylist,frame_rate_dict=frame_rate_dict,proc=proc,nbefore=nbefore,nafter=nafter,featurenames=featurenames,datasetnames=datasetnames,groupname=groupname)

    at.add_ret_to_data_struct(filename,keylist=keylist,proc=proc,grouplist=grouplist)
    return grouplist

def show_size_contrast(arr,show_labels=True,usize=np.array((5,8,13,22,36,60)),ucontrast=np.array((0,6,12,25,50,100)),vmin=None,vmax=None,flipud=False,cmap=plt.cm.viridis):
    this_usize = [str(int(np.floor(this))) for this in usize]
    this_ucontrast = [str(int(np.floor(this))) for this in ucontrast]
    nsize = len(usize)
    ncontrast = len(ucontrast)
    to_show = arr.copy()
    if flipud:
        to_show = np.flipud(to_show)
    plt.imshow(to_show,vmin=vmin,vmax=vmax,extent=[-0.5,ncontrast-0.5,-0.5,nsize-0.5],cmap=cmap)
    plt.xticks(np.arange(ncontrast),this_ucontrast)
    if flipud:
        plt.yticks(np.arange(nsize),this_usize)
    else:
        plt.yticks(np.arange(nsize)[::-1],this_usize)
    if show_labels:
        plt.xlabel('contrast (%)')
        plt.ylabel('size ($^o$)')

def scatter_size_contrast(y1,y2,nsize=5,ncontrast=6,alpha=1,equality_line=True,square=True,equate_0=False,dot_scale=10,colormap=plt.cm.viridis,mn=None,mx=None):
    if len(y1.shape)==2:
        nsize,ncontrast = y1.shape
    z = [y.reshape((nsize,ncontrast)) for y in [y1,y2]]
    if mn is None:
        mn = np.minimum(np.nanmin(y1),np.nanmin(y2))
    if mx is None:
        mx = np.maximum(np.nanmax(y1),np.nanmax(y2))
    colors = colormap(np.linspace(0,1,ncontrast))
    if equate_0:
        zero = [np.nanmean(z[idim][:,0]) for idim in range(2)]
        zero_color = colors[0:1]
        z = [z[idim][:,1:] for idim in range(2)]
        colors = colors[1:]
    if equate_0:
        plt.scatter(zero[0],zero[1],c=zero_color,s=(nsize+1)*dot_scale,alpha=alpha,edgecolors='k',linewidths=1)
    for s in range(nsize):
        plt.scatter(z[0][s],z[1][s],c=colors,s=(s+1)*dot_scale,alpha=alpha,edgecolors='k',linewidths=1)
    if equality_line:
        plt.plot((mn,mx),(mn,mx),c='k')
    if square:
        wiggle = 0.05*(mx-mn)
        plt.xlim((mn-wiggle,mx+wiggle))
        plt.ylim((mn-wiggle,mx+wiggle))

def scatter3d_size_contrast(y1,y2,y3,nsize=5,ncontrast=6,alpha=1,equality_line=True,square=True,equate_0=False,dot_scale=10,colormap=plt.cm.viridis,mn=None,mx=None):
    ax = plt.gcf().add_subplot(projection='3d')
    ndim = 3
    if len(y1.shape)==2:
        nsize,ncontrast = y1.shape
    z = [y.reshape((nsize,ncontrast)) for y in [y1,y2,y3]]
    if mn is None:
        mn = np.minimum(np.nanmin(y1),np.nanmin(y2))
    if mx is None:
        mx = np.maximum(np.nanmax(y1),np.nanmax(y2))
    colors = colormap(np.linspace(0,1,ncontrast))
    if equate_0:
        zero = [np.nanmean(z[idim][:,0]) for idim in range(ndim)]
        zero_color = colors[0:1]
        z = [z[idim][:,1:] for idim in range(ndim)]
        colors = colors[1:]
    if equate_0:
        ax.scatter(zero[0],zero[1],zero[2],c=zero_color,s=(nsize+1)*dot_scale,alpha=alpha,edgecolors='k',linewidths=1)
    for s in range(nsize):
        ax.scatter(z[0][s],z[1][s],z[2][s],c=colors,s=(s+1)*dot_scale,alpha=alpha,edgecolors='k',linewidths=1)
    if equality_line:
        ax.plot((mn,mx),(mn,mx),(mn,mx),c='k')
    if square:
        wiggle = 0.05*(mx-mn)
        ax.set_xlim((mn-wiggle,mx+wiggle))
        ax.set_ylim((mn-wiggle,mx+wiggle))
        ax.set_zlim((mn-wiggle,mx+wiggle))

def scatter_size_contrast_errorbar(x,y,equality_line=True,square=True,equate_0=False,nsize=5,ncontrast=6,dot_scale=10,colormap=plt.cm.viridis,mn=None,mx=None,alpha=1):
    def compute_mean_sem(x):
        xmean = np.nanmean(x,0)
        n_non_nan = np.sum(~np.isnan(x),0)
        xsem = np.nanstd(x,0)/np.sqrt(n_non_nan)
        return xmean,xsem
    def compute_mean_sem_zero_nonzero(x):
        xmean,xsem = compute_mean_sem(x)
        lkat = ~np.isnan(xsem[:,0])
        xmean[lkat,0] = np.nanmean(xmean[lkat,0])
        xsem[lkat,0] = np.sqrt(np.nansum(xsem[lkat,0]**2))/np.sum(lkat)
        return xmean,xsem
    if equate_0:
        xmean,xsem = compute_mean_sem_zero_nonzero(x)
        ymean,ysem = compute_mean_sem_zero_nonzero(y)
    else:
        xmean,xsem = compute_mean_sem(x)
        ymean,ysem = compute_mean_sem(y)
    plt.errorbar(xmean.flatten(),ymean.flatten(),yerr=ysem.flatten(),xerr=xsem.flatten(),fmt='none',c='k',zorder=1,alpha=alpha)
    scatter_size_contrast(xmean,ymean,equality_line=equality_line,square=square,equate_0=equate_0,nsize=nsize,ncontrast=ncontrast,dot_scale=dot_scale,colormap=colormap,mn=mn,mx=mx,alpha=alpha)

def scatter_size_contrast_pct_errorbar(x,y,equality_line=True,square=True,equate_0=False,nsize=5,ncontrast=6,dot_scale=10,colormap=plt.cm.viridis,mn=None,mx=None,alpha=1):
    def compute_mean_sem(x):
        xmean = np.nanmean(x,0)
        n_non_nan = np.sum(~np.isnan(x),0)
        xsem = np.nanstd(x,0)#/np.sqrt(n_non_nan)
        return xmean,xsem
    def compute_mean_sem_zero_nonzero(x):
        xmean,xsem = compute_mean_sem(x)
        lkat = ~np.isnan(xsem[:,0])
        xmean[lkat,0] = np.nanmean(xmean[lkat,0])
        xsem[lkat,0] = np.sqrt(np.nansum(xsem[lkat,0]**2))/np.sum(lkat)
        return xmean,xsem
    if equate_0:
        xmean,xsem = compute_mean_sem_zero_nonzero(x)
        ymean,ysem = compute_mean_sem_zero_nonzero(y)
    else:
        xmean,xsem = compute_mean_sem(x)
        ymean,ysem = compute_mean_sem(y)
    plt.errorbar(xmean.flatten(),ymean.flatten(),yerr=ysem.flatten(),xerr=xsem.flatten(),fmt='none',c='k',zorder=1)
    scatter_size_contrast(xmean,ymean,equality_line=equality_line,square=square,equate_0=equate_0,nsize=nsize,ncontrast=ncontrast,dot_scale=dot_scale,colormap=colormap,mn=mn,mx=mx,alpha=alpha)

def scatter_size_contrast_diff_errorbar(x1,x2,y,equality_line=True,square=True,equate_0=False,nsize=5,ncontrast=6,dot_scale=10,colormap=plt.cm.viridis,mn=None,mx=None):
    def compute_mean_sem(x):
        xmean = np.nanmean(x,0)
        n_non_nan = np.sum(~np.isnan(x),0)
        xsem = np.nanstd(x,0)/np.sqrt(n_non_nan)
        return xmean,xsem
    def compute_mean_sem_zero_nonzero(x):
        xmean,xsem = compute_mean_sem(x)
        lkat = ~np.isnan(xsem[:,0])
        xmean[lkat,0] = np.nanmean(xmean[lkat,0])
        xsem[lkat,0] = np.sqrt(np.nansum(xsem[lkat,0]**2))/np.sum(lkat)
        return xmean,xsem
    if equate_0:
        x1mean,x1sem = compute_mean_sem_zero_nonzero(x1)
        x2mean,x2sem = compute_mean_sem_zero_nonzero(x2)
        ymean,ysem = compute_mean_sem_zero_nonzero(y)
    else:
        x1mean,x1sem = compute_mean_sem(x1)
        x2mean,x2sem = compute_mean_sem(x2)
        ymean,ysem = compute_mean_sem(y)
    plt.errorbar((x1mean-x2mean).flatten(),ymean.flatten(),yerr=ysem.flatten(),xerr=np.sqrt(x1sem**2+x2sem**2).flatten(),fmt='none',c='k',zorder=1)
    scatter_size_contrast(x1mean-x2mean,ymean,equality_line=equality_line,square=square,equate_0=equate_0,nsize=nsize,ncontrast=ncontrast,dot_scale=dot_scale,colormap=colormap,mn=mn,mx=mx)

def compute_encoding_axes(dsname,expttype='size_contrast_0',cutoffs=(20,),alphas=np.logspace(-2,2,50),running_trials=False,training_set=None,equate_ori=False):
    # compute encoding axes based on training set trials
    na = len(alphas)
    with ut.hdf5read(dsname) as ds:
        keylist = list(ds.keys())
        nkey = len(keylist)
        R = [None for k in range(nkey)]
        reg = [None for k in range(nkey)]
        top_score = [None for k in range(nkey)]
        proc = [{} for k in range(len(keylist))]
        for k in range(len(keylist)):
            R[k] = [None for icutoff in range(len(cutoffs))]
            reg[k] = [None for icutoff in range(len(cutoffs))]
            sc0 = ds[keylist[k]][expttype]
            nbefore = sc0['nbefore'][()]
            nafter = sc0['nafter'][()]
            decon = np.nanmean(sc0['decon'][()][:,:,nbefore:-nafter],-1)
            data = sst.zscore(decon,axis=1)
            data[np.isnan(data)] = 0
            if training_set is None:
                train = np.ones((decon.shape[1],),dtype='bool')
            else:
                if isinstance(training_set[keylist[k]],list):
                    train = training_set[keylist[k]][0]
                else:
                    train = training_set[keylist[k]]

            u,sigma,v = np.linalg.svd(data)

            pval_ret = sc0['rf_mapping_pval'][()]
            dist_ret = sc0['rf_distance_deg'][()]
            ontarget_ret_lax = np.logical_and(dist_ret<40,pval_ret<0.05)
            running_speed_cm_s = sc0['running_speed_cm_s'][()]
            running = np.nanmean(running_speed_cm_s[:,nbefore:-nafter],1)>7
            if not running_trials:
                running = ~running
            size = sc0['stimulus_id'][()][0]
            contrast = sc0['stimulus_id'][()][1]
            angle = sc0['stimulus_id'][()][-1]
            if equate_ori:
                angle = np.remainder(angle,4).astype('int')

            proc[k]['u'] = u
            proc[k]['sigma'] = sigma
            proc[k]['v'] = v  
            proc[k]['pval_ret'] = pval_ret
            proc[k]['dist_ret'] = dist_ret
            proc[k]['ontarget_ret_lax'] = ontarget_ret_lax
            proc[k]['running_speed_cm_s'] = running_speed_cm_s
            proc[k]['running'] = running
            proc[k]['size'] = size
            proc[k]['contrast'] = contrast
            proc[k]['angle'] = angle
            proc[k]['cutoffs'] = cutoffs
            proc[k]['train'] = train

            uangle,usize,ucontrast = [sc0[key][()] for key in ['stimulus_direction_deg','stimulus_size_deg','stimulus_contrast']]

            proc[k]['uangle'],proc[k]['usize'],proc[k]['ucontrast'] = uangle,usize,ucontrast
            if equate_ori:
                proc[k]['uangle'] = proc[k]['uangle'][:int(len(proc[k]['uangle'])/2)]

            if np.logical_and(ontarget_ret_lax.sum()>100,running.mean()>0.5):
                uangle,usize,ucontrast = [np.unique(arr) for arr in [angle,size,contrast]]
                nsize = len(usize)
                ncontrast = len(ucontrast)
                nangle = len(uangle)
                top_score[k] = np.zeros((len(cutoffs),nsize,nangle))
                for icutoff,cutoff in enumerate(cutoffs):
                    R[k][icutoff] = np.zeros((nsize,nangle,cutoff))
                    reg[k][icutoff] = [None for s in range(nsize)]
                    for s in range(nsize):
                        reg[k][icutoff][s] = [None for i in range(nangle)]
                        for i in range(nangle):
                            stim_of_interest_all_contrast = ut.k_and(np.logical_or(np.logical_and(angle==i,size==s),contrast==0),running,train) #,eye_dist < np.nanpercentile(eye_dist,50))
                            X = (np.diag(sigma[:cutoff]) @ v[:cutoff,:]).T[stim_of_interest_all_contrast]
                            y = contrast[stim_of_interest_all_contrast] #>0

                            sc = np.zeros((na,))
                            for ia,alpha in enumerate(alphas):
                                linreg = sklearn.linear_model.Ridge(alpha=alpha,normalize=True)
                                reg1 = linreg.fit(X,y)
                                scores = sklearn.model_selection.cross_validate(linreg,X,y,cv=5)
                                pred = sklearn.model_selection.cross_val_predict(linreg,X,y,cv=5)
                                sc[ia] = scores['test_score'].mean()
                            best_alpha = np.argmax(sc)
                            top_score[k][icutoff,s,i] = sc.max()
                            linreg = sklearn.linear_model.Ridge(alpha=alphas[best_alpha],normalize=True)
                            reg[k][icutoff][s][i] = linreg.fit(X,y)
    return reg,proc,top_score

def compute_encoding_axis_auroc(reg,proc):
    return compute_encoding_axis_fn(reg,proc,ut.compute_auroc)

def hit_from_fa(y0,y1,fa):
    return np.nanmean(y1 > np.nanpercentile(y0,100*(1-fa)))

def compute_encoding_axis_hit_from_fa(reg,proc,fa=0.1):
    return compute_encoding_axis_fn(reg,proc,lambda x,y: hit_from_fa(x,y,fa))

def compute_encoding_axis_fn(reg,proc,fn):

    # compute auroc based on test set trials: take population activity vectors for 0 and target contrast, use regression
    # trained on training set to predict contrast for both sets of population activity vectors, compute AUROC of predicted contrasts for both

    auroc = [None for k in range(len(proc))]
    uangle,usize,ucontrast = [[None for k in range(len(proc))] for iparam in range(3)]
    icutoff = 0
    
    for iexpt in range(len(proc)):
        if not reg[iexpt][icutoff] is None:
            cutoff = proc[iexpt]['cutoffs'][icutoff]
            desired_outputs = ['angle','size','contrast','running','sigma','v','uangle','usize','ucontrast','train']
            angle,size,contrast,running,sigma,v,uangle[iexpt],usize[iexpt],ucontrast[iexpt],train = [proc[iexpt][output].copy() for output in desired_outputs]
            zero_contrast = ut.k_and(contrast==0,running) #,eye_dist < np.nanpercentile(eye_dist,50))
            nsize = len(usize[iexpt])
            ncontrast = len(ucontrast[iexpt])
            nangle = len(uangle[iexpt])
            auroc[iexpt] = np.zeros((nsize,ncontrast,nangle))
            for isize in range(nsize):
                for icontrast in range(ncontrast):
                    for iangle in range(nangle):
                        this_contrast = ut.k_and(angle==iangle,size==isize,contrast==icontrast,running,~train) #,eye_dist < np.nanpercentile(eye_dist,50))
                        if this_contrast.sum():
                            X0 = (np.diag(sigma[:cutoff]) @ v[:cutoff,:]).T[zero_contrast]
                            X1 = (np.diag(sigma[:cutoff]) @ v[:cutoff,:]).T[this_contrast]
                            # this_model = reg[iexpt][icutoff][isize][iangle].copy()
                            y0 = reg[iexpt][icutoff][isize][iangle].predict(X0)
                            y1 = reg[iexpt][icutoff][isize][iangle].predict(X1)
                            auroc[iexpt][isize,icontrast,iangle] = fn(y0,y1)
                        else:
                            auroc[iexpt][isize,icontrast,iangle] = np.nan
    return auroc

def compute_encoding_axis_projections(reg,proc):

    # compute auroc based on test set trials: take population activity vectors for 0 and target contrast, use regression
    # trained on training set to predict contrast for both sets of population activity vectors, compute AUROC of predicted contrasts for both

    auroc = {}#[None for k in range(len(proc))]
    uangle,usize,ucontrast = [[None for k in range(len(proc))] for iparam in range(3)]
    icutoff = 0
    
    for iexpt in range(len(proc)):
        if not reg[iexpt][icutoff] is None:
            cutoff = proc[iexpt]['cutoffs'][icutoff]
            desired_outputs = ['angle','size','contrast','running','sigma','v','uangle','usize','ucontrast','train']
            angle,size,contrast,running,sigma,v,uangle[iexpt],usize[iexpt],ucontrast[iexpt],train = [proc[iexpt][output].copy() for output in desired_outputs]
            zero_contrast = ut.k_and(contrast==0,running) #,eye_dist < np.nanpercentile(eye_dist,50))
            nsize = len(usize[iexpt])
            ncontrast = len(ucontrast[iexpt])
            nangle = len(uangle[iexpt])
            for isize in range(nsize):
                for icontrast in range(ncontrast):
                    for iangle in range(nangle):
                        if ucontrast[iexpt][icontrast]==0:
                            this_contrast = zero_contrast.copy()
                        else:
                            this_contrast = ut.k_and(angle==iangle,size==isize,contrast==icontrast,running,~train) #,eye_dist < np.nanpercentile(eye_dist,50))
                        if this_contrast.sum():
                            X1 = (np.diag(sigma[:cutoff]) @ v[:cutoff,:]).T[this_contrast]
                            y1 = reg[iexpt][icutoff][isize][iangle].predict(X1)
                            auroc[(iexpt,isize,icontrast,iangle)] = y1
                        else:
                            auroc[(iexpt,isize,icontrast,iangle)] = np.array(())
    return auroc

def show_auroc(auroc,usize=None,ucontrast=None,label='Population decoder detection AUROC'):
    show_size_contrast(auroc[:,1:],flipud=True,usize=usize,ucontrast=ucontrast[1:])
    plt.colorbar().set_label(label)
    plt.clim([0.5,1])
    
# def naka_rushton(c,params):
#     a = params[0]
#     b = params[1]
#     c50 = params[2]
#     n = params[3]
#     return (a*(c/c50)**n + b)/(1+(c/c50)**n)

def fit_opt_params_(c,r):
    if np.all(~np.isnan(r)):
        a_0 = r.max()
        b_0 = 1e-12
        c50_0 = 50
        n_0 = 1
        params_0 = np.array((a_0,b_0,c50_0,n_0))
        params_opt = sop.least_squares(lambda params: r-nra.naka_rushton(c,params),params_0,bounds=((0,0,0,1-1e-12),(np.inf,np.inf,500,1+1e-12)))
        return params_opt['x']
    else:
        return None
        
def fit_crfs(arr,cvals,contrast_axis=2):
    popt = np.zeros(arr.shape[0:contrast_axis]+(4,))
    for iroi in range(arr.shape[0]):
        for isize in range(arr.shape[1]):
            popt[iroi,isize] = fit_opt_params_(cvals,arr[iroi,isize])
    return popt #np.diff(arr,axis=contrast_axis)

def compute_size_contrast_deriv(arr,popt=None,cvals=np.array([0,6,12,25,50,100])):
    
    if popt is None:
        popt = fit_crfs(arr,cvals,contrast_axis=2)

    nr_contrast_deriv = np.zeros(arr.shape)
    for iroi in range(nr_contrast_deriv.shape[0]):
        for isize in range(nr_contrast_deriv.shape[1]):
            params = popt[iroi,isize]
            nr_contrast_deriv[iroi,isize] = egrad(lambda c: nra.naka_rushton(c,params))(cvals)

    modeled = np.array([[nra.naka_rushton(cvals,popt[iroi,isize]) for isize in range(popt.shape[1])] for iroi in range(popt.shape[0])])
    nr_size_slope = np.nanmean(np.abs(np.diff(modeled,axis=1)),axis=0)
    nr_size_deriv = np.zeros((nr_size_slope.shape[0]+1,nr_size_slope.shape[1]))
    nr_size_deriv[0] = nr_size_slope[0]
    nr_size_deriv[-1] = nr_size_slope[-1]
    nr_size_deriv[1:-1] = 0.5*(nr_size_slope[1:]+nr_size_slope[:-1])

    return popt,nr_contrast_deriv,nr_size_deriv

def plot_bootstrapped_size_tuning(data,icontrasts=np.arange(6),colors=None,usize=np.array((5,8,13,22,36,60))):
    if colors is None:
        colors = plt.cm.viridis(np.linspace(0,1,len(icontrasts)))
    size_tuning = sim_utils.gen_size_tuning(data)    
    usize0 = np.concatenate(((0,),usize))
    ut.plot_bootstrapped_errorbars_hillel(usize0,size_tuning[:,:,icontrasts].transpose((0,2,1)),pct=(16,84),colors=colors)

def symmetrize(arr):
    return np.concatenate((arr,arr[0:1]))

def plot_size_contrast_ori(arr):
    nsize,ncontrast,nangle = arr.shape
    for i in range(nsize):
        for j in range(ncontrast):
            plt.subplot(nsize,ncontrast,i*ncontrast+j+1)
            plt.fill_between(np.arange(nangle+1),symmetrize(arr[i,j]),alpha=0.5)
            plt.plot(np.arange(nangle+1),symmetrize(arr[i,j]))
            plt.ylim(arr.min(),1.1*arr.max())
            plt.axis('off')

def compute_diff_avg(zdata,axis=0):
    zdata_diff = np.diff(zdata,axis=axis)
    zdata_diff_avg = np.zeros_like(zdata)
    slicer = [slice(None) for idim in range(len(zdata.shape))]
    for axind in [0,-1]:
        slicer[axis] = axind
        zdata_diff_avg[slicer] = zdata_diff[slicer]
    slicer1 = slicer.copy()
    slicer1[axis] = slice(1,-1)
    slicer2 = slicer.copy()
    slicer2[axis] = slice(None,-1)
    slicer3 = slicer.copy()
    slicer3[axis] = slice(1,None)
    zdata_diff_avg[slicer1] = 0.5*zdata_diff[slicer2]+0.5*zdata_diff[slicer3]
    return zdata_diff_avg

def compute_slope_avg(tdata,zdata,axis=0):
    slicer = [np.newaxis for idim in range(len(zdata.shape))]
    slicer[axis] = slice(None)
    zdata_slope = np.diff(zdata,axis=axis)/np.diff(tdata)[slicer]
    zdata_slope_avg = np.zeros_like(zdata)
    slicer = [slice(None) for idim in range(len(zdata.shape))]
    for axind in [0,-1]:
        slicer[axis] = axind
        zdata_slope_avg[slicer] = zdata_slope[slicer]
    slicer1 = slicer.copy()
    slicer1[axis] = slice(1,-1)
    slicer2 = slicer.copy()
    slicer2[axis] = slice(None,-1)
    slicer3 = slicer.copy()
    slicer3[axis] = slice(1,None)
    zdata_slope_avg[slicer1] = 0.5*zdata_slope[slicer2]+0.5*zdata_slope[slicer3]
    return zdata_slope_avg
