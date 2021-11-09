#!/usr/bin/env python

import numpy as np
import pandas as pd
import h5py
import pyute as ut
from importlib import reload
reload(ut)
import pdb
import matplotlib.pyplot as plt
import nub_utils as utils
reload(utils)
import sklearn
import scipy.stats as sst
import sklearn.cluster as skc
import matplotlib.colors as mc
import matplotlib.patches as ptch
import matplotlib.collections as coll
import scipy.optimize as sop

keylist_v1_l23 = ['session_191108_M0403','session_191119_M0293','session_200311_M0403','session_200312_M0293','session_200312_M0807']

# movement condition: max velocity during trial > 1 cm/sec
move_fn = lambda x: np.nanmax(np.abs(x[:,8:-8]),axis=-1)>1
# running condition: mean velocity during trial > 10 cm/sec
run_fn = lambda x: np.nanmean(np.abs(x[:,8:-8]),axis=-1)>10

# pupil dilated condition: pupil area > 2% of eye mask area
dilation_fn = lambda x: np.nanmedian(x[:,8:-8],axis=-1)>0.02

# reordering stimuli to convert between the geometry of V1 stimulus labels, and geometry of S1 stimulus labels
order1 = np.argsort((utils.nubs_active*np.array((16,1,4,8,2))[np.newaxis]).sum(1),kind='stable')
order2 = np.argsort(utils.nubs_active[order1][::-1].sum(1),kind='stable')[::-1]
# order of stimuli for tuning curve display purposes
evan_order_actual = order1[order2]
# order of stimuli for stimulus identity display purposes
evan_order_apparent = np.argsort(utils.nubs_active[::-1].sum(1),kind='stable')[::-1]
nub_no = utils.nubs_active[evan_order_actual].sum(1)
#parula = ListedColormap(ut.loadmat('/Users/dan/Documents/code/adesnal/matlab_parula_colormap.mat','cmap'))

# similar to parula colormap, ported to python
parula_path = '/Users/dan/Documents/code/adesnal/'
parula_filename = parula_path+'matlab_parula_colormap.mat'
parula = mc.ListedColormap(ut.loadmat(parula_filename,'cmap'))

def compute_df(dsname,keylist,datafield,run_fn,dilation_fn=dilation_fn,gen_nub_selector=utils.gen_nub_selector_v1,trialwise_dfof=False,seed=0,include_all=False,train_test_mode='halves'):
    # return list of two (running/nonrunning) lists of nexpt lists of two (training/test) tuning curves (nroi x 32)
    # extract pandas dataframe from hdf5 file
    df, roi_info, trial_info = ut.compute_tavg_dataframe(dsname,'nub_0',datafield=datafield,keylist=keylist,run_fn=run_fn,dilation_fn=dilation_fn,trialwise_dfof=trialwise_dfof)
    return df, roi_info, trial_info

def compute_tuning(dsname,keylist,datafield,run_fn,dilation_fn=dilation_fn,gen_nub_selector=utils.gen_nub_selector_v1,trialwise_dfof=False,seed=0,include_all=False,train_test_mode='halves'):
    # return list of two (running/nonrunning) lists of nexpt lists of two (training/test) tuning curves (nroi x 32)
    # extract pandas dataframe from hdf5 file
    df, roi_info, trial_info = ut.compute_tavg_dataframe(dsname,'nub_0',datafield=datafield,keylist=keylist,run_fn=run_fn,dilation_fn=dilation_fn,trialwise_dfof=trialwise_dfof)
    keylist = list(roi_info.keys())
    print(list(trial_info.keys()))
    
    # separate trials into training set and test set
    train_test = [None for irun in range(2)]
    for irun in range(2):
        selector = gen_nub_selector(run=irun,dilated=1,centered=1)
        print(list(selector.keys()))
        if train_test_mode=='halves':
            training_frac = 0.5
        elif train_test_mode=='thirds':
            training_frac = np.array((0.33,0.33))
        elif train_test_mode=='quarters':
            training_frac = np.array((0.25,0.25))
        #print('training frac: '+str(training_frac))
        train_test[irun] = utils.select_trials(trial_info,selector,training_frac,seed=seed,include_all=include_all)
        
    # compute tuning curves by averaging across trials
    tuning = [None for irun in range(2)]
    for irun in range(2):
        tuning[irun] = utils.compute_tuning(df,trial_info,selector,include=train_test[irun])
        
    return tuning

def compute_tuning_after_blank(dsname,keylist,datafield,run_fn,dilation_fn=dilation_fn,gen_nub_selector=utils.gen_nub_selector_v1,trialwise_dfof=False,seed=0,include_all=False):
    # return list of two (running/nonrunning) lists of nexpt lists of two (training/test) tuning curves (nroi x 32)
    # extract pandas dataframe from hdf5 file
    df, roi_info, trial_info = ut.compute_tavg_dataframe(dsname,'nub_0',datafield=datafield,keylist=keylist,run_fn=run_fn,dilation_fn=dilation_fn,trialwise_dfof=trialwise_dfof)
    keylist = list(roi_info.keys())
    print(list(trial_info.keys()))
    for key in trial_info:
        na = trial_info[key]['stimulus_nubs_active']
        trial_info[key]['after_blank'] = np.zeros(na.shape,dtype='bool')
        trial_info[key]['after_blank'][1:] = (na[:-1]==0)
    print('after blank: ' + str(trial_info[key]['after_blank'].mean()))
    
    # separate trials into training set and test set
    train_test = [None for irun in range(2)]
    for irun in range(2):
        selector = gen_nub_selector(run=irun,dilated=1,centered=1)
        #print(list(selector.keys()))
        train_test[irun] = utils.select_trials(trial_info,selector,0.5,seed=seed,include_all=include_all)
        for ipart in range(3):
            print('partition %d: frac. after blank %.02f'%(ipart,trial_info[key]['after_blank'][train_test[irun][key][ipart]].mean()))
            #print(list(trial_info[key].keys()))
            print('partition %d: frac. running %.02f'%(ipart,trial_info[key]['running'][train_test[irun][key][ipart]].mean()))
        
    # compute tuning curves by averaging across trials
    tuning = [None for irun in range(2)]
    for irun in range(2):
        tuning[irun] = utils.compute_tuning(df,trial_info,selector,include=train_test[irun])
        
    return tuning

def compute_tunings_simple(dsname,keylist,datafield,run_fn,dilation_fn=dilation_fn,gen_nub_selector=utils.gen_nub_selector_v1,trialwise_dfof=False,seed=0,npartitioning=10):
    # return list of two (running/nonrunning) lists of nexpt lists of two (training/test) tuning curves (nroi x 32)
    # extract pandas dataframe from hdf5 file
    df, roi_info, trial_info = ut.compute_tavg_dataframe(dsname,'nub_0',datafield=datafield,keylist=keylist,run_fn=run_fn,dilation_fn=dilation_fn,trialwise_dfof=trialwise_dfof)
    keylist = list(roi_info.keys())
    print(list(trial_info.keys()))
    
    # separate trials into training set and test set
    train_test = [None for irun in range(2)]
    for irun in range(2):
        train_test[irun] = [None for ipartitioning in range(npartitioning)]
        selector = gen_nub_selector(run=irun,dilated=1,centered=1)
        for ipartitioning in range(npartitioning):
            print(list(selector.keys()))
            train_test[irun][ipartitioning] = utils.select_trials(trial_info,selector,0.5,seed=seed+ipartitioning)
        
    # compute tuning curves by averaging across trials
    tuning = [None for irun in range(2)]
    for irun in range(2):
        tuning[irun] = [None for ipartitioning in range(npartitioning)]
        for ipartitioning in range(npartitioning):
            tuning[irun][ipartitioning] = utils.compute_tuning(df,trial_info,selector,include=train_test[irun][ipartitioning])
        
    return tuning

###
# not used in the final analysis
def compute_bounds_faster(dsname,keylist,datafield,run_fn=None,gen_nub_selector=utils.gen_nub_selector_v1):
    if run_fn is None:
        run_fn = move_fn
    df, roi_info, trial_info = ut.compute_tavg_dataframe(dsname,'nub_0',datafield='decon',keylist=keylist,run_fn=run_fn)
    keylist = list(roi_info.keys())
    
    bounds = [None for irun in range(2)]
    for irun in range(2):
        selector = gen_nub_selector(run=irun)
        #bounds[irun] = utils.compute_bootstrap_error(df,trial_info,selector,pct=(2.5,97.5,50))
        bounds[irun] = utils.compute_bootstrap_error_faster(df,trial_info,selector,pct=(2.5,97.5,50))
        
    return bounds

# ##
#def compute_tuning(dsname,keylist,datafield,run_fn,dilation_fn=dilation_fn,gen_nub_selector=utils.gen_nub_selector_v1,trialwise_dfof=False,seed=0,include_all=False,train_test_mode='halves'):
#    # return list of two (running/nonrunning) lists of nexpt lists of two (training/test) tuning curves (nroi x 32)
#    # extract pandas dataframe from hdf5 file
#    df, roi_info, trial_info = ut.compute_tavg_dataframe(dsname,'nub_0',datafield=datafield,keylist=keylist,run_fn=run_fn,dilation_fn=dilation_fn,trialwise_dfof=trialwise_dfof)
#    keylist = list(roi_info.keys())
#    print(list(trial_info.keys()))
#    
#    # separate trials into training set and test set
#    train_test = [None for irun in range(2)]
#    for irun in range(2):
#        selector = gen_nub_selector(run=irun,dilated=1,centered=1)
#        print(list(selector.keys()))
#        if train_test_mode=='halves':
#            training_frac = 0.5
#        elif train_test_mode=='thirds':
#            training_frac = np.array((0.33,0.33))
#        elif train_test_mode=='quarters':
#            training_frac = np.array((0.25,0.25))
#        #print('training frac: '+str(training_frac))
#        train_test[irun] = utils.select_trials(trial_info,selector,training_frac,seed=seed,include_all=include_all)
#        
#    # compute tuning curves by averaging across trials
#    tuning = [None for irun in range(2)]
#    for irun in range(2):
#        tuning[irun] = utils.compute_tuning(df,trial_info,selector,include=train_test[irun])
#        
#    return tuning

#def compute_bounds(dsname,keylist,datafield,run_fn=None,gen_nub_selector=utils.gen_nub_selector_v1,pct=(2.5,97.5,50)):
def compute_bounds(dsname,keylist,datafield,run_fn=None,dilation_fn=dilation_fn,gen_nub_selector=utils.gen_nub_selector_v1,trialwise_dfof=False,seed=0,include_all=True,train_test_mode='halves',pct=(2.5,97.5,50),nreps=1000):
    # bounds: list of lists, running x expt x partition x pct (percentile) x (roi x size? x stim)
    # may need to tweak to reproduce previous plots
    # compute bootstrapped errorbars given by the percentiles in 'pct'
    # if not otherwise specified, split trials into moving and nonmoving
    if run_fn is None:
        run_fn = move_fn
    #df, roi_info, trial_info = ut.compute_tavg_dataframe(dsname,'nub_0',datafield='decon',keylist=keylist,run_fn=run_fn)
    #keylist = list(roi_info.keys())
    df, roi_info, trial_info = ut.compute_tavg_dataframe(dsname,'nub_0',datafield=datafield,keylist=keylist,run_fn=run_fn,dilation_fn=dilation_fn,trialwise_dfof=trialwise_dfof)
    keylist = list(roi_info.keys())
    print(list(trial_info.keys()))
    
    # separate trials into training set and test set
    train_test = [None for irun in range(2)]
    for irun in range(2):
        selector = gen_nub_selector(run=irun,dilated=1,centered=1)
        print(list(selector.keys()))
        if train_test_mode=='halves':
            training_frac = 0.5
        elif train_test_mode=='thirds':
            training_frac = np.array((0.33,0.33))
        elif train_test_mode=='quarters':
            training_frac = np.array((0.25,0.25))
        #print('training frac: '+str(training_frac))
        train_test[irun] = utils.select_trials(trial_info,selector,training_frac,seed=seed,include_all=include_all)
    
    bounds = [None for irun in range(2)]
    for irun in range(2):
        selector = gen_nub_selector(run=irun)
        #bounds[irun] = utils.compute_bootstrap_error(df,trial_info,selector,pct=pct)
        bounds[irun] = utils.compute_bootstrap_error(df,trial_info,selector,pct=pct,include=train_test[irun],nreps=nreps)
        
    return bounds

def compute_tunings(dsname,keylist,datafield,run_fn,gen_nub_selector=None,npartitionings=10,training_frac=0.5):
    df, roi_info, trial_info = ut.compute_tavg_dataframe(dsname,'nub_0',datafield='decon',keylist=keylist,run_fn=run_fn)
    keylist = list(roi_info.keys())
    
    if not gen_nub_selector is None:
        selector = [lambda x: gen_nub_selector(run=irun) for irun in range(2)]
    else:
        selector = [lambda x: utils.gen_nub_selector_v1(run=irun) for irun in range(2)]
    
    train_test = [None for irun in range(2)]
    for irun in range(2):
        selector = gen_nub_selector(run=irun)
        train_test[irun] = utils.select_trials(trial_info,selector,0.5)
        
    tuning = [None for irun in range(2)]
    for irun in range(2):
        selector = gen_nub_selector(run=irun)
        tuning[irun] = utils.compute_tuning_many_partitionings(df,trial_info,npartitionings,\
                                                               training_frac=training_frac,gen_nub_selector=selector)
    return tuning

def compute_lkat_dfof(dsname,keylist,run_fn=move_fn,gen_nub_selector=utils.gen_nub_selector_v1,dfof_cutoff=0.2):
    # return a boolean array identifying cells with mean dF/F >= dfof_cutoff
    tuning_dfof = compute_tuning(dsname,keylist,'F',run_fn,gen_nub_selector=gen_nub_selector)
    lkat = [None for irun in range(2)]
    for irun in range(2):
        nexpt = len(tuning_dfof[irun])
        lkat_list = [None for iexpt in range(nexpt)]
        for iexpt in range(nexpt):
            data = 0.5*(tuning_dfof[irun][iexpt][0] + tuning_dfof[irun][iexpt][1])
            lkat_list[iexpt] = np.nanmean(data-data[:,0:1],axis=1) >= dfof_cutoff
        lkat[irun] = np.concatenate(lkat_list,axis=0)
    return lkat

def compute_lkat_evan_style(dsname,keylist,run_fn=move_fn,pcutoff=0.05,dfof_cutoff=0.,datafield='decon',trialwise_dfof=False):
    # return a boolean array identifying cells significantly driven by at least one stimulus. 
    # t-test + Benjamini-Hochberg correction
    lkat = [None for irun in range(2)]
    for irun in range(2):
        df, roi_info, trial_info = ut.compute_tavg_dataframe(dsname,'nub_0',datafield=datafield,keylist=keylist,\
                                                             run_fn=run_fn,trialwise_dfof=trialwise_dfof)
        #if not irun:
            #for expt in trial_info:
                #trial_info[expt]['running'] = ~trial_info[expt]['running']
        roi_info = utils.test_sig_driven(df,roi_info,trial_info,pcutoff=pcutoff,dfof_cutoff=dfof_cutoff,running=irun)
        if not keylist is None:
            expts = keylist
        else:
            expts = list(roi_info.keys())
        nexpt = len(expts)
        lkat_list = [None for iexpt in range(nexpt)]
        for iexpt,expt in enumerate(expts):
            lkat_list[iexpt] = roi_info[expt]['sig_driven']
        lkat[irun] = np.concatenate(lkat_list,axis=0)
    return lkat

def plot_tuning_curves(targets,dsname,keylist,datafield='decon',run_fn=move_fn,\
                       gen_nub_selector=utils.gen_nub_selector_v1,dfof_cutoff=None,run_conditions=[0,1],colorbar=True,\
                       line=True,pcutoff=0.05,rcutoff=-1,trialwise_dfof=False,draw_stim_ordering=True,train_test_mode='halves'):
    # load data and average across trials to compute tuning curves
    tuning = compute_tuning(dsname,keylist,datafield,run_fn,gen_nub_selector=gen_nub_selector,trialwise_dfof=trialwise_dfof,train_test_mode=train_test_mode)
    lkat = compute_lkat_two_criteria(tuning,dsname,keylist,run_fn,dfof_cutoff=dfof_cutoff,pcutoff=pcutoff,rcutoff=rcutoff,\
                                     datafield=datafield,trialwise_dfof=trialwise_dfof)
    #lkat_roracle = [utils.compute_lkat_roracle(tuning[irun]) for irun in range(2)]
    ## figure out the neurons to include in the plot
    #if dfof_cutoff is None:
    #    #lkat = [None for irun in range(2)]
    #    lkat = lkat_roracle
    #else:
    #    lkat = compute_lkat_evan_style(dsname,keylist,run_fn=run_fn,dfof_cutoff=dfof_cutoff,pcutoff=pcutoff)
    #    lkat = [lkat[irun] & lkat_roracle[irun] for irun in range(2)]

    train_index,test_index = assign_train_test_index(train_test_mode)

    for irun in run_conditions:
        # get final list of neurons and plot their tuning curves
        plot_tuning_curves_(tuning[irun],targets[irun],lkat=lkat[irun],colorbar=colorbar,\
                            line=line,draw_stim_ordering=draw_stim_ordering)#,rcutoff=rcutoff,train_index=train_index,test_index=test_index)
    return tuning,lkat

def assign_train_test_index(train_test_mode):
    if train_test_mode=='halves':
        train_index = 0
        test_index = 1
    elif train_test_mode=='thirds' or train_test_mode=='quarters':
        train_index = np.array((0,1))
        test_index = 2
    return train_index,test_index

def plot_combined_tuning_curves(targets,dsnames,keylists,datafields='decon',run_fns=None,\
                                gen_nub_selectors=utils.gen_nub_selector_v1,dfof_cutoff=None,run_conditions=\
                                [0,1],colorbar=True,line=True,pcutoff=0.05,rcutoff=-1,trialwise_dfof=False\
                                ,draw_stim_ordering=True,train_test_mode='halves'):
    # combine data from multiple hdf5 files and plot it
    # function to separate movement trials from non-movement trials
    if run_fns is None:
        run_fns = [move_fn for ifn in range(len(dsnames))]
    tuning = []
    lkat = []
    for dsname,keylist,datafield,run_fn,gen_nub_selector in zip(dsnames,keylists,datafields,run_fns,gen_nub_selectors):
        # iterate through combinations of cell types to plot
        this_tuning = compute_tuning(dsname,keylist,datafield,run_fn,gen_nub_selector=gen_nub_selector,\
                                     trialwise_dfof=trialwise_dfof,train_test_mode=train_test_mode)
        tuning.append(this_tuning)
        this_lkat = compute_lkat_two_criteria(this_tuning,dsname,keylist,run_fn,dfof_cutoff=dfof_cutoff,pcutoff=pcutoff,\
                                              rcutoff=rcutoff,datafield=datafield,trialwise_dfof=trialwise_dfof)
        lkat.append(this_lkat)
        #if dfof_cutoff is None:
        #    lkat.append(None)
        #else:
        #    this_lkat = compute_lkat_evan_style(dsname,keylist,run_fn=run_fn,dfof_cutoff=dfof_cutoff,pcutoff=pcutoff)
        #    lkat.append(this_lkat)
    #tuning: [[irun=0,irun=1],[irun=0,irun=1]]
           
    tuning = [np.concatenate([t[irun] for t in tuning],axis=0) for irun in range(2)]
    if not lkat[0] is None:
        lkat = [np.concatenate([t[irun] for t in lkat],axis=0) for irun in range(2)]
        for irun in run_conditions[0]:
            print('# lkat: %d' %lkat[irun].sum())

    train_index,test_index = assign_train_test_index(train_test_mode)

    for irun in run_conditions[0]:
        plot_tuning_curves_(tuning[irun],targets[irun],lkat=lkat[irun],colorbar=colorbar,\
                            line=line,draw_stim_ordering=draw_stim_ordering,train_index=train_index,test_index=test_index)#,rcutoff=rcutoff)
def troubleshoot_lkat(dsname,keylist,datafield='decon',run_fn=None,gen_nub_selector=utils.gen_nub_selector_v1,dfof_cutoff=None,pcutoff=0.05):
    # choose neurons based on (1) significantly responding to at least one stimulus and (2) having similar tuning between training and test set, for comparison
    tuning = compute_tuning(dsname,keylist,datafield,run_fn,gen_nub_selector=gen_nub_selector)
    lkat_evan_style = compute_lkat_evan_style(dsname,keylist,run_fn=run_fn,dfof_cutoff=dfof_cutoff,pcutoff=pcutoff)
    lkat_roracle = [None for irun in range(2)]
    for irun in range(2):
        lkat_roracle[irun] = utils.compute_lkat_roracle(tuning[irun])
    return lkat_evan_style,lkat_roracle

def plot_tuning_curves_(tuning,target,lkat=None,colorbar=True,line=True,draw_stim_ordering=True,show_test=True,\
                       train_index=0,test_index=1): #,rcutoff=0.5
    # apply filter based on roracle and plot
    nexpt = len(tuning)

    # keep only cells with corrcoef btw. training and test tuning curves > 0.5
    #lkat_roracle = utils.compute_lkat_roracle(tuning,rcutoff=rcutoff)
    #if lkat is None:
    #    lkat = lkat_roracle
    #else:
    #    lkat = lkat & lkat_roracle
        #print((lkat & lkat_roracle).mean())

    # keep cells based on lkat, and separate training set tuning curve from test set

    #train_response = np.concatenate([tuning[iexpt][train_index] for iexpt in range(nexpt)],axis=0)[lkat]
    if np.isscalar(train_index):
        train_index = [train_index]
    train_response = [np.concatenate([tuning[iexpt][ti] for iexpt in range(nexpt)],axis=0)[lkat][np.newaxis] for ti in train_index]
    train_response = np.nanmean(np.concatenate(train_response,axis=0),axis=0)

    test_response = np.concatenate([tuning[iexpt][test_index] for iexpt in range(nexpt)],axis=0)[lkat]
    
    ndim = len(train_response.shape)-1
    
    if ndim==1:
        train_response = train_response[:,np.newaxis,:]
        test_response = test_response[:,np.newaxis,:]
    
    nsize = train_response.shape[1]
    fig = plt.figure(figsize=(6*nsize,6))
    ht = 6
    for idim in range(nsize):
        ax = fig.add_subplot(1,nsize,idim+1)

        if show_test:
            fig = show_evan_style(train_response[:,idim],test_response[:,idim],fig=fig,line=line,\
                              colorbar=colorbar,draw_stim_ordering=draw_stim_ordering)
        else:
            fig = show_evan_style(train_response[:,idim],train_response[:,idim],fig=fig,line=line,\
                              colorbar=colorbar,draw_stim_ordering=draw_stim_ordering)
            
    plt.tight_layout(pad=7)
    plt.savefig(target,dpi=300)

def plot_patch_no_pref(target,dsname,keylist,datafield='decon',run_fn=move_fn,gen_nub_selector=utils.gen_nub_selector_v1,dfof_cutoff=None,pcutoff=0.05,run_conditions=[0,1],colorbar=True,line=True,fig=None,cs=['C0','C1'],rcutoff=-1):
    # compute fraction of cells that prefer each number of patches, and plot them with errorbars
    # errorbars on N cells are computed as 1/sqrt(N)
    tuning = compute_tuning(dsname,keylist,datafield,run_fn,gen_nub_selector=gen_nub_selector)
    lkat = compute_lkat_two_criteria(tuning,dsname,keylist,run_fn,dfof_cutoff=dfof_cutoff,pcutoff=pcutoff,rcutoff=rcutoff)
    #if dfof_cutoff is None:
    #    lkat = [None for irun in [0,1]]
    #else:
    #    lkat = compute_lkat_evan_style(dsname,keylist,run_fn,dfof_cutoff=dfof_cutoff,pcutoff=pcutoff)
    if fig is None:
        fig = plt.figure(figsize=(6,6))
    for irun in run_conditions:
        plot_patch_no_pref_err_by_expt_(tuning[irun],lkat=lkat[irun],fig=fig,c=cs[irun])
    lbls = ['non-moving','moving']
    plt.legend([lbls[r] for r in run_conditions])
    plt.savefig(target,dpi=300)

def plot_patch_no_pref_(tuning,lkat=None,fig=None,c='C0'):
    nexpt = len(tuning)

    if lkat is None:
        lkat = utils.compute_lkat_roracle(tuning)

    
    train_response = np.concatenate([tuning[iexpt][0] for iexpt in range(nexpt)],axis=0)[lkat]
    test_response = np.concatenate([tuning[iexpt][1] for iexpt in range(nexpt)],axis=0)[lkat]
    all_response = 0.5*(train_response + test_response)
    
    sorteach = np.argsort(all_response[:,evan_order_actual],1)[:,::-1]
    pref_no = nub_no[sorteach[:,0]]
    
    nos = np.arange(1,6).astype('int')
    frac_pref_no = np.zeros(nos.shape)
    err_pref_no = np.zeros(nos.shape)
    for ino,no in enumerate(nos):
        norm_by = np.sum(nub_no==no)/(31/5)/100
        frac_pref_no[ino] = np.nansum(pref_no==no)/np.sum(~np.isnan(pref_no==no))/norm_by
        err_pref_no[ino] = np.sqrt(np.nansum(pref_no==no))/np.sum(~np.isnan(pref_no==no))/norm_by
    
    if fig is None:
        fig = plt.figure(figsize=(6,6))
    fig.add_subplot(1,1,1)
    print(frac_pref_no)
    plt.errorbar(nos,frac_pref_no,err_pref_no,c=c)
    plt.scatter(nos,frac_pref_no,c=c)
    plt.xticks(nos,nos)
    plt.xlabel('Preferred number of patches')
    plt.ylabel('% Neurons (norm. by # of stim)')
    #plt.tight_layout(pad=7)

def plot_patch_no_pref_err_by_expt_(tuning,lkat=None,fig=None,c='C0'):
    nexpt = len(tuning)

    if lkat is None:
        lkat = utils.compute_lkat_roracle(tuning)

    
    ctr = 0
    nexpt = len(tuning)
    frac_pref_no = np.zeros((nexpt,5))
    nos = np.arange(1,6).astype('int')
    for iexpt in range(nexpt):
        this_nroi = tuning[iexpt][0].shape[0]
        train_response = tuning[iexpt][0][lkat[ctr:ctr+this_nroi]]
        test_response = tuning[iexpt][1][lkat[ctr:ctr+this_nroi]]
        all_response = 0.5*(train_response + test_response)
        
        sorteach = np.argsort(all_response[:,evan_order_actual],1)[:,::-1]
        pref_no = nub_no[sorteach[:,0]]
        
        for ino,no in enumerate(nos):
            norm_by = np.sum(nub_no==no)/(31/5)/100
            frac_pref_no[iexpt,ino] = np.nansum(pref_no==no)/np.sum(~np.isnan(pref_no==no))/norm_by
        ctr = ctr+this_nroi
    lb,ub,mn = ut.bootstrap(frac_pref_no,pct=(16,84,50),axis=0,fn=np.nanmean)
    
    if fig is None:
        fig = plt.figure(figsize=(2,2))
    fig.add_subplot(1,1,1)
    print(frac_pref_no)
    plt.errorbar(nos,mn,yerr=(mn-lb,ub-mn),c=c,fmt='.-')
    #plt.scatter(nos,frac_pref_no,c=c)
    plt.xticks(nos,nos)
    plt.xlabel('Preferred number of patches')
    plt.ylabel('% Neurons (norm. by # of stim)')
    #plt.tight_layout(pad=7)

def show_evan_style(train_response,test_response,ht=6,cmap=parula,line=True,fig=None,colorbar=True,\
                    draw_stim_ordering=True,show_every=300):
    # show stimuli in order of increasing patch no (columns) and ROIs in order of preferred stim (rows)
    # for each ROI (rows), stimulus indices in order of decreasing preference (columns are preference rank)
    sorteach = np.argsort(train_response[:,evan_order_actual],1)[:,::-1]
    # sort ROIs according to their #1 ranked stimulus in the training set
    sortind = np.arange(train_response.shape[0])
    if fig is None:
        fig = plt.figure()
    for n in [3,2,1,0]:
        new_indexing = np.argsort(sorteach[:,n],kind='mergesort')
        sortind = sortind[new_indexing]
        sorteach = sorteach[new_indexing]
    nroi = test_response.shape[0]
    if nroi:
        # plot test set tuning curves normalized to their max index in the test set
        img = plt.imshow(test_response[sortind][:,evan_order_actual]/np.abs(test_response[sortind].max(1)[:,np.newaxis]),\
                         extent=[-0.5,31.5,0,5*ht],cmap=cmap,interpolation='none',vmin=0,vmax=1)
        if draw_stim_ordering:
            # underneath, plot graphic of stimuli in the same order as the columns
            utils.draw_stim_ordering(evan_order_apparent,invert=True)

        plt.yticks(5*ht-np.arange(0,5*ht,show_every/nroi*5*ht),['Neuron'] + list(np.arange(show_every,nroi,show_every)))
        for this_nub_no in range(2,6):
            first_ind = np.where(nub_no==this_nub_no)[0][0]
            plt.axvline(first_ind-0.45,c='w',linestyle='dotted')
            plt.axhline(5*ht*(1-np.where(sorteach[:,0]==first_ind)[0][0]/nroi),c='w',linestyle='dotted')
        #plt.ylabel('Neuron')
        plt.xlabel('Stimulus')
        if draw_stim_ordering:
            plt.ylim(-5,5*ht)
        else:
            plt.ylim(0,5*ht)
        plt.xlim(0.5,31.5)
        plt.xticks([])

        plt.text(-5,5*ht+1,'# patches: ')
        lbl_locs = [2.75,10.25,20.25,27.75,30.75]
        for inub,this_nub_no in enumerate(range(1,6)):
            plt.text(lbl_locs[inub],5*ht+1,this_nub_no)
        if line:
            plt.plot((0.5,31.5),(5*ht,0),c='k')
        if colorbar:
            cbaxes = fig.add_axes([0.12, 0.28, 0.03, 0.52])
            cb = plt.colorbar(img,cax=cbaxes)
            cbaxes.yaxis.set_ticks_position('left')
            cb.set_label('Normalized Response')
            cbaxes.yaxis.set_label_position('left')
        plt.tight_layout(pad=7)
    return fig

def extract_train_test(tuning,lkat=None,train_index=0,test_index=1):
    nexpt = len(tuning)
    if np.isscalar(train_index):
        train_index = [train_index]
    train_response = [np.concatenate([tuning[iexpt][ti] for iexpt in range(nexpt)],axis=0)[lkat][np.newaxis] for ti in train_index]
    train_response = np.nanmean(np.concatenate(train_response,axis=0),axis=0)
    test_response = np.concatenate([tuning[iexpt][test_index] for iexpt in range(nexpt)],axis=0)[lkat]
    ndim = len(train_response.shape)
    if ndim==2:
        train_response = train_response[:,np.newaxis,:]
        test_response = test_response[:,np.newaxis,:]
        ndim = 3
    return train_response,test_response

def normalize_response(tuning,lkat=None,train_index=0,test_index=1,fudge=1e-4):
    # subtract spont. activity, normalize to max for training and test set
    #nexpt = len(tuning)
    #if np.isscalar(train_index):
    #    train_index = [train_index]
    #train_response = [np.concatenate([tuning[iexpt][ti] for iexpt in range(nexpt)],axis=0)[lkat][np.newaxis] for ti in train_index]
    #train_response = np.nanmean(np.concatenate(train_response,axis=0),axis=0)
    #test_response = np.concatenate([tuning[iexpt][test_index] for iexpt in range(nexpt)],axis=0)[lkat]
    #ndim = len(train_response.shape)
    #if ndim==2:
    #    train_response = train_response[:,np.newaxis,:]
    #    test_response = test_response[:,np.newaxis,:]
    #    ndim = 3
    train_response,test_response = extract_train_test(tuning,lkat=lkat,train_index=train_index,test_index=test_index)
    # convert to evoked event rates, normalized to max
    train_norm_response = norm_to_max(train_response,fudge=fudge)
    test_norm_response = norm_to_max(test_response,fudge=fudge)
    #train_norm_response = train_response.copy() - train_response[:,:,0:1]
    #train_norm_response = train_norm_response/(fudge + train_norm_response[:,:,1:].max(2)[:,:,np.newaxis])
    #test_norm_response = test_response.copy() - test_response[:,:,0:1]
    #test_norm_response = test_norm_response/(fudge + test_norm_response[:,:,1:].max(2)[:,:,np.newaxis])
    return train_norm_response,test_norm_response

def norm_to_max(train_response,other_response=None,fudge=1e-4):
    if other_response is None:
        train_norm_response = train_response.copy() - train_response[:,:,0:1]
        train_norm_response = train_norm_response/(fudge + train_norm_response[:,:,1:].max(2)[:,:,np.newaxis])
    else:
        train_norm_response = train_response.copy() - other_response[:,:,0:1]
        other_norm_response = other_response.copy() - other_response[:,:,0:1]
        train_norm_response = train_norm_response/(fudge + other_norm_response[:,:,1:].max(2)[:,:,np.newaxis])
    return train_norm_response

def unpack_list_of_lists(lol,depth=0):
    # take list of list index at depth=depth, and bring it to the top. e.g. with depth=2, [i][j][k] becomes [k][i][j] (I think)
    if depth == 0:
        return lol
    elif depth == 1:
        return [[this_list[i] for this_list in lol] for i in range(len(lol[0]))]
    else:
        return unpack_list_of_lists([unpack_list_of_lists(this_list,depth=depth-1) for this_list in lol],depth=1)

def list_of_lists_dim(lol):
    # recursively concatenate len of each list in a nested list structure
    if isinstance(lol,list):
        return (len(lol),) + list_of_lists_dim(lol[0])
    else:
        return ()

def normalize_response_ab(tuning_a,tunings_pct,lkat=None,train_index=0,test_index=1,fudge=1e-4):
    # subtract spont. activity, normalize to max for training and test set
    train_response,test_response = extract_train_test(tuning_a,lkat=lkat,train_index=train_index,test_index=test_index)
    #train_norm_response = norm_to_max(train_response,fudge=fudge)
    #test_norm_response = norm_to_max(test_response,fudge=fudge)

    this_tunings_pct = unpack_list_of_lists(tunings_pct,depth=2)
    print('tuning_a shape: ' + str(list_of_lists_dim(tuning_a)))
    print('this_tunings_pct shape: ' + str(list_of_lists_dim(this_tunings_pct)))
    npct = len(this_tunings_pct)
    train_norm_response_pct,test_norm_response_pct = [[None for _ in range(npct)] for _ in range(2)]
    for ipct in range(npct):
        train_norm_response_pct[ipct],test_norm_response_pct[ipct] = extract_train_test(this_tunings_pct[ipct],lkat=lkat,
                train_index=train_index,test_index=test_index)
        # convert to evoked event rates, normalized to max
        train_norm_response_pct[ipct] = norm_to_max(train_norm_response_pct[ipct],other_response=train_response,fudge=fudge)
        test_norm_response_pct[ipct] = norm_to_max(test_norm_response_pct[ipct],other_response=test_response,fudge=fudge)

    return train_norm_response_pct,test_norm_response_pct

def compute_linear_prediction(test_norm_response):
    # linear_pred: (nroi,[nsize,]nstim)
    ndim = len(test_norm_response.shape)
    linear_pred = np.zeros_like(test_norm_response)
    slicer = [slice(None) for idim in range(ndim)]
    slicer[-1] = [2**n for n in range(5)][::-1]
    # single_whisker_responses: (nroi,[nsize,]nsingle)
    single_whisker_responses = test_norm_response[slicer]
    #single_whisker_responses = test_norm_response[:,[2**n for n in range(5)][::-1]]
    slicer_output = [slice(None) for idim in range(ndim)]
    slicer_input = [np.newaxis for idim in range(ndim-1)]
    slicer_input = slicer_input+[slice(None)]
    # utils.nubs_active: (nstim,nsingle)
    for istim in range(linear_pred.shape[-1]): # nstim
        slicer_output[-1] = istim
        linear_pred[slicer_output] = np.sum(single_whisker_responses*utils.nubs_active[istim][slicer_input],-1)
    return linear_pred

def compute_sorteach_sortind(train_norm_response):
    # sorteach: ith row gives indices to sort responses of ith neuron in descending order
    ndim = len(train_norm_response.shape)
    slicer = [slice(None) for idim in range(ndim)]
    slicer[-1] = evan_order_actual
    slicer1 = [slice(None) for idim in range(ndim)]
    slicer1[-1] = slice(None,None,-1)
    #sorteach = np.argsort(train_norm_response[:,evan_order_actual],1)[:,::-1]
    sorteach = np.argsort(train_norm_response[slicer],-1)[slicer1]
    # sortind: array that sorts neurons according to preferred stimulus

    ndim = len(train_norm_response.shape)
    if ndim==2:
        sorteach = np.argsort(train_norm_response[:,evan_order_actual],1)[:,::-1]
        sortind = np.arange(train_norm_response.shape[0])
        for n in [3,2,1,0]: #[3,2,1,0]:
            slicer[-1] = 0
            new_indexing = np.argsort(sorteach[slicer],kind='mergesort')
            sortind = sortind[new_indexing]
            sorteach = sorteach[new_indexing]
    if ndim==3:
        nsize = train_norm_response.shape[1]
        sorteach = np.array([np.argsort(train_norm_response[:,isize,evan_order_actual],1)[:,::-1]\
                             for isize in range(nsize)])
        sortind = np.array([np.arange(train_norm_response.shape[0]) for isize in range(nsize)])
        for isize in range(nsize):
            for n in [0]: #[3,2,1,0]:
                new_indexing = np.argsort(sorteach[isize][:,n],kind='mergesort')
                sortind[isize] = sortind[isize][new_indexing]
                sorteach[isize] = sorteach[isize][new_indexing]
        sortind = sortind.T
        #print(sortind.shape)
        sorteach = sorteach.transpose((1,0,2))
        #print(sorteach.shape)
    return sorteach,sortind

def subtract_lin(tuning,lkat=None,train_index=0,test_index=1):
    # compute evoked response, norm to max evoked response, subtract linear sum of single-patch normed responses
    train_norm_response,test_norm_response = normalize_response(tuning,lkat=lkat,
            train_index=train_index,test_index=test_index)

    linear_pred = compute_linear_prediction(test_norm_response)

    lin_subtracted = test_norm_response-linear_pred

    ndim = len(train_norm_response.shape)

    if ndim==2:
        utils.test_validity_of_linear_pred(test_norm_response,linear_pred)
    elif ndim==3:
        for isize in range(linear_pred.shape[1]):
            utils.test_validity_of_linear_pred(test_norm_response[:,isize],linear_pred[:,isize])

    sorteach,sortind = compute_sorteach_sortind(train_norm_response)
    
    return lin_subtracted,sorteach,sortind

    ## compute evoked response, norm to max evoked response, subtract linear sum of single-patch normed responses
    #nexpt = len(tuning)
    #
    ## to plot: cells that either have corrcoef training vs. test > 0.5, or both that and some other criterion
    #if np.isscalar(train_index):
    #    train_index = [train_index]
    #train_response = [np.concatenate([tuning[iexpt][ti] for iexpt in range(nexpt)],axis=0)[lkat][np.newaxis] for ti in train_index]
    #train_response = np.nanmean(np.concatenate(train_response,axis=0),axis=0)
    #test_response = np.concatenate([tuning[iexpt][test_index] for iexpt in range(nexpt)],axis=0)[lkat]
    #ndim = len(train_response.shape)
    #if ndim==2:
    #    train_response = train_response[:,np.newaxis,:]
    #    test_response = test_response[:,np.newaxis,:]
    #    ndim = 3
    ## convert to evoked event rates, normalized to max
    #fudge = 1e-4
    #train_norm_response = train_response.copy() - train_response[:,:,0:1]
    #train_norm_response = train_norm_response/(fudge + train_norm_response[:,:,1:].max(2)[:,:,np.newaxis])
    #test_norm_response = test_response.copy() - test_response[:,:,0:1]
    #test_norm_response = test_norm_response/(fudge + test_norm_response[:,:,1:].max(2)[:,:,np.newaxis])

    ## linear_pred: (nroi,[nsize,]nstim)
    #linear_pred = np.zeros_like(test_norm_response)
    #slicer = [slice(None) for idim in range(ndim)]
    #slicer[-1] = [2**n for n in range(5)][::-1]
    ## single_whisker_responses: (nroi,[nsize,]nsingle)
    #single_whisker_responses = test_norm_response[slicer]
    ##single_whisker_responses = test_norm_response[:,[2**n for n in range(5)][::-1]]
    #slicer_output = [slice(None) for idim in range(ndim)]
    #slicer_input = [np.newaxis for idim in range(ndim-1)]
    #slicer_input = slicer_input+[slice(None)]
    ## utils.nubs_active: (nstim,nsingle)
    #for istim in range(linear_pred.shape[-1]): # nstim
    #    slicer_output[-1] = istim
    #    linear_pred[slicer_output] = np.sum(single_whisker_responses*utils.nubs_active[istim][slicer_input],-1)
    #    lin_subtracted = test_norm_response-linear_pred

    ## sorteach: ith row gives indices to sort responses of ith neuron in descending order
    #slicer[-1] = evan_order_actual
    #slicer1 = [slice(None) for idim in range(ndim)]
    #slicer1[-1] = slice(None,None,-1)
    ##sorteach = np.argsort(train_norm_response[:,evan_order_actual],1)[:,::-1]
    #sorteach = np.argsort(train_norm_response[slicer],-1)[slicer1]
    ## sortind: array that sorts neurons according to preferred stimulus
    #if ndim==2:
    #    sorteach = np.argsort(train_norm_response[:,evan_order_actual],1)[:,::-1]
    #    sortind = np.arange(train_norm_response.shape[0])
    #    for n in [3,2,1,0]: #[3,2,1,0]:
    #        slicer[-1] = 0
    #        new_indexing = np.argsort(sorteach[slicer],kind='mergesort')
    #        sortind = sortind[new_indexing]
    #        sorteach = sorteach[new_indexing]
    #        
    #    # confirm that linear prediction is equal to the sum of single neuron 
    #    # normed evoked responses
    #    utils.test_validity_of_linear_pred(test_norm_response,linear_pred)
    #if ndim==3:
    #    nsize = linear_pred.shape[1]
    #    sorteach = np.array([np.argsort(train_norm_response[:,isize,evan_order_actual],1)[:,::-1]\
    #                         for isize in range(nsize)])
    #    sortind = np.array([np.arange(train_norm_response.shape[0]) for isize in range(nsize)])
    #    for isize in range(nsize):
    #        for n in [0]: #[3,2,1,0]:
    #            new_indexing = np.argsort(sorteach[isize][:,n],kind='mergesort')
    #            sortind[isize] = sortind[isize][new_indexing]
    #            sorteach[isize] = sorteach[isize][new_indexing]
    #            
    #        # confirm that linear prediction is equal to the sum of single neuron 
    #        # normed evoked responses
    #        utils.test_validity_of_linear_pred(test_norm_response[:,isize],linear_pred[:,isize])
    #    sortind = sortind.T
    #    print(sortind.shape)
    #    sorteach = sorteach.transpose((1,0,2))
    #    print(sorteach.shape)
    #
    #return lin_subtracted,sorteach,sortind

def subtract_lin_ab(tuning_a,tunings_pct,lkat=None,train_index=0,test_index=1):
    # compute evoked response, norm to max evoked response, subtract linear sum of single-patch normed responses
    train_norm_response_a,test_norm_response_a = normalize_response(tuning_a,lkat=lkat,
            train_index=train_index,test_index=test_index)
    train_norm_response_pct,test_norm_response_pct = normalize_response_ab(tuning_a,tunings_pct,lkat=lkat,
            train_index=train_index,test_index=test_index)

    linear_pred = compute_linear_prediction(test_norm_response_a)

    lin_subtracted_pct = [tnr-linear_pred for tnr in test_norm_response_pct]
    
    return lin_subtracted_pct

def plot_lin_subtracted_(target=None,lin_subtracted=None,sorteach=None,sortind=None,draw_stim_ordering=True):
    # plot heat map of linear differences, with stim visualization below, aligned to columns
    #fig = plt.figure(figsize=(6,6))
    #ax = fig.add_subplot(1,1,1)
    ndim = len(lin_subtracted.shape)-1
    if ndim==1:
        lin_subtracted = lin_subtracted[:,np.newaxis,:]
        sorteach = sorteach[:,np.newaxis,:]
        sortind = sortind[:,np.newaxis]
    nsize = lin_subtracted.shape[1]
    plt.figure(figsize=(6*nsize,6))
    for isize in range(nsize):
        plt.subplot(1,nsize,isize+1)
        ht = 6
        ld = plt.imshow(lin_subtracted[sortind[:,isize]][:,isize,evan_order_actual[6:]],extent=[-0.5,25.5,0,5*ht],cmap='bwr') #/lin_subtracted[sortind].max(1)[:,np.newaxis]
        if draw_stim_ordering:
            utils.draw_stim_ordering(evan_order_apparent[6:],invert=True)
    
        #mx = 1e-1*np.nanpercentile(np.abs(lin_subtracted),99)
        plt.ylabel('Neuron')
        plt.xlabel('Stimulus')
        #plt.clim(-mx,mx)
        plt.clim(-1.05,1.05)
        plt.colorbar()
        if draw_stim_ordering:
            plt.ylim(-5,5*ht)
        else:
            plt.ylim(0,5*ht)
        plt.xlim(-0.5,25.5)
        plt.xticks([])
        plt.yticks([])
        nroi = lin_subtracted.shape[0]
        show_every = 300
        plt.yticks(5*ht-np.arange(0,5*ht,show_every/nroi*5*ht),['Neuron'] + list(np.arange(show_every,nroi,show_every)))
        plt.yticks([])
        for this_nub_no in range(2,6):
            first_ind = np.where(nub_no[6:]==this_nub_no)[0][0]
            plt.axvline(first_ind-0.5,c='k',linestyle='dotted')
        for this_nub_no in range(2,6):
            first_ind = np.where(nub_no==this_nub_no)[0][0]
            plt.axhline(5*ht*(1-np.where(sorteach[:,isize,0]==first_ind)[0][0]/nroi),c='k',linestyle='dotted')
        #cbaxes = plt.gcf().add_axes([0.81, 0.235, 0.03, 0.645]) 
        #cb = plt.colorbar(ld,cax=cbaxes,ticks=np.linspace(-1,1,6))
        #cb.ax.set_yticks(np.linspace(-1,1,6))
        #cb.set_label('Normalized Linear Difference')
        #plt.imshow(lin_subtracted[sortind[:,isize]][:,isize,evan_order_actual[6:]],extent=[0,10,0,10],cmap='bwr')

#     plt.ylabel('neuron #')
#     plt.xlabel('stimulus #')
#     plt.clim(-1,1)
#     plt.xticks([])
#     plt.yticks([])
    plt.tight_layout(pad=7)
    if not target is None:
        plt.savefig(target,dpi=300)

def compute_lkat_two_criteria(tuning,dsname,keylist,run_fn,dfof_cutoff=None,pcutoff=0.05,rcutoff=-1,\
                              datafield='decon',trialwise_dfof=False):
    # return boolean arrays of cells that meet both (1) a criterion on correlation coefficient between 
    # training and test set, and (2) a criterion on significance of responses
    lkat_roracle = [utils.compute_lkat_roracle(tuning[irun],rcutoff=rcutoff) for irun in range(2)]
    print([lkat_roracle[irun].mean() for irun in range(2)])
    # figure out the neurons to include in the plot
    if dfof_cutoff is None:
        #lkat = [None for irun in range(2)]
        lkat = lkat_roracle
    else:
        lkat_sig_driven = compute_lkat_evan_style(dsname,keylist,run_fn=run_fn,dfof_cutoff=dfof_cutoff,\
                                                  pcutoff=pcutoff,datafield=datafield,trialwise_dfof=trialwise_dfof)
        print('driven: '+str([lkat_sig_driven[irun].mean() for irun in range(2)]))
        lkat = [lkat_sig_driven[irun] & lkat_roracle[irun] for irun in range(2)]
    return lkat

def compute_tuning_lkat_from_dsname(dsname,keylist,datafield='decon',run_fn=move_fn,\
                       gen_nub_selector=utils.gen_nub_selector_v1,dfof_cutoff=None,run_conditions=[0,1],\
                       pcutoff=0.05,rcutoff=-1):
    # load data and average across trials to compute tuning curves
    tuning = compute_tuning(dsname,keylist,datafield,run_fn,gen_nub_selector=gen_nub_selector,trialwise_dfof=False)
    lkat = compute_lkat_two_criteria(tuning,dsname,keylist,run_fn,dfof_cutoff=dfof_cutoff,pcutoff=pcutoff,rcutoff=rcutoff,\
                                     datafield=datafield,trialwise_dfof=False)
    return tuning,lkat

def plot_lin_subtracted(targets,dsname,keylist=None,datafield='decon',run_fn=move_fn,\
                        gen_nub_selector=utils.gen_nub_selector_v1,dfof_cutoff=None,pcutoff=0.05,\
                        run_conditions=[0,1],rcutoff=-1,trialwise_dfof=False,draw_stim_ordering=True,seed=0,train_test_mode='halves'):
    # compute and plot linear difference
    tuning = compute_tuning(dsname,keylist,datafield,run_fn,gen_nub_selector=gen_nub_selector,trialwise_dfof=trialwise_dfof,seed=seed,train_test_mode=train_test_mode)
    lkat = compute_lkat_two_criteria(tuning,dsname,keylist,run_fn,dfof_cutoff=dfof_cutoff,pcutoff=pcutoff,rcutoff=rcutoff,\
                                     datafield=datafield,trialwise_dfof=trialwise_dfof)
    #if dfof_cutoff is None:
    #    lkat = [None for irun in range(2)]
    #else:
    #    lkat = compute_lkat_evan_style(dsname,keylist,run_fn,dfof_cutoff=dfof_cutoff,pcutoff=pcutoff)
    for irun in run_conditions:
        expt_id = np.concatenate([iexpt*np.ones((tuning[irun][iexpt][0].shape[0],)) for iexpt in range(len(tuning[irun]))])
        expt_id = expt_id[lkat[irun]]
        #lin_subtracted,sorteach,sortind = subtract_lin(tuning[irun],lkat=lkat[irun])
        lin_subtracted,sorteach,sortind = subtract_lin(tuning[irun],lkat=lkat[irun])
        plot_lin_subtracted_(targets[irun],lin_subtracted,sorteach,sortind,draw_stim_ordering=draw_stim_ordering)
    return lin_subtracted,sorteach,sortind,expt_id

def plot_sorted_lin_subtracted_simple(targets,dsname,keylist=None,datafield='decon',run_fn=move_fn,\
                        gen_nub_selector=utils.gen_nub_selector_v1,dfof_cutoff=None,pcutoff=0.05,\
                        run_conditions=[0,1],rcutoff=-1,trialwise_dfof=False,draw_stim_ordering=True):
    # compute and plot linear difference
    tuning = compute_tuning(dsname,keylist,datafield,run_fn,gen_nub_selector=gen_nub_selector,trialwise_dfof=trialwise_dfof)
    lkat = compute_lkat_two_criteria(tuning,dsname,keylist,run_fn,dfof_cutoff=dfof_cutoff,pcutoff=pcutoff,rcutoff=rcutoff,\
                                     datafield=datafield,trialwise_dfof=trialwise_dfof)
    #if dfof_cutoff is None:
    #    lkat = [None for irun in range(2)]
    #else:
    #    lkat = compute_lkat_evan_style(dsname,keylist,run_fn,dfof_cutoff=dfof_cutoff,pcutoff=pcutoff)
    for irun in run_conditions:
        #lin_subtracted,sorteach,sortind = subtract_lin(tuning[irun],lkat=lkat[irun])
        lin_subtracted,sorteach,sortind = subtract_lin(tuning[irun],lkat=lkat[irun])
        plot_sorted_lin_subtracted_(targets[irun],lin_subtracted[:,0],sorteach[:,0],sortind[:,0])
    return lin_subtracted,sorteach,sortind

def plot_sorted_lin_subtracted_multiple_partitionings_simple(targets,dsname,keylist=None,datafield='decon',run_fn=move_fn,\
                        gen_nub_selector=utils.gen_nub_selector_v1,dfof_cutoff=None,pcutoff=0.05,\
                        run_conditions=[0,1],rcutoff=-1,trialwise_dfof=False,draw_stim_ordering=True,npartitioning=5):
    # compute and plot linear difference
    tuning = compute_tuning(dsname,keylist,datafield,run_fn,gen_nub_selector=gen_nub_selector,\
                                     trialwise_dfof=trialwise_dfof)
    
    lkat = compute_lkat_two_criteria(tuning,dsname,keylist,run_fn,dfof_cutoff=dfof_cutoff,pcutoff=pcutoff,rcutoff=rcutoff,\
                                     datafield=datafield,trialwise_dfof=trialwise_dfof)
    
    tunings = compute_tunings_simple(dsname,keylist,datafield,run_fn,gen_nub_selector=gen_nub_selector,\
                                     trialwise_dfof=trialwise_dfof,npartitioning=npartitioning)
    #if dfof_cutoff is None:
    #    lkat = [None for irun in range(2)]
    #else:
    #    lkat = compute_lkat_evan_style(dsname,keylist,run_fn,dfof_cutoff=dfof_cutoff,pcutoff=pcutoff)
    for irun in run_conditions:
        for ipartitioning in range(1):
            lin_subtracted,sorteach,sortind = subtract_lin(tunings[irun][ipartitioning],lkat=lkat[irun])
        for ipartitioning in range(1,npartitioning):
            this_lin_subtracted,this_sorteach,this_sortind = subtract_lin(tunings[irun][ipartitioning],lkat=lkat[irun])
            lin_subtracted = np.concatenate((lin_subtracted,this_lin_subtracted),axis=1)
            sorteach = np.concatenate((sorteach,this_sorteach),axis=1)
            sortind = np.concatenate((sortind,this_sortind),axis=1)
            
        plot_sorted_lin_subtracted_multiple_partitionings_simple_(targets[irun],lin_subtracted,sorteach,sortind)
    return lin_subtracted,sorteach,sortind

def plot_sorted_tuning_simple(targets,dsname,keylist=None,datafield='decon',run_fn=move_fn,\
                        gen_nub_selector=utils.gen_nub_selector_v1,dfof_cutoff=None,pcutoff=0.05,\
                        run_conditions=[0,1],rcutoff=-1,trialwise_dfof=False,draw_stim_ordering=True):
    # compute and plot linear difference
    tuning = compute_tuning(dsname,keylist,datafield,run_fn,gen_nub_selector=gen_nub_selector,trialwise_dfof=trialwise_dfof)
    lkat = compute_lkat_two_criteria(tuning,dsname,keylist,run_fn,dfof_cutoff=dfof_cutoff,pcutoff=pcutoff,rcutoff=rcutoff,\
                                     datafield=datafield,trialwise_dfof=trialwise_dfof)
    #if dfof_cutoff is None:
    #    lkat = [None for irun in range(2)]
    #else:
    #    lkat = compute_lkat_evan_style(dsname,keylist,run_fn,dfof_cutoff=dfof_cutoff,pcutoff=pcutoff)
    for irun in run_conditions:
        #lin_subtracted,sorteach,sortind = subtract_lin(tuning[irun],lkat=lkat[irun])
        lin_subtracted,sorteach,sortind = subtract_lin(tuning[irun],lkat=lkat[irun])
        this_tuning_train = np.concatenate([t[0] for t in tuning[irun]],axis=0)
        this_tuning_train = (this_tuning_train - this_tuning_train[:,0:1])/this_tuning_train.max(1)[:,np.newaxis]
        this_tuning_test = np.concatenate([t[1] for t in tuning[irun]],axis=0)
        this_tuning_test = (this_tuning_test - this_tuning_test[:,0:1])/this_tuning_test.max(1)[:,np.newaxis]
        plot_sorted_lin_subtracted_(targets[irun],this_tuning_test[lkat[irun]],sorteach[:,0],sortind[:,0])
    return lin_subtracted,sorteach,sortind

def plot_combined_lin_subtracted(targets,dsnames,keylists,datafields='decon',run_fns=None,\
                                 gen_nub_selectors=utils.gen_nub_selector_v1,dfof_cutoff=None,pcutoff=0.05,\
                                 run_conditions=[0,1],colorbar=True,line=True,\
                                 rcutoff=-1,trialwise_dfof=False,draw_stim_ordering=True,train_test_mode='halves'):
    # combine two datasets, and compute and plot linear differences
    if run_fns is None:
        run_fns = [move_fn for ifn in range(len(dsnames))]
    these_tunings = []
    lkat = []
    for dsname,keylist,datafield,run_fn,gen_nub_selector in zip(dsnames,keylists,datafields,run_fns,gen_nub_selectors):
        # iterate through combinations of cell types to plot
        this_tuning = compute_tuning(dsname,keylist,datafield,run_fn,gen_nub_selector=gen_nub_selector,\
                                     trialwise_dfof=trialwise_dfof,train_test_mode=train_test_mode)
        these_tunings.append(this_tuning)
        this_lkat = compute_lkat_two_criteria(this_tuning,dsname,keylist,run_fn,dfof_cutoff=dfof_cutoff,pcutoff=pcutoff,\
                                              rcutoff=rcutoff,datafield=datafield,trialwise_dfof=trialwise_dfof)
        lkat.append(this_lkat)
        #if dfof_cutoff is None:
        #    lkat.append(None)
        #else:
        #    this_lkat = compute_lkat_evan_style(dsname,keylist,run_fn=run_fn,dfof_cutoff=dfof_cutoff,pcutoff=pcutoff)
    
    tuning = [np.concatenate([t[irun] for t in these_tunings],axis=0) for irun in range(2)]
    if not lkat[0] is None:
        lkat = [np.concatenate([t[irun] for t in lkat],axis=0) for irun in range(2)]
        for irun in run_conditions[0]:
            print('# lkat: %d' %lkat[irun].sum())

    train_index,test_index = assign_train_test_index(train_test_mode)

    for irun in run_conditions[0]:
        lin_subtracted,sorteach,sortind = subtract_lin(tuning[irun],lkat=lkat[irun],train_index=train_index,test_index=test_index)
        plot_lin_subtracted_(targets[irun],lin_subtracted,sorteach,sortind,draw_stim_ordering=draw_stim_ordering)
        #plot_lin_subtracted_(targets[irun],lin_subtracted,sorteach[lkat[irun]],sortind[lkat[irun]])
    return lkat

def plot_tuning_curves_and_lin_subtracted(targets_tc,targets_ls,dsname,keylist,datafield='decon',run_fn=move_fn,\
                       gen_nub_selector=utils.gen_nub_selector_v1,dfof_cutoff=None,run_conditions=[0,1],colorbar=True,\
                       line=True,pcutoff=0.05,rcutoff=-1,trialwise_dfof=False,draw_stim_ordering=True,seed=0,\
                       train_test_mode='halves',bound_lin_subtracted=False,pct=(2.5,97.5,50)):
    # load data and average across trials to compute tuning curves
    tuning = compute_tuning(dsname,keylist,datafield,run_fn,gen_nub_selector=gen_nub_selector,\
                            trialwise_dfof=trialwise_dfof,seed=seed,train_test_mode=train_test_mode)
    if bound_lin_subtracted:
        bounds = compute_bounds(dsname,keylist,datafield,run_fn,gen_nub_selector=gen_nub_selector,\
                            trialwise_dfof=trialwise_dfof,seed=seed,train_test_mode=train_test_mode,include_all=False,\
                            pct=pct,nreps=10000)#pct=(2.5,97.5,50))
    ## figure out the neurons to include in the plot
    lkat = compute_lkat_two_criteria(tuning,dsname,keylist,run_fn,dfof_cutoff=dfof_cutoff,pcutoff=pcutoff,rcutoff=rcutoff,\
                                     datafield=datafield,trialwise_dfof=trialwise_dfof)

    train_index,test_index = assign_train_test_index(train_test_mode)
    
    for irun in run_conditions:
        # get final list of neurons and plot their tuning curves
        plot_tuning_curves_(tuning[irun],targets_tc[irun],lkat=lkat[irun],colorbar=colorbar,\
                            line=line,draw_stim_ordering=draw_stim_ordering,\
                            train_index=train_index,test_index=test_index)#,rcutoff=rcutoff)
        expt_id = np.concatenate([iexpt*np.ones((tuning[irun][iexpt][0].shape[0],)) for iexpt in range(len(tuning[irun]))])
        expt_id = expt_id[lkat[irun]]
        #lin_subtracted,sorteach,sortind = subtract_lin(tuning[irun],lkat=lkat[irun])
        lin_subtracted,sorteach,sortind = subtract_lin(tuning[irun],lkat=lkat[irun],\
                                                       train_index=train_index,test_index=test_index)#,rcutoff=rcutoff)
        if bound_lin_subtracted:
            lin_subtracted_bounds = subtract_lin_ab(tuning[irun],bounds[irun],lkat=lkat[irun],\
                                                       train_index=train_index,test_index=test_index)#,rcutoff=rcutoff)
        plot_lin_subtracted_(targets_ls[irun],lin_subtracted,sorteach,sortind,draw_stim_ordering=draw_stim_ordering)
    if compute_bounds:
        return tuning,lkat,lin_subtracted,sorteach,sortind,expt_id,bounds,lin_subtracted_bounds
    else:
        return tuning,lkat,lin_subtracted,sorteach,sortind,expt_id

def plot_combined_tuning_curves_and_lin_subtracted(targets_tc,targets_ls,dsnames,keylists,datafields='decon',run_fns=None,\
                                 gen_nub_selectors=utils.gen_nub_selector_v1,dfof_cutoff=None,pcutoff=0.05,\
                                 run_conditions=[0,1],colorbar=True,line=True,\
                                 rcutoff=-1,trialwise_dfof=False,draw_stim_ordering=True,\
                                 train_test_mode='halves'):
    # combine two datasets, and compute and plot linear differences
    if run_fns is None:
        run_fns = [move_fn for ifn in range(len(dsnames))]
    these_tunings = []
    lkat = []
    for dsname,keylist,datafield,run_fn,gen_nub_selector in zip(dsnames,keylists,datafields,run_fns,gen_nub_selectors):
        # iterate through combinations of cell types to plot
        this_tuning = compute_tuning(dsname,keylist,datafield,run_fn,gen_nub_selector=gen_nub_selector,\
                                     trialwise_dfof=trialwise_dfof,include_all=True)
        these_tunings.append(this_tuning)
        this_lkat = compute_lkat_two_criteria(this_tuning,dsname,keylist,run_fn,dfof_cutoff=dfof_cutoff,pcutoff=pcutoff,\
                                              rcutoff=rcutoff,datafield=datafield,trialwise_dfof=trialwise_dfof)
        lkat.append(this_lkat)
    
    tuning = [np.concatenate([t[irun] for t in these_tunings],axis=0) for irun in range(2)]
    if not lkat[0] is None:
        lkat = [np.concatenate([t[irun] for t in lkat],axis=0) for irun in range(2)]
        for irun in run_conditions[0]:
            print('# lkat: %d' %lkat[irun].sum())

    train_index,test_index = assign_train_test_index(train_test_mode)

    for irun in run_conditions[0]:
        plot_tuning_curves_(tuning[irun],targets_tc[irun],lkat=lkat[irun],colorbar=colorbar,\
                            line=line,draw_stim_ordering=draw_stim_ordering,train_index=train_index,test_index=test_index)
        lin_subtracted,sorteach,sortind = subtract_lin(tuning[irun],lkat=lkat[irun],\
                                                       train_index=train_index,test_index=test_index)
        plot_lin_subtracted_(targets_ls[irun],lin_subtracted,sorteach,sortind,draw_stim_ordering=draw_stim_ordering)
        
    return lkat

def sort_lin_subtracted(lin_subtracted,sorteach,sortind):
    lin_subtracted_sort = lin_subtracted.copy()
    for iroi in range(lin_subtracted.shape[0]):
        lin_subtracted_sort[iroi] = lin_subtracted[sortind][iroi][evan_order_actual][sorteach[iroi]]
    return lin_subtracted_sort

def plot_sorted_lin_subtracted_(target,lin_subtracted,sorteach,sortind):
    #if fig is None:
    #    fig = plt.figure()
    lin_subtracted_sort = sort_lin_subtracted(lin_subtracted,sorteach,sortind)
    lb,ub,mn = ut.bootstrap(lin_subtracted_sort[nub_no[sorteach[:,0]]>1],axis=0,pct=(16,84,50),fn=np.nanmean)
    plt.fill_between(np.arange(1,33),lb,ub)
    plt.xticks((1,5,10,15,20,25,30))
    plt.xlabel('Stimulus rank')
    plt.ylabel('Normalized linear difference')
    plt.axhline(0,linestyle='dotted',c='k')
    plt.savefig(target,dpi=300)

def plot_sorted_lin_subtracted_multiple_partitionings_simple_(target,lin_subtracted,sorteach,sortind):
    #if fig is None:
    #    fig = plt.figure()
    npartitioning = lin_subtracted.shape[1]
    lin_subtracted_sort = np.zeros_like(lin_subtracted)
    for ipartitioning in range(npartitioning):
        lin_subtracted_sort[:,ipartitioning] = sort_lin_subtracted(lin_subtracted[:,ipartitioning],sorteach[:,ipartitioning],\
                                                                 sortind[:,ipartitioning])
    lin_subtracted_sort = np.nanmean(lin_subtracted_sort,axis=1)
    #lb,ub,mn = ut.bootstrap(lin_subtracted_sort[nub_no[sorteach[:,0]]>1],axis=0,pct=(16,84,50),fn=np.nanmean)
    lb,ub,mn = ut.bootstrap(lin_subtracted_sort,axis=0,pct=(16,84,50),fn=np.nanmean)
    plt.fill_between(np.arange(1,33),lb,ub)
    plt.xticks((1,5,10,15,20,25,30))
    plt.xlabel('Stimulus rank')
    plt.ylabel('Normalized linear difference')
    plt.axhline(0,linestyle='dotted',c='k')
    plt.savefig(target,dpi=300)

def plot_sorted_lin_subtracted(targets,dsname,keylist=None,datafield='decon',\
                               run_fn=move_fn,gen_nub_selector=utils.gen_nub_selector_v1,\
                               npartitionings=10,tunings=None):
    # compute tuning with many different partitions of the data
    if tunings is None:
        tunings = compute_tunings(dsname,keylist,datafield,run_fn,gen_nub_selector=gen_nub_selector,\
                                  npartitionings=npartitionings)
    for irun in range(2):
        npartitioning = len(tunings[irun].partitioning.unique())
        # unique experiments in tunings dataframe
        sessions = tunings[irun].session_id.unique()
        lin_subtracted = [None for ipartitioning in range(npartitioning)]
        for ipartitioning in range(npartitioning):
            print('partition #' + str(ipartitioning))
            this_tuning = [None for session in sessions]
            for isession,session in enumerate(sessions):
                this_tuning[isession] = [None for ipartition in range(2)]
                for ipartition in range(2):
                    this_tuning[isession][ipartition] = tunings[irun].loc[\
                                                   (tunings[irun].session_id==session) \
                                                 & (tunings[irun].partitioning==ipartitioning) \
                                                 & (tunings[irun].partition ==ipartition)\
                                                                         ].iloc[:,:32].to_numpy()
            this_lin_subtracted,this_sorteach,this_sortind = subtract_lin(this_tuning)
            this_lin_subtracted,this_sorteach,this_sortind =\
            this_lin_subtracted[0],this_sorteach[0],this_sortind[0]
            lin_subtracted[ipartitioning] = np.zeros(this_lin_subtracted.shape)
            for iroi in range(this_lin_subtracted.shape[0]):
                lin_subtracted[ipartitioning][iroi] = this_lin_subtracted[iroi][this_sorteach[iroi]]
            #lin_subtracted[ipartitioning] = this_lin_subtracted[this_sorteach]
        #lin_subtracted = np.concatenate(lin_subtracted,axis=0)
        #sorteach = np.array([np.arange(32) for iroi in range(lin_subtracted.shape[0])])
        #sortind = np.arange(lin_subtracted.shape[0])
        print(lin_subtracted[0].shape)
        #print(np.nanmax(lin_subtracted))
        lin_subtracted = np.nanmean(np.array(lin_subtracted),axis=0)
        sorteach = np.array([np.arange(32) for iroi in range(lin_subtracted.shape[0])])
        sortind = np.arange(lin_subtracted.shape[0])
        plot_sorted_lin_subtracted_(targets[irun],lin_subtracted,sorteach,sortind)
    return tunings
    # reshape to iroi x ipartitioning x ipartition x istim
    # for each session
    # for each partitioning
    # compute sorting and linear difference of B based on A
    # average across partitionings
    # compute bootstrapped error bars across neurons

# ##

def plot_example_tuning_curves(target,dsname,selected_expts,selected_rois,keylist=None,datafield='decon',gen_nub_selector=utils.gen_nub_selector_v1,scale=1,irun=1):
    bounds = compute_bounds(dsname,keylist,gen_nub_selector=gen_nub_selector,datafield=datafield)
    plot_example_tuning_curves_(bounds[irun],selected_expts,selected_rois,scale=scale)
    plt.savefig(target,dpi=300)

###
# not used in final analysis
def plot_example_tuning_curves_faster(target,dsname,selected_expts,selected_rois,keylist=None,datafield='decon',gen_nub_selector=utils.gen_nub_selector_v1,scale=1,irun=1,pct=(16,84,50)):
    bounds = compute_bounds_faster(dsname,keylist,gen_nub_selector=gen_nub_selector,datafield=datafield,pct=pct)
    plot_example_tuning_curves_(bounds[irun],selected_expts,selected_rois,scale=scale)
    plt.savefig(target,dpi=300)

# ##

def plot_example_tuning_curves_(bounds,selected_expts,selected_rois,scale=1,ylim=None,xlim=None,to_plot=slice(None),\
                                zero_centered=False,aspect_ratio=(1,1.25),linewidth=None,linedash=None,isolated=True):
    if ylim is None:
        ylim = [None for iroi in selected_rois]
    else:
        ylim = [(0,0.5) for iroi in selected_rois]
    nroi = len(selected_rois)
    if isolated:
        plt.figure(figsize=(aspect_ratio[0]*len(selected_expts)*scale,aspect_ratio[1]*len(selected_expts)*scale))
    these_numbers_of_patches = [np.arange(1,6), np.arange(6,16), np.arange(16,26), np.arange(26,31), np.array((31,))]
    colors = [parula((itn)/3) for itn,tn in enumerate(these_numbers_of_patches[:-1])]
    colors = colors + [np.array((1,0.65,0))]
    iiroi = 0
    for iexpt,iroi in zip(selected_expts,selected_rois):#
        if isolated:
            plt.subplot(nroi,1,iiroi+1)
            markersize = 4
        else:
            markersize = 10
        yerr_down = bounds[iexpt][2][iroi][evan_order_actual]-bounds[iexpt][0][iroi][evan_order_actual]
        yerr_up = bounds[iexpt][1][iroi][evan_order_actual]-bounds[iexpt][2][iroi][evan_order_actual]
        yerr = np.concatenate((yerr_down[np.newaxis],yerr_up[np.newaxis]),axis=0)
        for itn,tn in enumerate(these_numbers_of_patches):
        # plt.fill_between(np.arange(32),bounds[0][0][0][iroi][evan_order_actual],bounds[0][0][1][iroi][evan_order_actual])
            plt.errorbar(tn,bounds[iexpt][2][iroi][evan_order_actual][tn],c=colors[itn],yerr=yerr[:,tn],\
                         capsize=1.5,fmt='.',markersize=markersize)
        if not linewidth is None:
            line = plt.plot(0.5+np.arange(32),np.zeros((32,)),linestyle='dashed',c='k',linewidth=linewidth)
        else:
            line = plt.plot(0.5+np.arange(32),np.zeros((32,)),linestyle='dashed',c='k')
        if isolated:
            scalebar = 0.1
            plt.plot((33.5,33.5),(0,scalebar),c='k')
        else:
            scalebar = 0
        if not linedash is None:
            line[0].set_dashes(linedash)
        plt.axis('off')
        if not ylim[iiroi] is None:
            plt.ylim(ylim[iiroi])
        else:
            mx_hi = np.max(np.abs(bounds[iexpt][1][iroi][evan_order_actual]))
            mx_lo = np.max(np.abs(bounds[iexpt][0][iroi][evan_order_actual]))
            mx = np.maximum(mx_lo,mx_hi)
            mx = np.maximum(mx,scalebar)
            if zero_centered:
                plt.ylim((-1.1*mx,1.1*mx))
            else:
                plt.ylim((0,1.1*mx))
        if not xlim is None:
            plt.xlim(xlim)
        iiroi = iiroi+1

def plot_example_lin_subtracted(target,dsname,selected_expts,selected_rois,keylist=None,datafield='decon',gen_nub_selector=utils.gen_nub_selector_v1,scale=1,irun=1,bounds=None,run_all=True,pct=(16,84,50)):
    if bounds is None:
        bounds = compute_bounds(dsname,keylist,gen_nub_selector=gen_nub_selector,datafield=datafield,pct=pct)
    if run_all:
        lin_subtracted_bounds = utils.compute_lin_subtracted_bounds(bounds[irun])
        plot_example_tuning_curves_(lin_subtracted_bounds,selected_expts,selected_rois,scale=scale,ylim=None,xlim=(5.5,34),zero_centered=True,aspect_ratio=(1*26/31,1.25),linewidth=0.8,linedash=[7,3])#,to_plot=slice(6,None))
        plt.savefig(target,dpi=300)
        return
    return bounds

def plot_example_sorted_tuning_curves_(bounds,selected_expts,selected_rois,scale=1,ylim=None,xlim=None,to_plot=slice(None),\
                                zero_centered=False,aspect_ratio=(1,1.25),linewidth=None,linedash=None,plot_order=None,\
                                       plot_order_multi=None,monochrome=False):
    if ylim is None:
        ylim = [None for iroi in selected_rois]
    else:
        ylim = [(0,0.5) for iroi in selected_rois]
    nroi = len(selected_rois)
    plt.figure(figsize=(aspect_ratio[0]*len(selected_expts)*scale,aspect_ratio[1]*len(selected_expts)*scale))
    these_numbers_of_patches = [np.arange(1,6), np.arange(6,16), np.arange(16,26), np.arange(26,31), np.array((31,))]
    colors = [parula((itn)/3) for itn,tn in enumerate(these_numbers_of_patches[:-1])]
    colors = colors + [np.array((1,0.65,0))]
    if monochrome:
        colors = ['k' for color in colors]
    iiroi = 0
    if plot_order is None:
        plot_order = np.zeros((len(selected_rois),32))
        plot_order_multi = np.zeros((len(selected_rois),32))
        plot_order_flag = True
    else:
        plot_order_flag = False
    for iexpt,iroi in zip(selected_expts,selected_rois):#
        plt.subplot(nroi,1,iiroi+1)
        #bds = [b[iroi] for b in bounds[iexpt]]
        #plot_sorted_colored_by_patchno(bds,colors,xlim=xlim,ylim=ylim[iiroi],zero_centered=zero_centered,linewidth=linewidth,linedash=linedash,plot_order=plot_order[iiroi],plot_order_flag=plot_order_flag)
        yerr_down = bounds[iexpt][2][iroi][evan_order_actual]-bounds[iexpt][0][iroi][evan_order_actual]
        yerr_up = bounds[iexpt][1][iroi][evan_order_actual]-bounds[iexpt][2][iroi][evan_order_actual]
        #yerr_down = bounds[iexpt][2][iroi]-bounds[iexpt][0][iroi]
        #yerr_up = bounds[iexpt][1][iroi]-bounds[iexpt][2][iroi]
        yerr = np.concatenate((yerr_down[np.newaxis],yerr_up[np.newaxis]),axis=0)
        if plot_order_flag:
            plot_order_ = np.argsort(bounds[iexpt][2][iroi][evan_order_actual])[::-1]
            plot_order[iiroi,plot_order_] = np.arange(32)
            plot_order_multi_ = 6 + np.argsort(bounds[iexpt][2][iroi][evan_order_actual][6:])[::-1]
            plot_order_multi[iiroi,plot_order_multi_] = np.arange(6,32)
            #plot_order_ = np.argsort(bounds[iexpt][2][iroi])[::-1]
            #plot_order[iiroi,plot_order_] = np.arange(32)
            #plot_order_multi_ = 6 + np.argsort(bounds[iexpt][2][iroi][6:])[::-1]
            #plot_order_multi[iiroi,plot_order_multi_] = np.arange(6,32)
        for itn,tn in enumerate(these_numbers_of_patches):
        # plt.fill_between(np.arange(32),bounds[0][0][0][iroi][evan_order_actual],bounds[0][0][1][iroi][evan_order_actual])
            ttn = plot_order[iiroi,tn]
            plt.errorbar(ttn,bounds[iexpt][2][iroi][evan_order_actual][tn],c=colors[itn],yerr=yerr[:,tn],\
            #plt.errorbar(ttn,bounds[iexpt][2][iroi][tn],c=colors[itn],yerr=yerr[:,tn],\
                         capsize=1.5,fmt='.',markersize=4)
        if not linewidth is None:
            line = plt.plot(0.5+np.arange(32),np.zeros((32,)),linestyle='dashed',c='k',linewidth=linewidth)
        else:
            line = plt.plot(0.5+np.arange(32),np.zeros((32,)),linestyle='dashed',c='k')
        if not linedash is None:
            line[0].set_dashes(linedash)
        plt.axis('off')
        if not ylim[iiroi] is None:
            plt.ylim(ylim[iiroi])
        else:
            mx_hi = np.max(np.abs(bounds[iexpt][1][iroi]))
            mx_lo = np.max(np.abs(bounds[iexpt][0][iroi]))
            mx = np.maximum(mx_lo,mx_hi)
            if zero_centered:
                plt.ylim((-1.1*mx,1.1*mx))
            else:
                plt.ylim((0,1.1*mx))
        if not xlim is None:
            plt.xlim(xlim)
        iiroi = iiroi+1
    return plot_order,plot_order_multi


def plot_sorted_colored_by_patchno(bds,xlim=None,ylim=None,zero_centered=False,linewidth=None,linedash=None,plot_order=None,plot_order_flag=False,monochrome=False,markersize=4):
    these_numbers_of_patches = [np.arange(1,6), np.arange(6,16), np.arange(16,26), np.arange(26,31), np.array((31,))]
    colors = [parula((itn)/3) for itn,tn in enumerate(these_numbers_of_patches[:-1])]
    colors = colors + [np.array((1,0.65,0))]
    if monochrome:
        colors = ['k' for color in colors]
    yerr_down = bds[2][evan_order_actual]-bds[0][evan_order_actual]
    yerr_up = bds[1][evan_order_actual]-bds[2][evan_order_actual]
    yerr = np.concatenate((yerr_down[np.newaxis],yerr_up[np.newaxis]),axis=0)
    if plot_order_flag:
        plot_order_ = np.argsort(bds[2][evan_order_actual])[::-1]
        plot_order[plot_order_] = np.arange(32)
        plot_order_multi_ = 6 + np.argsort(bds[2][evan_order_actual][6:])[::-1]
        plot_order_multi[plot_order_multi_] = np.arange(6,32)
    for itn,tn in enumerate(these_numbers_of_patches):
    # plt.fill_between(np.arange(32),bounds[0][0][0][iroi][evan_order_actual],bounds[0][0][1][iroi][evan_order_actual])
        ttn = plot_order[tn]
        plt.errorbar(ttn,bds[2][evan_order_actual][tn],c=colors[itn],yerr=yerr[:,tn],\
                     capsize=1.5,fmt='.',markersize=markersize)
    if not linewidth is None:
        line = plt.plot(0.5+np.arange(32),np.zeros((32,)),linestyle='dashed',c='k',linewidth=linewidth)
    else:
        line = plt.plot(0.5+np.arange(32),np.zeros((32,)),linestyle='dashed',c='k')
    if not linedash is None:
        line[0].set_dashes(linedash)
    plt.axis('off')
    if not ylim is None:
        plt.ylim(ylim)
    else:
        mx_hi = np.max(np.abs(bds[1][evan_order_actual]))
        mx_lo = np.max(np.abs(bds[0][evan_order_actual]))
        mx = np.maximum(mx_lo,mx_hi)
        if zero_centered:
            plt.ylim((-1.1*mx,1.1*mx))
        else:
            plt.ylim((0,1.1*mx))
    if not xlim is None:
        plt.xlim(xlim)

def plot_example_sorted_lin_subtracted(target,dsname,selected_expts,selected_rois,keylist=None,datafield='decon',\
                                       gen_nub_selector=utils.gen_nub_selector_v1,scale=1,irun=1,bounds=None,\
                                       run_all=True,pct=(16,84,50),plot_order=None,monochrome=False):
    if bounds is None:
        bounds = compute_bounds(dsname,keylist,gen_nub_selector=gen_nub_selector,datafield=datafield,pct=pct)
    if run_all:
        lin_subtracted_bounds = utils.compute_lin_subtracted_bounds(bounds[irun])
        plot_example_sorted_tuning_curves_(lin_subtracted_bounds,selected_expts,selected_rois,scale=scale,ylim=None,\
                                           xlim=(5.5,32),zero_centered=True,aspect_ratio=(1*26/31,1.25),linewidth=0.8,\
                                           linedash=[7,3],plot_order=plot_order,monochrome=monochrome)#,to_plot=slice(6,None))
        plt.savefig(target,dpi=300)
        return
    return bounds

def plot_roi_pref_patch_(dsname,expt,lkat,pref_nub):
    with ut.hdf5read(dsname) as ds:
        msk = ds[expt]['cell_mask'][:]
        nroi = msk.shape[0]
        #gd = np.random.randn(nroi)>0.5
        #pref_nub = np.random.randint(0,5,nroi)
        colors = [None for inub in range(5)]
        colors[0] = np.array((237,32,36))/255
        colors[1] = np.array((246,235,20))/255
        colors[2] = np.array((111,204,221))/255
        colors[3] = np.array((152,204,112))/255
        colors[4] = np.array((67,120,188))/255
        msk_image = np.ones(msk.shape[1:]+(3,))
        for inub in range(5):
            this_nub = np.where(lkat & (pref_nub==inub))[0]
            for ithis_nub in this_nub:
                msk_image[msk[ithis_nub]] = mc.to_rgb(colors[inub])
        msk_image[464:470,550:592,:] = 0
        plt.figure()
        plt.imshow(msk_image)
        plt.axis('off')
def plot_roi_pref_patch(target,dsname,expt,datafield='decon',run_fn=move_fn,gen_nub_selector=utils.gen_nub_selector_v1,dfof_cutoff=None,pcutoff=0.05,run_condition=0,rcutoff=-1):
    # compute fraction of cells that prefer each number of patches, and plot them with errorbars
    # errorbars on N cells are computed as 1/sqrt(N)
    tuning = compute_tuning(dsname,[expt],datafield,run_fn,gen_nub_selector=gen_nub_selector)
    lkat = compute_lkat_two_criteria(tuning,dsname,[expt],run_fn,dfof_cutoff=dfof_cutoff,pcutoff=pcutoff,rcutoff=rcutoff)
    pref_nub = np.argsort(tuning[run_condition][0][0][:,evan_order_actual[1:6]],axis=1)[:,-1]
    plot_roi_pref_patch_(dsname,expt,lkat[run_condition],pref_nub)
    plt.savefig(target,dpi=300)

def plot_nub_outlines():
    plt.plot((-15,15),(-5,-5),c='k')
    plt.plot((-15,15),(5,5),c='k')
    plt.plot((-5,-5),(-15,15),c='k')
    plt.plot((5,5),(-15,15),c='k')
    plt.plot((15,15),(-5,5),c='k')
    plt.plot((-15,-15),(-5,5),c='k')
    plt.plot((-5,5),(15,15),c='k')
    plt.plot((-5,5),(-15,-15),c='k')

def plot_rf_cloud(dsname,mag=3):
    with ut.hdf5read(dsname) as ds:
        keylist = list(ds.keys())
    #     plt.figure()
    #     im = ds[keylist[-2]]['mean_green_channel'][1]
        for key in keylist:
            if 'cell_center' in ds[key] and 'rf_ctr' in ds[key]['retinotopy_0']:
                rf_ctr = ds[key]['retinotopy_0']['rf_ctr'][:]
                roi_ctr = ds[key]['cell_center'][:].T
                rf_sigma = ds[key]['retinotopy_0']['rf_sigma'][:]
                lkat = (ds[key]['retinotopy_0']['rf_sq_error'][:] < 1) \
                & (rf_sigma > 5)
                plt.figure(figsize=(mag,mag))
                patches = []
                for iroi in range(lkat.sum()):
                    circ = ptch.Circle((rf_ctr[0][lkat][iroi],rf_ctr[1][lkat][iroi]),rf_sigma[lkat][iroi])#,c=roi_ctr[0][lkat])
                    patches.append(circ)
                p = coll.PatchCollection(patches,alpha=0.005)
                plt.gca().add_collection(p)
                #             plt.colorbar()
                plot_nub_outlines()
                plt.axis('off')
                plt.axis('equal')
                plt.xlim((-25,25))
                plt.ylim((-25,25))
                plt.savefig('figures/'+key+'_rf_locations_circles.jpg',dpi=300)
            else:
                print('could not do %s' % key)

def plot_rf_centers_colored(dsname,mag=3):
    with ut.hdf5read(dsnames[2]) as ds:
        keylist = list(ds.keys())
    #     plt.figure()
    #     im = ds[keylist[-2]]['mean_green_channel'][1]
        for key in keylist:
            if 'cell_center' in ds[key] and 'rf_ctr' in ds[key]['retinotopy_0']:
                rf_ctr = ds[key]['retinotopy_0']['rf_ctr'][:]
                roi_ctr = ds[key]['cell_center'][:].T
                lkat = (ds[key]['retinotopy_0']['rf_sq_error'][:] < 1) \
                & (ds[key]['retinotopy_0']['rf_sigma'][:] > 5)
                plt.figure(figsize=(2*mag,mag))
                for idim in range(2):
                    plt.subplot(1,2,idim+1)
                    plt.scatter(rf_ctr[0][lkat],rf_ctr[1][lkat],c=roi_ctr[idim][lkat]*670/796,s=10)
                    plt.colorbar()
                    plot_nub_outlines()
                    plt.axis('off')
                    plt.axis('equal')
    #             plt.savefig('figures/'+key+'_rf_locations_colored.jpg',dpi=300)
            else:
                print('could not do %s' % key)

def scatter_tuning_curves_decon_dfof(targets,dsname,keylist=None,datafield1='repwise_dFoF',datafield2='decon',\
                               run_fn=move_fn,gen_nub_selector=utils.gen_nub_selector_v1,\
                               rcutoff=-1,pcutoff=0.05,dfof_cutoff=0.):
    with ut.hdf5read(dsname) as ds:
        datafields = [datafield1,datafield2]
        tuning = [compute_tuning(dsname,keylist,d,run_fn,gen_nub_selector=gen_nub_selector,seed=0,\
                                 include_all=True) for d in datafields]
        lkat = [compute_lkat_two_criteria(t,dsname,keylist,run_fn,dfof_cutoff=dfof_cutoff,pcutoff=pcutoff,rcutoff=rcutoff)\
               for t in tuning]
    return tuning,lkat
   #     data = {}    
   #     if keylist is None:
   #         keylist = list(ds.keys())
   #     for key in keylist:
   #         nub_0 = ds[key]['nub_0']
   #         nbefore = nub_0['nbefore'][()]
   #         nafter = nub_0['nafter'][()]
   #         data[key] = [np.nanmean(nub_0[datafield][:,nbefore:-nafter],axis=1) for datafield in datafields]
   # return data
def scatter_tuning_curves_decon_dfof_after_blank(targets,dsname,keylist=None,datafield1='repwise_dFoF',datafield2='decon',\
                               run_fn=move_fn,gen_nub_selector=utils.gen_nub_selector_v1,\
                               rcutoff=-1,pcutoff=0.05,dfof_cutoff=0.):
    with ut.hdf5read(dsname) as ds:
        datafields = [datafield1,datafield2]
        tuning = [compute_tuning_after_blank(dsname,keylist,d,run_fn,gen_nub_selector=gen_nub_selector,seed=0,\
                                 include_all=True) for d in datafields]
        lkat = [compute_lkat_two_criteria(t,dsname,keylist,run_fn,dfof_cutoff=dfof_cutoff,pcutoff=pcutoff,rcutoff=rcutoff)\
               for t in tuning]
    return tuning,lkat

def exponential_fn(x,a,b):
    return a*np.exp(b*x)

def fit_exponentials_to_linear_diff(roiwise_lin_subtracted_sorted):
    popt = [None for itype in range(4)]

    for itype in range(4):
        print(itype)
        nroi,_,nstim = roiwise_lin_subtracted_sorted[itype].shape

        popt[itype] = np.zeros((nroi,2))
        for iroi in range(nroi):
            try:
                popt[itype][iroi],_ = sop.curve_fit(exponential_fn,np.arange(nstim),roiwise_lin_subtracted_sorted[itype][iroi,0],p0=(1,-0.1),bounds=([-np.inf,-np.inf],[np.inf,0]))
            except:
                print('could not do %d'%iroi)
    return popt

def plot_linear_diff_exp_sorted(roiwise_lin_subtracted_sorted):
    popt = fit_exponentials_to_linear_diff(roiwise_lin_subtracted_sorted)
    for itype in range(len(roiwise_lin_subtracted_sorted)):
        plt.figure()
        plt.imshow(roiwise_lin_subtracted_sorted[itype][np.argsort(popt[itype][:,1]),0],extent=[0,1,0,1],cmap='bwr',vmin=-1,vmax=1)
        plt.axis('off')

def plot_tuning_curve_exp_sorted(roiwise_tuning_sorted):
    popt = fit_exponentials_to_linear_diff(roiwise_tuning_sorted)
    for itype in range(len(roiwise_tuning_sorted)):
        plt.figure()
        order = np.argsort(popt[itype][:,1])
        data = roiwise_tuning_sorted[itype][order,0]
        plt.imshow(data/data.max(1)[:,np.newaxis],extent=[0,1,0,1],vmin=0,vmax=1,cmap=parula)
        plt.axis('off')

def reorder_stims(nub_ordering,flipped=None,nub_var=utils.nubs_active):
    if flipped is None:
        flipped = np.zeros(nub_ordering.shape,dtype='bool')
    nnub = nub_var.shape[1]
    binary_vals = nub_var.copy()
    for inub in range(len(nub_ordering)):
        if flipped[inub]:
            binary_vals[:,inub] = ~binary_vals[:,inub]
    binary_vals = binary_vals[:,list(nub_ordering)]
    multby = np.array([2**inub for inub in range(nnub)][::-1])[np.newaxis,:]
    ordering = np.argsort((multby*binary_vals).sum(1))
    return ordering

def show_plot_reordered_plot_and_fit(this_bounds,this_theta,cbd=1,nub_var=utils.nubs_active,rank_order=False,s1=False):
    nnub = 5
    this_prediction = utils.predict_output_theta_amplitude(this_theta,fn=utils.f_mt,nub_var=nub_var)
    
    flipped = this_theta[:-2] < 0
    if rank_order:
        stim_ordering = np.argsort(-this_prediction)
    else:
        nub_ordering = np.argsort(np.abs(this_theta[:-2]))[::-1]
        stim_ordering = reorder_stims(nub_ordering,flipped=flipped,nub_var=nub_var)
    
    plt.clf()
    plt.subplot(1,3,1)
    #if s1:#s1:
    this_expanded_bounds = [[tb[np.newaxis] for tb in this_bounds]]
    #else:
    #    this_expanded_bounds = [[tb[np.newaxis,evan_order_actual] for tb in this_bounds]]
    plot_example_tuning_curves_(this_expanded_bounds,[0],[0],isolated=False)
    plt.gca().set_ylim(bottom=-0.01*np.nanmax(this_expanded_bounds[0][1]))
    plt.title('Tuning curve')
    plt.ylabel('event rate (a.u.)')
    plt.subplot(1,3,3)
    #plot_order = stim_ordering[evan_order_actual]
    #plot_order = np.arange(32)
    #plot_order[evan_order_actual] = stim_ordering
    #this_stim_ordering = np.argsort(-this_prediction[evan_order_actual])
    this_stim_ordering = np.arange(evan_order_actual.shape[0])
    this_stim_ordering[np.argsort(-this_prediction[evan_order_actual])] = np.arange(evan_order_actual.shape[0])
    plot_order = this_stim_ordering
    plot_sorted_colored_by_patchno(this_bounds,xlim=None,ylim=None,zero_centered=False,linewidth=None,linedash=None,plot_order=plot_order,plot_order_flag=False,markersize=10)
    plt.gca().set_ylim(bottom=-0.01*np.nanmax(this_expanded_bounds[0][1]))
    plt.plot(np.arange(2**nnub),this_prediction[np.argsort(-this_prediction)],c='k',zorder=50)
    #plt.plot(np.arange(2**nnub),this_bounds[2][stim_ordering],label='data')
    #plt.fill_between(np.arange(2**nnub),this_bounds[0][stim_ordering],this_bounds[1][stim_ordering],alpha=0.25)
    #plt.gca().set_ylim(bottom=-0.01*np.nanmax(this_expanded_bounds[0][1]))
    #plt.xticks([])
    #plt.yticks([])
    #plt.plot(np.arange(2**nnub),this_prediction[stim_ordering],label='model')
    #plt.legend()
    #plt.ylabel('event rate (a.u.)')
    plt.title('Reordered tuning curve')

    ut.erase_top_right()
    if rank_order:
        plt.xlabel('Stimulus rank')
        xtk = np.array((1,5,10,15,20,25,30))
        plt.xticks(xtk-1,xtk)
    else:
        signs = [] #'' for inub in range(len(flipped))]
        for inub in range(len(nub_ordering)):
            if flipped[nub_ordering[inub]]:
                signs.append('-')
            else:
                signs.append('+')
        markers = ','.join([s+str(n) for s,n in zip(signs,nub_ordering)])
        plt.xlabel('stimulus # (ordered [' + markers + '])')
    plt.subplot(1,3,2)
    #if s1:
    #nub_order = np.array((0,1,2,3,4))
    #else:
    #nub_order = np.array((0,3,2,4,1))
    if s1:
        nub_lbls = ['C1',r'$\gamma$','B1','C2','D1']
    else:
        nub_lbls = ['1','5','3','2','4']
    utils.show_fit(this_theta,cbd=cbd,nub_lbls=nub_lbls)#add_1=True)#,nub_order=nub_order)
    plt.title('GLM fit parameters')
    plt.colorbar()
