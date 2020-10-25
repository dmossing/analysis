#!/usr/bin/env python

import pyute as ut
import autograd.numpy as np
import matplotlib.pyplot as plt
import sklearn
import h5py
import pdb
import scipy.optimize as sop
from autograd import elementwise_grad as egrad
from mpl_toolkits.mplot3d import Axes3D
import sklearn.discriminant_analysis as skd
import autograd.scipy.special as ssp
from autograd import jacobian
import size_contrast_analysis as sca
import scipy.stats as sst

def compute_tuning(dsfile,datafield='decon',running=True,expttype='size_contrast_0',run_cutoff=10,running_pct_cutoff=0.4):
    # take in an HDF5 data struct, and convert to an n-dimensional matrix
    # describing the tuning curve of each neuron. For size-contrast stimuli, 
    # the dimensions of this matrix are ROI index x size x contrast x direction x time. 
    # This outputs a list of such matrices, where each element is one imaging session
    with h5py.File(dsfile,mode='r') as f:
        keylist = [key for key in f.keys()]
        tuning = [None for i in range(len(keylist))]
        uparam = [None for i in range(len(keylist))]
        displacement = [None for i in range(len(keylist))]
        pval = [None for i in range(len(keylist))]
        for ikey in range(len(keylist)):
            session = f[keylist[ikey]]
            print(session)
            #print([key for key in session.keys()])
            if expttype in session and datafield in session[expttype]:
                sc0 = session[expttype]
                print(datafield)
                data = sc0[datafield][:]
                stim_id = sc0['stimulus_id'][:]
                nbefore = sc0['nbefore'][()]
                nafter = sc0['nafter'][()]
                if running:
                    trialrun = sc0['running_speed_cm_s'][:,nbefore:-nafter].mean(-1)>run_cutoff #
                else:
                    trialrun = sc0['running_speed_cm_s'][:,nbefore:-nafter].mean(-1)<run_cutoff
                #print(sc0['running_speed_cm_s'].shape)
                print(np.nanmean(trialrun))
                if np.nanmean(trialrun)>running_pct_cutoff:
                    tuning[ikey] = ut.compute_tuning(data,stim_id,trial_criteria=trialrun)[:]
                uparam[ikey] = []
                for param in sc0['stimulus_parameters']:
                    uparam[ikey] = uparam[ikey]+[sc0[param][:]]
                if 'rf_displacement_deg' in sc0:
                    pval[ikey] = sc0['rf_mapping_pval'][:]
                    sqerror = session['retinotopy_0']['rf_sq_error'][:]
                    sigma = session['retinotopy_0']['rf_sigma'][:]
                    X = session['cell_center'][:]
                    y = sc0['rf_displacement_deg'][:].T
                    rf_conditions = [ut.k_and(~np.isnan(X[:,0]),~np.isnan(y[:,0])),sqerror<0.75,sigma>3.3,pval[ikey]<0.1]
                    lkat = np.ones((X.shape[0],),dtype='bool')
                    for cond in rf_conditions:
                        lkat_new = (lkat & cond)
                        if lkat_new.sum()>=5:
                            lkat = lkat_new.copy()
                    linreg = sklearn.linear_model.LinearRegression().fit(X[lkat],y[lkat])
                    displacement[ikey] = np.zeros_like(y)
                    displacement[ikey][~np.isnan(X[:,0])] = linreg.predict(X[~np.isnan(X[:,0])])
    return tuning,uparam,displacement,pval

def compute_tuning_tavg_with_sem(dsfile,datafield='decon',running=True,expttype='size_contrast_0',run_cutoff=10,running_pct_cutoff=0.4):
    # take in an HDF5 data struct, and convert to an n-dimensional matrix
    # describing the tuning curve of each neuron. For size-contrast stimuli, 
    # the dimensions of this matrix are ROI index x size x contrast x direction x time. 
    # This outputs a list of such matrices, where each element is one imaging session
    with h5py.File(dsfile,mode='r') as f:
        keylist = [key for key in f.keys()]
        tuning = [None for i in range(len(keylist))]
        tuning_sem = [None for i in range(len(keylist))]
        uparam = [None for i in range(len(keylist))]
        displacement = [None for i in range(len(keylist))]
        pval = [None for i in range(len(keylist))]
        for ikey in range(len(keylist)):
            session = f[keylist[ikey]]
            print(session)
            #print([key for key in session.keys()])
            if expttype in session and datafield in session[expttype]:
                sc0 = session[expttype]
                print(datafield)
                data = sc0[datafield][:]
                stim_id = sc0['stimulus_id'][:]
                nbefore = sc0['nbefore'][()]
                nafter = sc0['nafter'][()]
                if running:
                    trialrun = sc0['running_speed_cm_s'][:,nbefore:-nafter].mean(-1)>run_cutoff #
                else:
                    trialrun = sc0['running_speed_cm_s'][:,nbefore:-nafter].mean(-1)<run_cutoff
                #print(sc0['running_speed_cm_s'].shape)
                print(np.nanmean(trialrun))
                if np.nanmean(trialrun)>running_pct_cutoff:
                    tuning[ikey],tuning_sem[ikey] = ut.compute_tuning_tavg_with_sem(data,stim_id,trial_criteria=trialrun,nbefore=nbefore,nafter=nafter)[:]
                uparam[ikey] = []
                for param in sc0['stimulus_parameters']:
                    uparam[ikey] = uparam[ikey]+[sc0[param][:]]
                if 'rf_displacement_deg' in sc0:
                    pval[ikey] = sc0['rf_mapping_pval'][:]
                    sqerror = session['retinotopy_0']['rf_sq_error'][:]
                    sigma = session['retinotopy_0']['rf_sigma'][:]
                    X = session['cell_center'][:]
                    y = sc0['rf_displacement_deg'][:].T
                    rf_conditions = [ut.k_and(~np.isnan(X[:,0]),~np.isnan(y[:,0])),sqerror<0.75,sigma>3.3,pval[ikey]<0.1]
                    lkat = np.ones((X.shape[0],),dtype='bool')
                    for cond in rf_conditions:
                        lkat_new = (lkat & cond)
                        if lkat_new.sum()>=5:
                            lkat = lkat_new.copy()
                    linreg = sklearn.linear_model.LinearRegression().fit(X[lkat],y[lkat])
                    displacement[ikey] = np.zeros_like(y)
                    displacement[ikey][~np.isnan(X[:,0])] = linreg.predict(X[~np.isnan(X[:,0])])
    return tuning,tuning_sem,uparam,displacement,pval

def compute_tunings(dsnames,datafield='decon',running=True,expttype='size_contrast_0',run_cutoff=10,running_pct_cutoff=0.4):
    # compute tuning as above, for each of a list of HDF5 files each corresponding to a particular cell type
    tunings = []
    uparams = []
    displacements = []
    pvals = []
    for dsname in dsnames:
        print(dsname)
        this_tuning,this_uparam,this_displacement,this_pval = compute_tuning(dsname,datafield=datafield,running=running,expttype=expttype,run_cutoff=run_cutoff,running_pct_cutoff=running_pct_cutoff)
        tunings.append(this_tuning)
        uparams.append(this_uparam)
        displacements.append(this_displacement)
        pvals.append(this_pval)
    return tunings,uparams,displacements,pvals

def compute_tunings_tavg_with_sem(dsnames,datafield='decon',running=True,expttype='size_contrast_0',run_cutoff=10,running_pct_cutoff=0.4):
    # compute tuning as above, for each of a list of HDF5 files each corresponding to a particular cell type
    tunings = []
    tunings_sem = []
    uparams = []
    displacements = []
    pvals = []
    for dsname in dsnames:
        print(dsname)
        this_tuning,this_tuning_sem,this_uparam,this_displacement,this_pval = compute_tuning_tavg_with_sem(dsname,datafield=datafield,running=running,expttype=expttype,run_cutoff=run_cutoff,running_pct_cutoff=running_pct_cutoff)
        tunings.append(this_tuning)
        tunings_sem.append(this_tuning)
        uparams.append(this_uparam)
        displacements.append(this_displacement)
        pvals.append(this_pval)
    return tunings,tunings_sem,uparams,displacements,pvals
        
def default_dsnames():
    dsbase = '/home/mossing/Documents/notebooks/shared_data/'
    dsnames = [dsbase+x+'_data_struct.hdf5' for x in ['pyr_l4','pyr_l23','sst_l23','vip_l23']]
    return dsnames

def default_selection():
    # select experiments which had all of the relevant stim conditions
    selection = [None, None, None, [1,2,3,4,5,9], None]
    return selection

def default_condition_inds():
    # select the relevant stim conditions for experiments that had extra ones
    slices = [slice(None,5),[0,-5,-4,-3,-2,-1]]
    return slices
    
def average_up(arr,nbefore=8,nafter=8,average_ori=True):
    # average across time points and directions
    ndim = len(arr.shape)
    slicer = [slice(None) for idim in range(ndim)]
    slicer[-1] = slice(nbefore,-nafter)
    tmean = np.nanmean(arr[slicer],-1)
    if average_ori:
        return np.nanmean(tmean,-1)
    else:
        return tmean

def columnize(arr):
    output = np.nanmean(arr,0).flatten()
    output = output/output.max()
    return output

def include_aligned(displacement,dcutoff,pval,pcutoff=0.05,less=True):
    # split data up into spatial pixels, according to the distance of the RF center from the stimulus center
    if less:
        criterion = lambda x: (x**2).sum(0) < dcutoff**2
    else:
        criterion = lambda x: (x**2).sum(0) > dcutoff**2
    return np.logical_and(criterion(displacement),pval < pcutoff)

def gen_rspatial(dsnames=None,selection=None,dcutoffs=[0,5,10,15],pval_cutoff=0.05,slices=None,datafield='decon',run_cutoff=10,running_pct_cutoff=0.4):
    # from a list of HDF5 files, split up the data into an arbitrary number of spatial pixels based on RF center location
    if dsnames is None:
        dsnames = default_dsnames()
    if selection is None:
        selection = default_selection()
    if slices is None:
        condition_inds = default_condition_inds()
        
    tunings,uparams,displacements,pvals = compute_tunings(dsnames,datafield=datafield,run_cutoff=run_cutoff,running_pct_cutoff=running_pct_cutoff)
    
    rs = []
    for icelltype in range(len(tunings)):
        rs.append([])
        these_tunings = tunings[icelltype]
        these_displacements = displacements[icelltype]
        these_pvals = pvals[icelltype]
        if not selection[icelltype] is None:
            sel = selection[icelltype]
        else:
            sel = np.arange(len(these_tunings))
        these_displacements = [these_displacements[i].T for i in sel if not these_tunings[i] is None]
        these_pvals = [these_pvals[i] for i in sel if not these_tunings[i] is None]
        these_tunings = [these_tunings[i] for i in sel if not these_tunings[i] is None]
        for idcutoff in range(len(dcutoffs)):
            dcutoff = dcutoffs[idcutoff]
            if len(dcutoffs)>idcutoff+1:
                dcuton = dcutoffs[idcutoff+1]
                aligned = [include_aligned(d,dcutoff,p,pval_cutoff,less=False) & include_aligned(d,dcuton,p,pval_cutoff,less=True) for d,p in zip(these_displacements,these_pvals)]
            else:
                aligned = [include_aligned(d,dcutoff,p,pval_cutoff,less=False) for d,p in zip(these_displacements,these_pvals)]
#         aligned = [include_aligned(d,dcutoff,p,pval_cutoff,less=True) for d,p in zip(these_displacements,these_pvals)]
#         misaligned = [include_aligned(d,dcutoff,p,pval_cutoff,less=False) for d,p in zip(these_displacements,these_pvals)]
            raligned = average_up(np.concatenate([x[aligned[i]][:,condition_inds[0],condition_inds[1]] for i,x in enumerate(these_tunings)],axis=0))
#         rmisaligned = average_up(np.concatenate([x[misaligned[i]][:,condition_inds[0],condition_inds[1]] for i,x in enumerate(these_tunings)],axis=0))
            rs[icelltype].append(raligned)
    return rs

def gen_rs(dsnames=None,selection=None,dcutoff=5,pval_cutoff=0.05,slices=None,running=True,expttype='size_contrast_0',run_cutoff=10,running_pct_cutoff=0.4):
    # same specifically for case of two spatial pixels
    if dsnames is None:
        dsnames = default_dsnames()
    if selection is None:
        selection = default_selection()
    if slices is None:
        condition_inds = default_condition_inds()
    nbefore = 8
    nafter = 8
        
    tunings,uparams,displacements,pvals = compute_tunings(dsnames,running=running,expttype=expttype,run_cutoff=run_cutoff,running_pct_cutoff=running_pct_cutoff)
    
    rs = []
    for icelltype in range(len(tunings)):
        these_tunings = tunings[icelltype]
        these_displacements = displacements[icelltype]
        these_pvals = pvals[icelltype]
        if not selection[icelltype] is None:
            sel = selection[icelltype]
        else:
            sel = np.arange(len(these_tunings))
        these_displacements = [these_displacements[i].T for i in sel if not these_tunings[i] is None]
        these_pvals = [these_pvals[i] for i in sel if not these_tunings[i] is None]
        these_tunings = [these_tunings[i] for i in sel if not these_tunings[i] is None]
        #these_tunings = [np.nanmean(np.nanmean(t[:,:,:,:,nbefore:-nafter],-1),-1) for t in these_tunings]
        aligned = [include_aligned(d,dcutoff,p,pval_cutoff,less=True) for d,p in zip(these_displacements,these_pvals)]
        misaligned = [include_aligned(d,dcutoff,p,pval_cutoff,less=False) for d,p in zip(these_displacements,these_pvals)]
        raligned = np.concatenate([average_up(x[aligned[i]][:,condition_inds[0],condition_inds[1]]) for i,x in enumerate(these_tunings)],axis=0)
        rmisaligned = np.concatenate([average_up(x[misaligned[i]][:,condition_inds[0],condition_inds[1]]) for i,x in enumerate(these_tunings)],axis=0)
        #raligned = average_up(np.concatenate([x[aligned[i]][:,condition_inds[0],condition_inds[1]] for i,x in enumerate(these_tunings)],axis=0))
        #rmisaligned = average_up(np.concatenate([x[misaligned[i]][:,condition_inds[0],condition_inds[1]] for i,x in enumerate(these_tunings)],axis=0))
        rs.append([raligned,rmisaligned])
    return rs

def gen_rs_modal_uparam(dsnames=None,selection=None,dcutoff=5,pval_cutoff=0.05,modal_uparam=None,running=True,expttype='size_contrast_0',average_ori=True,run_cutoff=10,running_pct_cutoff=0.4):
    # same specifically for case of two spatial pixels
    if dsnames is None:
        dsnames = default_dsnames()
    if selection is None:
        selection = default_selection()
    nbefore = 8
    nafter = 8
    
    nparam = np.array([mu.shape[0] for mu in modal_uparam])
        
    tunings,uparams,displacements,pvals = compute_tunings(dsnames,running=running,expttype=expttype,run_cutoff=run_cutoff,running_pct_cutoff=running_pct_cutoff)
    
    rs = []
    for icelltype in range(len(tunings)):
        these_tunings = tunings[icelltype]
        nexpt = len(these_tunings)
        these_uparams = uparams[icelltype]
        these_displacements = displacements[icelltype]
        these_pvals = pvals[icelltype]
        if not selection[icelltype] is None:
            sel = selection[icelltype]
        else:
            sel = np.arange(len(these_tunings))
        these_uparams = [these_uparams[i] for i in sel if not these_tunings[i] is None]
        these_displacements = [these_displacements[i].T for i in sel if not these_tunings[i] is None]
        these_pvals = [these_pvals[i] for i in sel if not these_tunings[i] is None]
        these_tunings = [these_tunings[i] for i in sel if not these_tunings[i] is None]
        nexpt = len(these_tunings)
        if nexpt>0:
            aligned = [include_aligned(d,dcutoff,p,pval_cutoff,less=True) for d,p in zip(these_displacements,these_pvals)]
            misaligned = [include_aligned(d,dcutoff,p,pval_cutoff,less=False) for d,p in zip(these_displacements,these_pvals)]
            this_rs = []
            for icriterion,criterion in enumerate([aligned,misaligned]):
                nrois = [get_nroi(tt[cc]) for tt,cc in zip(these_tunings,criterion)]
                nframes = [get_nframes(tt) for tt in these_tunings]
                ra = [np.nan*np.ones((nroi,)+tuple(nparam)+(nframe,)) for nroi,nframe in zip(nrois,nframes)]
                for iexpt in range(nexpt):
                    assign_from_uparam(ra[iexpt],modal_uparam,these_tunings[iexpt][criterion[iexpt]],these_uparams[iexpt])
                    ra[iexpt] = average_up(ra[iexpt],average_ori=average_ori)
                ra = np.concatenate(ra,axis=0)
                this_rs.append(ra)
        else:
            this_rs = [None for icriterion in range(2)]
        rs.append(this_rs)
    return rs

def gen_rs_modal_uparam_expt(dsnames=None,selection=None,dcutoff=5,pval_cutoff=0.05,modal_uparam=None,running=True,expttype='size_contrast_0',average_ori=True,run_cutoff=10,running_pct_cutoff=0.4):
    # same specifically for case of two spatial pixels
    if dsnames is None:
        dsnames = default_dsnames()
    if selection is None:
        selection = default_selection()
    nbefore = 8
    nafter = 8
    
    nparam = np.array([mu.shape[0] for mu in modal_uparam])
        
    tunings,uparams,displacements,pvals = compute_tunings(dsnames,running=running,expttype=expttype,run_cutoff=run_cutoff,running_pct_cutoff=running_pct_cutoff)
    
    rs = []
    expt_ids = []
    roi_ids = []
    for icelltype in range(len(tunings)):
        these_tunings = tunings[icelltype]
        #these_tunings_sem = tunings_sem[icelltype]
        nexpt = len(these_tunings)
        these_uparams = uparams[icelltype]
        these_displacements = displacements[icelltype]
        these_pvals = pvals[icelltype]
        if not selection[icelltype] is None:
            sel = selection[icelltype]
        else:
            sel = np.arange(len(these_tunings))
        these_uparams = [these_uparams[i] for i in sel if not these_tunings[i] is None]
        these_displacements = [these_displacements[i].T for i in sel if not these_tunings[i] is None]
        these_pvals = [these_pvals[i] for i in sel if not these_tunings[i] is None]
        these_tunings = [these_tunings[i] for i in sel if not these_tunings[i] is None]
        #these_tunings_sem = [these_tunings_sem[i] for i in sel if not these_tunings_sem[i] is None]
        nexpt = len(these_tunings)
        if nexpt>0:
            aligned = [include_aligned(d,dcutoff,p,pval_cutoff,less=True) for d,p in zip(these_displacements,these_pvals)]
            misaligned = [include_aligned(d,dcutoff,p,pval_cutoff,less=False) for d,p in zip(these_displacements,these_pvals)]
            this_rs = []
            this_expt_ids = []
            this_roi_ids = []
            for icriterion,criterion in enumerate([aligned,misaligned]):
                nrois = [get_nroi(tt[cc]) for tt,cc in zip(these_tunings,criterion)]
                nframes = [get_nframes(tt) for tt in these_tunings]
                ra = [np.nan*np.ones((nroi,)+tuple(nparam)+(nframe,)) for nroi,nframe in zip(nrois,nframes)]
                #ra_sem = [np.nan*np.ones((nroi,)+tuple(nparam)+(nframe,)) for nroi,nframe in zip(nrois,nframes)]
                expt_ida = [np.ones((nroi,))*iexpt for iexpt,nroi in enumerate(nrois)]
                roi_ida = [np.where(cc)[0] for cc in criterion]
                for iexpt in range(nexpt):
                    assign_from_uparam(ra[iexpt],modal_uparam,these_tunings[iexpt][criterion[iexpt]],these_uparams[iexpt])
                    #assign_from_uparam(ra_sem[iexpt],modal_uparam,these_tunings_sem[iexpt][criterion[iexpt]],these_uparams[iexpt])
                    ra[iexpt] = average_up(ra[iexpt],average_ori=average_ori)
                #ra = np.concatenate([np.nanmean(rra,0)[np.newaxis] for rra in ra],axis=0)
                ra = np.concatenate(ra,axis=0)
                expt_ida = np.concatenate(expt_ida,axis=0)
                roi_ida = np.concatenate(roi_ida,axis=0)
                this_rs.append(ra)
                this_expt_ids.append(expt_ida)
                this_roi_ids.append(roi_ida)
        else:
            this_rs = [None for icriterion in range(2)]
            this_expt_ids = [None for icriterion in range(2)]
            this_roi_ids = [None for icriterion in range(2)]
        rs.append(this_rs)
        expt_ids.append(this_expt_ids)
        roi_ids.append(this_roi_ids)
    return rs,expt_ids,roi_ids

def gen_rs_modal_uparam_expt_with_sem(dsnames=None,selection=None,dcutoff=5,pval_cutoff=0.05,modal_uparam=None,running=True,expttype='size_contrast_0',average_ori=True,run_cutoff=10,running_pct_cutoff=0.4):
    # same specifically for case of two spatial pixels
    if dsnames is None:
        dsnames = default_dsnames()
    if selection is None:
        selection = default_selection()
    nbefore = 8
    nafter = 8
    
    nparam = np.array([mu.shape[0] for mu in modal_uparam])
        
    tunings,tunings_sem,uparams,displacements,pvals = compute_tunings_tavg_with_sem(dsnames,running=running,expttype=expttype,run_cutoff=run_cutoff,running_pct_cutoff=running_pct_cutoff)
    
    rs = []
    rs_sem = []
    expt_ids = []
    roi_ids = []
    for icelltype in range(len(tunings)):
        these_tunings = tunings[icelltype]
        these_tunings_sem = tunings_sem[icelltype]
        nexpt = len(these_tunings)
        these_uparams = uparams[icelltype]
        these_displacements = displacements[icelltype]
        these_pvals = pvals[icelltype]
        if not selection[icelltype] is None:
            sel = selection[icelltype]
        else:
            sel = np.arange(len(these_tunings))
        these_uparams = [these_uparams[i] for i in sel if not these_tunings[i] is None]
        these_displacements = [these_displacements[i].T for i in sel if not these_tunings[i] is None]
        these_pvals = [these_pvals[i] for i in sel if not these_tunings[i] is None]
        these_tunings_sem = [these_tunings_sem[i] for i in sel if not these_tunings[i] is None]
        these_expt_ids = [i for i in sel if not these_tunings[i] is None]
        these_tunings = [these_tunings[i] for i in sel if not these_tunings[i] is None]
        nexpt = len(these_tunings)
        if nexpt>0:
            aligned = [include_aligned(d,dcutoff,p,pval_cutoff,less=True) for d,p in zip(these_displacements,these_pvals)]
            misaligned = [include_aligned(d,dcutoff,p,pval_cutoff,less=False) for d,p in zip(these_displacements,these_pvals)]
            this_rs = []
            this_rs_sem = []
            this_expt_ids = []
            this_roi_ids = []
            for icriterion,criterion in enumerate([aligned,misaligned]):
                nrois = [get_nroi(tt[cc]) for tt,cc in zip(these_tunings,criterion)]
                ra = [np.nan*np.ones((nroi,)+tuple(nparam)) for nroi in nrois]
                ra_sem = [np.nan*np.ones((nroi,)+tuple(nparam)) for nroi in nrois]
                expt_ida = [np.ones((nroi,))*these_expt_ids[iexpt] for iexpt,nroi in enumerate(nrois)]
                roi_ida = [np.where(cc)[0] for cc in criterion]
                for iexpt in range(nexpt):
                    assign_from_uparam(ra[iexpt],modal_uparam,these_tunings[iexpt][criterion[iexpt]],these_uparams[iexpt])
                    assign_from_uparam(ra_sem[iexpt],modal_uparam,these_tunings_sem[iexpt][criterion[iexpt]],these_uparams[iexpt])
                    #ra[iexpt] = average_up(ra[iexpt],average_ori=average_ori)
                    #ra_sem[iexpt] = np.sqrt(average_up(ra_sem[iexpt]**2,average_ori=average_ori))
                #ra = np.concatenate([np.nanmean(rra,0)[np.newaxis] for rra in ra],axis=0)
                ra = np.concatenate(ra,axis=0)
                ra_sem = np.concatenate(ra_sem,axis=0)
                expt_ida = np.concatenate(expt_ida,axis=0)
                roi_ida = np.concatenate(roi_ida,axis=0)
                this_rs.append(ra)
                this_rs_sem.append(ra_sem)
                this_expt_ids.append(expt_ida)
                this_roi_ids.append(roi_ida)
        else:
            this_rs = [None for icriterion in range(2)]
            this_rs_sem = [None for icriterion in range(2)]
            this_expt_ids = [None for icriterion in range(2)]
            this_roi_ids = [None for icriterion in range(2)]
        rs.append(this_rs)
        rs_sem.append(this_rs_sem)
        expt_ids.append(this_expt_ids)
        roi_ids.append(this_roi_ids)
    return rs,rs_sem,expt_ids,roi_ids

#    # same specifically for case of two spatial pixels
#    if dsnames is None:
#        dsnames = default_dsnames()
#    if selection is None:
#        selection = default_selection()
#    nbefore = 8
#    nafter = 8
#    
#    nparam = np.array([mu.shape[0] for mu in modal_uparam])
#        
#    tunings,uparams,displacements,pvals = compute_tunings(dsnames,running=running,expttype=expttype)
#    
#    rs = []
#    for icelltype in range(len(tunings)):
#        these_tunings = tunings[icelltype]
#        nexpt = len(these_tunings)
#        these_uparams = uparams[icelltype]
#        these_displacements = displacements[icelltype]
#        these_pvals = pvals[icelltype]
#        if not selection[icelltype] is None:
#            sel = selection[icelltype]
#        else:
#            sel = np.arange(len(these_tunings))
#        these_uparams = [these_uparams[i] for i in sel if not these_tunings[i] is None]
#        these_displacements = [these_displacements[i].T for i in sel if not these_tunings[i] is None]
#        these_pvals = [these_pvals[i] for i in sel if not these_tunings[i] is None]
#        these_tunings = [these_tunings[i] for i in sel if not these_tunings[i] is None]
#        nexpt = len(these_tunings)
#        if nexpt>0:
#            aligned = [include_aligned(d,dcutoff,p,pval_cutoff,less=True) for d,p in zip(these_displacements,these_pvals)]
#            misaligned = [include_aligned(d,dcutoff,p,pval_cutoff,less=False) for d,p in zip(these_displacements,these_pvals)]
#            this_rs = []
#            for icriterion,criterion in enumerate([aligned,misaligned]):
#                nrois = [get_nroi(tt[cc]) for tt,cc in zip(these_tunings,criterion)]
#                nframes = [get_nframes(tt) for tt in these_tunings]
#                ra = [np.nan*np.ones((nroi,)+tuple(nparam)+(nframe,)) for nroi,nframe in zip(nrois,nframes)]
#                for iexpt in range(nexpt):
#                    assign_from_uparam(ra[iexpt],modal_uparam,these_tunings[iexpt][criterion[iexpt]],these_uparams[iexpt])
#                    ra[iexpt] = average_up(ra[iexpt],average_ori=average_ori)
#                ra = np.concatenate(ra,axis=0)
#                this_rs.append(ra)
#        else:
#            this_rs = [None for icriterion in range(2)]
#        rs.append(this_rs)
#    return rs

def get_nroi(tt):
    return get_shape_dim(tt,0)
                                   
def get_nframes(tt):
    return get_shape_dim(tt,-1)
                                   
def get_shape_dim(tt,idim=0):
    if not tt is None:
        return tt.shape[idim]
    else:
        return None

def assign_to_modal_uparams(this_uparam,modal_uparam):
    try:
        mid_pts = 0.5*(modal_uparam[1:]+modal_uparam[:-1])
        bins = np.concatenate(((-np.inf,),mid_pts,(np.inf,)))
        inds_in_modal = np.digitize(this_uparam,bins)-1
        numerical = True
    except:
        print('non-numerical parameter')
        numerical = False
    if numerical:
        uinds = np.unique(inds_in_modal)
        inds_in_this = np.zeros((0,),dtype='int')
        for uind in uinds:
            candidates = np.where(inds_in_modal==uind)[0]
            dist_from_modal = np.abs(this_uparam[candidates]-modal_uparam[uind])
            to_keep = candidates[np.argmin(dist_from_modal)]
            inds_in_this = np.concatenate((inds_in_this,(to_keep,)))
        inds_in_modal = inds_in_modal[inds_in_this]
        bool_in_this = np.zeros((len(this_uparam),),dtype='bool')
        bool_in_modal = np.zeros((len(modal_uparam),),dtype='bool')
        bool_in_this[inds_in_this] = True
        bool_in_modal[inds_in_modal] = True
    else:
        assert(np.all(this_uparam==modal_uparam))
        bool_in_this,bool_in_modal = [np.ones(this_uparam.shape,dtype='bool') for iparam in range(2)]
    return bool_in_this,bool_in_modal

# def gen_slices_from_ind_list(ind_list,prepend_no=1):
#     nind = len(ind_list)
#     slc = [None for iind in range(nind)]
#     for iind in range(nind):
#         slc[iind] = [slice(None) for iiind in range(iind+prepend_no)]+[ind_list[iind]]
#     return slc

def gen_big_bool(bool_list):
    nind = len(bool_list)
    slicers = [[np.newaxis for iind in range(nind)] for iind in range(nind)]
    for iind in range(nind):
        slicers[iind][iind] = slice(None)
    big_ind = np.ones(tuple([iit.shape[0] for iit in bool_list]),dtype='bool')
    for iit,slc in zip(bool_list,slicers):
        big_ind = big_ind*iit[slc]
    return big_ind
    
def assign_(a,a_ind,b,b_ind,ignore_first=1):
    a_bool = gen_big_bool(a_ind)
    b_bool = gen_big_bool(b_ind)
    a[[slice(None) for iind in range(ignore_first)]+[a_bool]] = b[[slice(None) for iind in range(ignore_first)]+[b_bool]]
    
def assign_from_uparam(modal,modal_uparam,this,this_uparam):
    nparam = len(this_uparam)
    bool_in_this,bool_in_modal = [[None for iparam in range(nparam)] for ivar in range(2)]
    for iparam in range(nparam): #
        tu,mu = [a[iparam] for a in [this_uparam,modal_uparam]]
        bool_in_this[iparam],bool_in_modal[iparam] = assign_to_modal_uparams(tu,mu)
    #bool_in_this,bool_in_modal = assign_to_modal_uparams(this_uparam,modal_uparam)
    assign_(modal,bool_in_modal,this,bool_in_this)
    
# def index_(arr,ind_list,ignore_first=1):
#     a = arr
#     slcs = gen_slices_from_ind_list(ind_list,prepend_no=ignore_first)
#     for slc in slcs:
#         a = a[slc]
#     return a

def gen_size_tuning(sc):
    # add 0% contrast stimulus as if it were a 0 degree size
    gray = np.tile(sc[:,:,0].mean(1)[:,np.newaxis,np.newaxis],(1,1,sc.shape[2]))
    to_plot = np.concatenate((gray,sc),axis=1)
    print(to_plot.shape)
    return to_plot

def plot_size_tuning_by_contrast(arr):
    usize = np.array((0,5,8,13,22,36))
    ucontrast = np.array((0,6,12,25,50,100))
    arr_sz = gen_size_tuning(arr)
    arr_sz = arr_sz/arr_sz.max(1).max(1)[:,np.newaxis,np.newaxis]
    lb,ub = ut.bootstrap(arr_sz,np.mean,pct=(2.5,97.5))
    to_plot = arr_sz.mean(0)
    for ic in range(1,6):
        plt.subplot(1,5,ic)
        ut.plot_bootstrapped_errorbars_hillel(usize,arr_sz[:,:,ic].transpose((0,2,1)),colors=['k','r'],markersize=5)
#         plt.scatter((0,),to_plot[:,0,0].mean(0))
#         plt.scatter((0,),to_plot[:,0,1].mean(0))
        plt.ylim(0.5*to_plot.min(),1.2*to_plot.max())
        plt.title('%d%% contrast' % ucontrast[ic])
        plt.xlabel('size ($^o$)')
    plt.subplot(1,5,1)
    plt.ylabel('event rate / max event rate')
    plt.tight_layout()
    
def plot_size_tuning(arr,colors=None):
    usize = np.array((0,5,8,13,22,36))
    ucontrast = np.array((0,6,12,25,50,100))
    arr_sz = gen_size_tuning(arr)
    arr_sz = arr_sz/arr_sz.max(1).max(1)[:,np.newaxis,np.newaxis]
    lb,ub = ut.bootstrap(arr_sz,np.mean,pct=(2.5,97.5))
    to_plot = arr_sz.mean(0)
    ut.plot_bootstrapped_errorbars_hillel(usize,arr_sz[:,:,1::].transpose((0,2,1)),colors=colors)
    plt.ylim(to_plot.min()-0.1,to_plot.max()+0.1)
#     plt.title('%d%% contrast' % ucontrast[ic])
    plt.xlabel('size ($^o$)')
#     plt.subplot(1,2,1)
    
    plt.tight_layout()
    
def plot_size_tuning_peak_norm(arr,colors=None):
    usize = np.array((0,5,8,13,22,36))
    ucontrast = np.array((0,6,12,25,50,100))
    arr_sz = gen_size_tuning(arr)
    mx = arr_sz.max(1)[:,np.newaxis]
    mn = arr_sz.min(1)[:,np.newaxis]
    mn = 0
    arr_sz = (arr_sz-mn)/(mx-mn)
    lb,ub = ut.bootstrap(arr_sz,np.mean,pct=(2.5,97.5))
    to_plot = arr_sz.mean(0)
    ut.plot_bootstrapped_errorbars_hillel(usize,arr_sz[:,:,1::2].transpose((0,2,1)),colors=colors[::2])
    plt.ylim(to_plot.min()-0.1,to_plot.max()+0.1)
#     plt.title('%d%% contrast' % ucontrast[ic])
    plt.xlabel('size ($^o$)')
    plt.tight_layout()
    
def f_miller_troyer(mu,s2):
    # firing rate function, gaussian convolved with ReLU, derived in Miller and Troyer 2002
    u = mu/np.sqrt(2*s2)
    A = 0.5*mu*(1+ssp.erf(u))
    B = np.sqrt(s2)/np.sqrt(2*np.pi)*np.exp(-u**2)
    return A + B
#     return 0.5*mu*(1+np.exp(u)) + sigma/np.sqrt(2*np.pi)*np.exp(-u**2) # 0.5*mu*(1+ssp.erf(u))

def fprime_miller_troyer(mu,s2):
    return fprime_m_miller_troyer(mu,s2)

def fprime_m_miller_troyer(mu,s2):
    # firing rate function, gaussian convolved with ReLU, derived in Miller and Troyer 2002
    u = mu/np.sqrt(2*s2)
    A = 0.5*(1+ssp.erf(u))
    return A

def fprime_s_miller_troyer(mu,s2):
    # takes as argument the original value of s, not squared!
    u = mu/np.sqrt(2*s2)
    A = 1/np.sqrt(2*np.pi)*np.exp(-u**2)# *np.sign(s) is implicit, as s is understood to be positive
    return A

def fit_w(X,y,rate_fn,wm0=None,ws0=None,bounds=None):
    # X is (N,P), y is (N,). Finds w: (P,) weight matrix to explain y as y = f(X(wm),X(ws))
    # f is a static nonlinearity, given as a function of mean and std. of noise
    N,P = X.shape
    def parse_w(w):
        wm = w[:P]
        ws = w[P:]
#         return wm,ws,k
        return wm,ws
    def minusL(w):
#         wm,ws,k = parse_w(w)
        wm,ws = parse_w(w)
        return 0.5*np.sum((rate_fn(X @ wm,X @ ws)-y)**2) # k*
    def minusdLdw(w): 
        # sum in first dimension: (N,1) times (N,1) times (N,P)
        return egrad(minusL)(w)
    
    w0 = np.concatenate((wm0,ws0)) #,(k0,)))
    
    factr=1e7
    epsilon=1e-8
    pgtol=1e-5
    wstar = sop.fmin_l_bfgs_b(minusL,w0,fprime=minusdLdw,bounds=bounds,pgtol=pgtol,factr=factr,epsilon=epsilon)
    
    return wstar

def u_fn(X,Wx,Y,Wy,k):
    return X[0] @ Wx + X[1] @ (Wx*k) + Y[0] @ Wy + Y[1] @ (Wy*k)

def evaluate_f_mt(X,Ws,offset,k):
    # Ws: Wx,Wy,s02,Y
    return f_miller_troyer(u_fn(X,Ws[0],Ws[3],Ws[1],k)+offset,Ws[2])

def fit_w_data_loss(X,ydata,rate_fn,wm0=None,ws0=None,s020=None,k0=None,bounds=None,niter=int(1e4)):
    # X is (N,P), y is (N,). Finds w: (P,) weight matrix to explain y as y = f(X(wm),X(ws))
    # f is a static nonlinearity, given as a function of mean and std. of noise
    N,P = X[0].shape
    abd = 1

    nroi = ydata.shape[0]
    alpha_roi = sst.norm.ppf(np.arange(1,nroi+1)/(nroi+1))
    
    def sort_by_11(w):
        yalpha11 = rate_fn_wrapper(w,np.array((-abd,abd)))
        difference11 = compute_y_distance(yalpha11[np.newaxis,:,:],ydata[:,np.newaxis,:])
        sortby11 = np.argsort(difference11[:,0]-difference11[:,-1])
        return sortby11
    
    def compute_y_distance(y1,y2):
        return np.sum((y1-y2)**2,axis=-1)
    
    def compare_sorted_to_expected(w,sortind):
       
        yalpha_roi = rate_fn_wrapper(w,alpha_roi)
#         print(yalpha_roi.max())
        difference_roi = compute_y_distance(yalpha_roi,ydata[sortind])
        difference_roi_unsorted = compute_y_distance(yalpha_roi,ydata)
#         print(difference_roi.shape)
        return difference_roi

    def rate_fn_wrapper(w,alphas):
        wm,ws,s02,k = parse_w(w)
        inputs0 = [wm,np.array((0,)),s02,[np.array((0,)),np.array((0,))]]
        inputs1 = [X,ws,[np.array((0,)),np.array((0,))],np.array((0,))]
        yalpha = rate_fn(X,inputs0,alphas[:,np.newaxis]*u_fn(*inputs1,k),k)
        yalpha = normalize(yalpha)
        return yalpha

#     def compute_f_by_itself(w):

    def normalize(arr):
        arrsum = arr.sum(1)
#         arrnorm = np.ones_like(arr)
#         arrnorm = arrnorm/arrnorm.shape[1]
        well_behaved = (arrsum>0)[:,np.newaxis]
        arrnorm = well_behaved*arr/arrsum[:,np.newaxis] + (~well_behaved)*np.ones_like(arr)/arr.shape[1]
        return arrnorm
    
    def parse_w(w):
        wm = w[:P]
        ws = w[P:-2]
        s02 = w[-2]
        k = w[-1]
        return wm,ws,s02,k

    def minusL(w,sortind):
#         wm,ws,k = parse_w(w)
        difference_roi = compare_sorted_to_expected(w,sortind)
#         print(difference_roi.shape)
#         print(str(w) + ' -> ' + str(np.round(np.sum(difference_roi),decimals=2)))
        return 0.5*np.sum(difference_roi) # k*
    
    def minusdLdw(w,sortind): 
        # sum in first dimension: (N,1) times (N,1) times (N,P)
        return egrad(lambda w: minusL(w,sortind))(w)
    
    def fix_violations(w,bounds):
        lb = np.array([b[0] for b in bounds])
        ub = np.array([b[1] for b in bounds])
        w[w<lb] = lb[w<lb]
        w[w>ub] = ub[w>ub]
        return w
    
    w0 = np.concatenate((wm0,ws0,s020,k0)) #,(k0,)))
    
    factr=1e7
    epsilon=1e-8
    pgtol=1e-5
    this_w = w0
    for i in range(niter):
        sortind = sort_by_11(this_w)
        wstar = sop.fmin_l_bfgs_b(lambda w: minusL(w,sortind),this_w,fprime=lambda w: minusdLdw(w,sortind),bounds=bounds,pgtol=pgtol,factr=factr,epsilon=epsilon,maxiter=1)
        assert(~np.isnan(wstar[1]))
        if np.isnan(wstar[1]):
            this_w = old_w
        else:
            this_w = wstar[0].copy() + np.random.randn(*this_w.shape)*0.01*np.exp(-i/niter)
            old_w = wstar[0].copy()
        this_w = fix_violations(this_w,bounds)
        print(str(i) + ': ' + str(wstar[1]))
    
    max_alpha = alpha_roi.max()
    this_yalpha = rate_fn_wrapper(this_w,np.linspace(-max_alpha,max_alpha,101))
    ydist = compute_y_distance(ydata[sortind,np.newaxis],this_yalpha[np.newaxis,:])
    
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(ydist/ydist.max(1)[:,np.newaxis],extent=[-max_alpha,max_alpha,0,10])
    plt.plot(-alpha_roi,10*np.arange(nroi)/nroi,c='m')
    plt.subplot(2,2,2)
    plt.imshow(this_yalpha[25].reshape((5,6)))
    plt.subplot(2,2,4)
    plt.imshow(this_yalpha[75].reshape((5,6)))
#         print(wstar)
    
    return wstar

def gen_Weight_k_kappa(W,K,kappa):
    WW0 = np.concatenate((W,W*K[np.newaxis,:]),axis=1)
    WW1 = np.concatenate((W*K[np.newaxis,:]*kappa,W),axis=1)
    WW = np.concatenate((WW0,WW1),axis=0)
    return WW

def gen_Weight_k_kappa_t(W,K,kappa,T,nS=2,nT=2):
    MuT = np.array((1,1))
    MuK = np.array((1,kappa))
    WT = circulate(W,T,nT,Mu=MuT)
    KKlist = [K for iT in range(nT)]
    KK = np.concatenate(KKlist,axis=0)
    WW = circulate(WT,KK,nS,Mu=MuK)
    return WW

def circulate(V,M,nZ,Mu=None):
    if not M.size:
        return V
    Vpartlist = [V*(M[np.newaxis,:]**np.abs(iZ)) for iZ in range(-nZ+1,nZ)]
    if Mu is None:
        Mu = np.ones((nZ,))
    #VVlist = [np.concatenate([m*v for m,v in zip(Mu,Vpartlist[nZ-iZ-1:2*nZ-iZ-1])],axis=1) for iZ in range(nZ)]
    VVlist = [np.concatenate(Vpartlist[nZ-iZ-1:2*nZ-iZ-1],axis=1) for iZ in range(nZ)]
    #VV = np.concatenate(VVlist,axis=0)
    VV = np.concatenate([m*v for m,v in zip(Mu,VVlist)],axis=0)
    return VV

def u_fn_k_kappa(XX,YY,Wx,Wy,k,kappa):
    WWx,WWy = [gen_Weight_k_kappa(W,k,kappa) for W in [Wx,Wy]]
    return u_fn_WW(XX,YY,WWx,WWy)# XX @ WWx + YY @ WWy

def u_fn_k_kappa_t(XX,YY,Wx,Wy,k,kappa,T):
    WWx,WWy = [gen_Weight_k_kappa_t(W,k,kappa,T) for W in [Wx,Wy]]
    return u_fn_WW(XX,YY,WWx,WWy)# XX @ WWx + YY @ WWy

def u_fn_WW(XX,YY,WWx,WWy):
    return XX @ WWx + YY @ WWy

def print_labeled(lbl,var):
    print(lbl+': '+str(var))

def combine_inside_out(item,combine_fn,niter=0):
    combined_item = item.copy()
    for iiter in range(niter):
        for iel in range(len(combined_item)):
            combined_item[iel] = combine_fn(item[iel])
    return combined_item

def access_nested_list_element(this_list,imulti):
    current_el = this_list.copy()
    for ithis in imulti:
        current_el = current_el[ithis]
    return current_el

def flatten_nested_list_of_2d_arrays(this_list):
    # innermost index changing most quickly, outermost most slowly
    current_el = this_list.copy()
    shp = tuple(())
    while isinstance(current_el,list):
        shp = shp + (len(current_el),)
        current_el = current_el[0]
    nflat = np.prod(shp)
    n0,n1 = current_el.shape
    to_return = np.zeros((n0,n1*nflat))
    for iflat in range(nflat):
        imulti = np.unravel_index(iflat,shp)
        to_return[:,iflat*n1:(iflat+1)*n1] = access_nested_list_element(this_list,imulti)
    return to_return
        
def add_up_lists(lst):
    to_return = []
    for item in lst:
        to_return.append(item)
    return to_return

def parse_thing(V,shapes):
    for shape in shapes:
        if type(shape) is int:
            shape = (shape,)
    sizes = [np.prod(shape) for shape in shapes]
    sofar = 0
    outputs = []
    for size,shape in zip(sizes,shapes):
        if size > 1:
            new_element = V[sofar:sofar+size].reshape(shape)
        elif size==1:
            new_element = V[sofar] # if just a float
        elif size==0:
            new_element = np.array(())
        outputs.append(new_element)
        sofar = sofar + size
    return outputs
    
def compute_tr_siginv2_sig1(stim_deriv,noise,pc_list,nS=2,nT=2):
    tot = 0
    this_ncelltypes = len(pc_list[0][0])
    for iel in range(stim_deriv.shape[1]):
        iS,iT,icelltype = np.unravel_index(iel,(nS,nT,this_ncelltypes))
        sigma2 = np.sum(stim_deriv[:,iel]**2)
        #print_labeled('sigma2',sigma2)
        if sigma2>0 and not pc_list[iS][iT][icelltype] is None:
            inner_prod = np.sum([pc[0]**2*np.sqrt(sigma2)*inner_product_(stim_deriv[:,iel],pc[1]) for pc in pc_list[iS][iT][icelltype]])
        else:
            inner_prod = 0 
        tot = tot - 1/noise/(noise + sigma2)*inner_prod #.sum()
    return tot

def compute_log_det_sig2(stim_deriv,noise):
        sigma2 = inner_product_(stim_deriv,stim_deriv) #np.sum(stim_deriv**2,0)
        return np.sum([np.log(s2+noise) for s2 in sigma2])

def compute_mahalanobis_dist(stim_deriv,noise,mu_data,mu_model):
        # in the case where stim_deriv = 0 (no variability model) only the noise (sqerror) term contributes
    mu_dist = mu_model-mu_data
    inner_prod = inner_product_(stim_deriv,mu_dist) #np.einsum(stim_deriv,mu_dist,'ik,jk->ij')
    sigma2 = inner_product_(stim_deriv,stim_deriv) #np.sum(stim_deriv**2,0)
    noise_term = 1/noise*np.sum(inner_product_(mu_dist,mu_dist)) #np.inner(mu_dist,mu_dist)
    cov_term = np.sum([-1/noise/(noise+s2)*np.sum(ip**2) for s2,ip in zip(sigma2,inner_prod)])
    return noise_term + cov_term
    
def compute_kl_divergence(stim_deriv,noise,mu_data,mu_model,pc_list,nS=2,nT=2):
        # omitting a few terms: - d - log(sig1) # where d is the dimensionality
        # in the case where stim_deriv = 0 (no variability model) only the noise (sqerror) 
        # term in mahalanobis_dist contributes
    log_det = compute_log_det_sig2(stim_deriv,noise)
    tr_sig_quotient = compute_tr_siginv2_sig1(stim_deriv,noise,pc_list,nS=nS,nT=nT)
    maha_dist = compute_mahalanobis_dist(stim_deriv,noise,mu_data,mu_model)
    lbls = ['log_det','tr_sig_quotient','maha_dist']
    vars = [log_det,tr_sig_quotient,maha_dist]
    #for lbl,var in zip(lbls,vars):
    #    print_labeled(lbl,var)
    return 0.5*(log_det + tr_sig_quotient + maha_dist)
    
def inner_product_(a,b):
    return np.sum(a*b,0)

def minus_sum_log_ceil(log_arg,big_val):
    if np.all(log_arg>0):
        cost = -np.sum(np.log(log_arg))
    else:
        ok = (log_arg>0)
        cost = -np.sum(np.log(log_arg[ok])) + big_val*np.sum(~ok)
    return cost
