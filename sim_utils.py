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

default_running_pct_cutoff = 0.4

def compute_tuning(dsfile,datafield='decon',running=True,expttype='size_contrast_0',running_pct_cutoff=default_running_pct_cutoff,fill_nans_under_cutoff=False):
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
                    trialrun = sc0['running_speed_cm_s'][:,nbefore:-nafter].mean(-1)>10 #
                else:
                    trialrun = sc0['running_speed_cm_s'][:,nbefore:-nafter].mean(-1)<10
                #print(sc0['running_speed_cm_s'].shape)
                print(np.nanmean(trialrun))
                if np.nanmean(trialrun)>running_pct_cutoff:
                    tuning[ikey] = ut.compute_tuning(data,stim_id,trial_criteria=trialrun)[:]
                elif fill_nans_under_cutoff:
                    tuning[ikey] = ut.compute_tuning(data,stim_id,trial_criteria=trialrun)[:]
                    tuning[ikey] = np.nan*np.ones_like(tuning[ikey])
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

def get_ret_info(dsfile,expttype='size_contrast_0'):
    # take in an HDF5 data struct, and convert to an n-dimensional matrix
    # describing the tuning curve of each neuron. For size-contrast stimuli, 
    # the dimensions of this matrix are ROI index x size x contrast x direction x time. 
    # This outputs a list of such matrices, where each element is one imaging session
    with h5py.File(dsfile,mode='r') as f:
        keylist = [key for key in f.keys()]
        ret_info = [None for i in range(len(keylist))]
        for ikey in range(len(keylist)):
            session = f[keylist[ikey]]
            print(session)
            #print([key for key in session.keys()])
            if expttype in session:
                sc0 = session[expttype]
                if 'rf_displacement_deg' in sc0:
                    pval = sc0['rf_mapping_pval'][:]
                    sqerror = session['retinotopy_0']['rf_sq_error'][:]
                    sigma = session['retinotopy_0']['rf_sigma'][:]
                    try:
                        amplitude = session['retinotopy_0']['rf_amplitude'][:]
                    except:
                        amplitude = np.nan*np.ones_like(sigma)
                    cell_center = session['cell_center'][:]
                    rf_center = sc0['rf_displacement_deg'][:].T
                    X = cell_center
                    y = rf_center
                    rf_conditions = [ut.k_and(~np.isnan(X[:,0]),~np.isnan(y[:,0])),sqerror<0.75,sigma>3.3,pval<0.1]
                    lkat = np.ones((X.shape[0],),dtype='bool')
                    for cond in rf_conditions:
                        lkat_new = (lkat & cond)
                        if lkat_new.sum()>=5:
                            lkat = lkat_new.copy()
                    linreg = sklearn.linear_model.LinearRegression().fit(X[lkat],y[lkat])
                    ret_map_loc = np.zeros_like(y)
                    ret_map_loc[~np.isnan(X[:,0])] = linreg.predict(X[~np.isnan(X[:,0])])
                    ret_info[ikey] = {'pval':pval,'sqerror':sqerror,'sigma':sigma,\
                            'cell_center':cell_center,'rf_center':rf_center,\
                            'ret_map_loc':ret_map_loc,'amplitude':amplitude}
    return ret_info

def compute_tunings(dsnames,datafield='decon',running=True,expttype='size_contrast_0',running_pct_cutoff=default_running_pct_cutoff,fill_nans_under_cutoff=False):
    # compute tuning as above, for each of a list of HDF5 files each corresponding to a particular cell type
    tunings = []
    uparams = []
    displacements = []
    pvals = []
    for dsname in dsnames:
        print(dsname)
        this_tuning,this_uparam,this_displacement,this_pval = compute_tuning(dsname,datafield=datafield,running=running,expttype=expttype,running_pct_cutoff=running_pct_cutoff,fill_nans_under_cutoff=fill_nans_under_cutoff)
        tunings.append(this_tuning)
        uparams.append(this_uparam)
        displacements.append(this_displacement)
        pvals.append(this_pval)
    return tunings,uparams,displacements,pvals

def look_up_nroi(dsname):
    # compute tuning as above, for each of a list of HDF5 files each corresponding to a particular cell type
    with ut.hdf5read(dsname) as ds:
        keylist = list(ds.keys())
        nkey = len(keylist)
        nroi = [None for i in range(nkey)]
        for ikey in range(nkey):
            nroi[ikey] = len(ds[keylist[ikey]]['cell_id'][:])
    return nroi

def look_up_nrois(dsnames):
    # compute tuning as above, for each of a list of HDF5 files each corresponding to a particular cell type
    nrois = []
    for dsname in dsnames:
        print(dsname)
        this_nroi = look_up_nroi(dsname)    
        nrois.append(this_nroi)
    return nrois

def look_up_params(dsnames,expttype,paramnames):
    # compute tuning as above, for each of a list of HDF5 files each corresponding to a particular cell type
    params = []
    for dsname in dsnames:
        print(dsname)
        with ut.hdf5read(dsname) as ds:
            keylist = list(ds.keys())
            nkey = len(keylist)
            this_param = [None for i in range(nkey)]
            for ikey in range(nkey):
                if expttype in ds[keylist[ikey]]:
                    this_param[ikey] = np.array([ds[keylist[ikey]][expttype][paramname][:].flatten() for paramname in paramnames]).T
        params.append(this_param)
    return params
        
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
    
def average_up(arr,nbefore=8,nafter=8):
    # average across time points and directions
    ndim = len(arr.shape)
    slicer = [slice(None) for idim in range(ndim)]
    slicer[-1] = slice(nbefore,-nafter)
    return np.nanmean(np.nanmean(arr[slicer],-1),-1)

def average_time(arr,nbefore=8,nafter=8):
    # average across time points and directions
    ndim = len(arr.shape)
    slicer = [slice(None) for idim in range(ndim)]
    slicer[-1] = slice(nbefore,-nafter)
    return np.nanmean(arr[slicer],-1)

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

def gen_rspatial(dsnames=None,selection=None,dcutoffs=[0,5,10,15],pval_cutoff=0.05,slices=None,datafield='decon'):
    # from a list of HDF5 files, split up the data into an arbitrary number of spatial pixels based on RF center location
    if dsnames is None:
        dsnames = default_dsnames()
    if selection is None:
        selection = default_selection()
    if slices is None:
        condition_inds = default_condition_inds()
        
    tunings,uparams,displacements,pvals = compute_tunings(dsnames,datafield=datafield)
    
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

def gen_rs(dsnames=None,selection=None,dcutoff=5,pval_cutoff=0.05,slices=None,running=True,expttype='size_contrast_0',running_pct_cutoff=default_running_pct_cutoff):
    # same specifically for case of two spatial pixels
    if dsnames is None:
        dsnames = default_dsnames()
    if selection is None:
        selection = default_selection()
    if slices is None:
        condition_inds = default_condition_inds()
    nbefore = 8
    nafter = 8
        
    tunings,uparams,displacements,pvals = compute_tunings(dsnames,running=running,expttype=expttype,running_pct_cutoff=running_pct_cutoff)
    
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

def gen_rs_modal_uparam(dsnames=None,selection=None,dcutoff=5,pval_cutoff=0.05,modal_uparam=None,running=True,expttype='size_contrast_0',running_pct_cutoff=default_running_pct_cutoff):
    # same specifically for case of two spatial pixels
    if dsnames is None:
        dsnames = default_dsnames()
    if selection is None:
        selection = default_selection()
    nbefore = 8
    nafter = 8
    
    nparam = np.array([mu.shape[0] for mu in modal_uparam])
        
    tunings,uparams,displacements,pvals = compute_tunings(dsnames,running=running,expttype=expttype,running_pct_cutoff=running_pct_cutoff)
    
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
                    ra[iexpt] = average_up(ra[iexpt])
                ra = np.concatenate(ra,axis=0)
                this_rs.append(ra)
        else:
            this_rs = [None for icriterion in range(2)]
        rs.append(this_rs)
    return rs

def gen_rs_modal_uparam_all(dsnames=None,selection=None,modal_uparam=None,running=True,expttype='size_contrast_0',running_pct_cutoff=default_running_pct_cutoff):
    # same specifically for case of two spatial pixels
    if dsnames is None:
        dsnames = default_dsnames()
    if selection is None:
        selection = default_selection()
    nbefore = 8
    nafter = 8
    
    nparam = np.array([mu.shape[0] for mu in modal_uparam])
        
    tunings,uparams,displacements,pvals = compute_tunings(dsnames,running=running,expttype=expttype,running_pct_cutoff=running_pct_cutoff)
    nrois = look_up_nrois(dsnames)
    
    rs = []
    for icelltype in range(len(tunings)):
        these_tunings = tunings[icelltype]
        nexpt = len(these_tunings)
        these_uparams = uparams[icelltype]
        these_displacements = displacements[icelltype]
        these_pvals = pvals[icelltype]
        these_nrois = nrois[icelltype]
        if not selection[icelltype] is None:
            sel = selection[icelltype]
        else:
            sel = np.arange(len(these_tunings))
        these_uparams = [these_uparams[i] for i in sel]
        these_displacements = [safe_operate_(np.transpose,these_displacements[i]) for i in sel]
        these_pvals = [these_pvals[i] for i in sel]
        these_tunings = [these_tunings[i] for i in sel]
        these_nrois = [these_nrois[i] for i in sel]
        nexpt = len(these_tunings)
        if nexpt>0:
            isnone = np.array([x is None for x in these_tunings])
            nframes = np.zeros((len(these_tunings),),dtype='int')
            nframes[~isnone] = np.array([get_nframes(these_tunings[iexpt]) for iexpt in range(nexpt) if not isnone[iexpt]])
            nframes[isnone] = 1
            ra = [np.nan*np.ones((nroi,)+tuple(nparam)+(nframe,)) for nroi,nframe in zip(these_nrois,nframes)]
            for iexpt in range(nexpt):
                if not these_tunings[iexpt] is None:
                    assign_from_uparam(ra[iexpt],modal_uparam,these_tunings[iexpt],these_uparams[iexpt])
                ra[iexpt] = average_up(ra[iexpt])     
            this_rs = np.concatenate(ra,axis=0)
        else:
            this_rs = None
        rs.append(this_rs)
    return rs
        
def gen_rs_modal_uparam_ori(dsnames=None,selection=None,modal_uparam=None,running=True,expttype='size_contrast_0'):
    # same specifically for case of two spatial pixels
    if dsnames is None:
        dsnames = default_dsnames()
    if selection is None:
        selection = default_selection()
    nbefore = 8
    nafter = 8
    
    nparam = np.array([mu.shape[0] for mu in modal_uparam])
        
    tunings,uparams,displacements,pvals = compute_tunings(dsnames,running=running,expttype=expttype)
    nrois = look_up_nrois(dsnames)
    
    #aligned = [include_aligned(d,dcutoff,p,pval_cutoff,less=True) for d,p in zip(these_displacements,these_pvals)]
    #misaligned = [include_aligned(d,dcutoff,p,pval_cutoff,less=False) for d,p in zip(these_displacements,these_pvals)]
    
    rs = []
    rexpts = []
    displacements = []
    for icelltype in range(len(tunings)):
        these_tunings = tunings[icelltype]
        nexpt = len(these_tunings)
        these_uparams = uparams[icelltype]
        these_displacements = displacements[icelltype]
        these_pvals = pvals[icelltype]
        these_nrois = nrois[icelltype]
        if not selection[icelltype] is None:
            sel = selection[icelltype]
        else:
            sel = np.arange(len(these_tunings))
        these_uparams = [these_uparams[i] for i in sel]
        these_displacements = [safe_operate_(np.transpose,these_displacements[i]) for i in sel]
        these_pvals = [these_pvals[i] for i in sel]
        these_tunings = [these_tunings[i] for i in sel]
        these_nrois = [these_nrois[i] for i in sel]
        nexpt = len(these_tunings)
        if nexpt>0:
            isnone = np.array([x is None for x in these_tunings])
            nframes = np.zeros((len(these_tunings),),dtype='int')
            nframes[~isnone] = np.array([get_nframes(these_tunings[iexpt]) for iexpt in range(nexpt) if not isnone[iexpt]])
            nframes[isnone] = 1
            ra = [np.nan*np.ones((nroi,)+tuple(nparam)+(nframe,)) for nroi,nframe in zip(these_nrois,nframes)]
            rexpt = [iexpt*np.ones((nroi,)) for iexpt,nroi in enumerate(these_nrois)]
            for iexpt in range(nexpt):
                if not these_tunings[iexpt] is None:
                    print(these_uparams[iexpt])
                    assign_from_uparam(ra[iexpt],modal_uparam,these_tunings[iexpt],these_uparams[iexpt])
                ra[iexpt] = average_time(ra[iexpt])
            this_rs = np.concatenate(ra,axis=0)
            this_rexpt = np.concatenate(rexpt,axis=0)
            this_displacement = np.concatenate(these_displacements,axis=0)
        else:
            this_rs = None
        rs.append(this_rs)
        rexpts.append(this_rexpt)
        displacements.append(this_displacement)
    return rs,rexpts,displacements

def gen_rs_modal_uparam_ori_behavior(dsnames=None,selection=None,modal_uparam=None,expttype='size_contrast_0',running_pct_cutoff=0.2):
    # same specifically for case of two spatial pixels
    if dsnames is None:
        dsnames = default_dsnames()
    if selection is None:
        selection = default_selection()
    nbefore = 8
    nafter = 8
    
    nparam = np.array([mu.shape[0] for mu in modal_uparam])
    
    runnings = [False,True]
    rs = [[] for irun in range(len(runnings))]
    rexpts = [[] for irun in range(len(runnings))]
    displacements = [[] for irun in range(len(runnings))]
        
    for irun,running in enumerate(runnings):
        tunings,uparams,displacements,pvals = compute_tunings(dsnames,running=running,expttype=expttype,running_pct_cutoff=running_pct_cutoff,fill_nans_under_cutoff=True)
        nrois = look_up_nrois(dsnames)
    
        for icelltype in range(len(tunings)):
            these_tunings = tunings[icelltype]
            nexpt = len(these_tunings)
            these_uparams = uparams[icelltype]
            these_displacements = displacements[icelltype]
            these_pvals = pvals[icelltype]
            these_nrois = nrois[icelltype]
            if not selection[icelltype] is None:
                sel = selection[icelltype]
            else:
                sel = np.arange(len(these_tunings))
            these_uparams = [these_uparams[i] for i in sel]
            these_displacements = [safe_operate_(np.transpose,these_displacements[i]) for i in sel]
            these_pvals = [these_pvals[i] for i in sel]
            these_tunings = [these_tunings[i] for i in sel]
            these_nrois = [these_nrois[i] for i in sel]
            nexpt = len(these_tunings)
            if nexpt>0:
                isnone = np.array([x is None for x in these_tunings])
                nframes = np.zeros((len(these_tunings),),dtype='int')
                nframes[~isnone] = np.array([get_nframes(these_tunings[iexpt]) for iexpt in range(nexpt) if not isnone[iexpt]])
                nframes[isnone] = 1
                ra = [np.nan*np.ones((nroi,)+tuple(nparam)+(nframe,)) for nroi,nframe in zip(these_nrois,nframes)]
                rexpt = [iexpt*np.ones((nroi,)) for iexpt,nroi in enumerate(these_nrois)]
                for iexpt in range(nexpt):
                    if not these_tunings[iexpt] is None:
                        print(these_uparams[iexpt])
                        assign_from_uparam(ra[iexpt],modal_uparam,these_tunings[iexpt],these_uparams[iexpt])
                    ra[iexpt] = average_time(ra[iexpt])
                this_rs = np.concatenate(ra,axis=0)
                this_rexpt = np.concatenate(rexpt,axis=0)
            else:
                this_rs = None
            rs[irun].append(this_rs)
            rexpts[irun].append(this_rexpt)
            displacements[irun].append(this_rexpt)
    return rs,rexpts,displacements

def gen_params(dsnames=None,selection=None,paramnames=None,expttype='retinotopy_0',running=True):
    # same specifically for case of two spatial pixels
    if dsnames is None:
        dsnames = default_dsnames()
    if selection is None:
        selection = default_selection()
    
    tunings,uparams,displacements,pvals = compute_tunings(dsnames,running=running,expttype=expttype)
    nrois = look_up_nrois(dsnames)
    params = look_up_params(dsnames,expttype,paramnames)
    nparam = len(paramnames)
    
    rs = []
    for icelltype in range(len(params)):
        these_params = params[icelltype]
        nexpt = len(these_params)
        these_nrois = nrois[icelltype]
        if not selection[icelltype] is None:
            sel = selection[icelltype]
        else:
            sel = np.arange(len(these_params))
        these_params = [these_params[i] for i in sel]
        these_nrois = [these_nrois[i] for i in sel]
        if nexpt>0:
#             isnone = np.array([x is None for x in these_params])
#             ra = [np.nan*np.ones((nroi,nparam)) for nroi in these_nrois]
            this_rs = np.concatenate(these_params,axis=0)
        else:
            this_rs = None
        rs.append(this_rs)
    return rs

def safe_operate_(fn,inp):
    if not inp is None:
        return fn(inp)
    else:
        return inp

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
    # sc: (nroi,nsize,ncontrast)
    # add 0% contrast stimulus as if it were a 0 degree size
    gray = np.tile(np.nanmean(sc[:,:,0],1)[:,np.newaxis,np.newaxis],(1,1,sc.shape[2]))
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

def f_identity(mu,s2):
    return mu

def fprime_identity(mu,s2):
    return fprime_m_identity(mu,s2)

def fprime_m_identity(mu,s2):
    return np.ones(mu.shape)

def fprime_s_identity(mu,s2):
    return np.zeros(mu.shape)

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

def evaluate_f_identity(X,Ws,offset,k):
    # Ws: Wx,Wy,s02,Y
    return f_identity(u_fn(X,Ws[0],Ws[3],Ws[1],k)+offset,Ws[2])

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

def pca_denoise(arr,Npc):
    u,s,v = np.linalg.svd(arr)
    return u[:,:Npc] @ np.diag(s[:Npc]) @ v[:Npc,:]
