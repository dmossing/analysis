#!/usr/bin/env python
# %%
import numpy as np
import pyute as ut
import opto_utils
import naka_rushton_analysis as nra
import matplotlib.pyplot as plt
from numpy import maximum as npmaximum
import scipy.stats as sst
import size_contrast_analysis as sca
import size_contrast_figures as scf
import sim_utils

import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

def preprocess(*args,average=True):
    if average:
        return tuple([np.nanmean(arg,0)[np.newaxis] for arg in args])
    else:
        return args

def get_data(tuning,expt,ipart,irois=None,irun=0,shp=(4,6,8,2)):
    if irois is None:
        return tuning[irun].loc[expt,ipart].to_numpy().reshape((-1,)+shp)
    else:
        return tuning[irun].loc[expt,ipart].iloc[irois].to_numpy().reshape((-1,)+shp)

def compute_pref_angle(data,based_on='all'):
    data_mean = np.nanmean(np.nanmean(np.nanmean(data,1),1),-1)
    pref_angle = np.argmax(data_mean,axis=1)
    return pref_angle

def align_to_angle(data,pref_angle):
    output = np.zeros_like(data)
    nangle = data.shape[3]
    for iroi in range(data.shape[0]):
        slicer = list(np.arange(pref_angle[iroi],nangle))+list(np.arange(0,pref_angle[iroi]))
        output[iroi] = data[iroi][:,:,slicer]
    return output

def compute_slopes(csi,first_ind=0,xaxis=None,norm_to_max=False):
    nexpt = csi.shape[0]
    nsize = csi.shape[1]
    nlight = csi.shape[2]
    csislope = np.zeros((nexpt,nlight))
    for iexpt in range(nexpt):
        for ilight in range(nlight):
            csislope[iexpt,ilight] = scf.compute_mislope(csi[iexpt,:,ilight],first_ind=first_ind,last_ind=nsize-1,xaxis=xaxis,norm_to_max=norm_to_max)
    return csislope

def compute_summary_sc(tuning,exptlist,cands=None,irun=0):
    shp = (4,6,2)
    scbig,scanimal,semanimal = [np.zeros((0,) + shp) for _ in range(3)]
    expt_ids = np.zeros((0,))
    for iexpt,expt in enumerate(exptlist):
        if not cands is None:
            scs = [opto_utils.norm_to_mean_light_off(np.nanmean(get_data(tuning,expt,ipart,cands[expt],irun=irun),3)) for ipart in range(3)]
        else:
            scs = [opto_utils.norm_to_mean_light_off(np.nanmean(get_data(tuning,expt,ipart,irun=irun),3)) for ipart in range(3)]
        this_ipart = 2
        scbig = np.concatenate((scbig,scs[this_ipart]),axis=0)
        scanimal = np.concatenate((scanimal,np.nanmean(scs[this_ipart],0)[np.newaxis]),axis=0)
        semanimal = np.concatenate((semanimal,(np.nanstd(scs[this_ipart],0)/np.sqrt(scs[2].shape[0]))[np.newaxis]),axis=0)
        expt_ids = np.concatenate((expt_ids,iexpt*np.ones((scs[this_ipart].shape[0],))))
    return scbig,scanimal,semanimal,expt_ids

def compute_summary_sc_aligned(tuning,exptlist,displacement,cands=None,other_cands=None,irun=0,dcutoff=10,aligned=True):
    shp = (4,6,8,2)
    scbig_aligned,scanimal_aligned,semanimal_aligned = [np.zeros((0,) + shp) for _ in range(3)]
    expt_ids_aligned = np.zeros((0,))
    for iexpt,expt in enumerate(exptlist):
        if aligned:
            centered = np.sqrt(np.sum(displacement[expt]**2,1))<10
        else:
            centered = np.sqrt(np.sum(displacement[expt]**2,1))>10
        
        if not cands is None:
            centered = centered[cands[expt]]
            data = [opto_utils.norm_to_mean_light_off(get_data(tuning,expt,ipart,cands[expt],irun=irun)) for ipart in range(3)]
        else:
            data = [opto_utils.norm_to_mean_light_off(get_data(tuning,expt,ipart,irun=irun)) for ipart in range(3)]
        
        this_ipart = 2
        
        roilist = np.arange(data[this_ipart].shape[0])
        unlabeled = np.ones(roilist.shape,dtype='bool')
        if not other_cands is None:
            for oc in other_cands:
                unlabeled = unlabeled & ~np.in1d(roilist,oc[expt])
                
        pref_angle = compute_pref_angle(data[this_ipart])
        scs = [align_to_angle(data[ipart],pref_angle) for ipart in range(3)]
            
        scbig_aligned = np.concatenate((scbig_aligned,scs[this_ipart][centered & unlabeled]),axis=0)
        if centered.sum()>100 and cands is None:
            scanimal_aligned = np.concatenate((scanimal_aligned,np.nanmean(scs[this_ipart][centered & unlabeled],0)[np.newaxis]),axis=0)
        else:
            scanimal_aligned = np.concatenate((scanimal_aligned,np.nan*np.nanmean(scs[this_ipart][centered & unlabeled],0)[np.newaxis]),axis=0)
        expt_ids_aligned = np.concatenate((expt_ids_aligned,iexpt*np.ones((scs[this_ipart][centered & unlabeled].shape[0],))))
    return scbig_aligned,scanimal_aligned,semanimal_aligned,expt_ids_aligned

def compute_mis(xdata_norm, ydata_norm, average=False, norm_first=False, mi_fn=scf.smi_fn):
    if norm_first:
        this_xdata_norm, this_ydata_norm = norm_x_y(xdata_norm, ydata_norm)
    else:
        this_xdata_norm, this_ydata_norm = xdata_norm, ydata_norm
    mi = np.stack([mi_fn(x[:,:,:,np.newaxis]) for x in preprocess(this_xdata_norm,this_ydata_norm,average=average)],axis=-1)
    return mi

def compute_mimis(xdata_norm, ydata_norm, first_ind=0, average=False, norm_first=False, mi_fn=scf.smi_fn, **opt):
    mi = compute_mis(xdata_norm, ydata_norm, average=average, norm_first=norm_first, mi_fn=mi_fn)
    mimis = compute_slopes(mi, first_ind=first_ind, **opt)
    return mimis

def plot_mimi_bars_with_lines(xdata_norm,ydata_norm,mi_fn=scf.smi_fn,first_ind=0,c=None,average=False,save_string='',save=True,**opt):
    # make a bar plot, one black bar indicating metric with light off (calculated on xdata_norm)
    # and one colorful bar (c) indicating metric with light on (calculated on ydata_norm)
    # transparent lines indicate individual model fits
    # metric is computed using mi_fn
    # first_ind refers to the first index to use when computing the modulation index on the metric (here, slope)
    # 'average' indicates whether to average across model fits before plotting

    mi = np.stack([mi_fn(x[:,:,:,np.newaxis]) for x in preprocess(xdata_norm,ydata_norm,average=average)],axis=-1)
    mimis = compute_slopes(mi,first_ind=first_ind,**opt)

    # mimis = compute_mimis(xdata_norm,ydata_norm,first_ind=first_ind,average=average,**opt)

    plt.figure(figsize=(2.5,2.5))
    colors = [np.array((0,0,0))[np.newaxis],c[np.newaxis]]
    ut.plot_bars_with_lines(mimis,colors=colors,alpha=0.05,errorstyle='pct')
    ut.erase_top_right()
    if save:
        do_saving(save_string)
    _,p = sst.wilcoxon(mimis[:,0],mimis[:,1])
    print('p-value: '+str(p))
    print('%d/%d higher'%((mimis[:,1]>mimis[:,0]).sum(),mimis.shape[0]))

def plot_mi_errorbars(xdata_norm,ydata_norm,mi_fn=scf.smi_fn,first_ind=0,c=None,average=False,log_xaxis=True,xaxis=None,save_string='',save=True):
    # make error bar plot, one black curve indicating metric with light off (calculated on xdata_norm)
    # and one colorful curve (c) indicating metric with light on (calculated on ydata_norm)
    # metric is computed using mi_fn
    # first_ind refers to the first index to use when plotting the metric
    # 'average' indicates whether to average across model fits before plotting
    rg = slice(first_ind,None)
    mis = [mi_fn(x[:,:,:,np.newaxis]) for x in preprocess(xdata_norm,ydata_norm,average=average)]
    plt.figure(figsize=(2.5,2.5))
    colors = [np.array((0,0,0))[np.newaxis],c[np.newaxis]]
    for mi,color in zip(mis,colors):
        to_plot = mi[:,np.newaxis,rg]
        if log_xaxis:
            xaxis = np.arange(to_plot.shape[2])
        ut.plot_pct_errorbars_hillel(xaxis,to_plot,colors=color,pct=(16,84))#,markersize=10)
    ut.erase_top_right()
    if save:
        do_saving(save_string)

def gen_stim_light_df(arr):
    arr = arr[np.sum(np.sum(arr>0,-1),-1)>0]
    shp = arr.shape
    exptno,size,light = np.meshgrid(*[np.arange(s) for s in shp],indexing='ij')
    keys = ['response','exptno','size','light']
    vals = [arr,exptno,size,light]
    bool_vals = [np.isnan(vals[0])+0]+vals[1:]
    d = {k:v.flatten() for k,v in zip(keys,bool_vals)}
    df_is_nan = pd.DataFrame(data=d)
    
#     lkat = np.sum(np.isnan(vals[0]),axis=2)==0
    lkat = ~np.isnan(vals[0])
    print(lkat.shape)
    non_nan_vals = [v[lkat] for v in vals]
    d = {k:v.flatten() for k,v in zip(keys,non_nan_vals)}
    df_non_nan = pd.DataFrame(data=d)
    
    return df_is_nan,df_non_nan

def gen_stim_df(arr):
    arr = arr[np.sum(np.sum(arr>0,-1),-1)>0]
    shp = arr.shape[:-1]
    exptno,size = np.meshgrid(*[np.arange(s) for s in shp],indexing='ij')
    keys = ['response_off','response_on','exptno','size']
    vals = [arr[:,:,0],arr[:,:,1],exptno,size]
    bool_vals = [np.isnan(vals[0])+0]+vals[1:]
    d = {k:v.flatten() for k,v in zip(keys,bool_vals)}
    df_is_nan = pd.DataFrame(data=d)
    
#     lkat = np.sum(np.isnan(vals[0]),axis=2)==0
    lkat = ~np.isnan(vals[0])
    print(lkat.shape)
    non_nan_vals = [v[lkat] for v in vals]
    d = {k:v.flatten() for k,v in zip(keys,non_nan_vals)}
    df_non_nan = pd.DataFrame(data=d)
    
    return df_is_nan,df_non_nan

def gen_size_contrast_df(arr):
    arr = arr[np.sum(np.sum(np.sum(arr>0,-1),-1),-1)>0]
    shp = arr.shape[:-1]
    exptno,size,contrast = np.meshgrid(*[np.arange(s) for s in shp],indexing='ij')
    keys = ['response_off','response_on','exptno','size','contrast']
    vals = [arr[:,:,:,0],arr[:,:,:,1],exptno,size,contrast]
    bool_vals = [np.isnan(vals[0])+0]+vals[1:]
    d = {k:v.flatten() for k,v in zip(keys,bool_vals)}
    df_is_nan = pd.DataFrame(data=d)
    
#     lkat = np.sum(np.isnan(vals[0]),axis=2)==0
    lkat = ~np.isnan(vals[0])
    print(lkat.shape)
    non_nan_vals = [v[lkat] for v in vals]
    d = {k:v.flatten() for k,v in zip(keys,non_nan_vals)}
    df_non_nan = pd.DataFrame(data=d)
    
    return df_is_nan,df_non_nan

def gen_size_contrast_light_df(arr):
    arr = arr[np.sum(np.sum(np.sum(arr>0,-1),-1),-1)>0]
    shp = arr.shape
    exptno,size,contrast,light = np.meshgrid(*[np.arange(s) for s in shp],indexing='ij')
    keys = ['response','exptno','size','contrast','light']
    vals = [arr[:,:,:,:],exptno,size,contrast,light]
    bool_vals = [np.isnan(vals[0])+0]+vals[1:]
    d = {k:v.flatten() for k,v in zip(keys,bool_vals)}
    df_is_nan = pd.DataFrame(data=d)
    
#     lkat = np.sum(np.isnan(vals[0]),axis=2)==0
    lkat = ~np.isnan(vals[0])
    print(lkat.shape)
    non_nan_vals = [v[lkat] for v in vals]
    d = {k:v.flatten() for k,v in zip(keys,non_nan_vals)}
    df_non_nan = pd.DataFrame(data=d)
    
    return df_is_nan,df_non_nan

def scatter_size_contrast_x_dx(xdata_norm,ydata_norm,ylim=(-0.12,0.8),save_string='',c=None,save=True):
    plt.figure(figsize=(2.5,2.5))
    sca.scatter_size_contrast_errorbar(xdata_norm,ydata_norm-xdata_norm,equality_line=False,square=False)
    plt.axhline(0,c='k',linestyle='dashed')
    if not ylim is None:
        plt.ylim(ylim)
    ut.erase_top_right()
    plt.xlabel('baseline SST firing rate/mean')
    plt.ylabel('$\Delta$ SST firing rate/mean, \n VIP silencing')
    plt.tight_layout()
    if save:
        do_saving(save_string)

def plot_size_tuning_errorbars(xdata_norm,ydata_norm,c=None,these_contrasts=[1,5],bfactors=[0.5,1],ylim=None,save_string='',log_xaxis=False,save=True):
    usize = np.array((5,8,13,25,36,60))
    usize0 = np.concatenate(((0,),usize))
    if not log_xaxis:
        ux = usize0
    else:
        ux = np.arange(usize0.shape[0])
    xsize_tuning = sim_utils.gen_size_tuning(xdata_norm)
    ysize_tuning = sim_utils.gen_size_tuning(ydata_norm)
    for this_contrast,bfactor in zip(these_contrasts,bfactors):
        plt.figure(figsize=(2.5,2.5))
        ut.plot_pct_errorbars_hillel(ux,xsize_tuning[:,:,this_contrast:this_contrast+1].transpose((0,2,1)),colors=1-bfactor*(1-np.array((0,0,0))[np.newaxis]),pct=(16,84))#,markersize=10)
        ut.plot_pct_errorbars_hillel(ux,ysize_tuning[:,:,this_contrast:this_contrast+1].transpose((0,2,1)),colors=1-bfactor*(1-c[np.newaxis]),pct=(16,84))#,markersize=2)
        plt.gca().set_ylim(bottom=0)
        if log_xaxis:
            plt.xticks(ux,usize0)
        else:
            plt.xticks((0,20,40,60))
        ut.erase_top_right()
        plt.xlabel('size ($^o$)')
        plt.ylabel('event rate/mean')
        plt.tight_layout()
        if not ylim is None:
            plt.ylim(ylim)
        if save:
            do_saving(save_string % this_contrast)

def plot_contrast_tuning_errorbars(xdata_norm,ydata_norm,c=None,these_sizes=[0,5],bfactors=[0.5,1],ylim=None,save_string='',log_xaxis=True,save=True):
    ucontrast = np.array((0,6,12,25,50,100))
    if log_xaxis:
        ux = np.arange(ucontrast.shape[0])
    else:
        ux = ucontrast
    for this_size,bfactor in zip(these_sizes,bfactors):
        plt.figure(figsize=(2.5,2.5))
        ut.plot_pct_errorbars_hillel(ux,xdata_norm[:,this_size:this_size+1],colors=1-bfactor*(1-np.array((0,0,0))[np.newaxis]),pct=(16,84))#,markersize=10)
        ut.plot_pct_errorbars_hillel(ux,ydata_norm[:,this_size:this_size+1],colors=1-bfactor*(1-c[np.newaxis]),pct=(16,84))#,markersize=2)
        plt.gca().set_ylim(bottom=0)
        if log_xaxis:
            plt.xticks(ux,ucontrast)
        ut.erase_top_right()
        plt.xlabel('contrast (%)')
        plt.ylabel('event rate/mean')
        plt.tight_layout()
        if not ylim is None:
            plt.ylim(ylim)
        if save:
            do_saving(save_string % this_size)

def scatter_csi(xdata_norm,ydata_norm,alpha=1,save_string='',c=None,save=True):
    this_scsstanimal = np.stack((xdata_norm,ydata_norm),axis=3)
    cmax = 6
    csisst = np.zeros((this_scsstanimal.shape[0],this_scsstanimal.shape[1],this_scsstanimal.shape[3]))
    for ilight in range(2):
        csisst[:,:,ilight] = scf.csi_fn(this_scsstanimal[:,:,:cmax,ilight:ilight+1])#,sem=semsstanimal)
    to_plot = csisst.copy()
    plt.figure(figsize=(2.5,2.5))
    noise = 0e-3
    cs = ['C%d'%iexpt for iexpt in range(to_plot.shape[0])]
    for iexpt in range(to_plot.shape[0]):
        for isize in range(to_plot.shape[1]):
            plt.scatter(to_plot[iexpt,isize,0]+noise*np.random.randn(1),to_plot[iexpt,isize,1]+noise*np.random.randn(1),s=10*(isize+1),c=cs[iexpt],edgecolor='k',alpha=alpha)
    plt.plot((0,1),(0,1),c='k')
    plt.xticks(np.linspace(0,1,6))
    plt.yticks(np.linspace(0,1,6))
    ut.erase_top_right()
    plt.xlabel(r'CSI, light off')
    plt.ylabel(r'CSI, light on')
    plt.tight_layout()
    if save:
        plt.savefig(save_string)
    print(sst.wilcoxon(to_plot[:,:,0].mean(1),to_plot[:,:,1].mean(1)))

def plot_smi_errorbars(xdata_norm,ydata_norm,c=None,save_string='',save=True):
    # run plot_mi_errorbars using SMI as the metric
    ucontrast = np.array((0,6,12,25,50,100))
    first_ind = 1
    plot_mi_errorbars(xdata_norm,ydata_norm,mi_fn=scf.smi_fn,first_ind=first_ind,c=c)
    plt.xticks(np.arange(len(ucontrast[first_ind:])),ucontrast[first_ind:])
    plt.xlabel('contrast (%)')
    plt.ylabel('SMI')
    plt.tight_layout()
    if save:
        do_saving(save_string)

def plot_csi_errorbars(xdata_norm,ydata_norm,c=None,save_string='',save=True):
    # run plot_mi_errorbars using CSI as the metric
    usize = np.array((5,8,13,25,36,60))
    first_ind = 0
    plot_mi_errorbars(xdata_norm,ydata_norm,mi_fn=scf.csi_fn,first_ind=first_ind,c=c,log_xaxis=False,xaxis=usize)
    #plt.xticks(usize[first_ind:])
    #plt.xticks(np.arange(len(usize[first_ind:])),usize[first_ind:])
    plt.xlabel('size ($^o$)')
    plt.ylabel('CSI')
    plt.tight_layout()
    if save:
        do_saving(save_string)

def plot_smimi_bars_with_lines(xdata_norm,ydata_norm,c=None,save_string='',save=True,**opt):
    # run plot_mimi_bars_with_lines using SMI as the metric
    first_ind = 1
    plot_mimi_bars_with_lines(xdata_norm,ydata_norm,mi_fn=scf.smi_fn,first_ind=first_ind,c=c,**opt)
    plt.ylabel('slope, SMI vs. contrast')
    plt.tight_layout()
    if save:
        do_saving(save_string)

def plot_csimi_bars_with_lines(xdata_norm,ydata_norm,c=None,save_string='',save=True,**opt):
    # run plot_mimi_bars_with_lines using CSI as the metric
    first_ind = 0
    plot_mimi_bars_with_lines(xdata_norm,ydata_norm,mi_fn=scf.csi_fn,first_ind=first_ind,c=c,**opt)
    plt.ylabel('slope, CSI vs. size')
    plt.tight_layout()
    if save:
        do_saving(save_string)

def norm_x_y(xdata, ydata):
    ydata_norm = ydata/xdata.mean(1).mean(1)[:,np.newaxis,np.newaxis]
    xdata_norm = xdata/xdata.mean(1).mean(1)[:,np.newaxis,np.newaxis]
    return xdata_norm, ydata_norm

def run_mi_plotting(
        network_resps,
        target_bin,
        plot_mi_fn=plot_smi_errorbars,
        plot_mi_lbl='pc_smi_vs_contrast_l4',
        ext='eps',
        iconns=[0,2,3,4],
        itype=0,
        **opt
    ):
    # run a plotting function e.g. plot_mi_errorbars, or plot_mimi_bars_with_lines
    # on each of many opto stim directions, and many perturbations of the modeled network
    # with a given desired metric, plotting the outcome of that metric on PCs
    # network_resps a list of lists, opto stim direction x perturbation x (fit x size x contrast x cell type)
    # save the figure, indicating the type of plot, opto stim direction, perturbation, filename and extension
    ibaseline = 15

    idir = 2
    lkat = ut.k_and(*[network_resps[idir][iconn].max(1).max(1).max(1) < 10 for iconn in [0,2,3,4]])
    light_lbls = ['silencing','activation']
    conn_lbls = ['baseline','pcpc_deleted','pcpv_deleted','pcvip_deleted','pcsst_deleted','']
    cs = [np.array((1,0.8,0)),np.array((1,0,0))]
    for iconn in iconns:#[0,2,3,4]:
        if isinstance(target_bin, tuple):
            ilow = int(np.round(np.mean(target_bin[0][lkat])))
            ihigh = int(np.round(np.mean(target_bin[1][lkat])))#np.minimum(np.maximum(target_bin[lkat],0),20)#np.maximum(target_bin[lkat]-2,15)#ilight_drmax[2][iconn][lkat]
        else:
            ihigh = int(np.round(np.mean(target_bin[lkat])))#np.minimum(np.maximum(target_bin[lkat],0),20)#np.maximum(target_bin[lkat]-2,15)#ilight_drmax[2][iconn][lkat]
            ilow = ihigh#30-ihigh#4#ilight_drmax[1][iconn][lkat]
        idir = 0
        xdata = network_resps[idir][iconn][lkat][:,ibaseline,:,itype].reshape((-1,6,6))
        # xdata_norm = xdata/xdata.mean(1).mean(1)[:,np.newaxis,np.newaxis]
        ylims = [None,None]
        for idir,ilight,light_lbl,c,ylim in zip([1,2],[ilow,ihigh],light_lbls,cs,ylims):
            #ydata = network_resps[idir][iconn][lkat][:,:,:,itype][np.arange(lkat.sum()),ilight].reshape((-1,6,6))
            ydata = network_resps[idir][iconn][lkat][:,ilight,:,itype].reshape((-1,6,6))
            xdata_norm, ydata_norm = norm_x_y(xdata, ydata)
            # ydata_norm = ydata/xdata.mean(1).mean(1)[:,np.newaxis,np.newaxis]
            save_string = 'figures/%s_%s_%s.%s'%(plot_mi_lbl,light_lbl,conn_lbls[iconn],ext)
            opt['c'] = cs[idir-1]
            opt['save_string'] = save_string
            plot_mi_fn(xdata_norm,ydata_norm,**opt)

def do_saving(save_string):
    if len(save_string):
        if save_string[-3:] == 'jpg':
            plt.savefig(save_string,dpi=300)
        else:
            plt.savefig(save_string)

def compute_df_wrapper(dsfile='', keylist=[]):
    df,roi_info,trial_info = ut.compute_tavg_dataframe(dsfile,expttype='size_contrast_opto_0',keylist=keylist)
    output = {
        'df':df,
        'roi_info':roi_info,
        'trial_info':trial_info
    }
    return output