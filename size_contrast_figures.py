#!/usr/bin/env python

import numpy as np
import pyute as ut
import opto_utils
import naka_rushton_analysis as nra
import matplotlib.pyplot as plt

class ca_imaging(object):
    # contains parameters for generating figures relevant to calcium imaging data, in Mossing et al. 2021
    def __init__(self, rsfile, norm_cellwise=False, norm_all_but_60=True):
        npyfile = np.load(rsfile,allow_pickle=True)[()]
        self.rs = npyfile['rs']
        self.rs_sem = npyfile['rs_sem']
        self.roi_ids = npyfile['roi_ids']
        self.expt_ids = npyfile['expt_ids']

        self.usize = np.array((5,8,13,22,36,60))
        self.usize0 = np.concatenate(((0,),self.usize))
        self.ucontrast = np.array((0,6,12,25,50,100))
        
        self.run_dim = 0
        self.celltype_dim = 1
        self.align_dim = 2

        self.expt_dim = 0
        self.neuron_dim = 0
        self.size_dim = 1
        self.contrast_dim = 2
        self.ori_dim = 3

        self.celltype_lbls = ['pc_l4','pc_l23','sst','vip','pv']
        self.c_l4 = np.array((0,1,0))
        self.c_l23 = np.array((0.5,0.5,0.5))
        self.c_sst = np.array((1,0.65,0))
        self.c_vip = np.array((1,0,1))
        self.c_pv = np.array((0,0,1))
        self.celltype_colors = [self.c_l4,self.c_l23,self.c_sst,self.c_vip,self.c_pv]

        self.alignment_lbls = ['aligned', 'misaligned']

        self.running_lbls = ['non_running', 'running']

        self.rso, self.rso_sem = ut.apply_fn_to_nested_lists(opto_utils.sort_both_by_pref_dir,[None,None],self.rs,self.rs_sem)

        self.norm_all_but_60 = norm_all_but_60
        self.norm_msk = np.ones((6,6),dtype='bool')
        if self.norm_all_but_60:
            self.norm_msk[-1] = False
        self.norm_cellwise = norm_cellwise

        self.rso, self.rso_sem = ut.apply_fn_to_nested_lists(self.norm_to_mean,[None,None,None],self.expt_ids,self.rso,self.rso_sem)

        self.nexpt = ut.apply_fn_to_nested_lists_single_out(find_nexpt,[None,None,None],self.expt_ids)
        self.nexpt = accumulate_nexpt(self.nexpt)

        self.rsexpt = ut.apply_fn_to_nested_lists_single_out(average_expt_means,[None,None,None],self.expt_ids,self.rso,self.nexpt)
        self.rsexpt_sem = ut.apply_fn_to_nested_lists_single_out(average_expt_sems,[None,None,None],self.expt_ids,self.rso_sem,self.nexpt)

        self.rsexpt_dims = ut.list_of_lists_dim(self.rsexpt)
        self.rsexpt_dim_lbls = ['running','celltype','alignment']

    def norm_to_mean(self, expt_ids, *args):
        # given a series of args, normalize row i of each to np.nanmean(args[0][i])
        to_return = args 
        if not self.norm_cellwise:
            for expt_id in np.unique(expt_ids):
                mn = np.nanmean(args[0][expt_ids==expt_id][:,self.norm_msk])
                for iarg in range(len(args)):
                    to_return[iarg][expt_ids==expt_id] = args[iarg][expt_ids==expt_id]/mn
        else:
            mn = np.nanmean(args[0][:,self.norm_msk].reshape((args[0].shape[0],-1)),axis=1)
            mn = np.tile(mn,np.ones((len(args[0].shape),)))
            to_return = [arg/mn for arg in args]
        return to_return

    def fit_ayaz_models(self,norm_bottom=[False, False, True, False, False],norm_top=[False, False, False, True, False],two_n=False,sub=False):
        # run fit_ayaz_model, but with special functions for SST and VIP, where for SST the suppressive field is normalized
        # and for VIP, the driving field is normalized (a second time)
        self.norm_bottom,self.norm_top = [tile_list_of_lists(x,self.rsexpt_dims,self.celltype_dim) for x in [norm_bottom, norm_top]]
        self.two_n,self.sub = [tile_list_of_lists(x,self.rsexpt_dims) for x in [two_n, sub]]
        args = [self.norm_bottom, self.norm_top, self.two_n, self.sub]
        self.ams = ut.apply_fn_to_nested_lists_single_out(self.fit_ayaz_model,[None,None,None],self.rsexpt,*args)
        self.ams_simple = ut.apply_fn_to_nested_lists_single_out(self.fit_ayaz_model,[None,None,None],self.rsexpt)

    def fit_ayaz_model(self,rsexpt,norm_bottom=False,norm_top=False,two_n=False,sub=False):
        # fit normalization function as in Ayaz et al. (Carandini-Harris paper)
        data = np.nanmean(rsexpt,self.ori_dim)
        ams = nra.ayaz_model(data,usize=self.usize,ucontrast=self.ucontrast,norm_bottom=norm_bottom,norm_top=norm_top)
        return ams

    def gen_2d_colors(self,this_ncontrast,celltype_colors=None):
        # given a list of celltype colors, build a list of arrays shading those colors down to black, in this_ncontrast steps
        if celltype_colors is None:
            celltype_colors = self.celltype_colors
        cfrac = np.linspace(1,0,this_ncontrast+1)[:-1]
        colors = [cfrac[:,np.newaxis]*np.array(c)[np.newaxis,:] for c in celltype_colors]
        return colors

    def plot_ori_by_size(self,to_plot,colors,savefile=None,isizes=[0,2,4],icontrast=5):
        # at a fixed contrast, plot bootstrapped errorbars for isizes, colored using gen_2d_colors, and save to savefile
        plt.figure(figsize=(2.5,2.5))
        to_plot = mean_normalize(to_plot)
        colors2d = self.gen_2d_colors(len(isizes),celltype_colors=[colors])[0]
        ut.plot_bootstrapped_errorbars_hillel(np.arange(8),to_plot[:,isizes,icontrast,:],colors=colors2d)
        ut.erase_top_right()
        plt.xticks((0,2,4,6),(0,90,180,270))
        plt.xlabel('dir. relative to pref. dir.')
        plt.legend(['%d$^o$'%self.usize[isize] for isize in isizes])
        plt.ylabel('event rate/mean')
        plt.tight_layout()
        if not savefile is None:
            plt.savefig(savefile)

    def gen_rsexpt_filenames(self,filebase,rsexpt_inds=[0,1,2]):
        # generate filenames for list of lists, with formatting string filebase, where rsexpt_inds specifies the order in which
        # rsexpt_dim_lbls are invoked in filebase. E.g. [1,0] indicates rsexpt_dim_lbls[1] appears first and rsexpt_dim_lbls[0] 
        # appears second
        lbls = [getattr(self,'%s_lbls'%self.rsexpt_dim_lbls[ind]) for ind in rsexpt_inds]
        tiled_lbls = [tile_list_of_lists(lbl,self.rsexpt_dims,axis=ind) for lbl,ind in zip(lbls,rsexpt_inds)]
        gen_filenames_ = lambda *args: gen_filenames(filebase,*args)
        filenames = ut.apply_fn_to_nested_lists_single_out(gen_filenames_,[None,None,None],*tiled_lbls)
        return filenames

def gen_filenames(filebase,*args):
    # generate filenames from formatting string with enough blanks for args
    assert(filebase.count('%s')==len(args))
    return filebase % args

class extracellular_ephys(object):
    def __init__(self, rsfile):
        self.rrs,self.rfs = ut.loadmat(rsfile,['rrs','rfs'])
        self.ucontrast = [0,5,10,20,40,80]

def find_nexpt(expt_ids):
    return int(expt_ids.max()+1)

def accumulate_nexpt(nexpt):
    nrun = len(nexpt)
    ntype = len(nexpt[0])
    nalign = len(nexpt[0][0])
    for itype in range(len(nexpt[0])):
        all_nexpts = [[nexpt[irun][itype][ialign] for irun in range(nrun)] for ialign in range(nalign)]
        all_nexpts = np.max(np.array(all_nexpts).flatten())
        for irun in range(nrun):
            for ialign in range(nalign):
                nexpt[irun][itype][ialign] = all_nexpts
    return nexpt

def average_expt_means(expt_ids,rso,nexpt):
    rsexpt = np.nan*np.ones((nexpt,)+rso.shape[1:])
    for iexpt in range(nexpt):
        if np.any(expt_ids==iexpt):
            rsexpt[iexpt] = np.nanmean(rso[expt_ids==iexpt],0)
    print(rsexpt.shape)
    return rsexpt

def average_expt_sems(expt_ids,rso_sem,nexpt):
    rsexpt_sem = np.nan*np.ones((nexpt,)+rso_sem.shape[1:])
    for iexpt in range(nexpt):
        in_this_expt = (expt_ids==iexpt)
        if np.any(in_this_expt):
            n_this_expt = np.sum(in_this_expt)
            rsexpt_sem[iexpt] = np.sqrt(np.nansum(rso_sem[in_this_expt]**2,0))/n_this_expt
    return rsexpt_sem

#def tile_list_of_lists(this_list,dims,axis=None):
#    if not axis is None:
#        slicer = [np.newaxis for dim in range(axis)] + [slice(None)] + [np.newaxis for dim in range(axis+1,len(dims))]
#    else:
#        slicer = [np.newaxis for dim in range(1,len(dims))]
#        this_list = [this_list]
#    this_arr = np.array(this_list)[slicer]
#    this_arr = np.tile(this_arr,dims)
#    return this_arr.tolist()

def tile_list_of_lists(this_list,dims,axis=None):
    if not axis is None:
        starting_here = [repeat_value_list_of_lists(this_val,dims[axis+1:]) for this_val in this_list]
        return repeat_value_list_of_lists(starting_here,dims[:axis])
    else:
        return repeat_value_list_of_lists(this_list,dims)

def repeat_value_list_of_lists(this_val,dims):
    if not len(dims):
        return this_val
    else:
        return [repeat_value_list_of_lists(this_val,dims[1:]) for d in range(dims[0])]

def mean_normalize(to_plot):
    mn = np.nanmean(to_plot.reshape((to_plot.shape[0],-1)),-1)
    shp = to_plot.shape
    slicer = [slice(None)] + [np.newaxis for s in shp[1:]]
    normed = to_plot/mn[slicer]
    return normed


def fit_two_n_fn(data):
    return fit_nr_fn(data,model_fn=nra.nr_two_n_model)

def fit_two_c50_fn(data):
    return fit_nr_fn(data,model_fn=nra.nr_two_c50_model)

def fit_nr_fn(data,model_fn=nra.nr_two_n_model):
    ucontrast = np.array((0,6,12,25,50,100))
    if not data is None and data.size:
        this_data = np.nanmean(data,-1)
        this_model = model_fn(ucontrast,this_data)

        if np.sum(this_model.non_nan_rows):
            plt.figure()
            x = np.arange(100)
            nr_data = this_model.fn(x,1)
            iisize = 0
            for isize in range(6):
                if this_model.non_nan_rows[isize]:
                    plt.subplot(2,3,isize+1)
                    plt.plot(ucontrast,this_data[isize])
                    plt.plot(x,nr_data[iisize])
                iisize = iisize+1

        return this_model
    else:
        return None

def c50_monotonic_fn(data,clip_decreasing=clip_decreasing,clip_after=clip_after):
    if not data is None:
        this_data = np.nanmean(data,2)
        this_sem = np.nanmean(data,2)
        this_data[:,0] = np.nanmean(this_data[:,0])
        lkat = np.all(~np.isnan(this_data),1)
        nparams = np.sum(lkat)
        c50s = np.nan*np.ones((this_data.shape[0],))
        opt_params = nra.fit_opt_params_monotonic(ucontrast[:cmax],this_data[lkat,:cmax],clip_decreasing=clip_decreasing,clip_after=clip_after)
        c50s[lkat] = opt_params[nparams+1:2*nparams+1]

        if lkat.sum():
            plt.figure()
            x = np.arange(100)
            nr_data = nra.naka_rushton(x,opt_params,lkat.sum())
            iisize = 0
            for isize in range(6):
                if lkat[isize]:
                    plt.subplot(2,3,isize+1)
                    plt.plot(ucontrast,this_data[isize])
                    plt.plot(x,nr_data[iisize])
                    plt.plot(ucontrast,npmaximum.accumulate(this_data[isize]))
                    plt.ylim(0,1.1*this_data[isize].max())
                iisize = iisize+1
    else:
        c50s = None

    return c50s

def interp_numeric_c50_fn(data,resolution=501,thresh=0.5):
    if not data is None:
        this_data = np.nanmean(data,2)
        this_sem = np.nanmean(data,2)
        this_data[:,0] = np.nanmean(this_data[:,0])
        lkat = np.all(~np.isnan(this_data),1)
        c50s = np.nan*np.ones((this_data.shape[0],))
        for isize in range(this_data.shape[0]):
            if lkat[isize]:
                crf = sip.interp1d(ucontrast[:cmax],this_data[isize,:cmax])
                cinterp = np.linspace(0,ucontrast[cmax-1],resolution)
                c50s[isize] = nra.numeric_c50_fn(crf,cinterp,thresh=0.5)

        if lkat.sum():
            plt.figure()
            iisize = 0
            for isize in range(6):
                if lkat[isize]:
                    plt.subplot(2,3,isize+1)
                    plt.title('%.02f'%c50s[isize])
                    plt.plot(ucontrast,this_data[isize])
                    plt.axvline(c50s[isize],c='k',linestyle='dashed')
                    plt.ylim(0,1.1*this_data[isize].max())
                iisize = iisize+1
    else:
        c50s = None

    return c50s
