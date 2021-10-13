#!/usr/bin/env python

import numpy as np
import pyute as ut
import opto_utils
import naka_rushton_analysis as nra
import matplotlib.pyplot as plt
from numpy import maximum as npmaximum
import scipy.stats as sst
import size_contrast_analysis as sca

class ca_imaging(object):
    # contains parameters for generating figures relevant to calcium imaging data, in Mossing et al. 2021
    def __init__(self, rsfile=None, norm_cellwise=False, norm_all_but_60=True):
        if not rsfile is None:
            npyfile = np.load(rsfile,allow_pickle=True)[()]
            self.rs = npyfile['rs']
            self.rs_sem = npyfile['rs_sem']
            self.roi_ids = npyfile['roi_ids']
            self.expt_ids = npyfile['expt_ids']

        self.usize = np.array((5,8,13,22,36,60))
        self.usize0 = np.concatenate(((0,),self.usize))
        self.ucontrast = np.array((0,6,12,25,50,100))
        
        self.running_dim = 0
        self.celltype_dim = 1
        self.alignment_dim = 2

        self.expt_dim = 0
        self.neuron_dim = 0
        self.size_dim = 1
        self.contrast_dim = 2
        self.ori_dim = 3 

        self.full_celltype_lbls = ['L4 PC','L2/3 PC','SST','VIP','PV']
        self.celltype_lbls = ['pc_l4','pc_l23','sst','vip','pv']
        self.disp_order = [0,1,4,2,3]

        self.c_l4 = np.array((0,0.5,0))
        self.c_l23 = np.array((0.5,0.5,0.5))
        self.c_sst = np.array((1,0.65,0))
        self.c_vip = np.array((1,0,1))
        self.c_pv = np.array((0,0,1))
        self.celltype_colors = [self.c_l4,self.c_l23,self.c_sst,self.c_vip,self.c_pv]
        self.celltype_colors_for_2d = [2*self.c_l4,self.c_l23,self.c_sst,self.c_vip,self.c_pv]

        self.alignment_lbls = ['aligned', 'misaligned']

        self.running_lbls = ['non_running', 'running']
        
        this_fn = lambda x,y: opto_utils.sort_both_by_pref_dir(x,y,return_pref_dir=True)
        self.rso, self.rso_sem,self.pref_dir = ut.apply_fn_to_nested_lists(this_fn,[None,None],self.rs,self.rs_sem)

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

        self.rsexpt_colors = tile_list_of_lists(self.celltype_colors,dims=self.rsexpt_dims,axis=self.celltype_dim)
        self.rsexpt_colors_for_2d = tile_list_of_lists(self.celltype_colors_for_2d,dims=self.rsexpt_dims,axis=self.celltype_dim)
        self.rsexpt_full_celltype_lbls = tile_list_of_lists(self.full_celltype_lbls,dims=self.rsexpt_dims,axis=self.celltype_dim)

        self.nrunning = self.rsexpt_dims[self.running_dim]
        self.ncelltype = self.rsexpt_dims[self.celltype_dim]
        self.nalignment = self.rsexpt_dims[self.alignment_dim]

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

        this_fn = lambda *args: self.fit_ayaz_model(*args,nanmean_all=True)
        self.ams_all = ut.apply_fn_to_nested_lists_single_out(this_fn,[None,None,None],self.rsexpt,*args)

        self.ams_simple = ut.apply_fn_to_nested_lists_single_out(self.fit_ayaz_model,[None,None,None],self.rsexpt)

    def fit_ayaz_model(self,rsexpt,norm_bottom=False,norm_top=False,two_n=False,sub=False,nanmean_all=False):
        # fit normalization function as in Ayaz et al. (Carandini-Harris paper)
        data = np.nanmean(rsexpt,self.ori_dim)
        if nanmean_all:
            data = np.nanmean(data,self.expt_dim)
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
        savefig(savefile)

    def compute_c50s(self, clip_decreasing = True, clip_after = 2,first_ind=0,last_ind=4):

        this_fn = lambda x: c50_monotonic_fn(x,clip_decreasing=clip_decreasing,clip_after=clip_after,show_fig=False)
        self.c50s = ut.apply_fn_to_nested_list(this_fn,[None,None,None,None],self.rsexpt)

        self.c50_first_ind = first_ind
        self.c50_last_ind = last_ind
        this_fn = lambda x: compute_c50mi(x,first_ind=self.c50_first_ind,last_ind=self.c50_last_ind)
        self.c50mis = ut.apply_fn_to_nested_list(this_fn,[None,None,None,None],self.c50s)

        this_fn = lambda x: compute_c50sc(x,first_ind=self.c50_first_ind,last_ind=self.c50_last_ind,pval=False)
        self.c50scs = ut.apply_fn_to_nested_list(this_fn,[None,None,None,None],self.c50s)
        this_fn = lambda x: compute_c50sc(x,first_ind=self.c50_first_ind,last_ind=self.c50_last_ind,pval=True)
        self.c50scs_pval = ut.apply_fn_to_nested_list(this_fn,[None,None,None,None],self.c50s)

    def compute_smis(self,first_ind=1,last_ind=5):

        this_fn = lambda x: smi_fn(x)
        self.smis = ut.apply_fn_to_nested_list(this_fn,[None,None,None],self.rsexpt)

        self.smi_first_ind = first_ind
        self.smi_last_ind = last_ind
        this_fn = lambda x: compute_smimi(x,first_ind=self.smi_first_ind,last_ind=self.smi_last_ind)
        self.smimis = ut.apply_fn_to_nested_list(this_fn,[None,None,None],self.smis)

        this_fn = lambda x: compute_smisc(x,first_ind=self.smi_first_ind,last_ind=self.smi_last_ind,pval=False)
        self.smiscs = ut.apply_fn_to_nested_list(this_fn,[None,None,None,None],self.smis)
        this_fn = lambda x: compute_smisc(x,first_ind=self.smi_first_ind,last_ind=self.smi_last_ind,pval=True)
        self.smiscs_pval = ut.apply_fn_to_nested_list(this_fn,[None,None,None,None],self.smis)

    def compute_ccs(self,show_fig=False,pval_cutoff=0.05):
        self.cc = np.zeros((self.nrunning,self.nalignment,self.ncelltype,self.ncelltype))
        self.cc_list = {}
        self.pval_cc = np.zeros((self.nrunning,self.nalignment,self.ncelltype,self.ncelltype))
        for irun in range(self.nrunning):
            for ialign in range(self.nalignment):
                for itype1 in range(self.ncelltype):
                    for itype2 in range(self.ncelltype):
                        data1 = self.rsexpt[irun][itype1][ialign].copy()
                        data2 = self.rsexpt[irun][itype2][ialign].copy()
                        n1,n2 = data1.shape[0],data2.shape[0]
                        msk = np.zeros((n1,n2),dtype='bool')
                        if itype1==itype2:
                            msk[np.triu_indices(n1,k=1)] = True
                        else:
                            msk[:] = True
                        self.cc_list[(irun,ialign,itype1,itype2)] = np.corrcoef(np.nanmean(data1[:,self.norm_msk],-1).reshape((n1,-1)),np.nanmean(data2[:,self.norm_msk],-1).reshape((n2,-1)))[:n1,n1:][msk]
                        self.cc[irun,ialign,itype1,itype2] = np.nanmean(self.cc_list[irun,ialign,itype1,itype2])
        for irun in range(self.nrunning):
            for ialign in range(self.nalignment):
                for itype1 in range(self.ncelltype):
                    for itype2 in range(self.ncelltype):
                        xdata = self.cc_list[irun,ialign,itype1,itype1]
                        ydata = self.cc_list[irun,ialign,itype1,itype2]
                        if len(xdata[~np.isnan(xdata)])>1 and len(ydata[~np.isnan(ydata)])>1:
                            self.pval_cc[irun,ialign,itype1,itype2] = sst.mannwhitneyu(xdata[~np.isnan(xdata)],ydata[~np.isnan(ydata)]).pvalue
        if show_fig:
            self.show_cc_pval(pval_cutoff=pval_cutoff,irun=0,ialign=0)
            self.plot_cc(pval_cutoff=pval_cutoff,irun=0,ialign=0)

    def show_cc_pval(self,pval_cutoff=0.05,irun=0,ialign=0):
        plt.figure()
        plt.imshow(np.log10(self.pval_cc[irun,ialign][self.disp_order][:,self.disp_order])<np.log10(0.05))
        plt.colorbar()

    def plot_cc(self,pval_cutoff=0.05,irun=0,ialign=0):
        deltai = 0.5
        deltaj = -0.275
        disp_order = self.disp_order
        ntypes = self.ncelltype
        lbls = self.full_celltype_lbls
        ylbls = [lbls[d] for d in disp_order]
        xlbls = [yl.replace(' ',' \n ') for yl in ylbls]
        plt.figure(figsize=(2.5,2.5))
        ut.imshow_hot_cold(self.cc[irun,ialign][disp_order][:,disp_order])
        for i in range(ntypes):
            for j in range(ntypes):
                if self.pval_cc[irun,ialign][disp_order][:,disp_order][i,j]<pval_cutoff:
                    plt.text(j+deltaj,i+deltai,'*',fontsize=16)
        plt.xticks(np.arange(ntypes),xlbls)
        plt.yticks(np.arange(ntypes),ylbls)
        plt.ylim((ntypes-0.5,-0.5))
        plt.colorbar(ticks=np.linspace(-0.8,0.8,9))
        plt.tight_layout()
        plt.savefig('figures/avg_correlation_coefficient_btw_expts_by_celltype.eps')

    def plot_c50_vs_size(self,c50s,colors,savefile=None):
        bootstrap_opaque_plot_transparent(self.usize,c50s,colors)
        plt.xlabel('size ($^o$)')
        plt.ylabel('$C_{50}$')
        plt.tight_layout()
        savefig(savefile)

    def plot_smi_vs_contrast_semilogx(self,smis,colors,savefile=None):
        x = np.arange(1,6)
        bootstrap_opaque_plot_transparent(x,smis[:,1:],colors)
        plt.xticks(x,self.ucontrast[1:])
        plt.xlabel('contrast (%)')
        plt.ylabel('SMI = \n resp. to 60$^o$/max resp.')
        plt.tight_layout()
        savefig(savefile)

    def plot_pref_ori_hist(show_dir=False,split_by_expt_id=False,irun=0,ialign=0):
        def plot_data(data,opt):
            xlbl = opt['xlbl']
            pct_by = opt['pct_by']
            bins = -0.5 + np.arange(pct_by + 1)
            xticks = np.arange(8)[:pct_by]
            xticklbls = np.arange(0,360,45)[:pct_by]
            plt.hist(data % pct_by,bins=bins)
            plt.xticks(xticks,xticklbls)
            plt.xlabel(xlbl)
            plt.ylabel('# of neurons')
        opt = {}
        if show_dir:
            opt['stat_lbl'] = 'dir'
            opt['xlbl'] = 'direction'
            opt['pct_by'] = 8
        else:
            opt['stat_lbl'] = 'ori'
            opt['xlbl'] = 'orientation'
            opt['pct_by'] = 4
        if split_by_expt_id:
            split_lbl = 'by_expt'
        else:
            split_lbl = ''
        for icelltype in range(self.rsexpt_dims[self.celltype_dim]):
            plt.figure(figsize=(2.5,2.5))
            if split_by_expt_id:
                all_ids = np.unique(self.expt_ids[irun][icelltype][ialign])
                plot_data(self.prefdir[irun][icelltype][ialign])
            else:
                plot_data(self.prefdir[irun][icelltype][ialign])
                plt.title(self.celltype_lbls[icelltype])
            plt.tight_layout()
            plt.savefig('figures/%s_pref_%s_hist_%s.eps'%(self.celltype_lbls[icelltype],stat_lbl))

    def plot_sst_vip_spont_bars(self,irun=0,ialign=0,epsilon=0.02):
        rsexpt = self.rsexpt
        ratio = [None for itype in range(self.ncelltype)]
        for itype in range(1,self.ncelltype):
            spont_resp = np.nanmean(np.nanmean(rsexpt[irun][itype][ialign],self.ori_dim)[:,:,0],1)
            max_resp = np.nanmax(np.nanmax(np.nanmean(rsexpt[irun][itype][ialign],self.ori_dim),axis=1),axis=1)
            ratio[itype] = spont_resp/max_resp
        
        isst,ivip = 2,3
        colors = [self.c_sst,self.c_vip]#[(1,0.65,0),(1,0,1)]
        lbls = [self.full_celltype_lbls[isst],self.full_celltype_lbls[ivip]]
        plt.figure(figsize=(2.5,2.5))
        plot_bar_with_dots([ratio[i] for i in [isst,ivip]],colors=colors,epsilon=epsilon)
        plt.xticks(np.arange(2),lbls)
        plt.ylabel('spont./max. event rate')
        ut.erase_top_right()
        plt.tight_layout()
        plt.savefig('figures/spont_event_rate_vip_sst.jpg',dpi=300)
        sst.mannwhitneyu(ratio[isst][~np.isnan(ratio[isst])],ratio[ivip][~np.isnan(ratio[ivip])])

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

def gen_2d_colors(this_ncontrast,celltype_colors=None):
    # given a list of celltype colors, build a list of arrays shading those colors down to black, in this_ncontrast steps
    cfrac = np.linspace(1,0,this_ncontrast+1)[:-1]
    colors = [cfrac[:,np.newaxis]*np.array(c)[np.newaxis,:] for c in celltype_colors]
    return colors

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

def c50_monotonic_fn(data,clip_decreasing=True,clip_after=2,ucontrast=np.array((0,6,12,25,50,100)),show_fig=False):
    if not data is None:
        this_data = np.nanmean(data,2)
        this_sem = np.nanmean(data,2)
        this_data[:,0] = np.nanmean(this_data[:,0])
        lkat = np.all(~np.isnan(this_data),1)
        nparams = np.sum(lkat)
        c50s = np.nan*np.ones((this_data.shape[0],))
        cmax = len(ucontrast)
        opt_params = nra.fit_opt_params_monotonic(ucontrast[:cmax],this_data[lkat,:cmax],clip_decreasing=clip_decreasing,clip_after=clip_after)
        c50s[lkat] = opt_params[nparams+1:2*nparams+1]

        if lkat.sum():
            if show_fig:
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

def smi_fn(data):
    return np.nanmean(data,-1)[:,-1,:]/np.nanmax(np.nanmean(data,-1),1)

def interp_numeric_c50_fn(data,resolution=501,thresh=0.5,show_fig=False):
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
            if show_fig:
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

def compute_c50mi(c50,first_ind=0,last_ind=-2):
    return c50[first_ind] - c50[last_ind]

def compute_smimi(smi,first_ind=1,last_ind=5):
    return smi[:,first_ind] - smi[:,last_ind]

def compute_c50sc(c50,first_ind=0,last_ind=4,pval=False):
    spearman = sst.spearmanr(c50[first_ind:last_ind+1],np.arange(first_ind,last_ind+1))
    if not pval:
        return spearman.correlation
    else:
        return spearman.pval

def compute_smisc(smi,first_ind=1,last_ind=5,pval=False):
    spearman = sst.spearmanr(smi[first_ind:last_ind+1],np.arange(first_ind,last_ind+1))
    if not pval:
        return spearman.correlation
    else:
        return spearman.pval

class fig_gen(object):
    def __init__(self,opt,data_obj):
        self.savefiles = data_obj.gen_rsexpt_filenames(opt['filebase'],opt['rsexpt_inds'])
        this_fn = lambda *args: opt['fn'](*args, **opt['add_kwargs'])
        attr_args = [getattr(data_obj,attr) for attr in opt['attr_args']]
        ut.apply_fn_to_nested_lists_no_out(this_fn,opt['ind_list'],*attr_args,self.savefiles)

class pc_c50mi_bars(fig_gen):
    def __init__(self,data_obj,use_mi=True):
        opt = {}
        opt['filebase'] = 'figures/%s_c50mi_bars_%s.jpg'
        opt['rsexpt_inds'] = [1,0]#(lbls[itype],alignment_lbls[ialign])
        opt['ind_list'] = [None,[0,1],[0]]
        opt['fn'] = self.single_fn
        opt['attr_args'] = ['c50mis','rsexpt_colors','rsexpt_full_celltype_lbls']
        if not use_mi:
            opt['attr_args'][0] = 'c50scs'
        opt['add_kwargs'] = {'usizes':[data_obj.usize[data_obj.c50_first_ind],data_obj.usize[data_obj.c50_last_ind]],'use_mi':use_mi}
        super().__init__(opt,data_obj)
    def single_fn(self,c50mis,colors,full_celltype_lbl,savefile,usizes=[5,36],use_mi=True):
        plt.figure(figsize=(2.5,2.5))
        plot_bar_with_dots(c50mis,colors)
        plt.xticks((0,),[full_celltype_lbl])
        if use_mi:
            plt.ylabel('$C_{50}$ at %d$^o$ - $C_{50}$ at %d$^o$'%tuple(usizes))
        else:
            plt.ylabel('Spearman corr. coef., \n $C_{50}$ vs. size')
        plt.xlim((-1,1))
        ut.erase_top_right()
        plt.tight_layout()
        savefig(savefile.replace('pc_l23','pc'))

class pc_smimi_bars(fig_gen):
    def __init__(self,data_obj,use_mi=True):
        opt = {}
        opt['filebase'] = 'figures/%s_smimi_bars_%s.jpg'
        opt['rsexpt_inds'] = [1,0]#(lbls[itype],alignment_lbls[ialign])
        opt['ind_list'] = [None,[0,1],[0]]
        opt['fn'] = self.single_fn
        opt['attr_args'] = ['smimis','rsexpt_colors','rsexpt_full_celltype_lbls']
        if not use_mi:
            opt['attr_args'][0] = 'smiscs'
        opt['add_kwargs'] = {'ucontrasts':[data_obj.ucontrast[data_obj.smi_first_ind],data_obj.ucontrast[data_obj.smi_last_ind]],'use_mi':use_mi}
        super().__init__(opt,data_obj)
    def single_fn(self,smimis,colors,full_celltype_lbl,savefile,ucontrasts=[6,100],use_mi=True):
        plt.figure(figsize=(2.5,2.5))
        plot_bar_with_dots(smimis,colors)
        plt.xticks((0,),[full_celltype_lbl])
        if use_mi:
            plt.ylabel('SMI at %d%% - SMI at %d%%'%tuple(ucontrasts))
        else:
            plt.ylabel('Spearman corr. coef., \n SMI vs. contrast')
        plt.xlim((-1,1))
        ut.erase_top_right()
        plt.tight_layout()
        savefig(savefile.replace('pc_l23','pc'))

def plot_bar_with_dots(smimi,colors=None,epsilon=0.05,alpha=0.5,pct=(16,84)):
    if not isinstance(smimi,list):
        smimi = [smimi]
        colors = [colors]
    for iitype in range(len(smimi)):
        data = smimi[iitype]
        mn = np.nanmean(data,0)
        lb,ub = ut.bootstrap(data[~np.isnan(data)],pct=pct,axis=0,fn=np.mean)
        plt.bar((iitype,),mn,color=colors[iitype],alpha=alpha)
        plt.errorbar((iitype,),mn,np.nanstd(data,0)/np.sqrt(np.sum(~np.isnan(data),0)),fmt='none',c='k')
        plt.scatter(iitype*np.ones_like(data)+epsilon*np.random.randn(data.shape[0]),data,facecolor=colors[iitype],linewidth=1,edgecolor='k')
        _,p = sst.wilcoxon(data[~np.isnan(data)])
        print(p)

def bootstrap_opaque_plot_transparent(x,Y,colors,savefile=None,alpha=0.2):
    plt.figure(figsize=(2.5,2.5))
    plt.plot(x,Y.T,alpha=0.2,c=colors)
    ut.plot_bootstrapped_errorbars_hillel(x,Y[:,np.newaxis],pct=(16,84),colors=colors[np.newaxis])
    ut.erase_top_right()

def savefig(savefile):
    if not savefile is None:
        if '.eps' in savefile:
            plt.savefig(savefile)
        else:
            plt.savefig(savefile,dpi=300)

class interp_size_tuning(fig_gen):
    def __init__(self,data_obj):
        opt = {}
        opt['filebase'] = 'figures/%s_size_by_3_contrasts_%s_%s.eps'
        opt['rsexpt_inds'] = [1,0,2]
        opt['ind_list'] = [None,None,None]
        opt['fn'] = self.plot_interp_size_tuning
        opt['attr_args'] = ['ams_all','rsexpt_colors_for_2d']
        opt['add_kwargs'] = {'usize':data_obj.usize,'these_contrasts':[1,3,5]}
        super().__init__(opt,data_obj)
    def plot_interp_size_tuning(self,ams,colors,savefile=None,usize=None,these_contrasts=None):
        plt.figure(figsize=(3,2.5))
        colors2d = gen_2d_colors(len(these_contrasts),celltype_colors=[colors])[0]
        nra.plot_interp_size_tuning(ams=ams,usize=usize,these_contrasts=these_contrasts,colors=colors2d)
        savefig(savefile)

class interp_size_slope(fig_gen):
    def __init__(self,data_obj):
        opt = {}
        opt['filebase'] = 'figures/%s_size_slope_by_3_contrasts_%s_%s.eps'
        opt['rsexpt_inds'] = [1,0,2]
        opt['ind_list'] = [None,None,None]
        opt['fn'] = self.plot_interp_size_slope
        opt['attr_args'] = ['ams_all','rsexpt_colors_for_2d']
        opt['add_kwargs'] = {'usize':data_obj.usize,'these_contrasts':[1,3,5]}
        super().__init__(opt,data_obj)
    def plot_interp_size_slope(self,ams,colors,savefile=None,usize=None,these_contrasts=None):
        plt.figure(figsize=(3,2.5))
        usize0 = np.concatenate(((0,),usize))
        colors2d = gen_2d_colors(len(these_contrasts),celltype_colors=[colors])[0]
        nra.plot_interp_size_tuning(ams=ams,usize=usize,these_contrasts=these_contrasts,colors=colors2d,deriv=True)
        data = sca.compute_slope_avg(usize0,ams.data,axis=1)
        plt.gca().axhline(0,linestyle='dotted',c='k')
        plt.legend(['6%','25%','100%'])
        plt.xlabel('size ($^o$)')
        plt.ylabel('size tuning slope (mean/$^o$)')
        ut.erase_top_right()
        plt.tight_layout()
        #savefig(savefile)

class interp_contrast_slope_by_size(fig_gen):
    def __init__(self,data_obj):
        opt = {}
        opt['filebase'] = 'figures/%s_contrast_slope_by_3_contrasts_%s_%s.eps'
        opt['rsexpt_inds'] = [1,0,2]
        opt['ind_list'] = [None,None,None]
        opt['fn'] = self.plot_interp_contrast_slope
        opt['attr_args'] = ['ams_all','rsexpt_colors_for_2d']
        opt['add_kwargs'] = {'usize':data_obj.usize,'these_contrasts':[1,3,5]}
        super().__init__(opt,data_obj)
    def plot_interp_contrast_slope(self,ams,colors,savefile=None,usize=None,these_contrasts=None):
        plt.figure(figsize=(3,2.5))
        usize0 = np.concatenate(((0,),usize))
        nra.plot_interp_size_tuning(ams=ams,usize=usize,these_contrasts=these_contrasts,colors=colors,deriv=True,deriv_axis=2)
        data = sca.compute_slope_avg(usize0,starting_data,axis=1)
        plt.gca().axhline(0,linestyle='dotted',c='k')
        plt.legend(['6%','25%','100%'])
        plt.xlabel('size ($^o$)')
        plt.ylabel('contrast tuning slope (mean/%)')
        ut.erase_top_right()
        plt.tight_layout()
        #savefig(savefile)

class interp_contrast_slope_by_size(fig_gen):
    def __init__(self,data_obj):
        opt = {}
        opt['filebase'] = 'figures/%s_contrast_slope_by_3_contrasts_%s_%s.eps'
        opt['rsexpt_inds'] = [1,0,2]
        opt['ind_list'] = [None,None,None]
        opt['fn'] = self.plot_interp_size_slope
        opt['attr_args'] = ['ams_all','rsexpt_colors_for_2d']
        opt['add_kwargs'] = {'usize':data_obj.usize,'these_contrasts':[1,3,5]}
        super().__init__(opt,data_obj)
    def plot_interp_contrast_slope(self,ams,colors,savefile=None,ucontrast=None,these_sizes=None):
        plt.figure(figsize=(3,2.5))
        nra.plot_interp_contrast_tuning(ams=ams,ucontrast=ucontrast,these_sizes=these_sizes,colors=colors,deriv=True,deriv_axis=2)
        plt.gca().axhline(0,linestyle='dotted',c='k')
        plt.legend(['6%','25%','100%'])
        plt.xticks(np.arange(6),ucontrast)
        plt.ylabel('contrast tuning slope (mean/%)')
        ut.erase_top_right()
        plt.tight_layout()
        #savefig(savefile)
            #data = compute_slope_avg(ucontrast,starting_data,axis=2)
#             ut.plot_bootstrapped_errorbars_hillel(np.arange(6),data[:,these_contrasts,:],pct=(16,84),colors=colors[itype])
            #nra.plot_interp_contrast_tuning(ams=ams_all[ialign][irun][itype],usize=usize,these_sizes=these_sizes,colors=colors[itype],deriv=True,deriv_axis=2)
        plt.gca().axhline(0,linestyle='dotted',c='k')
        plt.legend(['5$^o$','13$^o$','36$^o$'])
        plt.xlabel('contrast (%)')
        plt.ylabel('contrast tuning slope (mean/%)')
        ut.erase_top_right()
        plt.tight_layout()
        plt.savefig('figures/%s_contrast_slope_by_3_sizes_%s_%s.eps'%(lbls[itype],running_lbls[irun],alignment_lbls[ialign]))

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
