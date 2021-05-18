#!/usr/bin/env python

import pyute as ut
import skimage.morphology as skm
import numpy as np
import sklearn
import scipy.ndimage.measurements as snm
import skimage.segmentation as sks
import matplotlib.pyplot as plt
import scipy.stats as sst
import size_contrast_analysis as sca

# ret_paramlist = ['rf_mapping_pval','rf_sq_error','rf_sigma','rf_distance_deg']

def compute_tuning(dsfile,running=True,center=True,fieldname='decon',keylist=None,pval_cutoff=np.inf,dcutoff=np.inf,nbefore=8,nafter=8,run_cutoff=10): 
    with ut.hdf5read(dsfile) as f:
        if keylist is None:
            keylist = [key for key in f.keys()]
        tuning = [None for ikey in range(len(keylist))]
        cell_criteria = [None for ikey in range(len(keylist))]
        uparam = [None for ikey in range(len(keylist))]
        for ikey in range(len(keylist)):
            session = f[keylist[ikey]]
            if 'size_contrast_opto_0' in session:
                sc0 = session['size_contrast_opto_0']
#                 print([key for key in session.keys()])
                data = sc0[fieldname][:]
                stim_id = sc0['stimulus_id'][:]
                paramnames = params = sc0['stimulus_parameters']
                uparam[ikey] = [sc0[paramname][:] for paramname in paramnames]
                trialrun = sc0['running_speed_cm_s'][:,nbefore:-nafter].mean(-1)>run_cutoff
                if not running:
                    trialrun = ~trialrun
                if 'rf_displacement_deg' in sc0:
                    pval = sc0['rf_mapping_pval'][:]
                    X = session['cell_center'][:]
                    y = sc0['rf_displacement_deg'][:].T
                    lkat = ut.k_and(pval<0.05,~np.isnan(X[:,0]),~np.isnan(y[:,0]))
                    linreg = sklearn.linear_model.LinearRegression().fit(X[lkat],y[lkat])
                    displacement = np.zeros_like(y)
                    displacement[~np.isnan(X[:,0])] = linreg.predict(X[~np.isnan(X[:,0])])
                if center:
                    cell_criteria[ikey] = (np.sqrt((displacement**2).sum(1))<dcutoff) & (pval < pval_cutoff)
                else:
                    cell_criteria[ikey] = (np.sqrt((displacement**2).sum(1))>dcutoff) & (pval < pval_cutoff)
                tuning[ikey] = ut.compute_tuning(data,stim_id,cell_criteria=cell_criteria[ikey],trial_criteria=trialrun) #cell_criteria
                print('%s: %.1f' % (keylist[ikey], trialrun.mean()))
            else:
                print('could not do '+keylist[ikey])
    return tuning,cell_criteria,uparam

def extract_fit_displacement(dsfile):
    displacement = {}
    with ut.hdf5read(dsfile) as f:
        keylist = [key for key in f.keys()]
        for ikey in range(len(keylist)):
            session = f[keylist[ikey]]
            if 'size_contrast_opto_0' in session:
                sc0 = session['size_contrast_opto_0']
                if 'rf_displacement_deg' in sc0:
                    pval = sc0['rf_mapping_pval'][:]
                    sigma = session['retinotopy_0']['rf_sigma'][:]
                    X = session['cell_center'][:]
                    y = sc0['rf_displacement_deg'][:].T
                    lkat = ut.k_and(pval<0.05,~np.isnan(X[:,0]),~np.isnan(y[:,0]),sigma>5)
                    linreg = sklearn.linear_model.LinearRegression().fit(X[lkat],y[lkat])
                    displacement[keylist[ikey]] = np.zeros_like(y)
                    displacement[keylist[ikey]][~np.isnan(X[:,0])] = linreg.predict(X[~np.isnan(X[:,0])])
    return displacement


def regression_slope(x,y):
    linreg = sklearn.linear_model.LinearRegression(fit_intercept=False).fit(x[:,np.newaxis],y[:,np.newaxis])
    return linreg.coef_[0,0],linreg.score(x[:,np.newaxis],y[:,np.newaxis])

def roiwise_regression_slope(sc):
    nroi = sc.shape[0]
    ndims = len(sc.shape)
    slicer0 = tuple([slice(None) for idim in range(ndims-2)]+[0])
    slicer1 = tuple([slice(None) for idim in range(ndims-2)]+[1])
    slope = np.zeros((nroi,))
    score = np.zeros((nroi,))
    for iroi in range(nroi):
        if np.sum(np.isnan(sc[iroi]))>0:
            slope[iroi],score[iroi] = np.nan,np.nan
        else:
            slope[iroi],score[iroi] = regression_slope(sc[iroi][slicer0].flatten(),sc[iroi][slicer1].flatten())
    return slope,score

def get_median_slope(slopescore):
    slope,score = slopescore
    return np.median(slope[score>0])

def show_light_on_light_off(sc,usize=None,vmin=None,vmax=None):
    if vmax is None:
        mx = np.nanmax(sc)
    else:
        mx = vmax
    if vmin is None:
        mn = np.nanmin(sc)
    else:
        mn = vmin
    for ilight in range(2):
        plt.subplot(1,2,ilight+1)
        if usize is None:
            plt.imshow(sc[:,:,ilight],vmin=mn,vmax=mx)
        else:
            sca.show_size_contrast(sc[:,:,ilight],vmin=mn,vmax=mx,flipud=True,usize=usize)
    plt.tight_layout()

def show_on_off(criterion):
    r = sc[lkat][criterion].mean(0)
    mx = r.max()
    mn = r.min()
    for ilight in range(2):
        plt.subplot(1,2,ilight+1)
        plt.imshow(r[:,:,ilight],vmin=mn,vmax=mx)
        
def show_on_off_diff(criterion):
    r = sc[lkat][criterion].mean(0)
    mx = r.max()
    mn = r.min()
    for ilight in range(2):
        plt.subplot(1,3,ilight+1)
        plt.imshow(r[:,:,ilight],vmin=mn,vmax=mx)
    plt.subplot(1,3,3)
    plt.imshow(r[:,:,1]-r[:,:,0])
    
def scatter_on_off(criterion):
    r = sc[lkat][criterion].mean(0)
    sca.scatter_size_contrast(r[:,:,0],r[:,:,1])
    
def redilate(im,iterations=1):
    to_return = im.copy()
    for ii in range(iterations):
        to_return = skm.binary_dilation(to_return)
    return to_return

def reerode(im,iterations=1):
    to_return = im.copy()
    for ii in range(iterations):
        to_return = skm.binary_erosion(to_return)
    return to_return

def compute_image_features(r,g,msk,depth,summary_stat=np.median):
    nroi = msk.shape[0]
    feature_dict = {}
    for featurename in ['rednucleus','redaround','redlvl','greenlvl','redgreencorr','redmembrane','redcv']:
        feature_dict[featurename] = np.zeros((nroi,))
    for iroi in range(nroi):
        annulus = redilate(msk[iroi],iterations=3) & ~msk[iroi]
        nucleus = reerode(msk[iroi],iterations=3)
        membrane = msk[iroi] & ~nucleus
        redvals = r[depth[iroi]][msk[iroi]]
        greenvals = g[depth[iroi]][msk[iroi]]
        feature_dict['redcv'][iroi] = np.std(redvals)/np.mean(redvals)
        feature_dict['rednucleus'][iroi] = summary_stat(r[depth[iroi]][nucleus]) #.mean()
        feature_dict['redmembrane'][iroi] = summary_stat(r[depth[iroi]][membrane]) #.mean()
        feature_dict['redaround'][iroi] = summary_stat(r[depth[iroi]][annulus]) #.mean()
        feature_dict['greenlvl'][iroi] = summary_stat(greenvals) #.mean()
        feature_dict['redlvl'][iroi] = summary_stat(redvals) #.mean()
        feature_dict['redgreencorr'][iroi] = np.corrcoef(redvals,greenvals)[0,1]
    return feature_dict

def show_roi_boundaries(red_green,msk,ctr=None,frame_width=25,show_outline=True):
    if len(red_green.shape)==2:
        rgb = np.zeros(red_green.shape+(3,))
        rgb[:,:,0] = red_green
        red_green = rgb
    if show_outline:
        img = sks.mark_boundaries(red_green,skm.binary_dilation(msk).astype('int'),color=(1,1,1))
    else:
        img = red_green.copy()
    if not ctr is None:
        c = ctr.astype('int')
    else:
        c = np.array(snm.center_of_mass(msk)).astype('int')
    small_img = np.zeros((2*frame_width,2*frame_width) + img.shape[2:])
    top_bd = np.maximum(0,c[0]-frame_width)
    bottom_bd = np.minimum(img.shape[0],c[0]+frame_width)
    left_bd = np.maximum(0,c[1]-frame_width)
    right_bd = np.minimum(img.shape[1],c[1]+frame_width)
    img_content = img[top_bd:bottom_bd,left_bd:right_bd]
    small_img[:img_content.shape[0],:img_content.shape[1]] = img_content
    plt.imshow(small_img)
    #plt.imshow(red_green)
    plt.axis('off')
    return small_img
    
def norm_to_mean_light_off(sc):
    ndim = len(sc.shape)
    slicer = [slice(None) for idim in range(ndim)]
    mean_light_off = sc.copy()
    while len(mean_light_off.shape)>2:
        mean_light_off = np.nanmean(mean_light_off,1)
    mean_light_off = mean_light_off[tuple([slice(None)]+[np.newaxis for idim in range(ndim-2)]+[slice(0,1)])]
    return sc/mean_light_off

def gen_rsdict_opto_light_off(rsdict_opto,optotype=None):
    rsdict_opto_light_off = {}
    rsdict_opto_light_off['retinotopy_0'] = rsdict_opto['retinotopy_0'].copy()
    rsdict_opto_light_off['size_contrast_0'] = [sc0[:,:,:,:,0].copy() for sc0 in rsdict_opto['size_contrast_opto_0']]
    rsdict_opto_light_off['figure_ground_0'] = [fg0[:,:,:,0].copy() for fg0 in rsdict_opto['figure_ground_opto_0']]

    #session_dict_opto_light_off = {}
    #session_dict_opto_light_off['retinotopy_0'] = session_dict_opto['retinotopy_0'].copy()
    #session_dict_opto_light_off['size_contrast_0'] = [sc0.copy() for sc0 in session_dict_opto['size_contrast_opto_0'][optotype]]
    #session_dict_opto_light_off['figure_ground_0'] = [fg0.copy() for fg0 in session_dict_opto['figure_ground_opto_0'][optotype]]
    
    if not optotype is None:
        for key in rsdict_opto_light_off:
            rsdict_opto_light_off[key] = rsdict_opto_light_off[key][optotype]
            #session_dict_opto_light_off[key] = session_dict_opto_light_off[key][optotype]
            
    return rsdict_opto_light_off #,session_dict_opto_light_off

def spont_effect(size_contrast_light,up=True):
    # (n,6,6,8,2)
    data_spont = compute_spont_light_on_light_off(size_contrast_light)
    pval = sst.ttest_ind(data_spont[:,:,0],data_spont[:,:,1],axis=1).pvalue
    if up:
        goes_up = np.nanmean(data_spont[:,:,1],1)>np.nanmean(data_spont[:,:,0],1)
    else:
        goes_up = np.nanmean(data_spont[:,:,1],1)<np.nanmean(data_spont[:,:,0],1)
    return goes_up,pval

def compute_spont_light_on_light_off(size_contrast_light):
    data = norm_to_mean_light_off(size_contrast_light)     
    if len(data.shape)==5:
        data_spont = data[:,:,0,:,:].reshape((data.shape[0],-1,2))
    elif len(data.shape)==4:
        data_spont = data[:,0,:,:].reshape((data.shape[0],-1,2))
    data_spont = data_spont[:,~np.isnan(data_spont.sum(2).sum(0))]
    return data_spont

def ret_aligned(ret,pcutoff=0.05,dcutoff=10,dcutoff_fn=None):
    if dcutoff_fn is None:
        return (ret[:,0]<pcutoff) & (ret[:,3]<dcutoff)
    else:
        return (ret[:,0]<pcutoff) & dcutoff_fn(ret[:,3])

def plot_size_contrast_ori_light(this_data,colorset=['k','b'],mag=1):
    data = this_data.copy()
    if len(data.shape)==4:
        data = data[:,:,:,:,np.newaxis]
    nsize,ncontrast,nori,nlight = data.shape[1:]
    plt.figure(figsize=(ncontrast*mag,nsize*mag))
    for isize in range(nsize):
        for icontrast in range(ncontrast):
            plt.subplot(nsize,ncontrast,(nsize-isize-1)*ncontrast+icontrast+1)
            for ilight in range(nlight):
                plt.plot(np.nanmean(data,0)[isize,icontrast,:,ilight],c=colorset[ilight])#,c=colorsets[optotype])
            plt.ylim((0,np.nanmean(data,0).max()))
            plt.plot((0,nori-0.5),(0,0),linestyle='dotted',c='k')
            plt.axis('off')

# 
def sort_by_pref_dir(rs,ori_axis=3):
    rs_sort = rs.copy()
    for ir in range(len(rs_sort)):
        rs_ori = rs_sort[ir].copy()
        dims_to_avg = np.flip(np.arange(len(rs_ori.shape)))
        dims_to_avg = dims_to_avg[dims_to_avg!=0]
        dims_to_avg = dims_to_avg[dims_to_avg!=ori_axis]
        for idim in dims_to_avg: #while len(rs_ori.shape)>2:
            rs_ori = np.nanmean(rs_ori,idim) #np.nanmean(rs_ori,1)
        nroi = rs_ori.shape[0]
        nori = rs_ori.shape[1]
        pref_dir = np.argmax(rs_ori,axis=1)
#         ii,jj = np.meshgrid(np.arange(rs_ori.shape[0]),np.arange(rs_ori.shape[1]),indexing='ij')
        slicer = [slice(None) for idim in range(len(rs_sort[ir].shape)-1)]
        for iroi in range(nroi):
            order = np.array(list(np.arange(pref_dir[iroi],nori))+list(np.arange(0,pref_dir[iroi])))
            slicer[ori_axis-1] = order
            rs_sort[ir][iroi] = rs_sort[ir][iroi][slicer]
    return rs_sort

def sort_both_by_pref_dir(rs,other,ori_axis=3):
    return sort_all_by_pref_dir((rs,ori_axis),(other,ori_axis))
    #rs_sort = rs.copy()
    #other_sort = other.copy()
    #for ir in range(len(rs_sort)):
    #    rs_ori = rs_sort[ir].copy()
    #    dims_to_avg = np.flip(np.arange(len(rs_ori.shape)))
    #    dims_to_avg = dims_to_avg[dims_to_avg!=0]
    #    dims_to_avg = dims_to_avg[dims_to_avg!=ori_axis]
    #    for idim in dims_to_avg: #while len(rs_ori.shape)>2:
    #        rs_ori = np.nanmean(rs_ori,idim) #np.nanmean(rs_ori,1)
    #    nroi = rs_ori.shape[0]
    #    nori = rs_ori.shape[1]
    #    pref_dir = np.argmax(rs_ori,axis=1)
#   #      ii,jj = np.meshgrid(np.arange(rs_ori.shape[0]),np.arange(rs_ori.shape[1]),indexing='ij')
    #    slicer = [slice(None) for idim in range(len(rs_sort[ir].shape)-1)]
    #    for iroi in range(nroi):
    #        order = np.array(list(np.arange(pref_dir[iroi],nori))+list(np.arange(0,pref_dir[iroi])))
    #        slicer[ori_axis-1] = order
    #        rs_sort[ir][iroi] = rs_sort[ir][iroi][slicer]
    #        other_sort[ir][iroi] = other_sort[ir][iroi][slicer]
    #return rs_sort,other_sort

def sort_all_by_pref_dir(rs,*others):
    rs_sort = rs[0].copy()
    ori_axis = rs[1]
    others_sort = [other[0].copy() for other in others]
    other_ori_axes = [other[1] for other in others]
    for ir in range(len(rs_sort)):
        rs_ori = rs_sort[ir].copy()
        dims_to_avg = np.flip(np.arange(len(rs_ori.shape)))
        dims_to_avg = dims_to_avg[dims_to_avg!=0]
        dims_to_avg = dims_to_avg[dims_to_avg!=ori_axis]
        for idim in dims_to_avg: #while len(rs_ori.shape)>2:
            rs_ori = np.nanmean(rs_ori,idim) #np.nanmean(rs_ori,1)
        nroi = rs_ori.shape[0]
        nori = rs_ori.shape[1]
        pref_dir = np.argmax(rs_ori,axis=1)
        slicer = [slice(None) for idim in range(len(rs_sort[ir].shape)-1)]
        other_slicers = [[slice(None) for idim in range(len(other_sort[ir].shape)-1)] for other_sort in others_sort]
        for iroi in range(nroi):
            order = np.array(list(np.arange(pref_dir[iroi],nori))+list(np.arange(0,pref_dir[iroi])))
            slicer[ori_axis-1] = order
            rs_sort[ir][iroi] = rs_sort[ir][iroi][slicer]
            for iother in range(len(others)):
                other_slicers[iother][other_ori_axes[iother]-1] = order
                others_sort[iother][ir][iroi] = others_sort[iother][ir][iroi][other_slicers[iother]]
    return [rs_sort]+others_sort

def fix_fg_dir(rs):
    rs_sort = rs.copy()
    order = np.array(list(np.arange(2,8))+list(np.arange(0,2)))
    for ir in range(len(rs_sort)):
        nroi = rs_sort[ir].shape[0]
        for iroi in range(nroi):
            for idim in (2,3):
                rs_sort[ir][iroi,idim,:] = rs_sort[ir][iroi,idim,order]
    return rs_sort

def extract_outlines(dii,frame_width=25,rois=None):
    iroi = 0
    nroi = dii['image_data']['depth'].shape[0]
    if rois is None:
        rois = np.arange(nroi)
    img = dii['image_data']['r2'][dii['image_data']['depth'][iroi]]
    outlines0 = show_roi_boundaries(np.minimum(1.5*img/img.max(),1),dii['image_data']['msk'][iroi],frame_width=frame_width)
    outlines = np.zeros((len(rois),)+outlines0.shape)
    for iiroi,iroi in enumerate(rois):
        outlines[iiroi] = show_roi_boundaries(np.minimum(5*img/img.max(),1),dii['image_data']['msk'][iroi],frame_width=frame_width)
    return outlines

def safe_predict_proba_(logreg,features):
    output = np.zeros((features.shape[0],))
    lkat = ~np.isnan(features.sum(1)) & np.all(np.abs(features)<np.inf,axis=1)
    output[~lkat] = np.nan
    output[lkat] = logreg.predict_proba(features[lkat])[:,1]
    return output

def compute_slope(x,y,axis=0):
    return (np.nanmean(x*y,axis=axis)/np.nanmean(x**2,axis=axis))[np.newaxis]

def compute_slope_cols(xy,axis=0):
    x,y = xy[:,0],xy[:,1]
    return compute_slope(x,y,axis=axis)

def compute_slope_(animal_data,axis=0):
    x = np.nanmean(animal_data[:,:,:,0],0)
    x = np.reshape(x,(x.shape[0]*x.shape[1],-1))
    y = np.nanmean(animal_data[:,:,:,1],0)
    y = np.reshape(y,(y.shape[0]*y.shape[1],-1))
    return compute_slope(x,y,axis=axis)

def compute_slope_w_intercept(x,y,axis=0):
    xy_ = np.nanmean(x*y,axis=axis)
    x_ = np.nanmean(x,axis=axis)
    y_ = np.nanmean(y,axis=axis)
    x2_ = np.nanmean(x**2,axis=axis)
    astar = ((xy_-x_*y_)/(x2_-x_**2))
    return astar[np.newaxis]

def compute_slope_w_intercept_cols(xy,axis=0):
    x,y = xy[:,0],xy[:,1]
    return compute_slope_w_intercept(x,y,axis=axis)

def compute_slope_w_intercept_(animal_data,axis=0):
    x = np.nanmean(animal_data[:,:,:,0],0)
    x = np.reshape(x,(x.shape[0]*x.shape[1],-1))
    y = np.nanmean(animal_data[:,:,:,1],0)
    y = np.reshape(y,(y.shape[0]*y.shape[1],-1))
    return compute_slope_w_intercept(x,y,axis=axis)

def compute_intercept(x,y,axis=0):
    xy_ = np.nanmean(x*y,axis=axis)
    x_ = np.nanmean(x,axis=axis)
    y_ = np.nanmean(y,axis=axis)
    x2_ = np.nanmean(x**2,axis=axis)
    astar = ((xy_-x_*y_)/(x2_-x_**2))
    bstar = y_ - astar*x_
    return bstar[np.newaxis]

def compute_intercept_cols(xy,axis=0):
    x,y = xy[:,0],xy[:,1]
    return compute_intercept(x,y,axis=axis)

def compute_intercept_(animal_data,axis=0):
    x = np.nanmean(animal_data[:,:,:,0],0)
    x = np.reshape(x,(x.shape[0]*x.shape[1],-1))
    y = np.nanmean(animal_data[:,:,:,1],0)
    y = np.reshape(y,(y.shape[0]*y.shape[1],-1))
    return compute_intercept(x,y,axis=axis)

def scatter_size_contrast_errorbar(animal_data,pct=(16,84),mn_plot=None,mx_plot=None,opto_color='b',equality_line=True,square=True,xlabel=None,ylabel=None,alpha=1):
    if xlabel is None:
        xlabel = 'PC event rate, light off'
    if ylabel is None:
        ylabel = 'PC event rate, light on'
    lb,ub,mn = ut.bootstrap(animal_data,pct=pct+(50,),fn=np.nanmean,axis=0)
    stats = ut.bootstat(animal_data,fns=[compute_slope_w_intercept_,compute_intercept_])
    npt = mn[:,:,0].size
    xerr = np.zeros((2,npt))
    xerr[0] = mn[:,:,0].flatten()-lb[:,:,0].flatten()
    xerr[1] = ub[:,:,0].flatten()-mn[:,:,0].flatten()
    yerr = np.zeros((2,npt))
    yerr[0] = mn[:,:,1].flatten()-lb[:,:,1].flatten()
    yerr[1] = ub[:,:,1].flatten()-mn[:,:,1].flatten()
    if mx_plot is None:
        mx_plot = 1.1*np.nanmax(mn)
    if mn_plot is None:
        mn_plot = np.nanmin(mn)
    xx = np.linspace(mn_plot,mx_plot,100)[:,np.newaxis]
    YY = xx*stats[0]+stats[1]
    lb_YY = np.nanpercentile(YY,16,axis=1)
    ub_YY = np.nanpercentile(YY,84,axis=1)
    mn_YY = np.nanpercentile(YY,50,axis=1)
    plt.fill_between(xx[:,0],lb_YY,ub_YY,alpha=0.5*alpha,facecolor=opto_color)
    plt.plot(xx[:,0],mn_YY,c=opto_color,alpha=alpha)
    plt.errorbar(mn[:,:,0].flatten(),mn[:,:,1].flatten(),xerr=xerr,yerr=yerr,fmt='none',zorder=1,c='k',alpha=0.5*alpha)
    sca.scatter_size_contrast(mn[:,:,0],mn[:,:,1],equality_line=equality_line,square=square,alpha=alpha)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

def plot_bootstrapped_regression_lines(animal_data,c='k',alpha=1,nreps=1000,pct=(16,84),incl_intercept=True,equal_cond=True,flipxy=False):
    xx = np.linspace(animal_data[:,0].min(),animal_data[:,0].max(),101)
    if flipxy:
        animal_data = animal_data[:,::-1]
    if incl_intercept:
        if equal_cond:
            stats = ut.bootstat_equal_cond(animal_data,fns=[compute_slope_w_intercept_cols,compute_intercept_cols],nreps=nreps)
        else:
            stats = ut.bootstat(animal_data,fns=[compute_slope_w_intercept_cols,compute_intercept_cols],nreps=nreps)
        YY = xx[:,np.newaxis]*stats[0]+stats[1]
    else:
        if equal_cond:
            stats = ut.bootstat_equal_cond(animal_data,fns=[compute_slope_cols],nreps=nreps)
        else:
            stats = ut.bootstat(animal_data,fns=[compute_slope_cols],nreps=nreps)
        YY = xx[:,np.newaxis]*1/stats[0]
    lb_YY = np.nanpercentile(YY,pct[0],axis=1)
    ub_YY = np.nanpercentile(YY,pct[1],axis=1)
    mn_YY = np.nanpercentile(YY,50,axis=1)
    plt.fill_between(xx,lb_YY,ub_YY,alpha=0.5*alpha,facecolor=c)
    plt.plot(xx,mn_YY,c=c,alpha=alpha)
    return stats
    
def scatter_size_contrast_x_dx_errorbar(animal_data,pct=(16,84),opto_color='b',xlabel=None,ylabel=None,mn_plot=None,mx_plot=None,alpha=1):
    # axis 0: roi, axis 1: size, axis 2: contrast, axis 3: light
    if xlabel is None:
        xlabel = 'PC event rate/mean, \n light off'
    if ylabel is None:
        ylabel = 'PC event rate/mean, \n light on $-$ light off'
    if mn_plot is None:
        mn_plot = np.nanmin(animal_data[:,:,:,0])
    if mx_plot is None:
        mx_plot = np.nanmax(animal_data[:,:,:,0])
    diff_data = animal_data.copy()
    diff_data[:,:,:,1] = diff_data[:,:,:,1] - diff_data[:,:,:,0]
    scatter_size_contrast_errorbar(diff_data,pct=(16,84),mn_plot=mn_plot,mx_plot=mx_plot,opto_color=opto_color,equality_line=False,square=False,xlabel=xlabel,ylabel=ylabel,alpha=alpha)
    plt.axhline(0,c='k',linestyle='dashed')
