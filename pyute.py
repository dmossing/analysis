#!/usr/bin/env python

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import sklearn.cluster as skc
import os
import scipy.misc as smi
import scipy.io as sio
import h5py
import scipy.ndimage.filters as sfi
from oasis.functions import deconvolve
import scipy.optimize as sop
import scipy.ndimage.measurements as snm
import re
import pickle as pkl

def norm01(arr,dim=1):
    # normalize each row of arr to [0,1]
    dim = np.minimum(dim,len(arr.shape)-1)
    try:
        mnm = arr.min(dim)[:,np.newaxis]
        mxm = arr.max(dim)[:,np.newaxis]
    except:
        mnm = arr.min(dim)
        mxm = arr.max(dim)
    return (arr-mnm)/(mxm-mnm)

def zscore(arr):
    # zscore each row of arr
    if len(arr.shape)==1:
        mn = np.nanmean(arr)
        st = np.nanstd(arr)
#        mn = arr.mean()
#        st = arr.std()
    elif len(arr.shape)==2:
        mn = np.nanmean(arr,axis=1)[:,np.newaxis]
        st = np.nanstd(arr,axis=1)[:,np.newaxis]
        #mn = arr.mean(1)[:,np.newaxis]
        #st = arr.std(1)[:,np.newaxis]
    elif len(arr.shape)==3:
        aux = arr.reshape((arr.shape[0],-1))
        mn = np.nanmean(aux,axis=1)[:,np.newaxis,np.newaxis]
        st = np.nanstd(aux,axis=1)[:,np.newaxis,np.newaxis]
        #mn = aux.mean(1)[:,np.newaxis,np.newaxis]
        #st = aux.std(1)[:,np.newaxis,np.newaxis]
    return (arr-mn)/st

def threeand(a,b,c):
    return np.logical_and(np.logical_and(a,b),c)

def print_multipage(args,fn,filename):
    if type(args) is np.ndarray:
        args = [args]
    plt.close()
    with PdfPages(filename+'.pdf') as pdf:
        for i in range(args[0].shape[0]):
            #fn(arr[i])
            fn([el[i] for el in args])
            pdf.savefig()
            plt.close()

def sortbymultiple(variables):
    N = len(variables)
    fieldnames = ['a'*(i+1) for i in range(N)]
    xtype = np.dtype([(name,var.dtype) for name,var in zip(fieldnames,variables)])
    x = np.empty((variables[0].shape[0],),dtype=xtype)
    for i in range(N):
        x[fieldnames[i]] = variables[i]
    return np.argsort(x,order=tuple(fieldnames))

def shapeup(arr,variables):
    x = sortbymultiple(variables)
    star = tuple([len(np.unique(var)) for var in variables])
    return arr[:,x].reshape((arr.shape[0],)+star+(-1,)+(arr.shape[-1],))

def overlay_mg(a,b,normalize=True):
    # overlay 2 grayscale images, with image a as magenta and image b as green
    if normalize:
        imr = norm01(a,dim=None)
        img = norm01(b,dim=None)
        imb = norm01(a,dim=None)
    else:
        aux = np.concatenate((a[:,:,np.newaxis],b[:,:,np.newaxis]),axis=2)
        aux = norm01(aux,dim=None)
        a1 = aux[:,:,0]
        b1 = aux[:,:,1] 
        imr = a1
        img = b1
        imb = a1
    return np.dstack((imr,img,imb))

def bootstrap(arr,fn,axis=0,nreps=1000,pct=(2.5,97.5)):
    # given arr 1D of size N, resample nreps sets of N of its elements with replacement. Compute fn on each of the samples
    # and report percentiles pct
    N = arr.shape[axis]
    c = np.random.choice(np.arange(N),size=(N,nreps))
    L = len(arr.shape)
#    if len(arr.shape)==1:
#        resamp = arr[c]
#    elif len(arr.shape)==2:
#        if axis==0:
#            resamp = arr[c]
#        elif axis==1:
#            resamp = arr[:,c]
#    elif len(arr.shape)==3:
#        if axis==0:
#            resamp = arr[c]
#        elif axis==1:
#            resamp = arr[:,c]
#        elif axis==2:
#            resamp = arr[:,:,c]
    resamp=np.rollaxis(arr,axis,0)
    resamp=resamp[c]
    resamp=np.rollaxis(resamp,0,axis+2) # plus 1 due to rollaxis syntax. +1 due to extra resampled axis
    resamp=np.rollaxis(resamp,0,L+1)
    stat = fn(resamp,axis=axis)
    #lb = np.percentile(stat,pct[0],axis=axis) # after computing fn, dimension axis+1 shifted down to axis
    lb = np.percentile(stat,pct[0],axis=-1) # resampled axis rolled to last position
    #ub = np.percentile(stat,pct[1],axis=axis)
    ub = np.percentile(stat,pct[1],axis=-1) # resampled axis rolled to last position
    return lb,ub

def subsample(arr,fn,axis=0,nsubsample=None,nreps=1000,pct=(2.5,97.5)):
    # given arr 1D of size N, resample nreps sets of N of its elements with replacement. Compute fn on each of the samples
    # and report percentiles pct
    N = arr.shape[axis]
    assert(nsubsample<=N)
    c = np.zeros((nsubsample,nreps),dtype=int)
    for i in range(nreps):
        c[:,i] = np.random.choice(np.arange(N),size=(nsubsample,),replace=False)

    #c = np.random.choice(np.arange(N),size=(nsubsample,nreps))
    L = len(arr.shape)
    resamp=np.rollaxis(arr,axis,0)
    resamp=resamp[c]
    resamp=np.rollaxis(resamp,0,axis+2) # plus 1 due to rollaxis syntax. +1 due to extra resampled axis
    resamp=np.rollaxis(resamp,0,L+1)
    resamp = np.nanmean(resamp,axis=axis)
    stat = fn(resamp,axis=-1)
    #lb = np.percentile(stat,pct[0],axis=axis) # after computing fn, dimension axis+1 shifted down to axis
    lb = np.percentile(stat,pct[0],axis=-1) # resampled axis rolled to last position
    #ub = np.percentile(stat,pct[1],axis=axis)
    ub = np.percentile(stat,pct[1],axis=-1) # resampled axis rolled to last position
    return lb,ub

def reindex(C,inds):
    return C[inds,:][:,inds]

def showclustered(C,n_clusters=3,cmap=plt.cm.viridis):
    km = skc.KMeans(n_clusters=n_clusters).fit(C)
    plt.imshow(reindex(C,np.argsort(km.labels_)),interpolation='nearest',cmap=cmap)
    return km.labels_

def ddigit(n,d):
    q = str(n)
    z = '0'*(d-len(q))
    return z + str(n)

def genmovie(mchunk,movpath='temp_movie/',vmax=None,K=4):
    T = mchunk.shape[0]
    if vmax is None:
        try:
            vmax = np.iinfo(mchunk.dtype).max
        except:
            vmax = mchunk.max()
#    if movpath is None:
#        movpath = 'temp_movie/'
    if not os.path.exists(movpath):
        os.mkdir(movpath)
    for t in range(T):
        print(t)
        smi.toimage(mchunk[t],cmin=0,cmax=vmax).save(movpath+ddigit(t,K)+'.tif')
    print('this process finished')

def saveaspdf(filename):
    with PdfPages(filename+'.pdf') as pdf:
        pdf.savefig()

def heatmap(x,y,bins=20):
    h,xedges,yedges = np.histogram2d(x,y,bins=bins)
    plt.imshow(np.rot90(np.log(h),1),interpolation='nearest',cmap=plt.cm.viridis)
    plt.axis('off')
    return xedges,yedges

def plotstacked(arr,offset=1):
    addto = offset*np.arange(arr.shape[1])[np.newaxis,:]
    plt.plot(arr+addto)

def trialize(arr,frm,nbefore=15,nafter=30):
    if len(frm.shape)==1:
        frm = frm.reshape((-1,2))
    triallen = np.diff(frm,axis=1)
    triallen = np.round(triallen.mean()).astype('int')
    trialwise = np.zeros((arr.shape[0],frm.shape[0],triallen+nbefore+nafter))
    for i in range(trialwise.shape[1]):
        trialwise[:,i,:] = arr[:,frm[i,0]-nbefore:frm[i,0]+triallen+nafter]
    return trialwise

def resample(signal1,trig1,trig2):
        assert(trig1.sum()==trig2.sum())
        frametrig1 = np.where(trig1)[0]
        frametrig2 = np.where(trig2)[0]
        signal2 = np.zeros(trig2.shape,dtype=signal1.dtype)
        for i,tr in enumerate(frametrig1[:-1]):
            ptno1 = frametrig1[i+1]-frametrig1[i]
            ptno2 = frametrig2[i+1]-frametrig2[i]
            signal2[frametrig2[i]:frametrig2[i+1]] = np.interp(np.linspace(0,1,ptno2),np.linspace(0,1,ptno1),signal1[frametrig1[i]:frametrig1[i+1]])
        return signal2[frametrig2[0]:frametrig2[-1]]

#def gen_trialwise(datafiles,nbefore=0,nafter=0,blcutoff=5,blspan=3000,ds=10,rg=None):
#    def tack_on(to_add,trialwise,ctrialwise,strialwise,dfof):
#        to_add[np.isnan(to_add)] = 0
#        baseline = sfi.percentile_filter(to_add[:,::ds],blcutoff,(1,int(blspan/ds)))
#        baseline = np.repeat(baseline,ds,axis=1)
#        if baseline.shape[1]>to_add.shape[1]:
#            baseline = baseline[:,:to_add.shape[1]]
#        c = np.zeros_like(to_add)
#        s = np.zeros_like(to_add)
#        this_dfof = np.zeros_like(to_add)
#        for i in range(c.shape[0]):
#            this_dfof[i] = (to_add[i]-baseline[i,:])/baseline[i,:]
#            c[i],s[i],_,_,_  = deconvolve(this_dfof[i],penalty=1)
#            to_add[np.isnan(to_add)] = 0
#            to_add = trialize(to_add,frm,nbefore,nafter)
#        c = trialize(c,frm,nbefore,nafter)
#        s = trialize(s,frm,nbefore,nafter)
#        try:
#            trialwise = np.concatenate((trialwise,to_add),axis=0)
#            ctrialwise = np.concatenate((ctrialwise,c),axis=0)
#            strialwise = np.concatenate((strialwise,s),axis=0)
#            dfof = np.concatenate((dfof,this_dfof),axis=0)
#        except:
#            trialwise = to_add.copy()
#            ctrialwise = c.copy()
#            strialwise = s.copy()
#            dfof = this_dfof.copy()
#        return trialwise,ctrialwise,strialwise,dfof
#        
#    trialwise = np.array(())
#    ctrialwise = np.array(())
#    strialwise = np.array(())
#    dfof = np.array(())
#    try:
#        for datafile in datafiles:
#            if not rg is None:
#                frm = sio.loadmat(datafile.replace('.rois','.mat'),squeeze_me=True)['info']['frame'][()][rg[0]:rg[1]]
#            else:
#                frm = sio.loadmat(datafile.replace('.rois','.mat'),squeeze_me=True)['info']['frame'][()]
#            to_add = sio.loadmat(datafile,squeeze_me=True)['corrected']
#            trialwise,ctrialwise,strialwise,dfof = tack_on(to_add,trialwise,ctrialwise,strialwise,dfof)   
#    except:
#        for datafile in datafiles:
#            if not rg is None:
#                frm = sio.loadmat(datafile.replace('.rois','.mat'),squeeze_me=True)['info']['frame'][()][rg[0]:rg[1]]
#            else:
#                frm = sio.loadmat(datafile.replace('.rois','.mat'),squeeze_me=True)['info']['frame'][()]
#            with h5py.File(datafile,mode='r') as f:
#                to_add = f['corrected'][:].T[:,1:]
#                print(to_add.shape)
#                trialwise,ctrialwise,strialwise,dfof = tack_on(to_add,trialwise,ctrialwise,strialwise,dfof)   
#   #             baseline = sfi.percentile_filter(to_add[:,::ds],blcutoff,(1,int(blspan/ds)))
#   #             baseline = np.repeat(baseline,ds,axis=1)
#   #             if baseline.shape[1]>to_add.shape[1]:
#   #                 baseline = baseline[:,:to_add.shape[1]]
#   #             c = np.zeros_like(to_add)
#   #             s = np.zeros_like(to_add)
#   #             to_add[np.isnan(to_add)] = 0
#   #             for i in range(c.shape[0]):
#   #                 dfof = (to_add[i]-baseline[i,:])/baseline[i,:]
#   #                 try:
#   #                     c[i],s[i],_,_,_  = deconvolve(dfof,penalty=1)
#   #                 except:
#   #                     print("in "+datafile+" couldn't do "+str(i))
#   #             to_add = trialize(to_add,frm,nbefore,nafter)
#   #             c = trialize(c,frm,nbefore,nafter)
#   #             s = trialize(s,frm,nbefore,nafter)
#   #             try:
#   #                 trialwise = np.concatenate((trialwise,to_add),axis=0)
#   #                 ctrialwise = np.concatenate((ctrialwise,c),axis=0)
#   #                 strialwise = np.concatenate((strialwise,s),axis=0)
#   #             except:
#   #                 trialwise = to_add.copy()
#   #                 ctrialwise = c.copy()
#   #                 strialwise = s.copy()
#    return trialwise,ctrialwise,strialwise,dfof

def gen_trialwise(datafiles,nbefore=4,nafter=8,blcutoff=1,blspan=3000,ds=10,rg=None):
    
    def tack_on(to_add,existing):
        try:
            existing = np.concatenate((existing,to_add),axis=0)
        except:
            existing = to_add.copy()
        return existing
    
    def process(to_add):
        to_add[np.isnan(to_add)] = np.nanmin(to_add) #0
        if to_add.max():
            baseline = sfi.percentile_filter(to_add[:,::ds],blcutoff,(1,int(blspan/ds)))
            topline = sfi.percentile_filter(to_add[:,::ds],99,(1,int(blspan/ds))) # dan added 18/10/30
            baseline = np.maximum(baseline,topline/10) # dan added 18/10/30
            baseline = np.repeat(baseline,ds,axis=1)
            if baseline.shape[1]>to_add.shape[1]:
                baseline = baseline[:,:to_add.shape[1]]
            c = np.zeros_like(to_add)
            s = np.zeros_like(to_add)
            this_dfof = np.zeros_like(to_add)
            for i in range(c.shape[0]):
                this_dfof[i] = (to_add[i]-baseline[i,:])/baseline[i,:]
                this_dfof[i][np.isnan(this_dfof[i])] = 0
                c[i],s[i],_,_,_  = deconvolve(this_dfof[i].astype(np.float64),penalty=1)
        else:
            this_dfof = np.zeros_like(to_add)
            c = np.zeros_like(to_add)
            s = np.zeros_like(to_add)
        to_add = trialize(to_add,frm,nbefore,nafter)
        c = trialize(c,frm,nbefore,nafter)
        s = trialize(s,frm,nbefore,nafter)
        return to_add,c,s,this_dfof
        
    trialwise = np.array(())
    ctrialwise = np.array(())
    strialwise = np.array(())
    dfof = np.array(())
    for datafile in datafiles:
        if not rg is None:
            frm = sio.loadmat(datafile.replace('.rois','.mat'),squeeze_me=True)['info']['frame'][()]-1
            frm = frm[rg[0]:frm.size+rg[1]]
        else:
            frm = sio.loadmat(datafile.replace('.rois','.mat'),squeeze_me=True)['info']['frame'][()]-1
        try:
            to_add = sio.loadmat(datafile,squeeze_me=True)['corrected']
        except:
            with h5py.File(datafile,mode='r') as f:
                to_add = f['corrected'][:].T
        to_add,c,s,this_dfof = process(to_add)
        trialwise = tack_on(to_add,trialwise)
        ctrialwise = tack_on(c,ctrialwise)
        strialwise = tack_on(s,strialwise)
        dfof = tack_on(this_dfof,dfof)

    return trialwise,ctrialwise,strialwise,dfof

def twoD_Gaussian(xy, xo, yo, amplitude, sigma_x, sigma_y, theta, offset):
    x = xy[0]
    y = xy[1]
    xo = float(xo)
    yo = float(yo)    
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) 
                            + c*((y-yo)**2)))
    return g.ravel()

def assign_tuple(tup,ind,tgt):
    lis = [list(x) for x in tup]
    ctr = 0
    if len(ind)==1:
        lis[ind[0]] =  tgt
    if len(ind)==2:
        lis[ind[0]][ind[1]] = tgt
    return tuple([tuple(x) for x in bounds])

# TEMPORARILY constraining gaussian to have positive amplitude for PC data (19/1/30)
def fit_2d_gaussian(locs,ret,verbose=False,bounds=((None,None,0,0,0,0,0),(None,None,np.inf,None,None,2*np.pi,np.inf))):
    
    xx,yy = np.meshgrid(locs[0],locs[1])
    x = xx.flatten()
    y = yy.flatten()
    bounds = [list(x) for x in bounds]
    if not bounds[0][0]:
        bounds[0][0] = x.min()
    if not bounds[1][0]:
        bounds[1][0] = x.max()
    if not bounds[0][1]:
        bounds[0][1] = y.min()
    if not bounds[1][1]:
        bounds[1][1] = y.max()
    if not bounds[1][3]:
        bounds[1][3] = x.max()
    if not bounds[1][4]:
        bounds[1][4] = y.max()
    bounds = tuple([tuple(x) for x in bounds])
    msk_surr = np.zeros(ret.shape[1:3],dtype='bool')
    msk_surr[0,:] = 1
    msk_surr[:,0] = 1
    msk_surr[-1,:] = 1
    msk_surr[:,-1] = 1 
    i = 0
    can_be_negative = bounds[0][2]<0
    #if ret[i][msk_surr].mean()<ret[i].mean() or not can_be_negative:
    #    extremum = np.argmax(ret[i],axis=None)
    #    initial_guess = (x[extremum],y[extremum],ret[i].max()-ret[i].min(),10,10,0,ret[i].min())
    #else:
    #    extremum = np.argmin(ret[i],axis=None)
    #    initial_guess = (x[extremum],y[extremum],ret[i].min()-ret[i].max(),10,10,0,ret[i].max())
    params = np.zeros((2,ret.shape[0],7))
    sqerror = np.zeros((2,ret.shape[0]))
    for i in range(ret.shape[0]):
        for k in range(2):
            try:
                data = ret[i].flatten()
                #if ret[i][msk_surr].mean()<ret[i].mean() or not can_be_negative:
                if k==0:
                    extremum = np.argmax(ret[i],axis=None)
                    initial_guess = (x[extremum],y[extremum],ret[i].max()-ret[i].min(),10,10,0,np.maximum(0,ret[i].min()))                  
                    assign_tuple(bounds,(0,2),0)
                    assign_tuple(bounds,(1,2),np.inf)
                    popt, pcov = sop.curve_fit(twoD_Gaussian, (x,y), data, p0=initial_guess, bounds=bounds)
                else:
                    extremum = np.argmin(ret[i],axis=None)
                    initial_guess = (x[extremum],y[extremum],ret[i].min()-ret[i].max(),10,10,0,ret[i].max())
                    assign_tuple(bounds,(0,2),-np.inf)
                    assign_tuple(bounds,(1,2),0)
                    popt, pcov = sop.curve_fit(twoD_Gaussian, (x,y), data, p0=initial_guess, bounds=bounds)
                # xo, yo, amplitude, sigma_x, sigma_y, theta, offset
                modeled = twoD_Gaussian((x,y),*popt)
                sqerror[k,i] = ((modeled-data)**2).mean()/np.var(data)
                params[k,i] = popt
            except:
                if verbose:
                    print("couldn't do "+str(i))
    best_option = np.argmin(sqerror,axis=0)
    paramdict = {}
    paramdict['sqerror'] = np.zeros(ret.shape[0:1])
    paramdict['xo'] = np.zeros(ret.shape[0:1])
    paramdict['yo'] = np.zeros(ret.shape[0:1])
    paramdict['amplitude'] = np.zeros(ret.shape[0:1])
    paramdict['sigma_x'] = np.zeros(ret.shape[0:1])
    paramdict['sigma_y'] = np.zeros(ret.shape[0:1])
    paramdict['theta'] = np.zeros(ret.shape[0:1])
    paramdict['offset'] = np.zeros(ret.shape[0:1])    
    for i in range(sqerror.shape[1]):
        bo = best_option[i]
        paramdict['sqerror'][i] = sqerror[bo,i]
        paramdict['xo'][i] = params[bo,i,0]
        paramdict['yo'][i] = params[bo,i,1]
        paramdict['amplitude'][i] = params[bo,i,2]
        paramdict['sigma_x'][i] = params[bo,i,3]
        paramdict['sigma_y'][i] = params[bo,i,4]
        paramdict['theta'][i] = params[bo,i,5]
        paramdict['offset'][i] = params[bo,i,6]
    return paramdict

def model_2d_gaussian(locs,paramdict):
    N = paramdict['xo'].shape[0]
    params = np.zeros((N,7))
    params[:,0] = paramdict['xo']
    params[:,1] = paramdict['yo']
    params[:,2] = paramdict['amplitude']
    params[:,3] = paramdict['sigma_x']
    params[:,4] = paramdict['sigma_y']
    params[:,5] = paramdict['theta']
    params[:,6] = paramdict['offset']

    xx,yy = np.meshgrid(locs[0],locs[1])
    x = xx.flatten()
    y = yy.flatten()

    modeled = np.zeros((N,)+xx.shape)
    for i in range(N):
        modeled[i] = twoD_Gaussian((x,y),*params[i]).reshape(xx.shape)
    return modeled
    

def add_to_array(starting,to_add):
    if starting.size:
        return np.concatenate((starting,to_add),axis=0)
    else:
        return to_add

def imshow_in_rows(arr,rowlen=10,scale=0.5):
    nrows = np.ceil(arr.shape[0]/rowlen)
    plt.figure(figsize=(scale*rowlen,scale*nrows))
    for k in range(arr.shape[0]):
        plt.subplot(nrows,rowlen,k+1)
        plt.imshow(arr[k])
        plt.axis('off')

def plot_in_rows(arr,rowlen=10,scale=0.5):
    nrows = np.ceil(arr.shape[0]/rowlen)
    plt.figure(figsize=(scale*rowlen,scale*nrows))
    for k in range(arr.shape[0]):
        plt.subplot(nrows,rowlen,k+1)
        plt.plot(arr[k])
        plt.axis('off')

def imshow_in_pairs(arr1,arr2,rowlen=10,scale=0.5):
    nrows = np.ceil(arr1.shape[0]/rowlen)
    rowlen = rowlen*2
    plt.figure(figsize=(scale*rowlen,scale*nrows))
    for k in range(arr1.shape[0]):
        mn = np.minimum(arr1[k].min(),arr2[k].min())
        mx = np.maximum(arr1[k].max(),arr2[k].max())
        plt.subplot(nrows,rowlen,2*k+1)
        plt.imshow(arr1[k],vmin=mn,vmax=mx)
        plt.axis('off')
        plt.subplot(nrows,rowlen,2*k+2)
        plt.imshow(arr2[k],vmin=mn,vmax=mx)
        plt.axis('off')

def dist_from_center(ret,gridsize=5,center=(0,0)):
    com = np.zeros((ret.shape[0],2))
    for i in range(ret.shape[0]):
        com[i] = snm.center_of_mass(norm01(ret[i],dim=None))-np.array((ret.shape[1]-1)/2) # center def. as 0,0
    d = gridsize*np.sqrt(np.sum((com-np.array(center))**2,axis=1))
    return d

def dict_select(dict_of_arrs,dict_of_inds):
    # arrays are stored in a dict, and sets of indices to them are stored in a dict. Create a stitched together array (along axis 0) containing the concatenated arr[ind]
    key_list = list(dict_of_inds.keys())
    total_inds = 0
    for key in key_list:
        total_inds = total_inds + dict_of_inds[key].sum()
    template = dict_of_arrs[key_list[0]]
    summary_arr = np.zeros((total_inds,)+template.shape[1:],dtype=template.dtype)
    total_inds = 0
    for key in key_list:
        inds = dict_of_inds[key]
        old_total_inds = total_inds
        total_inds = total_inds+inds.sum()
        summary_arr[old_total_inds:total_inds] = dict_of_arrs[key][inds]
    return summary_arr

def get_dict_of_booleans(dict_of_vals,fn):
    # fn returns a logical. Go through a dict of arrs, and select which entries satisfy fn(arr)
    dict_of_booleans = {}
    for key in dict_of_vals.keys():
        dict_of_booleans[key] = fn(dict_of_vals[key])
    return dict_of_booleans

def fn_select(dict_of_arrs,dict_of_vals,fn):
    # fn returns a logical. Select the entries in arrs in dict A, where the entries of arrs in dict B satisfy fn(arrB)
    return dict_select(dict_of_arrs,get_dict_of_booleans(dict_of_vals,fn))

def precise_trialize(traces,frame,line,roilines,nlines=512,nplanes=4,nbefore=4,nafter=8,continuous=False):
    
    def convert_frame_line(frame,line,nlines,nplanes,continuous=False,matlab_format=True):
        
        def repeat_internal_values(x):
            return np.concatenate((x,np.repeat(x[1:-1],2),x[-1]),axis=None)
        
        if matlab_format:
            frame = frame-1
            line = line-1
        if continuous: # no dead time between stims; the end of one is the start of the next
            frame = repeat_internal_values(frame)
            line = repeat_internal_values(line)
        frame = frame.reshape((-1,2))
        line = line.reshape((-1,2))
        new_frame = np.floor(frame/nplanes).astype('<i4')
        new_line = line + np.remainder(frame,nplanes)*nlines
        new_nlines = nlines*nplanes
        
        return new_frame,new_line,new_nlines

    frame = frame.astype('<i4')

    while np.min(np.diff(frame)) < 0:
        brk = np.argmin(np.diff(frame))+1
        frame[brk:] = frame[brk:] + 65536

    with open('temp_frame.p', 'wb') as handle:
        pkl.dump(frame, handle, protocol=pkl.HIGHEST_PROTOCOL)
    
    frame,line,nlines = convert_frame_line(frame,line,nlines,nplanes,continuous=continuous)
    
    trigtime = frame+line/nlines
    triallen = np.diff(trigtime,axis=1)
    trigrate = np.mean(triallen)
    interpbetweentrigs = int(np.round(trigrate))
    roitime = roilines/nlines
    desired_offsets = np.arange(-nbefore,interpbetweentrigs+nafter) # frames relative to each trigger to sample
    
    trialwise = np.zeros((traces.shape[0],trigtime.shape[0],desired_offsets.size))
    for trial in range(trigtime.shape[0]):
        desired_frames = frame[trial,0]+desired_offsets
        try:
            for cell in range(traces.shape[0]):
                trialwise[cell,trial] = np.interp(trigtime[trial,0]+desired_offsets,desired_frames+roitime[cell],traces[cell][desired_frames])
        except:
            print('could not do trial #'+str(trial))
            trialwise[cell,trial] = np.nan
    
    return trialwise
    #return trialize(traces,frame,nbefore=nbefore,nafter=nafter) # TEMPORARY!! SEEING IF INTERPOLATION IS THE PROBLEM. Seems not to be at first glance...

def loadmat(filename,desired_vars):
    
    # take either a single string as input, or a tuple/list of strings. In the first case, the variable value is returned. In the second case, a tuple is returned.
    tuple_flag = True
    if type(desired_vars) is list:
        desired_vars = tuple(desired_vars)
    if not type(desired_vars) is tuple:
        tuple_flag = False
        desired_vars = (desired_vars,)
        
    try:
        matfile = sio.loadmat(filename,squeeze_me=True)
        to_return = tuple([matfile[var] for var in desired_vars])
    except:
        with h5py.File(filename,mode='r') as f:
            to_return = tuple([f[var][:].T for var in desired_vars])
            
    if not tuple_flag:
        to_return = to_return[0]
        
    return to_return

def gen_precise_trialwise(datafiles,nbefore=4,nafter=8,blcutoff=1,blspan=3000,ds=10,rg=None,frame_adjust=None):
    
    def tack_on(to_add,existing):
        try:
            existing = np.concatenate((existing,to_add),axis=0)
        except:
            existing = to_add.copy()
        return existing
    
    def process(to_add,roilines):
        to_add_copy = to_add.copy()
        to_add[np.isnan(to_add)] = np.minimum(np.nanmin(to_add),0)
#        to_add[to_add<0] = 0
        baseline = sfi.percentile_filter(to_add[:,::ds],blcutoff,(1,int(blspan/ds)))
        baseline = np.repeat(baseline,ds,axis=1)
        if baseline.shape[1]>to_add.shape[1]:
            baseline = baseline[:,:to_add.shape[1]]
        #to_correct = to_add<0 # commented out 19/2/5
        #to_correct = baseline<0 # changed 19/2/4 # commented out 19/2/5
        #to_add[to_correct] = to_add[to_correct] - baseline[to_correct] # commented out 19/2/5
        #baseline[to_correct] = 0 # commented out 19/2/5
        c = np.zeros_like(to_add)
        s = np.zeros_like(to_add)
        this_dfof = np.zeros_like(to_add)
        for i in range(c.shape[0]):
            #try:
            fudge = 5e-2*np.nanmax(to_add[i])
            if to_add[i].max()>0:
                this_dfof[i] = (to_add[i]-baseline[i,:])/np.maximum(fudge,baseline[i,:])
            else:
                print('roi '+str(i)+' all zeros')
            c[i],s[i],_,_,_  = deconvolve(this_dfof[i].astype(np.float64),penalty=1)
            #except:
            #    print("couldn't do "+str(i))
        to_add = precise_trialize(to_add,frm,line,roilines,nbefore=nbefore,nafter=nafter)
        cc = precise_trialize(c,frm,line,roilines,nbefore=nbefore,nafter=nafter)
        ss = precise_trialize(s,frm,line,roilines,nbefore=nbefore,nafter=nafter)
        dd = precise_trialize(this_dfof.astype(np.float64),frm,line,roilines,nbefore=nbefore,nafter=nafter)
        return to_add,cc,ss,this_dfof,s,dd
        
    trialwise = np.array(())
    ctrialwise = np.array(())
    strialwise = np.array(())
    dtrialwise = np.array(())
    dfof = np.array(())
    straces = np.array(())
    for datafile in datafiles:
        thisdepth = int(datafile.split('_ot_')[-1].split('.rois')[0])
        info = loadmat(re.sub('_ot_[0-9]*.rois','.mat',datafile),'info')
        frm = info['frame'][()]
        line = info['line'][()]
        if not rg is None:
            frm = frm[rg[0]:frm.size+rg[1]]
            line = line[rg[0]:line.size+rg[1]]
        if not frame_adjust is None:
            frm = frame_adjust(frm)
            line = frame_adjust(line)
        (to_add,ctr) = loadmat(datafile,('corrected','ctr'))
        print(datafile)
        print(to_add.shape)
        nlines = loadmat(datafile,'msk').shape[0]
        roilines = ctr[0] + nlines*thisdepth
        to_add,c,s,this_dfof,this_straces,dtr = process(to_add,roilines)
        trialwise = tack_on(to_add,trialwise)
        ctrialwise = tack_on(c,ctrialwise)
        strialwise = tack_on(s,strialwise)
        dtrialwise = tack_on(dtr,dtrialwise)
        dfof = tack_on(this_dfof,dfof)
        straces = tack_on(this_straces,straces)

    return trialwise,ctrialwise,strialwise,dfof,straces,dtrialwise

def plot_errorbars(x,mn_tgt,lb_tgt,ub_tgt,colors=None):
    if colors is None:
        colors = plt.cm.viridis(np.linspace(0,1,mn_tgt.shape[0]))
    errorplus = ub_tgt-mn_tgt
    errorminus = mn_tgt-lb_tgt
    errors = np.concatenate((errorplus[np.newaxis],errorminus[np.newaxis]),axis=0)
    for i in range(mn_tgt.shape[0]):
        plt.errorbar(x,mn_tgt[i],yerr=errors[:,i,:],c=colors[i])

def plot_bootstrapped_errorbars_hillel(x,arr,pct=(2.5,97.5),colors=None,linewidth=None,markersize=None):
    mn_tgt = np.nanmean(arr,0)
    lb_tgt,ub_tgt = bootstrap(arr,fn=np.nanmean,pct=pct)
    plot_errorbars_hillel(x,mn_tgt,lb_tgt,ub_tgt,colors=colors,linewidth=linewidth,markersize=markersize)
    #if colors is None:
    #    colors = plt.cm.viridis(np.linspace(0,1,mn_tgt.shape[0]))
    #for i in range(mn_tgt.shape[0]):
    #    plot_errorbar_hillel(x,mn_tgt[i],lb_tgt[i],ub_tgt[i],c=colors[i])

def plot_errorbars_hillel(x,mn_tgt,lb_tgt,ub_tgt,colors=None,linewidth=None,markersize=None):
    if colors is None:
        colors = plt.cm.viridis(np.linspace(0,1,mn_tgt.shape[0]))
    for i in range(mn_tgt.shape[0]):
        plot_errorbar_hillel(x,mn_tgt[i],lb_tgt[i],ub_tgt[i],c=colors[i],linewidth=linewidth,markersize=markersize)

def plot_errorbars_in_rows(x,mn_tgt,lb_tgt,ub_tgt,rowlen=10,scale=0.5):
    nrows = np.ceil(mn_tgt.shape[0]/rowlen)
    plt.figure(figsize=(scale*rowlen,scale*nrows))
    for k in range(mn_tgt.shape[0]):
        plt.subplot(nrows,rowlen,k+1)
        errorplus = ub_tgt[k]-mn_tgt[k]
        errorminus = mn_tgt[k]-lb_tgt[k]
        errors = np.concatenate((errorplus[np.newaxis],errorminus[np.newaxis]),axis=0)
        for i in range(mn_tgt[k].shape[0]):
            plt.errorbar(x,mn_tgt[k][i],yerr=errors[:,i,:])
            #plt.plot(mn_tgt[k])
        plt.axis('off')

def parse_options(opt,opt_keys,*args):
    # create a dict opt with keys opt_keys specifying the options listed
    # options specified in *args will overwrite the original entries of opt if they are not None

    if opt is None:
        opt = {}

    for i,key in enumerate(opt_keys):
        if not args[i] is None or not key in opt:
            opt[key] = args[i]

    for key in opt_keys:
        if not key in opt:
            opt[key] = None

    return opt

def plot_errorbar_hillel(x,mn_tgt,lb_tgt,ub_tgt,plot_options=None,c=None,linestyle=None,linewidth=None,markersize=None):
    opt_keys = ['c','linestyle','linewidth','markersize']
    opt = parse_options(plot_options,opt_keys,c,linestyle,linewidth,markersize)
    c,linestyle,linewidth,markersize = [opt[key] for key in opt_keys]

    errorplus = ub_tgt-mn_tgt
    errorminus = mn_tgt-lb_tgt
    errors = np.concatenate((errorplus[np.newaxis],errorminus[np.newaxis]),axis=0)
    plt.errorbar(x,mn_tgt,yerr=errors,c=c,linestyle=linestyle,linewidth=linewidth)
    plt.scatter(x,mn_tgt,c=c,s=markersize)

def get_dict_ind(opt,i):
    opt_temp = {}
    for key in opt.keys():
        isarr = type(opt[key]) is np.ndarray
        if not type(opt[key]) is list and not isarr:
            opt_temp[key] = opt[key]
        else:
            if isarr and i>=opt[key].shape[0]:
                opt_temp[key] = opt[key]
            else:
                opt_temp[key] = opt[key][i]
    return opt_temp

def plot_nothing(plot_options):
    opt_keys = ['c','linestyle','linewidth']
    opt = parse_options(plot_options,opt_keys,None,None,None)
    c,linestyle,linewidth = [opt[key] for key in opt_keys]
    plt.plot(0,0,c=c,linestyle=linestyle,linewidth=linewidth)

def keylist(dict):
    return list(dict.keys())
