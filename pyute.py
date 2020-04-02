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
import glob
import fnmatch
import shutil
import pandas as pd
import scipy.stats as sst
from pympler import asizeof
import sklearn.metrics as skm
import sklearn

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
    resamp=np.rollaxis(arr,axis,0)
    resamp=resamp[c]
    resamp=np.rollaxis(resamp,0,axis+2) # plus 1 due to rollaxis syntax. +1 due to extra resampled axis
    resamp=np.rollaxis(resamp,0,L+1)
    stat = fn(resamp,axis=axis)
    #lb = np.percentile(stat,pct[0],axis=-1) # resampled axis rolled to last position
    #ub = np.percentile(stat,pct[1],axis=-1) # resampled axis rolled to last position
    return tuple([np.percentile(stat,p,axis=-1) for p in pct])

def bootstrap_df(arr,fn,axis=0,nreps=1000,pct=(2.5,97.5)):
    # given arr 1D of size N, resample nreps sets of N of its elements with replacement. Compute fn on each of the samples
    # and report percentiles pct
    N = df.shape[axis]
    c = np.random.choice(np.arange(N),size=(N,nreps))
    L = len(df.shape)
    if axis==1:
        resamp=df.T
    resamp=resamp.iloc[c]
    resamp=np.rollaxis(resamp,0,axis+2) # plus 1 due to rollaxis syntax. +1 due to extra resampled axis
    resamp=np.rollaxis(resamp,0,L+1)
    stat = fn(resamp,axis=axis)
    #lb = np.percentile(stat,pct[0],axis=-1) # resampled axis rolled to last position
    #ub = np.percentile(stat,pct[1],axis=-1) # resampled axis rolled to last position
    return tuple([np.percentile(stat,p,axis=-1) for p in pct])

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
    if arr is None:
        return None
    if len(arr.shape)==1:
        arr = arr[np.newaxis]
        singleton = True
    else:
        singleton = False
    triallen = np.diff(frm,axis=1)
    triallen = np.round(triallen.mean()).astype('int')
    trialwise = np.zeros((arr.shape[0],frm.shape[0],triallen+nbefore+nafter))
    trialwise[:] = np.nan
    for i in range(trialwise.shape[1]):

        begin_arr = frm[i,0]-nbefore
        begin_t = np.maximum(-begin_arr,0)

        end_arr = frm[i,0]+triallen+nafter
        end_t = triallen+nbefore+nafter + np.minimum(arr.shape[1]-end_arr,0)
            
        try:
            trialwise[:,i,begin_t:end_t] = arr[:,begin_arr:end_arr]
        except:
            print('problem with trial #'+str(i))
    if singleton:
        trialwise = trialwise[0]
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
def process_ca_traces(to_add,ds=10,blspan=3000,blcutoff=1,frm=None,nbefore=4,nafter=4):
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
    d = trialize(this_dfof,frm,nbefore,nafter)
    return to_add,c,s,d #,this_dfof #(non-trialized)

def gen_trialwise(datafiles,nbefore=4,nafter=8,blcutoff=1,blspan=3000,ds=10,rg=None):
    
    def tack_on(to_add,existing):
        try:
            existing = np.concatenate((existing,to_add),axis=0)
        except:
            existing = to_add.copy()
        return existing
    
   # def process(to_add):
   #     to_add[np.isnan(to_add)] = np.nanmin(to_add) #0
   #     if to_add.max():
   #         baseline = sfi.percentile_filter(to_add[:,::ds],blcutoff,(1,int(blspan/ds)))
   #         topline = sfi.percentile_filter(to_add[:,::ds],99,(1,int(blspan/ds))) # dan added 18/10/30
   #         baseline = np.maximum(baseline,topline/10) # dan added 18/10/30
   #         baseline = np.repeat(baseline,ds,axis=1)
   #         if baseline.shape[1]>to_add.shape[1]:
   #             baseline = baseline[:,:to_add.shape[1]]
   #         c = np.zeros_like(to_add)
   #         s = np.zeros_like(to_add)
   #         this_dfof = np.zeros_like(to_add)
   #         for i in range(c.shape[0]):
   #             this_dfof[i] = (to_add[i]-baseline[i,:])/baseline[i,:]
   #             this_dfof[i][np.isnan(this_dfof[i])] = 0
   #             c[i],s[i],_,_,_  = deconvolve(this_dfof[i].astype(np.float64),penalty=1)
   #     else:
   #         this_dfof = np.zeros_like(to_add)
   #         c = np.zeros_like(to_add)
   #         s = np.zeros_like(to_add)
   #     to_add = trialize(to_add,frm,nbefore,nafter)
   #     c = trialize(c,frm,nbefore,nafter)
   #     s = trialize(s,frm,nbefore,nafter)
   #     return to_add,c,s,this_dfof
        
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
        to_add,c,s,this_dfof = process_ca_traces(to_add,nbefore=nbefore,nafter=nafter,blcutoff=blcutoff,blspan=blspan,ds=ds,frm=frm)
        trialwise = tack_on(to_add,trialwise)
        ctrialwise = tack_on(c,ctrialwise)
        strialwise = tack_on(s,strialwise)
        dfof = tack_on(this_dfof,dfof)

    return trialwise,ctrialwise,strialwise,dfof

def twoD_Gaussian(xy, xo, yo, amplitude, sigma_x, sigma_y, theta, offset):
    x = xy[0]
    y = xy[1]
    xo = np.array(xo).astype('float')
    yo = np.array(yo).astype('float') 
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
    return tuple([tuple(x) for x in lis])

def fit_2d_gaussian_before_after(locs,ret,verbose=False,bounds=((None,None,0,0,0,0),(None,None,np.inf,None,None,2*np.pi)),nbefore=8,nafter=8,xoyo_predictions=None,xoyo_prediction_wt=None):
#     
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
    #i = 0
    #can_be_negative = bounds[0][2]<0
    #if ret[i][msk_surr].mean()<ret[i].mean() or not can_be_negative:
    #    extremum = np.argmax(ret[i],axis=None)
    #    initial_guess = (x[extremum],y[extremum],ret[i].max()-ret[i].min(),10,10,0,ret[i].min())
    #else:
    #    extremum = np.argmin(ret[i],axis=None)
    #    initial_guess = (x[extremum],y[extremum],ret[i].min()-ret[i].max(),10,10,0,ret[i].max())
    params = np.zeros((2,ret.shape[0],len(bounds[0])))
    sqerror = np.inf*np.ones((2,ret.shape[0]))
    slcs = [slice(None,nbefore),slice(nbefore,-nafter),slice(-nafter,None)]
    ret_before,ret_during,ret_after = [np.nanmean(ret[:,:,:,slc],-1) for slc in slcs]
    if not xoyo_predictions is None:
        fit_fn = lambda xy,p: np.concatenate((twoD_Gaussian(xy,*p),np.sqrt(xoyo_prediction_wt)*np.array((p[0],p[1]))))
    else:
        fit_fn = lambda xy,*p: twoD_Gaussian(xy,*p,0)
    #assert(True==False)
    for i in range(ret.shape[0]):
        #assert(i!=0)
        for k in range(2):
            try:
                data = (ret_during[i]-0.5*ret_before[i]-0.5*ret_after[i]).flatten()
                #if ret[i][msk_surr].mean()<ret[i].mean() or not can_be_negative:
                if not xoyo_predictions is None:
                    fit_data = np.concatenate((data,np.sqrt(xoyo_prediction_wt)*xoyo_predictions[i]))
                else:
                    fit_data = data
                if k==0:
                    # xo,yo,amplitude,sigma_x,sigma_y,theta,offset
                    extremum = np.argmax(data,axis=None)
                    initial_guess = (x[extremum],y[extremum],ret[i].max()-ret[i].min(),10,10,0)                  
                    bounds = assign_tuple(bounds,(0,2),0)
                    bounds = assign_tuple(bounds,(1,2),np.inf)
                   # bounds = assign_tuple(bounds,(0,6),0) # peg offset to 0
                   # bounds = assign_tuple(bounds,(1,6),0) # peg offset to 0
                    popt, pcov = sop.curve_fit(fit_fn, (x,y), fit_data, p0=initial_guess, bounds=bounds)
                else:
                    # xo,yo,amplitude,sigma_x,sigma_y,theta,offset
                    initial_guess = (x[extremum],y[extremum],ret[i].min()-ret[i].max(),10,10,0)
                    bounds = assign_tuple(bounds,(0,2),-np.inf)
                    bounds = assign_tuple(bounds,(1,2),0)
                   # bounds = assign_tuple(bounds,(0,6),0) # peg offset to 0
                   # bounds = assign_tuple(bounds,(1,6),0) # peg offset to 0
                    popt, pcov = sop.curve_fit(fit_fn, (x,y), fit_data, p0=initial_guess, bounds=bounds)
                # xo, yo, amplitude, sigma_x, sigma_y, theta, offset
                modeled = twoD_Gaussian((x,y),*popt,0)
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
    #paramdict['offset'] = np.zeros(ret.shape[0:1])    
    for i in range(sqerror.shape[1]):
        bo = best_option[i]
        paramdict['sqerror'][i] = sqerror[bo,i]
        paramdict['xo'][i] = params[bo,i,0]
        paramdict['yo'][i] = params[bo,i,1]
        paramdict['amplitude'][i] = params[bo,i,2]
        paramdict['sigma_x'][i] = params[bo,i,3]
        paramdict['sigma_y'][i] = params[bo,i,4]
        paramdict['theta'][i] = params[bo,i,5]
        #paramdict['offset'][i] = params[bo,i,6]
    return paramdict

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
    #i = 0
    #can_be_negative = bounds[0][2]<0
    #if ret[i][msk_surr].mean()<ret[i].mean() or not can_be_negative:
    #    extremum = np.argmax(ret[i],axis=None)
    #    initial_guess = (x[extremum],y[extremum],ret[i].max()-ret[i].min(),10,10,0,ret[i].min())
    #else:
    #    extremum = np.argmin(ret[i],axis=None)
    #    initial_guess = (x[extremum],y[extremum],ret[i].min()-ret[i].max(),10,10,0,ret[i].max())
    params = np.zeros((2,ret.shape[0],7))
    sqerror = np.inf*np.ones((2,ret.shape[0]))
    for i in range(ret.shape[0]):
        #assert(i!=0)
        for k in range(2):
            try:
                data = ret[i].flatten()
                #if ret[i][msk_surr].mean()<ret[i].mean() or not can_be_negative:
                if k==0:
                    extremum = np.argmax(ret[i],axis=None)
                    initial_guess = (x[extremum],y[extremum],ret[i].max()-ret[i].min(),10,10,0,np.maximum(0,ret[i].min()))                  
                    bounds = assign_tuple(bounds,(0,2),0)
                    bounds = assign_tuple(bounds,(1,2),np.inf)
                    popt, pcov = sop.curve_fit(twoD_Gaussian, (x,y), data, p0=initial_guess, bounds=bounds)
                else:
                    extremum = np.argmin(ret[i],axis=None)
                    initial_guess = (x[extremum],y[extremum],ret[i].min()-ret[i].max(),10,10,0,ret[i].max())
                    bounds = assign_tuple(bounds,(0,2),-np.inf)
                    bounds = assign_tuple(bounds,(1,2),0)
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
        mn = np.minimum(np.nanmin(arr1[k]),np.nanmin(arr2[k]))
        mx = np.maximum(np.nanmax(arr1[k]),np.nanmax(arr2[k]))
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
        #to_return = tuple([matfile[var] for var in desired_vars])
        to_return = tuple([safe_get_(matfile,var,h5=False) for var in desired_vars])
    except:
        with h5py.File(filename,mode='r') as f:
            #to_return = tuple([f[var][:].T for var in desired_vars])
            to_return = tuple([safe_get_(f,var,h5=True) for var in desired_vars])
            
    if not tuple_flag:
        to_return = to_return[0]
        
    return to_return

def safe_get_(matfile,var,h5=False):
    if var in matfile:
        if not h5:
            return matfile[var]
        else:
            return matfile[var][:].T
    else:
        return None

def gen_precise_trialwise(datafiles,nbefore=4,nafter=8,blcutoff=1,blspan=3000,ds=10,rg=None,frame_adjust=None):
    
    def tack_on(to_add,existing):
        try:
            existing = np.concatenate((existing,to_add),axis=0)
        except:
            existing = to_add.copy()
        return existing
    
    def process(to_add,uncorrected,neuropil,roilines):
        to_add_copy = to_add.copy()
        to_add = interp_nans(to_add,axis=-1)
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
        #to_add = precise_trialize(to_add,frm,line,roilines,nbefore=nbefore,nafter=nafter)
        #cc = precise_trialize(c,frm,line,roilines,nbefore=nbefore,nafter=nafter)
        #ss = precise_trialize(s,frm,line,roilines,nbefore=nbefore,nafter=nafter)
        #dd = precise_trialize(this_dfof.astype(np.float64),frm,line,roilines,nbefore=nbefore,nafter=nafter)
        to_add,trialwise_t_offset = precise_trialize_no_interp(to_add,frm,line,roilines,nbefore=nbefore,nafter=nafter,nplanes=len(datafiles))
        raw_traces,_ = precise_trialize_no_interp(uncorrected,frm,line,roilines,nbefore=nbefore,nafter=nafter,nplanes=len(datafiles))
        neuropil,_ = precise_trialize_no_interp(neuropil,frm,line,roilines,nbefore=nbefore,nafter=nafter,nplanes=len(datafiles))
        cc,_ = precise_trialize_no_interp(c,frm,line,roilines,nbefore=nbefore,nafter=nafter,nplanes=len(datafiles))
        ss,_ = precise_trialize_no_interp(s,frm,line,roilines,nbefore=nbefore,nafter=nafter,nplanes=len(datafiles))
        dd,_ = precise_trialize_no_interp(this_dfof.astype(np.float64),frm,line,roilines,nbefore=nbefore,nafter=nafter,nplanes=len(datafiles))
        return to_add,cc,ss,this_dfof,s,dd,trialwise_t_offset,raw_traces,neuropil
        
    trialwise = np.array(())
    ctrialwise = np.array(())
    strialwise = np.array(())
    dtrialwise = np.array(())
    dfof = np.array(())
    straces = np.array(())
    trialwise_t_offset = np.array(())
    proc = {}
    proc['raw_trialwise'] = np.array(())
    proc['neuropil_trialwise'] = np.array(())
    proc['trialwise_t_offset'] = np.array(())
    for datafile in datafiles:
        thisdepth = int(datafile.split('_ot_')[-1].split('.rois')[0])
        info = loadmat(re.sub('_ot_[0-9]*.rois','.mat',datafile),'info')
        frm = info['frame'][()]
        line = info['line'][()]
        event_id = info['event_id'][()]
        ignore_first = 0
        ignore_last = 0
        while event_id[0]==2:
            event_id = event_id[1:]
            frm = frm[1:]
            line = line[1:]
            ignore_first = ignore_first+1
        while event_id[-1]==2:
            event_id = event_id[:-1]
            frm = frm[:-1]
            line = line[:-1]
            ignore_last = ignore_last+1
        if not rg is None:
            thisrg = (rg[0]-ignore_first,rg[1]+ignore_last)
            print(thisrg)
            frm = frm[thisrg[0]:frm.size+thisrg[1]]
            line = line[thisrg[0]:line.size+thisrg[1]]
        else:
            frm = frm[event_id==1]
            line = line[event_id==1]
        if not frame_adjust is None:
            frm = frame_adjust(frm)
            line = frame_adjust(line)
        (to_add,ctr,uncorrected,neuropil) = loadmat(datafile,('corrected','ctr','Data','Neuropil'))
        print(datafile)
        print(to_add.shape)
        nlines = loadmat(datafile,'msk').shape[0]
        roilines = ctr[0] + nlines*thisdepth
        #to_add,c,s,this_dfof,this_straces,dtr = process(to_add,roilines)
        to_add,c,s,this_dfof,this_straces,dtr,tt,uncorrected,neuropil = process(to_add,uncorrected,neuropil,roilines)
        trialwise = tack_on(to_add,trialwise)
        ctrialwise = tack_on(c,ctrialwise)
        strialwise = tack_on(s,strialwise)
        dtrialwise = tack_on(dtr,dtrialwise)
        dfof = tack_on(this_dfof,dfof)
        straces = tack_on(this_straces,straces)
        #trialwise_t_offset = tack_on(tt,trialwise_t_offset)
        proc['raw_trialwise'] = tack_on(uncorrected,proc['raw_trialwise'])
        proc['neuropil_trialwise'] = tack_on(neuropil,proc['neuropil_trialwise'])
        proc['trialwise_t_offset'] = tack_on(tt,proc['trialwise_t_offset'])

    #return trialwise,ctrialwise,strialwise,dfof,straces,dtrialwise
    return trialwise,ctrialwise,strialwise,dfof,straces,dtrialwise,proc # trialwise_t_offset

def plot_errorbars(x,mn_tgt,lb_tgt,ub_tgt,colors=None):
    if colors is None:
        colors = plt.cm.viridis(np.linspace(0,1,mn_tgt.shape[0]))
    errorplus = ub_tgt-mn_tgt
    errorminus = mn_tgt-lb_tgt
    errors = np.concatenate((errorplus[np.newaxis],errorminus[np.newaxis]),axis=0)
    for i in range(mn_tgt.shape[0]):
        plt.errorbar(x,mn_tgt[i],yerr=errors[:,i,:],c=colors[i])

def plot_bootstrapped_errorbars_hillel(x,arr,pct=(2.5,97.5),colors=None,linewidth=None,markersize=None,norm_to_max=False):
    mn_tgt = np.nanmean(arr,0)
    lb_tgt,ub_tgt = bootstrap(arr,fn=np.nanmean,pct=pct)
    if norm_to_max:
        normby = mn_tgt.max(-1)[:,np.newaxis] - mn_tgt.min(-1)[:,np.newaxis]
        baseline = mn_tgt.min(-1)[:,np.newaxis]
        mn_tgt = mn_tgt - baseline
        lb_tgt = lb_tgt - baseline
        ub_tgt = ub_tgt - baseline
        plot_errorbars_hillel(x,mn_tgt/normby,lb_tgt/normby,ub_tgt/normby,colors=colors,linewidth=linewidth,markersize=markersize)
    else:
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
    if not plot_options is None:
        opt = parse_options(plot_options,opt_keys,c,linestyle,linewidth,markersize)
        c,linestyle,linewidth,markersize = [opt[key] for key in opt_keys]

    errorplus = ub_tgt-mn_tgt
    errorminus = mn_tgt-lb_tgt
    errors = np.concatenate((errorplus[np.newaxis],errorminus[np.newaxis]),axis=0)
    plt.errorbar(x,mn_tgt,yerr=errors,c=c,linestyle=linestyle) #,fmt=None)
    plt.plot(x,mn_tgt,c=c,linestyle=linestyle,linewidth=linewidth)
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

def precise_trialize_no_interp(traces,frame,line,roilines,nlines=512,nplanes=4,nbefore=4,nafter=8,continuous=False):
    
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
    
    frame,line,nlines = convert_frame_line(frame,line,nlines,nplanes,continuous=continuous)
    
    ftrigtime = line/nlines
    trigtime = frame+ftrigtime
    triallen = np.diff(trigtime,axis=1)
    trigrate = np.mean(triallen)
    interpbetweentrigs = int(np.round(trigrate))
    roitime = roilines/nlines
    desired_offsets = np.arange(-nbefore,interpbetweentrigs+nafter) # frames relative to each trigger to sample
    
    trialwise = np.zeros((traces.shape[0],trigtime.shape[0],desired_offsets.size))
    trialwise_t_offset = np.zeros((traces.shape[0],trigtime.shape[0]))
    for trial in range(trigtime.shape[0]):
        desired_frames = frame[trial,0]+desired_offsets
        stop_early = desired_frames.max()+1 >= traces.shape[1] # last frame is beyond the end
        if stop_early:
            stopat = traces.shape[1]-1-desired_frames.max()-1 # take a subset of the frames
        else:
            stopat = desired_frames.size # take all the frames
        #try:
        for cell in range(traces.shape[0]):
            trialwise_t_offset[cell,trial] = np.mod(roitime[cell]-ftrigtime[trial,0],1)
            if roitime[cell] > ftrigtime[trial,0]: # if cell is recorded after stim comes in, take the stim frame
                trialwise[cell,trial][:stopat] = traces[cell][desired_frames[:stopat]]
            else: # if cell is recorded before stim comes in, take the subsequent frame
                trialwise[cell,trial][:stopat] = traces[cell][desired_frames[:stopat]+1]
        #except:
        #    print('could not do trial #'+str(trial))
        #    trialwise[cell,trial] = np.nan
    
    return trialwise,trialwise_t_offset

def hdf5edit(filename):
    if not os.path.exists(filename):
        return h5py.File(filename,mode='w')
    else:
        return h5py.File(filename,mode='r+')

def hdf5read(filename):
    if not os.path.exists(filename):
        return h5py.File(filename,mode='w')
    else:
        return h5py.File(filename,mode='r')

def dict_to_hdf5(filename,groupname,dicti):
    #with h5py.File(filename,mode='r+') as f:
    with hdf5edit(filename) as f:
        assert(not groupname in f.keys())
        this_group = f.create_group(groupname)
        dict_to_hdf5_(this_group,dicti)

def dict_to_hdf5_(this_group,dicti):
    for key in dicti:
        this_type = type(dicti[key])
        if this_type is str or this_type is float or this_type is int:
            this_group[key] = dicti[key]
        elif this_type is np.ndarray:
            this_group.create_dataset(key,data=dicti[key])
        elif this_type is dict:
            new_group = this_group.create_group(key)
            dict_to_hdf5_(new_group,dicti[key])

def matfile_to_dict(matfile):
    dicti = {}
    for key in list(matfile.dtype.fields.keys()):
        dicti[key] = matfile[key][()]
    return dicti

def hdf5_to_dict(grp):
    print(grp[()])
    assert(True==False)
    if isinstance(grp[()],h5py.Group):
        print('it is a group')
        dicti = {}
        for key in grp[()]:
            dicti[key] = hdf5_to_dict(grp[()][key])
    else:
        dicti = grp[:]
    return dicti

def k_and(*args):
    if len(args)>2:
        return np.logical_and(args[0],k_and(*args[1:]))
    elif len(args)==2:
        return np.logical_and(args[0],args[1])
    else:
        return args[0]

def compute_tuning(data,stim_id,cell_criteria=None,trial_criteria=None):
    ndims = stim_id.shape[0]
    maxind = tuple(stim_id.max(1).astype('int')+1)
    if cell_criteria is None:
        cell_criteria = np.ones((data.shape[0],),dtype='bool')
    if trial_criteria is None:
        trial_criteria = np.ones((data.shape[1],),dtype='bool')
    nparams = len(maxind)
    ntrialtypes = int(np.prod(maxind))
    tuning = np.zeros((data[cell_criteria].shape[0],ntrialtypes)+data.shape[2:])
    for itype in range(ntrialtypes):
        imultitype = np.unravel_index(itype,maxind)
        these_trials = trial_criteria.copy()
        for iparam in range(nparams):
            these_trials = np.logical_and(these_trials,stim_id[iparam]==imultitype[iparam])
        tuning[:,itype] = np.nanmean(data[cell_criteria][:,these_trials],1)
    tuning = np.reshape(tuning,(tuning.shape[0],)+maxind+tuning.shape[2:])
    return tuning

def compute_tavg_dataframe(dsfile,expttype='size_contrast_0',datafield='decon',nbefore_default=None,nafter_default=None,keylist=None,run_fn=None):
    # will return a pandas dataframe, consisting of data from every trial in every expt
    # and two dicts: each indexed by session id, one listing roi parameters (location, rf center, rf pval), and one listing trialwise parameters (run speed, eye position)
    with h5py.File(dsfile,mode='r') as f:
        if keylist is None:
            keylist = [key for key in f.keys()]
        df = [None for i in range(len(keylist))]
        roi_info = {}
        trial_info = {}
        for ikey in range(len(keylist)):
            session = f[keylist[ikey]]
            print(session)
            print([key for key in session.keys()])
            if expttype in session:
                sc0 = session[expttype]
                if nbefore_default is None:
                    nbefore = sc0['nbefore'][()]
                else:
                    nbefore = nbefore_default
                if nafter_default is None:
                    nafter = sc0['nafter'][()]
                else:
                    nafter = nafter_default
                if run_fn is None:
                    run_fn = lambda x: x[:,nbefore:-nafter].mean(-1)>10
                data = np.nanmean(sc0[datafield][:,:,nbefore:-nafter][:],-1) # N neurons x P trials (previously x T timepoints)
                stim_id = sc0['stimulus_id'][:]
                if 'running_speed_cm_s' in sc0:
                    trialrun = run_fn(sc0['running_speed_cm_s'][:]) #[:,nbefore:-nafter].mean(-1)>10 #
                else:
                    trialrun = sc0['trialrun'][:]
                #uparam[ikey] = [None for iparam in range(len(sc0['stimulus_parameters']))]
                dfdict = {}
                dfdict['data'] = data.flatten()
                roi_index = session['cell_id'][:]
                trial_index = np.arange(data.shape[1])
                dfdict['roi_index'] = np.tile(roi_index[:,np.newaxis],(1,data.shape[1])).flatten()
                dfdict['trial_index'] = np.tile(trial_index[np.newaxis,:],(data.shape[0],1)).flatten()
                dfdict['session_id'] = keylist[ikey]

                roidict = {}
                if 'rf_displacement_deg' in sc0:
                    roidict['rf_displacement'] = sc0['rf_displacement_deg'][:]
                    roidict['pval'] = sc0['rf_mapping_pval'][:]
                if 'cell_center' in session:
                    roidict['cell_center'] = session['cell_center'][:].T
                if 'cell_depth' in session:
                    roidict['cell_depth'] = session['cell_depth'][:].T

                trialdict = {}
                for iparam,param in enumerate(sc0['stimulus_parameters']):
                    this_info = sc0[param][:][stim_id[iparam]]
                    trialdict[param.decode('UTF-8')] = this_info
                    trialdict['running'] = trialrun
                    #dfdict[param.decode('UTF-8')] = np.tile(trial_info[np.newaxis,:],(data.shape[0],1)).flatten()

                df[ikey] = pd.DataFrame(dfdict)
                roi_info[keylist[ikey]] = roidict
                trial_info[keylist[ikey]] = trialdict

    df = pd.concat(df)
    
    return df,roi_info,trial_info

def array_to_flat_plus_indices(data):
    flat = data.flatten()
    indices = np.meshgrid(*[np.arange(shp) for shp in data.shape])
    indices = [ind.flatten() for ind in indices]
    return [flat] + indices

def invert_list_ordering(list1):
# swap 0th and 1st level list indices
    return [[*a] for a in zip(*list1)]

def nested_list_to_flat_plus_indices(nested):
    if isinstance(nested,list):
        output = [nested_list_to_flat_plus_indices(x) for x in nested]
        intermediate = [[outp[0],ioutp*np.ones(outp[0].shape,dtype='int'),*outp[1:]] for ioutp,outp in enumerate(output)]
        final = [np.concatenate(a,axis=0) for a in invert_list_ordering(intermediate)]
    else:
        final = array_to_flat_plus_indices(nested)
    return final
    
def noisy_scatter(x,y,noise=0.1):
    def prepare_to_plot(u):
        return u + noise*np.random.randn(*u.shape)
    xplot = prepare_to_plot(x)
    yplot = prepare_to_plot(y)
    plt.scatter(xplot,yplot,s=5,alpha=2e2/x.size)

def plot_bin_stat(x,y,nbins=20):
    binmean,binedge,_ = sst.binned_statistic(x,y,bins=nbins)
    binstd,_,_ = sst.binned_statistic(x,y,statistic=sst.sem,bins=nbins)
    plt.errorbar(0.5*(binedge[:-1]+binedge[1:]),binmean,binstd,c='r')
    plt.plot(0.5*(binedge[:-1]+binedge[1:]),np.zeros_like(binmean),c='k')

def mkdir(d):
    if not os.path.exists(d):
        os.mkdir(d)
        
def copy_directory_structure(source,target):
    for root,dirs,files in os.walk(source):
        for name in dirs:
            thispath = os.path.join(root,name)
            thatpath = thispath.replace(source,target)
            mkdir(thatpath)
            
def copy_pattern_ds(source,target,pattern,exclude=[]):
    if type(exclude) is str:
        exclude = [exclude]
    copy_directory_structure(source,target)
    for root,dirs,files in os.walk(source):
        for name in files:
            exclude_this = False
            if exclude:
                for bad_item in exclude:
                    if fnmatch.fnmatchcase(name,bad_item):
                        exclude_this = True
            if fnmatch.fnmatchcase(name,pattern) and not exclude_this:
                thispath = os.path.join(root,name)
                thatpath = thispath.replace(source,target)
                print(' --> '.join([thispath,thatpath]))
                shutil.copyfile(thispath,thatpath)

def plot_ellipse(x,y,ctr_fn=np.mean,rad_fn=np.std,alpha=None,c=None,edge=True):
    ell = mp.Ellipse(xy=(ctr_fn(x),ctr_fn(y)),width=2*rad_fn(x),height=2*rad_fn(y))
    plt.gca().add_artist(ell)
    if not alpha is None:
        ell.set_alpha(alpha)
    if not c is None:
        ell.set_facecolor(colors[i])
    if edge:
        ell.set_edgecolor('k')

def plot_ellipse(x,y,ctr_fn=np.mean,rad_fn=np.std,alpha=None,c=None,edge=True):
    ell = mp.Ellipse(xy=(ctr_fn(x),ctr_fn(y)),width=2*rad_fn(x),height=2*rad_fn(y))
    plt.gca().add_artist(ell)
    if not alpha is None:
        ell.set_alpha(alpha)
    if not c is None:
        ell.set_facecolor(colors[i])
    if edge:
        ell.set_edgecolor('k')

def combine_rg(r,g,redfactor=1,greenfactor=1):
    rn = redfactor*(r/r.max())[:,:,np.newaxis]
    gn = greenfactor*(g/g.max())[:,:,np.newaxis]
    rn[rn>1] = 1
    gn[gn>1] = 1
    rgb = np.concatenate((rn,gn,np.zeros_like(rn)),axis=2)
    return rgb

def compute_kernel_density(ctr,bw=0.1,grid_pts=None):
    if grid_pts is None:
        x,y = np.meshgrid(np.arange(0,sz[1],resolution),np.arange(0,sz[0],resolution))
    else:
        y,x = np.meshgrid(grid_pts[0],grid_pts[1],indexing='ij')
    posns = np.vstack((y.ravel(),x.ravel()))
    kernel_density = sst.gaussian_kde(ctr,bw_method=bw)(posns).reshape(x.shape)
    return kernel_density

def gen_size_list(v):
    var_list = list(v.keys())
    size_list = {}
    for name in var_list:
        sz = asizeof.asizeof(v[name])
#         print(name + ':' + str(sz))
        size_list[name] = sz
    return size_list

def plot_traces_grid(arr):
    # arr is nc1 x nc2 x T (x ntrials)
    nc1,nc2 = arr.shape[:2]
    for ic1 in range(nc1):
        for ic2 in range(nc2):
            plt.subplot(nc1,nc2,ic1*nc2+ic2+1)
            plt.plot(arr[ic1,ic2])
            plt.ylim((arr.min(),arr.max()))
            plt.axis('off')

def interp_nans(arr,axis=-1,verbose=False):
    nan_loc = np.where(np.isnan(arr))
    arr_new = arr.copy()
    for loc in zip(*nan_loc):
        loc_before,loc_after = np.array(loc).copy(),np.array(loc).copy()
        loc_before[axis] = loc_before[axis]-1
        loc_after[axis] = loc_after[axis]+1
        try:
            arr_new[loc] = 0.5*(arr[tuple(loc_before)]+arr[tuple(loc_after)])
        except:
            if verbose:
                print('could not interpolate '+str(loc))
    return arr_new

def compute_tuning_ret_run(dsfile,running=True,center=True,fieldname='decon',keylist=None,expttype='size_contrast_opto_0'): #,subsample=1.):
    #output = compute_tuning_ret_run_subsample(dsfile,running=running,center=center,fieldname=fieldname,keylist=keylist,expttype=expttype): #,subsample=1.):
    with h5py.File(dsfile,mode='r') as f:
        if keylist is None:
            keylist = [key for key in f.keys()]
        tuning = [None]*len(keylist)
        for ikey in range(len(keylist)):
            session = f[keylist[ikey]]
            if expttype in session:
#                 print([key for key in session.keys()])
                data = session[expttype][fieldname][:]
                nbefore = session[expttype]['nbefore'][()]
                nafter = session[expttype]['nafter'][()]
                stim_id = session[expttype]['stimulus_id'][:]
                trialrun = session[expttype]['running_speed_cm_s'][:,nbefore:-nafter].mean(-1)>10
                if not running:
                    trialrun = ~trialrun

                #trialrun = trialrun & np.random.rand(trialrun.shape)<subsample
                    
                if 'rf_displacement_deg' in session[expttype]:
                    pval = session[expttype]['rf_mapping_pval'][:]
                    X = session['cell_center'][:]
                    y = session[expttype]['rf_displacement_deg'][:].T
                    lkat = k_and(pval<0.05,~np.isnan(X[:,0]),~np.isnan(y[:,0]))
                    linreg = sklearn.linear_model.LinearRegression().fit(X[lkat],y[lkat])
                    displacement = np.zeros_like(y)
                    displacement[~np.isnan(X[:,0])] = linreg.predict(X[~np.isnan(X[:,0])])
                if center:
                    cell_criteria = np.sqrt((displacement**2).sum(1))<10
                else:
                    cell_criteria = np.sqrt((displacement**2).sum(1))>10
                tuning[ikey] = compute_tuning(data,stim_id,cell_criteria=cell_criteria,trial_criteria=trialrun)
                print('%s: %.1f' % (keylist[ikey], trialrun.mean()))
            else:
                print('could not do '+keylist[ikey])
    return tuning

def compute_tuning_ret_run_subsample(dsfile,running=True,center=True,fieldname='decon',keylist=None,expttype='size_contrast_opto_0',dcutoff=10,sample=None):
    with h5py.File(dsfile,mode='r') as f:
        if keylist is None:
            keylist = [key for key in f.keys()]
        tuning = [None]*len(keylist)
        for ikey in range(len(keylist)):
            session = f[keylist[ikey]]
            if expttype in session:
#                 print([key for key in session.keys()])
                data = session[expttype][fieldname][:]
                nbefore = session[expttype]['nbefore'][()]
                nafter = session[expttype]['nafter'][()]
                stim_id = session[expttype]['stimulus_id'][:]
                if not 'trialrun' in session[expttype]:
                    trialrun = session[expttype]['running_speed_cm_s'][:,nbefore:-nafter].mean(-1)>10
                else:
                    trialrun = session[expttype]['trialrun'][:]
                if not running:
                    trialrun = ~trialrun

                if sample is None:
                    this_sample = np.random.randn(*trialrun.shape)>0
                else:
                    this_sample = sample

                these_samples = [this_sample,~this_sample]
                    
                if 'rf_displacement_deg' in session[expttype]:
                    pval = session[expttype]['rf_mapping_pval'][:]
                    X = session['cell_center'][:]
                    y = session[expttype]['rf_displacement_deg'][:].T
                    lkat = k_and(pval<0.05,~np.isnan(X[:,0]),~np.isnan(y[:,0]))
                    linreg = sklearn.linear_model.LinearRegression().fit(X[lkat],y[lkat])
                    displacement = np.zeros_like(y)
                    displacement[~np.isnan(X[:,0])] = linreg.predict(X[~np.isnan(X[:,0])])
                if not center is None:
                    if center:
                        cell_criteria = np.sqrt((displacement**2).sum(1))<dcutoff
                    else:
                        cell_criteria = np.sqrt((displacement**2).sum(1))>dcutoff
                else:
                    cell_criteria = np.ones(data.shape[0:1],dtype='bool')
                nsamples = 2
                tuning[ikey] = [None for isample in range(nsamples)]
                for isample in range(nsamples):
                    tuning[ikey][isample] = compute_tuning(data,stim_id,cell_criteria=cell_criteria,trial_criteria=(trialrun & these_samples[isample]))
                print('%s: %.1f' % (keylist[ikey], trialrun.mean()))
            else:
                print('could not do '+keylist[ikey])
    return tuning

def compute_roc(a1,a2):
    # computes area under receiver operating characteristic curve
    # for two arrays of numbers a1 and a2. returns 1 if numbers in a2
    # are strictly larger than a1, 0 if strictly smaller, and intermediate
    # values for varying degrees of overlap in the distributions
    vals = np.concatenate((a1,a2))
    grps = np.concatenate((np.zeros_like(a1),np.ones_like(a2)))
    vals_argsort = np.argsort(-vals)
    fp,tp = [np.cumsum(grps[vals_argsort]==thing)/np.sum(grps[vals_argsort]==thing) for thing in (0,1)]
    vals_sorted = vals[vals_argsort]
    return vals_sorted,fp,tp

def compute_auroc(a1,a2):
    # computes area under receiver operating characteristic curve
    # for two arrays of numbers a1 and a2. returns 1 if numbers in a2
    # are strictly larger than a1, 0 if strictly smaller, and intermediate
    # values for varying degrees of overlap in the distributions
    vals_sorted,fp,tp = compute_roc(a1,a2)
   # vals = np.concatenate((a1,a2))
   # grps = np.concatenate((np.zeros_like(a1),np.ones_like(a2)))
   # vals_argsort = np.argsort(-vals)
   # fp,tp = [np.cumsum(grps[vals_argsort]==thing)/np.sum(grps[vals_argsort]==thing) for thing in (0,1)]
    return skm.auc(fp,tp)

def select_trials(trial_info,selector,training_frac,include_all=False):
    # dict saying what to do with each trial type. If a function, apply that function to the trial info column to
    # obtain a boolean indexing variable
    # if 0, then the tuning output should be indexed by that variable
    # if 1, then that variable will be marginalized over in the tuning output
    def gen_train_test_exptwise(ti):
        ntrials = ti[params[0]].size
        gd = np.ones((ntrials,),dtype='bool')
        for param in params:
            if callable(selector[param]): # all the values of selector that are functions, ignore trials where that function evaluates to False
                exclude = ~selector[param](ti[param])
                gd[exclude] = False
        condition_list,_ = gen_condition_list(ti,selector) # automatically, separated out such that each half of the data gets an equivalent fraction of trials with each condition type
        condition_list = [c[gd] for c in condition_list]
        in_training_set = np.zeros((ntrials,),dtype='bool')
        in_test_set = np.zeros((ntrials,),dtype='bool')
        to_keep = output_training_test(condition_list,training_frac)
        in_training_set[gd] = to_keep
        in_test_set[gd] = ~to_keep
        if include_all:
            train_test = [in_training_set,in_test_set,np.logical_or(in_training_set,in_test_set)]
        else:
            train_test = [in_training_set,in_test_set]
        return train_test,ntrials

    params = list(selector.keys())
    keylist = list(trial_info.keys())
    if isinstance(trial_info[keylist[0]],dict):
        ntrials = {}
        train_test = {}
        for key in trial_info.keys():
            ti = trial_info[key]
            train_test[key],ntrials[key] = gen_train_test_exptwise(ti)
    else:
        ti = trial_info
        train_test,ntrials = gen_train_test_exptwise(ti)

    return train_test

def gen_condition_list(ti,selector,filter_selector=lambda x:True):
# ti: trial_info generated by compute_tavg_dataframe
# selector: dict where each key is a param in ti.keys(), and each value is either a callable returning a boolean,
# to be applied to ti[param], or an input to the function filter_selector
# filter selector: if filter_selector(selector[param]), the tuning curve will be separated into the unique elements of ti[param].
    params = list(selector.keys())
    condition_list = []
    for param in params:
        if not callable(selector[param]) and filter_selector(selector[param]):
            condition_list.append(ti[param])
    return condition_list,params

def output_training_test(condition_list,training_frac):
    # output training and test sets balanced for conditions
    # condition list, generated by gen_condition_list, has a row for each condition that should be equally assorted
    if not isinstance(condition_list,list):
        condition_list = [condition_list.copy()]
    iconds,uconds = zip(*[pd.factorize(c,sort=True) for c in condition_list])
    #uconds = [np.sort(u) for u in uconds]
    nconds = np.array([u.size for u in uconds])
    in_training_set = np.zeros(condition_list[0].shape,dtype='bool')
    for iflat in range(np.prod(nconds)):
        coords = np.unravel_index(iflat,tuple(nconds))
        lkat = np.where(k_and(*[iconds[ic] == coords[ic] for ic in range(len(condition_list))]))[0]
        n_train = int(np.round(training_frac*len(lkat)))
        to_train = np.random.choice(lkat,n_train,replace=False)
        in_training_set[to_train] = True
    #assert(True==False)
    return in_training_set

def compute_size_tuning(sc,contrast_axis=2):
    slc = [slice(None) for idim in range(len(sc.shape))]
    slc[contrast_axis] = slice(0,1)
    gray = sc[slc].mean(contrast_axis-1) #(nroi,1,nother...)
    slc = [slice(None) for idim in range(len(sc.shape))]
    slc[contrast_axis-1] = np.newaxis
    slc[contrast_axis] = [0 for icontrast in range(sc.shape[contrast_axis])]
    size_by_contrast = np.concatenate((gray[slc],sc),axis=contrast_axis-1)
    slc = [slice(None) for idim in range(len(sc.shape))]
    slc[contrast_axis] = slice(1,None)
    size_by_contrast = size_by_contrast[slc]
    return size_by_contrast

def assign_from_uparam(modal,modal_uparam,this,this_uparam,ignore_first=0):
    nparam = len(this_uparam)
    bool_in_this,bool_in_modal = [[None for iparam in range(nparam)] for ivar in range(2)]
    for iparam in range(nparam): #
        tu,mu = [a[iparam] for a in [this_uparam,modal_uparam]]
        bool_in_this[iparam],bool_in_modal[iparam] = assign_to_modal_uparams(tu,mu)
    #bool_in_this,bool_in_modal = assign_to_modal_uparams(this_uparam,modal_uparam)
    assign_(modal,bool_in_modal,this,bool_in_this,ignore_first=ignore_first)
    
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
    
def assign_(a,a_ind,b,b_ind,ignore_first=1):
    a_bool = gen_big_bool(a_ind)
    b_bool = gen_big_bool(b_ind)
    a[[slice(None) for iind in range(ignore_first)]+[a_bool]] = b[[slice(None) for iind in range(ignore_first)]+[b_bool]]
    
def gen_big_bool(bool_list):
    nind = len(bool_list)
    slicers = [[np.newaxis for iind in range(nind)] for iind in range(nind)]
    for iind in range(nind):
        slicers[iind][iind] = slice(None)
    big_ind = np.ones(tuple([iit.shape[0] for iit in bool_list]),dtype='bool')
    for iit,slc in zip(bool_list,slicers):
        big_ind = big_ind*iit[slc]
    return big_ind

def set_lims(*arrs,wiggle_pct=0.05):
    mn = np.inf
    mx = -np.inf
    for arr in arrs:
        mn = np.minimum(np.nanmin(arr),mn)
        mx = np.maximum(np.nanmax(arr),mx)
    wiggle = wiggle_pct*(mx-mn)
    plt.xlim((mn-wiggle,mx+wiggle))
    plt.ylim((mn-wiggle,mx+wiggle))

def pca_denoise(arr,Npc):
    u,s,vh = np.linalg.svd(arr.T,full_matrices=False)
    return (u[:,:Npc] @ np.diag(s[:Npc]) @ vh[:Npc,:]).T

def bar_pdf(data,bins=None,alpha=1):
    h,_ = np.histogram(data,bins=bins)
    plt.bar(0.5*(bins[:-1]+bins[1:]),h/h.sum(),width=bins[1]-bins[0],alpha=0.5)

def compute_tuning_many_partitionings(df,trial_info,npartitionings,training_frac=0.5): 
    selector_s1 = gen_nub_selector_s1() 
    selector_v1 = gen_nub_selector_v1() 
    keylist = list(trial_info.keys()) 
    train_test = {} 
    for key in keylist: 
        if trial_info[key]['area'][:2]=='s1': 
            selector = gen_nub_selector_s1() 
        else: 
            selector = gen_nub_selector_v1() 
        train_test[key] = [None for ipartitioning in range(npartitionings)] 
        for ipartitioning in range(npartitionings): 
            train_test[key][ipartitioning] = select_trials(trial_info[key],selector,training_frac) 
    tuning = pd.DataFrame() 
    ttls = ['s1_l4','s1_l23','v1_l4','v1_l23'] 
    selectors = [selector_s1, selector_s1, selector_v1, selector_v1] 
    tt = [{k:v[ipartitioning] for k,v in zip(train_test.keys(),train_test.values())} for ipartitioning in range(npartitionings)] 
    for ttl,selector in zip(ttls,selectors): 
        for ipartitioning in range(npartitionings): 
            new_tuning = compute_tuning_df(df.loc[df.area==ttl],trial_info,selector,include=tt[ipartitioning]) 
            new_tuning.insert(new_tuning.shape[1],'partitioning',ipartitioning) 
            #new_tuning['partitioning'] = ipartitioning 
            tuning = tuning.append(new_tuning) 
    return tuning

def select_trials(trial_info,selector,training_frac,include_all=False):
    # dict saying what to do with each trial type. If a function, apply that function to the trial info column to
    # obtain a boolean indexing variable
    # if 0, then the tuning output should be indexed by that variable
    # if 1, then that variable will be marginalized over in the tuning output
    def gen_train_test_exptwise(ti):
        ntrials = ti[params[0]].size
        gd = np.ones((ntrials,),dtype='bool')
        for param in params:
            if callable(selector[param]): # all the values of selector that are functions, ignore trials where that function evaluates to False
                exclude = ~selector[param](ti[param])
                gd[exclude] = False
        condition_list,_ = gen_condition_list(ti,selector) # automatically, separated out such that each half of the data gets an equivalent fraction of trials with each condition type
        condition_list = [c[gd] for c in condition_list]
        in_training_set = np.zeros((ntrials,),dtype='bool')
        in_test_set = np.zeros((ntrials,),dtype='bool')
        to_keep = output_training_test(condition_list,training_frac)
        in_training_set[gd] = to_keep
        in_test_set[gd] = ~to_keep
        if include_all:
            train_test = [in_training_set,in_test_set,np.logical_or(in_training_set,in_test_set)]
        else:
            train_test = [in_training_set,in_test_set]
        return train_test,ntrials

    params = list(selector.keys())
    keylist = list(trial_info.keys())
    if isinstance(trial_info[keylist[0]],dict):
        ntrials = {}
        train_test = {}
        for key in trial_info.keys():
            ti = trial_info[key]
            train_test[key],ntrials[key] = gen_train_test_exptwise(ti)
    else:
        ti = trial_info
        train_test,ntrials = gen_train_test_exptwise(ti)

    return train_test

def compute_tuning_df(df,trial_info,selector,include=None):
    params = list(selector.keys())
#     expts = list(trial_info.keys())
    expts = df.session_id.unique()
    nexpt = len(expts)
    tuning = pd.DataFrame()
    if include is None:
        include = {expt:None for expt in expts}
    for iexpt,expt in enumerate(expts):
        in_this_expt = (df.session_id == expt)
        trialwise = df.loc[in_this_expt].pivot(values='data',index='roi_index',columns='trial_index')
        nroi = trialwise.shape[0]
        ntrial = trialwise.shape[1]
        if include[expt] is None:
            print('including all trials in one partition')
            include[expt] = np.ones((ntrial,),dtype='bool')
        if not isinstance(include[expt],list):
            include[expt] = [include[expt]]
        npart = len(include[expt])
#         if isinstance(include[expt],list):
#             tuning[iexpt] = [None for ipart in range(npart)]
        #condition_list = []
        # args to gen_condition_list
        # ti
# selector: dict where each key is a param in ti.keys(), and each value is either a callable returning a boolean,
# to be applied to ti[param], or an input to the function filter_selector
# filter selector: if filter_selector(selector[param]), the tuning curve will be separated into the unique elements of ti[param]. 
        condition_list,_ = gen_condition_list(trial_info[expt],selector,filter_selector=np.logical_not)
        iconds,uconds = zip(*[pd.factorize(c,sort=True) for c in condition_list])
        nconds = [len(u) for u in uconds]
        for ipart in range(npart):
            tip = np.zeros((nroi,)+tuple(nconds))
            for iflat in range(np.prod(nconds)):
                coords = np.unravel_index(iflat,tuple(nconds))
                lkat = k_and(include[expt][ipart],*[iconds[ic] == coords[ic] for ic in range(len(condition_list))])
                tip[(slice(None),)+coords] = np.nanmean(trialwise.loc[:,lkat],-1)
            shp = [np.arange(s) for s in tip.shape[1:]]
            column_labels = pd.MultiIndex.from_product(shp,names=params[1:])
            index = pd.MultiIndex.from_tuples([(expt,ipart,ii) for ii in range(tip.shape[0])],names=['session_id','partition','roi_index'])
            #tip_df = pd.DataFrame(tip.reshape((tip.shape[0],-1)),index=np.arange(tip.shape[0]),columns=column_labels)
            tip_df = pd.DataFrame(tip.reshape((tip.shape[0],-1)),index=index,columns=column_labels)
            #tip_df['partition'] = ipart
            #tip_df['session_id'] = expt
            #tip_df['area'] = trial_info[expt]['area']
            tuning = tuning.append(tip_df)
    return tuning
