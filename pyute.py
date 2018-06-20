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

def norm01(arr,dim=1):
    # normalize each row of arr to [0,1]
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

def gen_trialwise(datafiles,nbefore=0,nafter=0,blcutoff=5,blspan=3000,ds=10,rg=None):
    def tack_on(to_add,trialwise,ctrialwise,strialwise,dfof):
        to_add[np.isnan(to_add)] = 0
        baseline = sfi.percentile_filter(to_add[:,::ds],blcutoff,(1,int(blspan/ds)))
        baseline = np.repeat(baseline,ds,axis=1)
        if baseline.shape[1]>to_add.shape[1]:
            baseline = baseline[:,:to_add.shape[1]]
        c = np.zeros_like(to_add)
        s = np.zeros_like(to_add)
        this_dfof = np.zeros_like(to_add)
        for i in range(c.shape[0]):
            this_dfof[i] = (to_add[i]-baseline[i,:])/baseline[i,:]
            c[i],s[i],_,_,_  = deconvolve(this_dfof[i],penalty=1)
            to_add[np.isnan(to_add)] = 0
            to_add = trialize(to_add,frm,nbefore,nafter)
        c = trialize(c,frm,nbefore,nafter)
        s = trialize(s,frm,nbefore,nafter)
        try:
            trialwise = np.concatenate((trialwise,to_add),axis=0)
            ctrialwise = np.concatenate((ctrialwise,c),axis=0)
            strialwise = np.concatenate((strialwise,s),axis=0)
            dfof = np.concatenate((dfof,this_dfof),axis=0)
        except:
            trialwise = to_add.copy()
            ctrialwise = c.copy()
            strialwise = s.copy()
            dfof = this_dfof.copy()
        return trialwise,ctrialwise,strialwise,dfof
        
    trialwise = np.array(())
    ctrialwise = np.array(())
    strialwise = np.array(())
    dfof = np.array(())
    try:
        for datafile in datafiles:
            if not rg is None:
                frm = sio.loadmat(datafile.replace('.rois','.mat'),squeeze_me=True)['info']['frame'][()][rg[0]:rg[1]]
            else:
                frm = sio.loadmat(datafile.replace('.rois','.mat'),squeeze_me=True)['info']['frame'][()]
            to_add = sio.loadmat(datafile,squeeze_me=True)['corrected']
            trialwise,ctrialwise,strialwise,dfof = tack_on(to_add,trialwise,ctrialwise,strialwise,dfof)   
    except:
        for datafile in datafiles:
            if not rg is None:
                frm = sio.loadmat(datafile.replace('.rois','.mat'),squeeze_me=True)['info']['frame'][()][rg[0]:rg[1]]
            else:
                frm = sio.loadmat(datafile.replace('.rois','.mat'),squeeze_me=True)['info']['frame'][()]
            with h5py.File(datafile,mode='r') as f:
                to_add = f['corrected'][:].T[:,1:]
                print(to_add.shape)
                trialwise,ctrialwise,strialwise,dfof = tack_on(to_add,trialwise,ctrialwise,strialwise,dfof)   
   #             baseline = sfi.percentile_filter(to_add[:,::ds],blcutoff,(1,int(blspan/ds)))
   #             baseline = np.repeat(baseline,ds,axis=1)
   #             if baseline.shape[1]>to_add.shape[1]:
   #                 baseline = baseline[:,:to_add.shape[1]]
   #             c = np.zeros_like(to_add)
   #             s = np.zeros_like(to_add)
   #             to_add[np.isnan(to_add)] = 0
   #             for i in range(c.shape[0]):
   #                 dfof = (to_add[i]-baseline[i,:])/baseline[i,:]
   #                 try:
   #                     c[i],s[i],_,_,_  = deconvolve(dfof,penalty=1)
   #                 except:
   #                     print("in "+datafile+" couldn't do "+str(i))
   #             to_add = trialize(to_add,frm,nbefore,nafter)
   #             c = trialize(c,frm,nbefore,nafter)
   #             s = trialize(s,frm,nbefore,nafter)
   #             try:
   #                 trialwise = np.concatenate((trialwise,to_add),axis=0)
   #                 ctrialwise = np.concatenate((ctrialwise,c),axis=0)
   #                 strialwise = np.concatenate((strialwise,s),axis=0)
   #             except:
   #                 trialwise = to_add.copy()
   #                 ctrialwise = c.copy()
   #                 strialwise = s.copy()
    return trialwise,ctrialwise,strialwise,dfof

def fit_2d_gaussian(locs,ret,verbose=False):
    
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
    
    xx,yy = np.meshgrid(locs[0],locs[1])
    x = xx.flatten()
    y = yy.flatten()
    initial_guess = (0,0,ret[0].max(),10,10,0,0)
    params = np.zeros((ret.shape[0],)+(len(initial_guess),))
    sqerror = np.zeros((ret.shape[0],))
    for i in range(ret.shape[0]):
        try:
            data = ret[i].flatten()
            initial_guess = (0,0,data.max(),10,10,0,0)
            popt, pcov = sop.curve_fit(twoD_Gaussian, (x,y), data, p0 = initial_guess)
            modeled = twoD_Gaussian((x,y),*popt)
            sqerror[i] = ((modeled-data)**2/popt[2]**2).sum()
            params[i] = popt
        except:
            if verbose:
                print("couldn't do "+str(i))
    paramdict = {}
    paramdict['sqerror'] = sqerror
    paramdict['xo'] = params[:,0]
    paramdict['yo'] = params[:,1]
    paramdict['amplitude'] = params[:,2]
    paramdict['sigma_x'] = params[:,3]
    paramdict['sigma_y'] = params[:,4]
    paramdict['theta'] = params[:,5]
    paramdict['offset'] = params[:,6]
    return paramdict
