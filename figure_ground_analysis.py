#!/usr/bin/env python

import os
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import h5py
from oasis.functions import deconvolve
from oasis import oasisAR1, oasisAR2
import pyute as ut

from importlib import reload
import scipy.ndimage.filters as sfi
import scipy.stats as sst
import scipy.ndimage.measurements as snm
from mpl_toolkits.mplot3d import Axes3D


def analyze_figure_ground(datafiles,stimfile,retfile=None,frame_adjust=None,nbefore=4,nafter=4):
    nbydepth = get_nbydepth(datafiles)
    trialwise,ctrialwise,strialwise,dfof,straces = ut.gen_precise_trialwise(datafiles,frame_adjust=frame_adjust)
    print(strialwise.shape)
    zstrialwise = sst.zscore(strialwise.reshape((strialwise.shape[0],-1)).T).T.reshape(strialwise.shape)
    
    result = sio.loadmat(stimfile,squeeze_me=True)['result'][()]
    
    infofile = sio.loadmat(datafiles[0][:-12]+'.mat',squeeze_me=True)
    frame = infofile['info'][()]['frame'][()]
    if frame_adjust:
        print('adjusted')
        frame = frame_adjust(frame)
    if np.remainder(frame.shape[0],2):
        print('deleted one')
        frame = frame[:-1]
    
    data = strialwise #[:,:,nbefore:-nafter]
    
    try:
        dxdt = sio.loadmat(datafiles[1],squeeze_me=True)['dxdt']
    except:
        with h5py.File(datafiles[1],mode='r') as f:
            dxdt = f['dxdt'][:].T
            
    trialrun = np.zeros(frame[0::2].shape)
    for i in range(len(trialrun)):
        trialrun[i] = dxdt[frame[0::2][i]:frame[1::2][i]].mean()
    runtrial = trialrun>100

    pval = np.zeros(strialwise.shape[0])
    for i in range(strialwise.shape[0]):
        _,pval[i] = sst.ttest_rel(strialwise[i,:,nbefore-1],strialwise[i,:,nbefore+1])
        
    stimparams = result['stimParams']

    order = ['ctrl','fig','grnd','iso','cross']
    norder = len(order)
    ori = stimparams[0]
    sz = stimparams[1]
    figContrast = stimparams[-2]
    grndContrast = stimparams[-1]

    paramdict = {}
    paramdict['ctrl'] = np.logical_and(figContrast==0,grndContrast==0)
    paramdict['fig'] = np.logical_and(figContrast==1,grndContrast==0)
    paramdict['grnd'] = np.logical_and(np.logical_and(figContrast==0,grndContrast==1),sz>0)
    paramdict['iso'] = sz==0
    paramdict['cross'] = np.logical_and(figContrast==1,grndContrast==1)
        
    indexlut,stimp = np.unique(stimparams,axis=1,return_inverse=True)

    angle = stimparams[0]
    size = stimparams[1]
    contrast = stimparams[4]

    ucontrast = np.unique(contrast)
    uangle = np.unique(angle)
    usize = np.unique(size)
    ncontrast = len(ucontrast)
    nangle = len(uangle)
    nsize = len(usize)

    angle180 = np.remainder(angle,180)
    uangle180 = np.unique(angle180)
    nangle180 = len(uangle180)

    Smean = np.zeros((strialwise.shape[0],norder,nangle180,strialwise.shape[2]))
    Stavg = np.zeros((strialwise.shape[0],norder,nangle180,int(strialwise.shape[1]/nangle/norder)))

    Strials = {}
    Sspont = {}
    print(runtrial.shape)
    for i,name in enumerate(order):
        for j,theta in enumerate(uangle180):
            lkat = np.logical_and(runtrial,np.logical_and(angle180==theta,paramdict[name]))
            if lkat.sum()==1:
                print('problem')
            Smean[:,i,j,:] = data[:,lkat,:].mean(1)
            Strials[(i,j)] = data[:,lkat,nbefore:-nafter].mean(2)
            Sspont[(i,j)] = data[:,lkat,:nbefore].mean(2)

    lb = np.zeros((strialwise.shape[0],norder,nangle180))
    ub = np.zeros((strialwise.shape[0],norder,nangle180))

    for i in range(norder):
        print(i)
        for j in range(nangle180):
            lb[:,i,j],ub[:,i,j] = ut.bootstrap(Strials[(i,j)],np.mean,axis=1,pct=(16,84))
            # mn[:,i,j,k] = np.nanmean(Strials[(i,j,k)],axis=1)

    pval_fig = np.zeros((strialwise.shape[0],nangle180))
    for j,theta in enumerate(uangle180):
        print(theta)
        figind = int(np.where(np.array([x=='fig' for x in order]))[0])
        _,pval_fig[:,j] = sst.ttest_rel(Sspont[(figind,j)],Strials[(figind,j)],axis=1)
        
    pval_grnd = np.zeros((strialwise.shape[0],nangle180))
    for j,theta in enumerate(uangle180):
        print(theta)
        grndind = int(np.where(np.array([x=='grnd' for x in order]))[0])
        _,pval_grnd[:,j] = sst.ttest_rel(Sspont[(grndind,j)],Strials[(grndind,j)],axis=1)
                
    Savg = np.nanmean(np.nanmean(Smean[:,:,:,nbefore:-nafter],axis=-1),axis=2)

    Storiavg = Stavg.mean(1)
    # _,pval = sst.ttest_ind(Storiavg[:,0,-1].T,Storiavg[:,0,0].T)

    #suppressed = np.logical_and(pval<0.05,Savg[:,0,-1]<Savg[:,0,0])
    #facilitated = np.logical_and(pval<0.05,Savg[:,0,-1]>Savg[:,0,0])

    return Savg,Smean,lb,ub,pval_fig,pval_grnd,trialrun

def get_nbydepth(datafiles):
    nbydepth = np.zeros((len(datafiles),))
    for i,datafile in enumerate(datafiles):
        with h5py.File(datafile,mode='r') as f:
            nbydepth[i] = (f['corrected'][:].T.shape[0])
    return nbydepth

def analyze_everything(folds,files,rets,adjust_fns):
    soriavg = {}
    strialavg = {}
    lb = {}
    ub = {}
    pval_fig = {}
    pval_grnd = {}
    nbydepth = {}
    ret_vars = {}
    trialrun = {}
    datafoldbase = '/home/mossing/scratch/2Pdata/'
    datafoldbase_old = '/home/mossing/excitation/2P/'
    stimfoldbase = '/home/mossing/scratch/visual_stim/'
    for thisfold,thisfile,retnumber,frame_adjust in zip(folds,files,rets,adjust_fns):
        if thisfold[:2]=='18':
             datafold = datafoldbase+thisfold+'ot/'
        else:
             datafold = datafoldbase_old+thisfold+'ot/'
        datafiles = [thisfile+'_ot_'+number+'.rois' for number in ['000','001','002','003']]

        stimfold = stimfoldbase+thisfold
        stimfile = thisfile+'.mat'

        datafiles = [datafold+file for file in datafiles]
        stimfile = stimfold+stimfile
        retfile = datafoldbase+thisfold+'retinotopy_{}.mat'.format(retnumber)

        nbefore = 4
        nafter = 4
        soriavg[thisfold],strialavg[thisfold],lb[thisfold],ub[thisfold],pval_fig[thisfold],pval_grnd[thisfold],trialrun[thisfold] = analyze_figure_ground(datafiles,stimfile,retfile,frame_adjust)
        try: 
            ret_vars[thisfold] = sio.loadmat(retfile,squeeze_me=True)#['ret'][:]
            print('loaded retinotopy file')
            ret_vars[thisfold]['position'] = sio.loadmat(stimfile,squeeze_me=True)['result'][()]['position']
        except:
            print('retinotopy not saved for '+thisfile)
            ret_vars[thisfold] = None
        nbydepth[thisfold] = get_nbydepth(datafiles)
    return soriavg,strialavg,lb,ub,pval_fig,pval_grnd,nbydepth,ret_vars,trialrun
