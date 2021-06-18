#!/usr/bin/env python

import matplotlib.pyplot as plt
import autograd.numpy as np
import os
import scipy.io.wavfile as siw
import scipy.signal as ssi
from autograd import grad
import scipy.optimize as sop
import pyute as ut
import scipy.interpolate as sip
import song_conversion_script as scs
import glob

def run_fitting(wavfile):
    print('running trace3d fit...')
    fs,data = siw.read(wavfile)
    freq,t,spec = ssi.spectrogram(data)
    nt = t.shape[0]
    spec = spec/np.max(np.mean(spec,axis=0))
    minpad = 1000
    maxlen = 2000
    npad = minpad + np.argmax(ssi.convolve(np.sum(spec[:,minpad:],0),np.ones((maxlen,)),mode='valid'))
    endat = npad+maxlen#np.minimum(npad+maxlen,nt-npad)
    rg = slice(npad,endat)
    ntpad = endat-npad
    freq_min = 5
    def trace3d_to_spec(freq,t,trace,fsigma,fudge=1e-6):
        amp = trace[:,0,:]
        fmu = trace[:,1,:]
#         fsigma = trace[:,2,:]
        dfreq = freq[np.newaxis,:,np.newaxis] - fmu[:,np.newaxis,:]
        fs2 = fsigma[:,np.newaxis,np.newaxis]**2 + fudge
#         print((dfreq.max(),fmu.mean(),fsigma.mean()))
        return np.sum(amp[:,np.newaxis,:]*np.exp(-0.5*dfreq**2/fs2)/np.sqrt(2*np.pi*fs2),axis=0)
    def parse_trace(trace):
        this_trace = np.reshape(trace[:-nharmonics],(nharmonics,ndim,ntpad))
        fsigma = trace[-nharmonics:]
        return this_trace,fsigma
    def cost_lsq(trace,fsigma):
        spec_modeled = trace3d_to_spec(freq,t[rg],trace,fsigma)
        return np.sum((spec[freq_min:,npad:endat] - spec_modeled[freq_min:])**2)
    def cost_tv_l1(trace):
        return np.sum(np.abs(np.diff(trace,axis=2)),axis=2)
    def cost_tv_l2(trace):
        return np.sum(np.abs(np.diff(trace,axis=2))**2,axis=2)
    def cost_l1(trace):
        return np.sum(np.abs(trace),axis=2)
    # lam_tv_l1 = np.array((1,0,0))
    # lam_tv_l2 = np.array((0,10000,10000))
    # lam_l1 = np.array((0.1,0,0))
    lam_tv_l1 = np.array((1e2,0))
    lam_tv_l2 = np.array((0,3e5))
    lam_l1 = np.array((1e2,0))
    def cost_total(trace):
        this_trace,fsigma = parse_trace(trace)
        tv_l1_term = np.sum(lam_tv_l1[np.newaxis,:]*cost_tv_l1(this_trace)) 
        tv_l2_term = np.sum(lam_tv_l2[np.newaxis,:]*cost_tv_l2(this_trace))
        l1_term = np.sum(lam_l1[np.newaxis,:]*cost_l1(this_trace))
        lsq_term = cost_lsq(this_trace,fsigma)
        #print((lsq_term,tv_l1_term,tv_l2_term,l1_term))
        return lsq_term + tv_l1_term + tv_l2_term + l1_term
    nharmonics = 1
    ndim = 2
    trace0 = np.zeros((nharmonics,ndim,ntpad))
    fsigma0 = 1*np.mean(np.diff(freq))*np.ones((nharmonics,))
    fsigma0[-1] = 2*fsigma0[-1]
    # trace0[0,2,:] = fsigma0
    trace0[:,0,:] = (np.std(spec[:,rg],axis=0)/np.max(np.std(spec[:,rg],axis=0))*np.max(spec[:,rg]))[np.newaxis,:]*np.sqrt(2*np.pi*fsigma0[:,np.newaxis]**2)
    trace0[:,1,:] = (np.sum(freq[:,np.newaxis]*spec[:,rg],axis=0)/np.sum(spec[:,rg],axis=0))[np.newaxis]
    trace0 = np.concatenate((trace0.flatten(),fsigma0))
    bds = sop.Bounds(lb=np.zeros_like(trace0),ub=np.inf*np.ones_like(trace0))
    # start_time = timeit.default_timer()
    res = sop.minimize(cost_total,trace0,jac=grad(cost_total),bounds=bds)
    trace,fsigma = parse_trace(res.x)
    return trace,fsigma

def run_on_fold(folder):
    file_list = glob.glob(folder+'/*.mp3')
    file_list.sort()
    for filename in file_list:
        print(filename)
        wavfile = scs.run(filename.split('.mp3')[0])
        trace,fsigma = run_fitting(wavfile)
        np.save(wavfile.replace(".wav","_trace3d.npy"),{'trace':trace,'fsigma':fsigma},allow_pickle=True)

