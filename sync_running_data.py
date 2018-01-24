#!/usr/bin/env python

import numpy as np
import os
import glob
import scipy.io as sio
import sys

def get_run_signal(matfile_running,matfile_stiminfo):
    def resample(signal1,trig1,trig2):
        assert(trig1.sum()==trig2.sum())
        frametrig1 = np.where(trig1)[0]
        frametrig2 = np.where(trig2)[0]
        signal2 = np.zeros_like(trig2)
        for i,tr in enumerate(frametrig1[:-1]):
            ptno1 = frametrig1[i+1]-frametrig1[i]
            ptno2 = frametrig2[i+1]-frametrig2[i]
            signal2[frametrig2[i]:frametrig2[i+1]] = np.interp(np.linspace(0,1,ptno2),np.linspace(0,1,ptno1),signal1[frametrig1[i]:frametrig1[i+1]])
        return signal2[frametrig2[0]:frametrig2[-1]]
    matfile = sio.loadmat(matfile_running)
    dxdt = matfile['dx_dt'][:,0]
    trigrun = matfile['stim_trigger']
    info = sio.loadmat(matfile_stiminfo)['info']
    frame2p = info[0]['frame'][0][:,0]
    trig2p = np.zeros((frame2p.max()+1,),dtype=int)
    trig2p[frame2p] = 1
    dxdt2p = resample(dxdt,trigrun,trig2p)
    roifile = matfile_stiminfo.split('.mat')[0]+'.rois'
    with h5py.File(roifile,'r') as f:
        traces = f['Data'][()].T
        traces2p = traces[:,frame2p[0]:frame2p[-1]]
#    traces = sio.loadmat(matfile_stiminfo.replace('.mat','_ftotal.mat'))['ftotal']
    traces2p = traces[frame2p[0]:frame2p[-1]]
    return dxdt2p,traces2p

if __name__ == "__main__":
    pattern2p = sys.argv[1]
    print('2P file regex: '+pattern2p)
    patternrun = sys.argv[2]
    print('running file regex: '+patternrun)
    fnames2p = glob.glob(pattern2p)
    fnames2p.sort()
    fnamesrun = glob.glob(patternrun)
    fnamesrun.sort() 
    for namerun,name2p in zip(fnamesrun,fnames2p):
        print('running file: '+namerun)
        print('trace file: '+name2p)
        dx_dt,traces = get_run_signal(namerun,name2p)
        sio.savemat(namerun.replace('.mat','_2p.mat'),{'dx_dt':dx_dt,'ftotal':traces})
