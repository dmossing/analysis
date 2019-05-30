import os
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import h5py
from oasis.functions import deconvolve
from oasis import oasisAR1, oasisAR2
import pyute as ut
from importlib import reload
import matplotlib
import scipy.stats as sst
from mpl_toolkits.mplot3d import Axes3D
import retinotopy_analysis as rt
import scipy.optimize as sop
import pdb

def get_ops(thisfold,thisfile,s2p_base='/media/mossing/backup_0/data/suite2P/results/'):
    animalid = thisfile.split('_')[0]
    date = thisfold.split('/')[0]
    analyses = os.listdir('/'.join([s2p_base,animalid,date]))
    assert(len(analyses)==1)
    analysis_fold = analyses[0]
    ops = np.load('/'.join([s2p_base,animalid,date,analysis_fold,'suite2p','ops1.npy']))
    ops = [x for x in ops]
    return ops

def get_msk_properties(thisfold,thisfile,mat_base='/media/mossing/backup_0/data/2P/',nplanes=4):
#     props['redratio'] = [None]*nplanes
    props = [None]*nplanes
    for iplane in range(nplanes):
        props[iplane] = {}
        with h5py.File('/'.join([mat_base,thisfold,'ot',thisfile+'_ot_00'+str(iplane)+'.rois']),mode='r') as matfile:
#             props['redratio'][iplane] = matfile['redratio'][:][:,0]
            props[iplane]['msk'] = matfile['msk'][:].transpose((0,2,1)).astype('bool')
            props[iplane]['ctr'] = matfile['ctr'][:]
    return props
