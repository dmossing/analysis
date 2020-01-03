#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np
import sys
# option to import from github folder
sys.path.insert(0, '/home/mossing/code/downloads/s2p_github')
import suite2p
from suite2p.run_s2p import run_s2p
from suite2p import utils
from suite2p.io.tiff import tiff_to_binary
from suite2p.io import save
from suite2p.registration import register
from suite2p.detection import chan2detect
import timeit
import os
import shutil
import pdb
from importlib import reload
reload(suite2p.run_s2p)
reload(suite2p)
import glob

fast_disk = '/home/mossing/data_ssd/suite2P/bin'

# set your options for running
# overwrites the run_s2p.default_ops
ops = {
        'fast_disk': fast_disk, # used to store temporary binary file, defaults to save_path0 (set as a string NOT a list)
        'save_path0': [], # stores results, defaults to first item in data_path
        'delete_bin': True, # whether to delete binary file after processing
        # main settings
        'nplanes' : 4, # each tiff has these many planes in sequence
        'nchannels' : 1, # each tiff has these many channels per plane
        'functional_chan' : 1, # this channel is used to extract functional ROIs (1-based)
        'diameter':15, # this is the main parameter for cell detection, 2-dimensional if Y and X are different (e.g. [6 12])
        'tau':  1., # this is the main parameter for deconvolution
        'fs': 7.75,  # sampling rate (total across planes)
        # output settings
        'save_mat': True, # whether to save output as matlab files
        'combined': True, # combine multiple planes into a single result /single canvas for GUI
        # parallel settings
        'num_workers': 0, # 0 to select num_cores, -1 to disable parallelism, N to enforce value
        'num_workers_roi': -1, # 0 to select number of planes, -1 to disable parallelism, N to enforce value
        # registration settings
        'do_registration': True, # whether to register data
        'nimg_init': 200, # subsampled frames for finding reference image
        'batch_size': 500, # number of frames per batch
        'maxregshift': 0.05, # max allowed registration shift, as a fraction of frame max(width and height)
        'align_by_chan' : 1, # when multi-channel, you can align by non-functional channel (1-based)
        'reg_tif': True, # whether to save registered tiffs
        'subpixel' : 10, # precision of subpixel registration (1/subpixel steps)
        # cell detection settings
        'connected': True, # whether or not to keep ROIs fully connected (set to 0 for dendrites)
        'navg_frames_svd': 5000, # max number of binned frames for the SVD
        'nsvd_for_roi': 2000, # max number of SVD components to keep for ROI detection
        'max_iterations': 25, # maximum number of iterations to do cell detection
        'ratio_neuropil': 6., # ratio between neuropil basis size and cell radius
        'ratio_neuropil_to_cell': 3, # minimum ratio between neuropil radius and cell radius
        'tile_factor': 1., # use finer (>1) or coarser (<1) tiles for neuropil estimation during cell detection
        'threshold_scaling': 1., # adjust the automatically determined threshold by this scalar multiplier
        'max_overlap': 0.75, # cells with more overlap than this get removed during triage, before refinement
        'inner_neuropil_radius': 2, # number of pixels to keep between ROI and neuropil donut
        'outer_neuropil_radius': np.inf, # maximum neuropil radius
        'min_neuropil_pixels': 350, # minimum number of pixels in the neuropil
        # deconvolution settings
        'baseline': 'maximin', # baselining mode
        'win_baseline': 60., # window for maximin
        'sig_baseline': 10., # smoothing constant for gaussian filter
        'prctile_baseline': 8.,# optional (whether to use a percentile baseline)
        'neucoeff': .7,  # neuropil coefficient
      }


def process_data(animalid,date,expt_ids,raw_base='/home/mossing/data/suite2P/raw/',result_base='/home/mossing/data/suite2P/results/',fast_disk='/home/mossing/data_ssd/suite2P/bin',nchannels=1,delete_raw=False,diameter=15):
#    save_path0 = result_base+animalid+'/'+date+'/'+'_'.join(expt_ids)
#    data_path = [raw_base+animalid+'/'+date+'/'+lbl for lbl in expt_ids]

    db = prepare_db(animalid,date,expt_ids,raw_base=raw_base,result_base=result_base,fast_disk=fast_disk,nchannels=nchannels,diameter=diameter)

    # provide an h5 path in 'h5py' or a tiff path in 'data_path'
    # db overwrites any ops (allows for experiment specific settings)
    #db = {
    #      'h5py': [], # a single h5 file path
    #      'h5py_key': 'data',
    #      'look_one_level_down': False, # whether to look in ALL subfolders when searching for tiffs,
    #      'save_path0': save_path0,
    #      'data_path': data_path, # a list of folders with tiffs 
    #                                             # (or folder of folders with tiffs if look_one_level_down is True, or subfolders is not empty)                                     
    #      'subfolders': [], # choose subfolders of 'data_path' to look in (optional)
    #      'fast_disk': fast_disk, # string which specifies where the binary file will be stored (should be an SSD)
    #      'nchannels': nchannels,
    #      'diameter': diameter
    #    }

    try:
        shutil.rmtree(fast_disk+'/suite2p')
        print('fast disk contents deleted')
    except:
        print('fast disk location empty')

    opsEnd=run_s2p(ops=ops,db=db)
    if delete_raw:
        for fold in db['data_path']:
            for old_file in glob.glob(fold + '/*.tif'):
                os.remove(old_file)

def process_data_1ch_2ch(animalid,date,expt_ids_1ch,expt_ids_2ch,raw_base='/home/mossing/data/suite2P/raw/',result_base='/home/mossing/data/suite2P/results/',fast_disk='/home/mossing/data_ssd/suite2P/bin',delete_raw=False,diameter=15):
#    save_path0 = result_base+animalid+'/'+date+'/'+'_'.join(expt_ids)
#    data_path = [raw_base+animalid+'/'+date+'/'+lbl for lbl in expt_ids]

    # process 1ch data first
    db = prepare_db(animalid,date,expt_ids_1ch,raw_base=raw_base,result_base=result_base,fast_disk=fast_disk,nchannels=1,diameter=diameter,combined=False)

    try:
        shutil.rmtree(fast_disk+'/suite2p')
        print('fast disk contents deleted')
    except:
        print('fast disk location empty')

    # run suite2p part
    opsEnd=run_s2p(ops=ops,db=db)
    if delete_raw:
        for fold in db['data_path']:
            for old_file in glob.glob(fold + '/*.tif'):
                os.remove(old_file)

    add_2ch_data(animalid,date,expt_ids_1ch,expt_ids_2ch,raw_base=raw_base,result_base=result_base,fast_disk=fast_disk,delete_raw=delete_raw,diameter=diameter)
    # do stuff with 2ch data and combine everything together

def add_2ch_data(animalid,date,expt_ids_1ch,expt_ids_2ch,raw_base='/home/mossing/data/suite2P/raw/',result_base='/home/mossing/data/suite2P/results/',fast_disk='/home/mossing/data_ssd/suite2P/bin',delete_raw=False,diameter=15):

    mouse = animalid
    blk  = expt_ids_1ch #int(dat[mouse]['blk'][k])
    blk_ = '_'.join(blk)
    redblk  = expt_ids_2ch #int(dat[mouse]['redblk'][k])
    redblk_ = '_'.join(redblk)
            
    # green experiment save_path0
    froot = result_base+'%s/%s/%s/' % (mouse, date, blk_)
    print(froot)
    ops1 = np.load(os.path.join(froot, "suite2p/ops1.npy"),allow_pickle=True)
    
    # 2-channel experiment raw data saved here
    data_path = [raw_base+'%s/%s/%s/' % (mouse, date, rb) for rb in redblk]
    print(data_path[0])
    ops = ops1[0].copy()
    ops['fast_disk'] = []
    ops['save_path0'] = fast_disk # save binary locally on SSD
    ops['data_path'] = data_path
    ops['nchannels'] = 2
    ops['functional_chan'] = 1
    
    # compute binaries for 2 channels       
    ops2 = tiff_to_binary(ops)

    
    for iplane in range(0,len(ops1)):
        #ops2 = np.load("/media/carsen/SSD/BIN/suite2p/plane%d/ops.npy"%iplane).item()
        #opsOut = register.register_binary(ops2, ops1[iplane]['refImg'])
        if 'yoff' in ops2[iplane]:
            del ops2[iplane]['yoff']
        
        opsOut = register.register_binary(ops2[iplane], ops1[iplane]['refImg'])
        ops1[iplane]['meanImg_chan2'] = opsOut['meanImg_chan2']
        stat = np.load(os.path.join(froot, "suite2p/plane%d/stat.npy"%iplane), allow_pickle=True)
        ops1[iplane]['chan2_thres'] = 0.65
        ops1[iplane],redcell = chan2detect.detect(ops1[iplane],stat)
        np.save(os.path.join(froot, "suite2p/plane%d/ops.npy"%iplane), ops1[iplane], allow_pickle=True)
        np.save(os.path.join(froot, "suite2p/plane%d/redcell.npy"%iplane), redcell, allow_pickle=True)
        ops1[iplane]['save_path'] = os.path.join(froot, "suite2p/plane%d"%iplane)
        ops1[iplane]['save_path0'] = os.path.join(froot)
    save.combined(ops1) 

    if delete_raw:
        for fold in data_path:
            for old_file in glob.glob(fold + '/*.tif'):
                os.remove(old_file)
    
def prepare_db(animalid,date,expt_ids,raw_base='/home/mossing/data/suite2P/raw/',result_base='/home/mossing/data/suite2P/results/',fast_disk='/home/mossing/data_ssd/suite2P/bin',nchannels=1,diameter=15,combined=True):
    save_path0 = result_base+animalid+'/'+date+'/'+'_'.join(expt_ids)
    data_path = [raw_base+animalid+'/'+date+'/'+lbl for lbl in expt_ids]

    # provide an h5 path in 'h5py' or a tiff path in 'data_path'
    # db overwrites any ops (allows for experiment specific settings)
    db = {
          'h5py': [], # a single h5 file path
          'h5py_key': 'data',
          'look_one_level_down': False, # whether to look in ALL subfolders when searching for tiffs,
          'save_path0': save_path0,
          'data_path': data_path, # a list of folders with tiffs 
                                                 # (or folder of folders with tiffs if look_one_level_down is True, or subfolders is not empty)                                     
          'subfolders': [], # choose subfolders of 'data_path' to look in (optional)
          'fast_disk': fast_disk, # string which specifies where the binary file will be stored (should be an SSD)
          'nchannels': nchannels,
          'diameter': diameter,
        'combined': combined
        }

    return db


## In[15]:
#
#animalid = ['M0090','M0090','M0002','M0002','M10368','M0094']
#date = ['190321','190326','190503','190411','190506','190508']
#expt_ids = [['1','2','3'],['2','3','4'],['5','6','7','8'],['4','5','6'],['2','3','4'],['2','3','4','5']]
#for i in range(1,2):
#    process_data(animalid[i],date[i],expt_ids[i],nchannels=2)
##for i in range(3,6):
##    process_data(animalid[i],date[i],expt_ids[i],nchannels=1)
#
#
#
#def main(animalid,date,expt_ids):
#    process_data
#
#if __name__ == "__main__":
#    main()	
