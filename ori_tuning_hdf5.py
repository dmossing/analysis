#!/usr/bin/env python

# before running this script, need to have converted 2P videos to hdf5 format.

import h5py
import numpy as np
import sys
import glob
import LFutils as ut
import os
import tkinter as tk
from tkinter import filedialog
import scipy.io as sio
import shutil

filenames = glob.glob('frames/*.dat')
filenames.sort()

N = len(filenames)
(L1,L2) = (2048,2048)

tgt_fname = input('HDF5 filename: ')
if not tgt_fname.endswith('.hdf5'):
    tgt_fname = tgt_fname+'.hdf5'
tgt_fold = os.path.dirname(tgt_fname)
if tgt_fold:
    tgt_fold = tgt_fold + '/'
    if not os.path.isdir(tgt_fold):
        os.mkdir(tgt_fold)
    os.chdir(tgt_fold)
print(tgt_fname)
assert(not os.path.exists(tgt_fname))

# save LF acquisition params
f = h5py.File(tgt_fname,'w')
g = f.create_group('ori_tuning')
g = f['ori_tuning']
g['date'] = input('date: ')
g['frame_rate'] = float(input('frame rate (Hz): '))
g['animalid'] = input('animal ID: ')
g['stim_type'] = input('stim delivery: ')
g['notes'] = input('notes: ')
g['depth'] = input('depth (um): ')
# save LF stim timing
stim_frames = [int(line.strip()) for line in open('stims.txt')]
glf = g.create_group('LF')
glf['stim_frames'] = np.asarray(stim_frames)

# save LF stim orientations
print('select stim file')
root = tk.Tk()
root.withdraw()
filepath = filedialog.askopenfilename()
matfile = sio.loadmat(filepath,squeeze_me=True)
result = matfile['result']
glf['stim_orientations'] = result['stimParams'][()][0]
raw_fname = 'rawLF.hdf5'
with h5py.File(tgt_fold+raw_fname,'w') as fraw:
    for attr in ['date','animalid','notes']:
        fraw[attr] = g[attr]
    frames = fraw.create_dataset('frames',(N,L1,L2),dtype=np.uint16)
    #glf.create_dataset('raw',(N,L1,L2),dtype=np.uint16)
    for j,name in enumerate(filenames):
        with open(name,'rb') as fid:
            frame = np.fromfile(fid,dtype=np.uint16).reshape((2048,2048))
        frames[j] = frame
        print(j)
glf['raw_frames'] = h5py.ExternalLink(raw_fname,tgt_fold)

g2p = g.create_group('2P')

# save 2P stim timing
print('select 2P .mat file')
root = tk.Tk()
root.withdraw()
filepath = filedialog.askopenfilename()
matfile = sio.loadmat(filepath,squeeze_me=True)
info = matfile['info']
g2p['stim_frames'] = info['frame'][()]-2
# -1 from matlab indexing -1 from skipping the first frame

# save 2P stim orientations
print('select 2P stim file')
root = tk.Tk()
root.withdraw()
filepath = filedialog.askopenfilename()
matfile = sio.loadmat(filepath,squeeze_me=True)
result = matfile['result']
g2p['stim_orientations'] = result['stimParams'][()][0]

# save 2P hdf5 link
print('select 2P .hdf5 file')
root = tk.Tk()
root.withdraw()
filepath2p = filedialog.askopenfilename()
foldname2p,filename2p = os.path.split(filepath2p)
if not filename2p.contains('.hdf5'):
    print('not an HDF5 file, please fix')
f2p = open(tgt_fold+filename2p,'a')
f2p.close()
g2p['raw_frames'] = h5py.ExternalLink(filename2p,tgt_fold)

gz = g.create_group('Z')
print('select Z .hdf5 file')
root = tk.Tk()
root.withdraw()
filepathZ = filedialog.askopenfilename()
foldnameZ,filenameZ = os.path.split(filepathZ)
if not filenameZ.contains('.hdf5'):
    print('not an HDF5 file, please fix')
fz = open(tgt_fold+filenameZ,'a')
fz.close()
gz['raw_frames'] = h5py.ExternalLink(filenameZ,tgt_fold)

try:
    shutil.copyfile(filepath2p,tgt_fold+filename2p)
try:
    shutil.copyfile(filepathZ,tgt_fold+filenameZ)

with h5py.File(tgt_fold+filenameZ,'r') as fz:
    data = fz['raw_frames']
    zm = gz.create_dataset('mean_stack',data.shape[:-1],dtype=np.float64)
    for z in range(zm.shape[0]):
        zm[z] = data[z].astype(np.float64).mean(2)
    gz['depths'] = fz['depths']

f.close()
