#!/usr/bin/env python

import h5py
import scipy.io as sio
import os
import numpy as np

def return_dims(fname):
    # return size of sbx file, with indices (t,c,z,y,x), skipping the first (incomplete) frame
    szbytes = os.path.getsize(fname+'.sbx')
    info = sio.loadmat(fname+'.mat',squeeze_me=True)['info']
    chan = int(info['channels'])
    rpb = int(info['recordsPerBuffer'])
    sz = info['sz'][()]
    if chan is 1:
        factor = 1
        nchan = 2
    elif chan is 2:
        factor = 2
        nchan = 1
    elif chan is 3:
        factor = 2
        nchan = 1
    try:
        zno = info['otparam'][-1]
    except:
        zno = 1
    return (int(szbytes/rpb/sz[1]*factor/4/zno)-1,nchan,zno,sz[0],sz[1])

def gen_hdf5(fname):
    dims = return_dims(fname)
    framesz = np.prod(dims[1:]) # each digit is 2 bytes, by #channels,#planes,#rows,#columns
    with h5py.File(fname+'.hdf5','w') as f:
        raw = f.create_dataset('raw_frames',dims,dtype='<u2')
#         frame = np.zeros(dims[1:],dtype='<u2')
        with open(fname+'.sbx','rb') as fsbx:
            fsbx.seek(2*framesz) # skip the first (incomplete) frame
            for t in range(dims[0]):
                raw[t] = np.fromfile(fsbx,dtype='<u2',count=framesz).reshape(dims[1:])
#                 raw[t] = frame
                print(t)

def load_sbx(fname):
    dims = return_dims(fname)
    framesz = np.prod(dims[1:]) # #channels,#planes,#rows,#columns
    raw = np.zeros(dims,dtype='<u2')
    with open(fname+'.sbx','rb') as fsbx:
        fsbx.seek(2*framesz) # each digit is 2 bytes; skip the first (incomplete) frame
        for t in range(dims[0]):
            raw[t] = np.fromfile(fsbx,dtype='<u2',count=framesz).reshape(dims[1:])
            print(t)
    return raw
