#!/usr/bin/env python

# tools to convert between .sbx, .tif, and .hdf5 formats

import h5py
import scipy.io as sio
import os
import numpy as np
from tifffile import imsave
import sys

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
        zno = info['otparam'][()][-1]
    except:
        zno = 1
    return (int(szbytes/rpb/sz[1]*factor/4/zno)-1,nchan,zno,sz[0],sz[1])

def sbxtohdf5(fname):
    # save sbx to hdf5 dataset f['raw_frames']
    dims = return_dims(fname)
    framesz = np.prod(dims[1:]) # each digit is 2 bytes, by #channels,#planes,#rows,#columns
    with h5py.File(fname+'.hdf5','w') as f:
        raw = f.create_dataset('raw_frames',dims,dtype='<u2')
#         frame = np.zeros(dims[1:],dtype='<u2')
        with open(fname+'.sbx','rb') as fsbx:
            fsbx.seek(2*framesz) # skip the first (incomplete) frame
            for t in range(dims[0]):
                raw[t] = (np.iinfo('<u2').max-np.fromfile(fsbx,dtype='<u2',count=framesz).reshape(dims[2:5]+dims[1:2])).transpose((3,0,1,2))
#                 raw[t] = frame
                print(t)

def loadsbx(fname,frames=None):
    # load an sbx file into RAM, either the entire file or specified frames
    dims = return_dims(fname)
    if frames is not None:
        dims = (len(frames),) + dims[1:]
    framesz = np.prod(dims[1:]) # #channels,#planes,#rows,#columns
    raw = np.zeros(dims,dtype='<u2')
    with open(fname+'.sbx','rb') as fsbx:
        fsbx.seek(2*framesz) # each digit is 2 bytes; skip the first (incomplete) frame
        if frames is not None:
            ctr = 0
            for t,frame in enumerate(frames):
                fsbx.seek(2*framesz*(frame-ctr),1)
                ctr = frame+1
                raw[t] = (np.iinfo('<u2').max-np.fromfile(fsbx,dtype='<u2',count=framesz).reshape(dims[2:5]+dims[1:2])).transpose((3,0,1,2))
        else:
            for t in range(dims[0]):
                raw[t] = (np.iinfo('<u2').max-np.fromfile(fsbx,dtype='<u2',count=framesz).reshape(dims[2:5]+dims[1:2])).transpose((3,0,1,2))
                print(t)
    return raw

def sbxtotiff(fname,chunksize=200):
    # save sbx file to RAM. Due to memory limitations, load 'chunksize' frames at a time
    chunksize = 200
    dims = return_dims(fname)
    startind = np.arange(0,dims[0],chunksize)
    endind = np.minimum(np.arange(chunksize,dims[0]+chunksize,chunksize),dims[0])
    try:
        for i in range(len(startind)):
            print(i)
            raw = np.iinfo('<u2').max-load_sbx(fname,frames=np.arange(startind[i],endind[i]))
            imsave('green_channel.tif',raw[:,0,0],append=True)
            imsave('red_channel.tif',raw[:,1,0],append=True)
    except:
        for i in range(len(startind)):
            print(i)
            raw = loadsbx(fname,frames=np.arange(startind[i],endind[i]))
            imsave('green_channel.tif',raw[:,0,0],append=True)

if __name__=="__main__":
    sbxfile = sys.argv[1]
    if '.sbx' in sbxfile:
        sbxfile = sbxfile.split('.sbx')[0]
    sbxtohdf5(sbxfile)
