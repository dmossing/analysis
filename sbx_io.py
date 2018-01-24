#!/usr/bin/env python

import scipy.io as sio
import os
from math import inf
import numpy as np
import h5py

def dimensions(fname):
    # return dimensions of sbx file, with indices (c,x,y,f) or (c,x,y,z,f) in the case of optotune
    info = sio.loadmat(fname+'.mat',squeeze_me=True)['info']
    sz = info['sz'][()]
    chan = int(info['channels'])
    if chan is 1:
        nchan = 2
    elif chan is 2:
        nchan = 1
    elif chan is 3:
        nchan = 1
    rpb = int(info['recordsPerBuffer'])
    szbytes = os.path.getsize(fname+'.sbx')
    nframes = int(szbytes/(nchan*np.prod(sz)*2))
	try:
		zno = info['otparam'][-1]
	except:
		zno = 1
	if zno > 1:
    	return [nchan,sz[1],sz[0],zno,nframes]
	else:
		return [nchan,sz[1],sz[0],nframes]

def loadsbx(fname,frames=[0,inf],invert=True):
    dims = dimensions(fname)
    framesz = np.prod(dims[0:3])
    # determine frames to load
    if frames[-1]==inf:
        del(frames[-1])
        frames.extend(range(frames[-1]+1,dims[-1]))
    numFrames = len(frames)
    # determine number of seek operations
    jumps = np.diff([-1]+frames)-1
    startID = jumps.nonzero()[0]
    if jumps[0]==0:
        startID = np.concatenate((np.array([0]),startID))
    N = np.diff(np.concatenate((startID-1,np.array([numFrames-1]))))
    # load requested frames
    images = np.empty(dims[0:3]+[numFrames],dtype=np.uint16) #initialize output
    with open(fname+'.sbx','rb') as fsbx:
        for idx in range(len(startID)):
            fsbx.seek(2*framesz*frames[startID[idx]])
            images[:,:,:,startID[idx]:startID[idx]+N[idx]] = np.fromfile(fsbx,dtype=np.uint16,count=framesz*N[idx]).reshape(dims[0:3]+[N[idx]],order='F')
    # format output
    images = images.transpose(2,1,0,3)
    if invert:
        images = np.iinfo(images.dtype).max - images
    print('Loaded '+str(numFrames)+'  frames from '+fname+'.sbx')
    return images #return requested frames
    
def savetohdf5(fname,zno=1):
    dims = dimensions(fname)
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

