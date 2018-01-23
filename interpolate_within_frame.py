#!/usr/bin/env python

import os
import sys
import scipy.io as sio
import os
import pyute as ut
import h5py
import numpy as np
import scipy.ndimage.filters as sfi

def interpolate_within_frame(fname,trigspertrial=2,runfname=None):
	roifname = fname+'.rois'
	infofname = fname+'.mat'

	roifile = h5py.File(roifname,mode='r+')
	matfile = sio.loadmat(infofname,squeeze_me=True)

	data = roifile['Data'][:].T
	npil = roifile['Neuropil'][:].T

	# frame and line when each trigger comes in
	frame = matfile['info']['frame'][()]
	line = matfile['info']['line'][()]
	nlines = matfile['info']['sz'][()][0]

	frametime = np.tile(np.arange(data.shape[0])[:,np.newaxis],(1,data.shape[1])) + np.tile(roiline[np.newaxis,:]/nlines,(data.shape[0],1))

	trigtime = frame-1+line/nlines # -1 matlab to python indexing
	trigrate = np.mean(np.diff(trigtime[::trigspertrial]))
	interpbetweentrigs = int(np.round(trigrate))

	extra_trigs = 1
	interpframetime = np.linspace(trigtime[0],trigtime[-1]+extra_trigs*trigrate,(len(trigtime)+extra_trigs-1)*interpbetweentrigs+1)

	data_interp = np.zeros((len(interpframetime),data.shape[1]))
	npil_interp = np.zeros((len(interpframetime),npil.shape[1]))
	for cell in range(data.shape[1]):
		data_interp[:,cell] = np.interp(interpframetime,frametime[:,cell],data[:,cell])
		npil_interp[:,cell] = np.interp(interpframetime,frametime[:,cell],npil[:,cell])
	roifile['Data_interp'] = data_interp
	roifile['Neuropil_interp'] = npil_interp
	
	if runfname:
		runfile = sio.loadmat(runfname,squeeze_me=True)
		dxdt = runfile['dxdt']
		dxdt_interp = np.interp(interpframetime,np.arange(dxdt.shape[0]),dxdt)
		roifile['dxdt_interp'] = dxdt_interp
		roifile['frame_interp'] = interpframetime
	
	roifile.close()
	
