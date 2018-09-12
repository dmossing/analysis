#!/usr/bin/env python

import os
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import h5py
from oasis.functions import deconvolve
from oasis import oasisAR1, oasisAR2
import pyute as ut
import scipy.stats as sst
import scipy.ndimage.filters as sfi

blcutoff = 5
ds = 10
blspan = 3000
nbefore = 0
nafter = 0

def analyze_precise_retinotopy(datafiles,stimfile,retfile,criterion=lambda x: x>100,rg=(2,-10),nbefore=nbefore,nafter=nafter):
	nbydepth = np.zeros((len(datafiles),))
	for i,datafile in enumerate(datafiles):
		corrected = ut.loadmat(datafile,'corrected')
		nbydepth[i] = corrected.shape[0]
#		 with h5py.File(datafile,mode='r') as f:
#			 nbydepth[i] = (f['corrected'][:].T.shape[0])
	trialwise,ctrialwise,strialwise,dfof = ut.gen_precise_trialwise(datafiles,rg=rg,nbefore=nbefore,nafter=nafter)
	zstrialwise = sst.zscore(strialwise.reshape((strialwise.shape[0],-1)).T).T.reshape(strialwise.shape)

	result = sio.loadmat(stimfile,squeeze_me=True)['result'][()]

	infofile = sio.loadmat(datafiles[0][:-12]+'.mat',squeeze_me=True)
	retfile = sio.loadmat(retfile,squeeze_me=True)

	locinds = retfile['locinds']

	frame = infofile['info'][()]['frame'][()]
	frame = np.unique(frame[rg[0]:frame.size+rg[1]]) # this format for all the retinotopic mapping through 12/12

	data = strialwise[:,:,nbefore:strialwise.shape[-1]-nafter]

	Ny = locinds[:,0].max()
	Nx = locinds[:,1].max()
	
	try:
		try:
			dxdt = sio.loadmat(datafiles[1],squeeze_me=True)['dxdt']
		except:
			with h5py.File(datafiles[1],mode='r') as f:
				dxdt = f['dxdt'][:].T
	except:
		print('no running data saved; assuming all running')
		dxdt = 101*np.ones((frame.max(),))

	trialrun = np.zeros(frame[0::2].shape)
	for i in range(len(trialrun)):
		trialrun[i] = dxdt[frame[0::2][i]:frame[1::2][i]].mean()
	runtrial = criterion(np.abs(trialrun)) #>0

	ret = np.zeros((data.shape[0],Ny,Nx)) #,2))
	for j in range(Ny):
		for k in range(Nx):
			lkat_reg = np.logical_and(np.logical_and(locinds[:,0]==j+1,locinds[:,1]==k+1),runtrial)
			# lkat_reg = np.logical_and(lkat,np.logical_not(inverted))
			# lkat_inv = np.logical_and(lkat,inverted)
			n_reg = lkat_reg.sum()
			# n_inv = lkat_inv.sum()
			print((n_reg,))#n_inv))
			for idx in np.where(lkat_reg)[0]:
				ret[:,j,k] = ret[:,j,k] + data[:,idx].mean(1)/n_reg
			assert(~np.isnan(np.nanmax(ret[:,j,k])))
			# for idx in np.where(lkat_inv)[0]:
				# ret[:,j,k,1] = ret[:,j,k,1] + data[:,idx].mean(1)/n_inv

	gridsize = 10
	xrg = np.arange(-(Nx-1)*gridsize/2,(Nx+1)*gridsize/2,gridsize)
	yrg = np.arange(-(Ny-1)*gridsize/2,(Ny+1)*gridsize/2,gridsize)

	paramdict = ut.fit_2d_gaussian((xrg,yrg),ret) #,ut.fit_2d_gaussian((xrg,yrg),ret[:,:,:,1])]

	pval = np.zeros(strialwise.shape[0])
	for i in range(strialwise.shape[0]):
		_,pval[i] = sst.ttest_rel(strialwise[i,:,nbefore-1],strialwise[i,:,nbefore+1])
	
	return ret,paramdict,pval,trialrun


def analyze_retinotopy(datafiles,stimfile,retfile,criterion=lambda x: x>100,rg=(2,-10),nbefore=nbefore,nafter=nafter):
	nbydepth = np.zeros((len(datafiles),))
	for i,datafile in enumerate(datafiles):
		with h5py.File(datafile,mode='r') as f:
			nbydepth[i] = (f['corrected'][:].T.shape[0])
	trialwise,ctrialwise,strialwise,dfof = ut.gen_trialwise(datafiles,rg=rg,nbefore=nbefore,nafter=nafter)
	zstrialwise = sst.zscore(strialwise.reshape((strialwise.shape[0],-1)).T).T.reshape(strialwise.shape)

	result = sio.loadmat(stimfile,squeeze_me=True)['result'][()]

	infofile = sio.loadmat(datafiles[0][:-5]+'.mat',squeeze_me=True)
	retfile = sio.loadmat(retfile,squeeze_me=True)

	locinds = retfile['locinds']

	frame = infofile['info'][()]['frame'][()]
	frame = np.unique(frame[rg[0]:frame.size+rg[1]]) # this format for all the retinotopic mapping through 12/12

	data = strialwise[:,:,nbefore:strialwise.shape[-1]-nafter]

	Ny = locinds[:,0].max()
	Nx = locinds[:,1].max()
	
	try:
		try:
			dxdt = sio.loadmat(datafiles[1],squeeze_me=True)['dxdt']
		except:
			with h5py.File(datafiles[1],mode='r') as f:
				dxdt = f['dxdt'][:].T
	except:
		print('no running data saved; assuming all running')
		dxdt = 101*np.ones((frame.max(),))

	trialrun = np.zeros(frame[0::2].shape)
	for i in range(len(trialrun)):
		trialrun[i] = dxdt[frame[0::2][i]:frame[1::2][i]].mean()
	runtrial = criterion(np.abs(trialrun)) #>0

	ret = np.zeros((data.shape[0],Ny,Nx)) #,2))
	for j in range(Ny):
		for k in range(Nx):
			lkat_reg = np.logical_and(np.logical_and(locinds[:,0]==j+1,locinds[:,1]==k+1),runtrial)
			# lkat_reg = np.logical_and(lkat,np.logical_not(inverted))
			# lkat_inv = np.logical_and(lkat,inverted)
			n_reg = lkat_reg.sum()
			# n_inv = lkat_inv.sum()
			print((n_reg,))#n_inv))
			for idx in np.where(lkat_reg)[0]:
				ret[:,j,k] = ret[:,j,k] + data[:,idx].mean(1)/n_reg
			assert(~np.isnan(np.nanmax(ret[:,j,k])))
			# for idx in np.where(lkat_inv)[0]:
				# ret[:,j,k,1] = ret[:,j,k,1] + data[:,idx].mean(1)/n_inv

	gridsize = 10
	xrg = np.arange(-(Nx-1)*gridsize/2,(Nx+1)*gridsize/2,gridsize)
	yrg = np.arange(-(Ny-1)*gridsize/2,(Ny+1)*gridsize/2,gridsize)

	paramdict = ut.fit_2d_gaussian((xrg,yrg),ret) #,ut.fit_2d_gaussian((xrg,yrg),ret[:,:,:,1])]

	pval = np.zeros(strialwise.shape[0])
	for i in range(strialwise.shape[0]):
		_,pval[i] = sst.ttest_rel(strialwise[i,:,nbefore-1],strialwise[i,:,nbefore+1])
	
	return ret,paramdict,pval,trialrun

def get_nbydepth(datafiles):
	nbydepth = np.zeros((len(datafiles),))
	for i,datafile in enumerate(datafiles):
		with h5py.File(datafile,mode='r') as f:
			nbydepth[i] = (f['corrected'][:].T.shape[0])
	return nbydepth

def gen_traces(datafiles,blcutoff=blcutoff,blspan=blspan): #nbefore=nbefore,nafter=nafter
	trialwise = np.array(())
	ctrialwise = np.array(())
	strialwise = np.array(())
	dfofall = np.array(())
	baselineall = np.array(())
	for datafile in datafiles:
		frm = sio.loadmat(datafile.replace('.rois','.mat'),squeeze_me=True)['info']['frame'][()][2:-10]
		with h5py.File(datafile,mode='r') as f:
			to_add = f['corrected'][:].T
			to_add[np.isnan(to_add)] = 0
#			 baseline = np.percentile(to_add,blcutoff,axis=1)
			baseline = sfi.percentile_filter(to_add[:,::ds],blcutoff,(1,int(blspan/ds)))
			baseline = np.repeat(baseline,ds,axis=1)
			for i in range(baseline.shape[0]):
				baseline[i] = sfi.gaussian_filter1d(baseline[i],blspan/2)
#			 if baseline.shape[1]<to_add.shape[1]:
#				 baseline = np.hstack((baseline,np.repeat(baseline[:,-1],to_add.shape[1]-baseline.shape[1])))
			if baseline.shape[1]>to_add.shape[1]:
				baseline = baseline[:,:to_add.shape[1]]
			c = np.zeros_like(to_add)
			s = np.zeros_like(to_add)
			dfof = np.zeros_like(to_add)
			for i in range(c.shape[0]):
#				 dfof = (to_add[i]-baseline[i,np.newaxis])/baseline[i,np.newaxis]
				dfof[i] = (to_add[i]-baseline[i,:])/baseline[i,:]
				try:
					c[i],s[i],_,_,_  = deconvolve(dfof[i],penalty=1,sn=5e-3)
				except:
					throwaway = 0
					print("in "+datafile+" couldn't do "+str(i))
			try:
				trialwise = np.concatenate((trialwise,to_add),axis=0)
				ctrialwise = np.concatenate((ctrialwise,c),axis=0)
				strialwise = np.concatenate((strialwise,s),axis=0)
				dfofall = np.concatenate((dfofall,dfof),axis=0)
				baselineall = np.concatenate((baselineall,baseline),axis=0)
			except:
				trialwise = to_add.copy()
				ctrialwise = c.copy()
				strialwise = s.copy()
				dfofall = dfof.copy()
				baselineall = baseline.copy()
	return trialwise,ctrialwise,strialwise,dfofall,baselineall
