#!/usr/bin/env python
 
import scipy.ndimage.interpolation as snip
import skimage.io as skio
import matplotlib.pyplot as plt
import skimage.transform as skt
import numpy as np
import pyute as ut
import os

def gen_movie(delta=100,scaley=0.5,spacebetween=5,source_fold=None,source_file=None,frame_file=None,offset=1000,frame_no=200,target_fold='/home/mossing/Documents/notebooks/temp/',frame_rg=(1,0)):
    # source_fold and source_file are format strings, like 'whatever{0}whatever', where {0} specifies where a digit should go

    if source_fold is None or source_file is None:
        date,animalid,exptno = parse_sbx_filename(frame_file)
        source_fold = make_suite2p_fold(date,animalid,exptno)
        source_file = make_suite2p_file(date,animalid,exptno)

    filenames = [(source_fold+'/'+source_file).format(n) for n in range(1,5)]
    imgs = [skio.MultiImage(filename) for filename in filenames]

    (Ny,Nx) = imgs[0][0].shape

    tr = gen_transform(delta,scaley,(Ny,Nx))

    frm_on = gen_frm_on(frame_file,rg=frame_rg)

    warped = [skt.warp(imgs[n][offset],tr) for n in range(4)]
    green_frame = np.vstack([x[:int(Ny/2+spacebetween)] for x in warped[::-1]])
    stim_on = gen_stim_on(green_frame.shape)
    
    if not os.path.exists(target_fold):
        os.mkdir(target_fold)

    for t in range(frame_no):
        print(t)
        warped = [skt.warp(imgs[n][offset+t],tr) for n in range(4)]
        green_frame = np.vstack([x[:int(Ny*scaley+spacebetween)] for x in warped[::-1]])
        green_frame = green_frame*(green_frame>0)
        red_frame = np.in1d(offset+t,frm_on)*stim_on
        rgb = np.dstack((red_frame,green_frame,red_frame))
        plt.imsave(target_fold+'/{0:04d}.tif'.format(t),rgb)

def gen_transform(delta=100,scaley=0.5,shp=(512,796)):
    (Ny,Nx) = shp
    trmat = np.array(((1-delta/Nx,-delta/Ny,delta),(0,scaley,0),(0,0,1)))
    tr = skt.AffineTransform(np.linalg.inv(trmat))
    return tr

def gen_frm_on(frame_file,rg=(1,0)):
    info = ut.loadmat(frame_file,'info')[()]
    frame = info['frame']
    frm = np.floor(frame[rg[0]:frame.size+rg[1]]/4).astype('int')
    frm_on = np.concatenate([np.arange(x,y) for x,y in zip(frm[0::2],frm[1::2])])
    return frm_on

def gen_stim_on(shp,width=50,from_corner=25):
    stim_on = np.zeros(shp)
    stim_on[-from_corner-width:-from_corner,-from_corner-width:-from_corner] = 1
    return stim_on

def parse_sbx_filename(path):
    path = path.split('.')[0]
    parts = path.split('/')
    date = parts[5]
    animalid = parts[6]
    exptname = parts[7]
    exptno = str(int(exptname.split('_')[-1]))
    return date,animalid,exptno

def make_suite2p_fold(date,animalid,exptno,base='/home/mossing/scratch/2Pdata/suite2P/registered/'):
    return '/'.join([base,animalid,date,exptno,'Plane{0}/'])

def make_suite2p_file(date,animalid,exptno):
    return '_'.join([date,exptno,animalid,'2P','plane{0}','1.tif'])
