#!/usr/bin/env python

import scipy.ndimage.interpolation as snip
import scipy.misc as smi
from PIL import Image

def load_planes(folds,files,frames):
    # folds: folders containing the planes of interest
    # files: tif file index(es) within each folder to generate a movie from
    # frames: frame indices within each file to generate a movie from
    def make_lists(*args):
        for thing in args:
            if not type(thing) is list:
                thing = [thing]

    make_lists(folds,files,frames)
    
    for imfold in folds:
	for i,imfile in enumerate(files):
            img = Image.open(imfold+'/'+imfile)
            img.seek(frames[i][0])
            for frame in frames:
                
