#!/usr/bin/env python
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import numpy as np


def read_exptlist(filename,lines_per_expt=3,foldline=0,fileline=1):
    with open(filename,mode='r') as f:
        content = f.read().splitlines()
    foldname = []
    filenames = []
    fileline = np.array(fileline)
    samefold = False
    for iline,line in enumerate(content):
        if line and not line[0]=='#':
            if np.mod(iline,lines_per_expt)==foldline:
                foldname.append(line)
                samefold = False
            elif np.in1d(np.mod(iline,lines_per_expt),fileline):
                filenos = [int(x) for x in line.split(',')]
                if samefold:
                    filenames[-1] = filenames[-1] + filenos
                else: 
                    filenames.append(filenos)
                samefold = True
    for filelist in filenames:
        filelist.sort()
    return foldname,filenames


