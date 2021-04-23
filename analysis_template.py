#!/usr/bin/env python

import numpy as np
import pyute as ut
import os
import h5py

blcutoff = 20 # 1 before
ds = 10
blspan = 3000
nbefore = 4
nafter = 4

#def loadmat_from_names_(matfile,varnames):
#    return ut.loadmat(matfile,varnames)

def load_roi_info(datafiles):
    nplanes = len(datafiles)
    msk,ctr = load_msk_ctr(datafiles[0])
    cell_mask = np.zeros((0,)+msk.shape[1:],dtype='bool')
    cell_center = np.zeros((0,2))
    cell_depth = np.zeros((0,))
    for iplane in range(nplanes):
        msk,ctr = load_msk_ctr(datafiles[iplane])
        cell_mask = np.concatenate((cell_mask,msk),axis=0)
        cell_center = np.concatenate((cell_center,ctr),axis=0)
        cell_depth = np.concatenate((cell_depth,iplane*np.ones((msk.shape[0],))))

    varnames1 = ['green_mean','red_mean']
    varnames2= ['meanImg','meanImgE','meanImg_chan2','meanImg_chan2_corrected']
    outputs1 = ut.loadmat(datafiles[0],varnames1)
    outputs2 = ut.loadmat(datafiles[0],varnames2)
    use_first,use_second = [not outputs[0] is None for outputs in [outputs1,outputs2]]

    if use_first:
        iplane = 0
        mean_image_green,mean_image_red = ut.loadmat(datafiles[iplane],varnames1)
        shp = mean_image_green.shape
        mean_red_channel = np.zeros((nplanes,)+shp)
        mean_green_channel = np.zeros((nplanes,)+shp)
        for iplane in range(nplanes):
            mean_image_green,mean_image_red = ut.loadmat(datafiles[iplane],varnames1)
            mean_green_channel[iplane] = mean_image_green
            mean_red_channel[iplane] = mean_image_red
        mean_red_channel_corrected = None
        mean_green_channel_enhanced = None
    elif use_second:
        iplane = 0
        mean_image_green,mean_image_green_enhanced,mean_image_red,mean_image_red_corrected = ut.loadmat(datafiles[iplane],varnames2)
        shp = mean_image_green.shape
        mean_green_channel = np.zeros((nplanes,)+shp)
        mean_green_channel_enhanced = np.zeros((nplanes,)+shp)
        mean_red_channel = np.zeros((nplanes,)+shp)
        mean_red_channel_corrected = np.zeros((nplanes,)+shp)
        for iplane in range(nplanes):
            mean_image_green,mean_image_green_enhanced,mean_image_red,mean_image_red_corrected = ut.loadmat(datafiles[iplane],varnames2)
            #mean_image_green,mean_image_red = ut.loadmat(datafiles[iplane],['meanImg','meanImg_chan2_corrected'])
            mean_red_channel[iplane] = mean_image_red
            mean_red_channel_corrected[iplane] = mean_image_red_corrected
            mean_green_channel[iplane] = mean_image_green
            mean_green_channel_enhanced[iplane] = mean_image_green_enhanced
    else:
        print('no mean image data for ' + datafiles[0])
        mean_red_channel = None
        mean_red_channel_corrected = None
        mean_green_channel = None
        mean_green_channel_enhanced = None

    proc = {}

    proc['mean_red_channel'] = mean_red_channel
    proc['mean_red_channel_corrected'] = mean_red_channel_corrected
    proc['mean_green_channel'] = mean_green_channel
    proc['mean_green_channel_enhanced'] = mean_green_channel_enhanced
    proc['cell_depth'] = cell_depth
    proc['cell_center'] = cell_center
    proc['cell_mask'] = cell_mask

    return proc

def analyze(datafiles,stimfile,frame_adjust=None,rg=(1,0),nbefore=nbefore,nafter=nafter,stim_params=None):
    # stim_params: list (or similar) of tuples, where first element is a string corresponding to a field of the
    # output hdf5 file proc, and second element is a function taking result as an input, to yield the correct data

    # find number of ROIs in each plane
    nbydepth = get_nbydepth(datafiles)

    nplanes = len(nbydepth)

    # get trialized fluorescence data
    trialwise,ctrialwise,strialwise,dfof,straces,dtrialwise,proc1 = ut.gen_precise_trialwise(datafiles,rg=rg,frame_adjust=frame_adjust,nbefore=nbefore,nafter=nafter,blcutoff=blcutoff) # , trialwise_t_offset

    # load stimulus data
    #result = sio.loadmat(stimfile,squeeze_me=True)['result'][()]
    result = ut.loadmat(stimfile,'result')[()]
    #result = result[()]
    
    # correct stim trigger frames if necessary
    #infofile = sio.loadmat(datafiles[0][:-12]+'.mat',squeeze_me=True) # original .mat file
    info = ut.loadmat(datafiles[0][:-12]+'.mat','info')[()] # original .mat file
    #frame = infofile['info'][()]['frame'][()]
    frame = info['frame'][()].astype(np.int64)
    if not rg is None:
        frame = frame[rg[0]:frame.size+rg[1]]
    else:
        event_id = info['event_id'][()].astype(np.int64)
        frame = frame[event_id==1]
    if frame_adjust:
        frame = frame_adjust(frame)

    while np.min(np.diff(frame)) < 0:
        brk = np.argmin(np.diff(frame))+1
        frame[brk:] = frame[brk:] + 65536
         
    # load running and pupil data
    dxdt = ut.loadmat(datafiles[0],'dxdt').flatten()
    try:
        # first entry of pupil_ctr is x, second entry is y
        pupil_ctr,pupil_area,pupil_frac_ctr,pupil_frac_area = ut.loadmat(datafiles[0],['pupil_ctr','pupil_area','pupil_frac_ctr','pupil_frac_area'])
        pupil_area = pupil_area.flatten()
        pupil_frac_area = pupil_frac_area.flatten()
    except:
        print('no eye tracking data for ' + stimfile)
        pupil_ctr = None
        pupil_frac_ctr = None
        pupil_area = None
        pupil_frac_area = None

    nplanes = len(datafiles)

    msk,ctr = load_msk_ctr(datafiles[0])
    cell_mask = np.zeros((0,)+msk.shape[1:],dtype='bool')
    cell_center = np.zeros((0,2))
    cell_depth = np.zeros((0,))
    for iplane in range(nplanes):
        msk,ctr = load_msk_ctr(datafiles[iplane])
        cell_mask = np.concatenate((cell_mask,msk),axis=0)
        cell_center = np.concatenate((cell_center,ctr),axis=0)
        cell_depth = np.concatenate((cell_depth,iplane*np.ones((msk.shape[0],))))

#    try:
#        try:
#        #if True:
#            #mean_image_red,mean_image_green = ut.loadmat(datafiles[0],['red_mean','green_mean'])
#            #mean_red_channel = np.zeros((len(datafiles),)+mean_image_red.shape)
#            #mean_green_channel = np.zeros((len(datafiles),)+mean_image_green.shape)
#            mean_red_channel = np.zeros((nplanes,)+cell_mask.shape[1:])
#            mean_green_channel = np.zeros((nplanes,)+cell_mask.shape[1:])
#            for iplane in range(nplanes):
#                mean_image_red,mean_image_green = ut.loadmat(datafiles[iplane],['red_mean','green_mean'])
#                mean_red_channel[iplane] = mean_image_red
#                mean_green_channel[iplane] = mean_image_green
#        except:
#            mean_red_channel = np.zeros((nplanes,)+cell_mask.shape[1:])
#            mean_red_channel_corrected = np.zeros((nplanes,)+cell_mask.shape[1:])
#            mean_green_channel = np.zeros((nplanes,)+cell_mask.shape[1:])
#            mean_green_channel_enhanced = np.zeros((nplanes,)+cell_mask.shape[1:])
#            for iplane in range(nplanes):
#                mean_image_green,mean_image_green_enhanced,mean_image_red,mean_image_red_corrected = ut.loadmat(datafiles[iplane],['meanImg','meanImgE','meanImg_chan2','meanImg_chan2_corrected'])
#                #mean_image_green,mean_image_red = ut.loadmat(datafiles[iplane],['meanImg','meanImg_chan2_corrected'])
#                mean_red_channel[iplane] = mean_image_red
#                mean_red_channel_corrected[iplane] = mean_image_red_corrected
#                mean_green_channel[iplane] = mean_image_green
#                mean_green_channel_enhanced[iplane] = mean_image_green_enhanced
#    except:
#        print('no mean image data for ' + stimfile)
#        mean_red_channel = None
#        mean_red_channel_corrected = None
#        mean_green_channel = None
#        mean_green_channel_enhanced = None

    #varnames1 = ['green_mean','red_mean']
    #varnames2= ['meanImg','meanImgE','meanImg_chan2','meanImg_chan2_corrected']
    #outputs1 = ut.loadmat(datafiles[0],varnames1)
    #outputs2 = ut.loadmat(datafiles[0],varnames2)
    #use_first,use_second = [not outputs[0] is None for outputs in [outputs1,outputs2]]

    #if use_first:
    #    mean_red_channel = np.zeros((nplanes,)+cell_mask.shape[1:])
    #    mean_green_channel = np.zeros((nplanes,)+cell_mask.shape[1:])
    #    for iplane in range(nplanes):
    #        mean_image_green,mean_image_red = ut.loadmat(datafiles[iplane],varnames1)
    #        mean_green_channel[iplane] = mean_image_green
    #        mean_red_channel[iplane] = mean_image_red
    #    mean_red_channel_corrected = None
    #    mean_green_channel_enhanced = None
    #elif use_second:
    #    mean_green_channel = np.zeros((nplanes,)+cell_mask.shape[1:])
    #    mean_green_channel_enhanced = np.zeros((nplanes,)+cell_mask.shape[1:])
    #    mean_red_channel = np.zeros((nplanes,)+cell_mask.shape[1:])
    #    mean_red_channel_corrected = np.zeros((nplanes,)+cell_mask.shape[1:])
    #    for iplane in range(nplanes):
    #        mean_image_green,mean_image_green_enhanced,mean_image_red,mean_image_red_corrected = ut.loadmat(datafiles[iplane],['meanImg','meanImgE','meanImg_chan2','meanImg_chan2_corrected'])
    #        #mean_image_green,mean_image_red = ut.loadmat(datafiles[iplane],['meanImg','meanImg_chan2_corrected'])
    #        mean_red_channel[iplane] = mean_image_red
    #        mean_red_channel_corrected[iplane] = mean_image_red_corrected
    #        mean_green_channel[iplane] = mean_image_green
    #        mean_green_channel_enhanced[iplane] = mean_image_green_enhanced
    #else:
    #    print('no mean image data for ' + stimfile)
    #    mean_red_channel = None
    #    mean_red_channel_corrected = None
    #    mean_green_channel = None
    #    mean_green_channel_enhanced = None
    # trialize running and pupil data
    #try:
    roi_proc = load_roi_info(datafiles)
    #except:
    #    roi_proc = None
    frame_div = np.floor(2*frame/nplanes).astype(np.int64)
    trialrun = ut.trialize(dxdt.T,frame,nbefore=nbefore,nafter=nafter)
    trialctr = ut.trialize(pupil_ctr,frame_div,nbefore=nbefore,nafter=nafter)
    trialfracctr = ut.trialize(pupil_frac_ctr,frame_div,nbefore=nbefore,nafter=nafter)
    trialarea = ut.trialize(pupil_area,frame_div,nbefore=nbefore,nafter=nafter)
    trialfracarea = ut.trialize(pupil_frac_area,frame_div,nbefore=nbefore,nafter=nafter)

    proc = {}
    proc['trialrun'] = trialrun
    proc['trialctr'] = trialctr
    proc['trialarea'] = trialarea
    proc['trialfracctr'] = trialfracctr
    proc['trialfracarea'] = trialfracarea
    proc['trialwise'] = trialwise
    proc['strialwise'] = strialwise
    proc['nbydepth'] = nbydepth
    proc['dtrialwise'] = dtrialwise
    proc['dfof'] = dfof
    proc['trialwise_t_offset'] = proc1['trialwise_t_offset']
    proc['raw_trialwise'] = proc1['raw_trialwise']
    proc['neuropil_trialwise'] = proc1['neuropil_trialwise']
    if roi_proc:
        for key in roi_proc:
            proc[key] = roi_proc[key]
    else:
        print('could not compute roi info')
    #proc['mean_red_channel'] = mean_red_channel
    #proc['mean_red_channel_corrected'] = mean_red_channel_corrected
    #proc['mean_green_channel'] = mean_green_channel
    #proc['mean_green_channel_enhanced'] = mean_green_channel_enhanced
    #proc['cell_depth'] = cell_depth
    #proc['cell_center'] = cell_center
    #proc['cell_mask'] = cell_mask
    proc['nbefore'] = nbefore
    proc['nafter'] = nafter
              
    # define extra parameters based on 'result' variable
    for param in stim_params:
        name,function = param
        proc[name] = function(result)
    
    return proc

def trialavg(arr,nbefore=nbefore,nafter=nafter):
    return np.nanmean(arr[:,nbefore:-nafter],axis=1)

def get_nbydepth(datafiles):
    nbydepth = np.zeros((len(datafiles),))
    for i,datafile in enumerate(datafiles):
        corrected = ut.loadmat(datafile,'corrected')[()]
        nbydepth[i] = corrected.shape[0]
        #with h5py.File(datafile,mode='r') as f:
            #nbydepth[i] = (f['corrected'][:].T.shape[0])
    return nbydepth

def assign_(dicti,field,val):
    if not field in dicti:
        dicti[field] = val

def add_ret_to_data_struct(filename, keylist=None, proc=None, grouplist=None):
    with ut.hdf5edit(filename) as data_struct:
        for key,group in zip(keylist,grouplist):
            print((key,group))
            print(list(proc[key].keys()))

            if 'ret_vars' in proc[key]:
                ret_vars = proc['/'.join([key,'ret_vars'])]
                paramdict = proc['/'.join([key,'ret_vars','paramdict_normal'])]
                rf_ctr = np.concatenate((paramdict['xo'][:][np.newaxis,:],-paramdict['yo'][:][np.newaxis,:]),axis=0)
                
                rf_sq_error = ret_vars['paramdict_normal']['sqerror'][:]
                sx = ret_vars['paramdict_normal']['sigma_x'][:]
                sy = ret_vars['paramdict_normal']['sigma_y'][:]
                amp = ret_vars['paramdict_normal']['amplitude'][:]

                this_expt = data_struct[group]

                assign_(this_expt,'rf_mapping_pval',ret_vars['pval_ret'][:])
                assign_(this_expt,'rf_ctr',rf_ctr)
                assign_(this_expt,'rf_sq_error',rf_sq_error)
                assign_(this_expt,'rf_sigma',np.sqrt(sx**2+sy**2))
                assign_(this_expt,'rf_amplitude',amp)
                #this_expt['rf_mapping_pval'] = ret_vars['pval_ret'][:]
               # this_expt['rf_ctr'] = rf_ctr
               # this_expt['rf_sq_error'] = rf_sq_error
               # this_expt['rf_sigma'] = np.sqrt(sx**2+sy**2)
                if 'position' in ret_vars:
                    stim_offset = ret_vars['position'][:] - paramdict['ctr'][:]
                    rf_distance_deg = np.sqrt(((rf_ctr-stim_offset[:,np.newaxis])**2).sum(0))
                    rf_displacement_deg = rf_ctr-stim_offset[:,np.newaxis]
                    for kkey,vval in zip(['rf_distance_deg','rf_displacement_deg','stim_offset_deg'],[rf_distance_deg,rf_displacement_deg,stim_offset]):
                        assign_(this_expt,kkey,vval)
                   # this_expt['rf_distance_deg'] = rf_distance_deg
                   # this_expt['rf_displacement_deg'] = rf_displacement_deg
                   # this_expt['stim_offset_deg'] = stim_offset
                    

def add_data_struct_h5(filename, cell_type='PyrL23', keylist=None, frame_rate_dict=None, proc=None, nbefore=8, nafter=8,featurenames=['size','contrast','angle'],datasetnames=None,groupname='size_contrast',replace=False):
    if datasetnames is None:
        datasetnames = ['stimulus_'+name for name in featurenames]
    #with h5py.File(filename,mode='w+') as data_struct:
    with ut.hdf5edit(filename) as data_struct:
    #data_struct = {}
        ret_vars = {}
        grouplist = [None]*len(keylist)
        for ikey,key in enumerate(keylist):
            dfof = proc[key]['dtrialwise'][:]
            decon = proc[key]['strialwise'][:] 
            t_offset = proc[key]['trialwise_t_offset'][:] 
            running_speed_cm_s = 4*np.pi/180*proc[key]['trialrun'][:] # 4 cm from disk ctr to estimated mouse location

            #ucontrast,icontrast = np.unique(proc[key]['contrast'][:],return_inverse=True)
            #usize,isize = np.unique(proc[key]['size'][:],return_inverse=True)
            #uangle,iangle = np.unique(proc[key]['angle'][:],return_inverse=True)
            #stimulus_id = np.concatenate((isize[np.newaxis],icontrast[np.newaxis],iangle[np.newaxis]),axis=0)
            #stimulus_size_deg = usize
            #stimulus_contrast = ucontrast
            #stimulus_direction = uangle
    
            ufeature_list,stimulus_id = gen_stimulus_id(proc[key],featurenames)

            cell_id = np.arange(dfof.shape[0])
            session_id = key
            mouse_id = key.split('_')[1]
            
            if session_id in data_struct.keys():
                if len(cell_id) != len(data_struct[session_id]['cell_id']):
                    del data_struct[session_id]
            
            if not session_id in data_struct.keys():
                this_session = data_struct.create_group(session_id)
                this_session['mouse_id'] = mouse_id
                this_session['cell_type'] = cell_type
                this_session.create_dataset('cell_id',data=cell_id)
                
                for field in ['cell_depth','cell_mask','cell_center','mean_red_channel','mean_red_channel_corrected','mean_green_channel','mean_green_channel_enhanced']:
                    if field in proc[key]:
                        this_session.create_dataset(field,data=proc[key][field])
            else:
                this_session = data_struct[session_id]

            if 'ret_vars' in proc[key]:
                ret_vars[key] = proc['/'.join([key,'ret_vars'])]
                paramdict = proc['/'.join([key,'ret_vars','paramdict_normal'])]
                #rf_ctr = np.concatenate((paramdict['xo'][:][np.newaxis,:],-paramdict['yo'][:][np.newaxis,:]),axis=0)
                rf_ctr = np.concatenate((paramdict['xo'][:][np.newaxis,:],paramdict['yo'][:][np.newaxis,:]),axis=0)# new version uses y+ is up sign convention from the beginning

                stim_offset = np.array((0,0)) #ret_vars[key]['position'][:] - paramdict['ctr'][:]

                pval_ret = proc['/'.join([key,'ret_vars','pval_ret'])]

                rf_distance_deg = np.sqrt(((rf_ctr-stim_offset[:,np.newaxis])**2).sum(0))

                rf_displacement_deg = rf_ctr-stim_offset[:,np.newaxis]
            elif 'retinotopy_0' in this_session and 'rf_ctr' in this_session['retinotopy_0']:
                rf_ctr = this_session['retinotopy_0']['rf_ctr'][:]

                #ctr = this_session['retinotopy_0']['ctr'][:] # center of retinotopic mapping stimuli
                stim_offset = proc[key]['position'][:]# - ctr # center of size contrast stimuli e.g. w/r/t retino center -- NOW WRT CENTER OF SCREEN
                #stim_offset = proc[key]['position'][:][::-1]# - ctr # center of size contrast stimuli e.g. w/r/t retino center -- NOW WRT CENTER OF SCREEN; NOW IN Y,X ORDER, 21/4/22

                pval_ret = this_session['retinotopy_0']['rf_mapping_pval'][:]

                rf_distance_deg = np.sqrt(((rf_ctr-stim_offset[:,np.newaxis])**2).sum(0))

                rf_displacement_deg = rf_ctr-stim_offset[:,np.newaxis]
            #elif 'retinotopy_0' in this_session and 'rf_ctr' in this_session['retinotopy_0']:
            #    rf_ctr = this_session['retinotopy_0']['rf_ctr'][:]*np.array((1,-1))[:,np.newaxis]
            #    ctr = this_session['retinotopy_0']['ctr'][:] # center of retinotopic mapping stimuli
            #    stim_offset = proc[key]['position'][:] - ctr # center of size contrast stimuli e.g. w/r/t retino center
            #    rf_distance_deg = np.sqrt(((rf_ctr-stim_offset[:,np.newaxis])**2).sum(0))
            #    rf_displacement_deg = rf_ctr-stim_offset[:,np.newaxis]

            exptno = 0
            if replace and groupname+'_0' in this_session.keys():
                del this_session[groupname+'_0']
            else:
                while groupname+'_'+str(exptno) in this_session.keys():
                    exptno = exptno+1
            this_expt = this_session.create_group(groupname+'_'+str(exptno))

            grouplist[ikey] = session_id + '/' + groupname + '_' + str(exptno)

            this_expt.create_dataset('stimulus_id',data=stimulus_id)
            stimulus_parameters = [n.encode('ascii','ignore') for n in datasetnames]
            for name,ufeature in zip(datasetnames,ufeature_list):
                this_expt.create_dataset(name,data=ufeature)

            #if 'ret_vars' in proc[key]:
            #    this_expt['rf_mapping_pval'] = ret_vars[key]['pval_ret'][:]
            #    this_expt['rf_distance_deg'] = rf_distance_deg
            #    this_expt['rf_displacement_deg'] = rf_displacement_deg
            #    this_expt['rf_ctr'] = rf_ctr
            #    this_expt['stim_offset_deg'] = stim_offset
            this_expt['rf_mapping_pval'] = pval_ret[:]
            this_expt['rf_distance_deg'] = rf_distance_deg
            this_expt['rf_displacement_deg'] = rf_displacement_deg
            this_expt['rf_ctr'] = rf_ctr
            this_expt['stim_offset_deg'] = stim_offset

            this_expt.create_dataset('running_speed_cm_s',data=running_speed_cm_s)
            this_expt.create_dataset('F',data=dfof)
            this_expt.create_dataset('raw_trialwise',data=proc[key]['raw_trialwise'][:])
            this_expt.create_dataset('neuropil_trialwise',data=proc[key]['neuropil_trialwise'][:])
            if 'trialctr' in proc[key]:
                this_expt.create_dataset('pupil_ctr_trialwise_pix',data=proc[key]['trialctr'][:])
                this_expt.create_dataset('pupil_ctr_trialwise_pct_eye_diam',data=proc[key]['trialfracctr'][:])
                this_expt.create_dataset('pupil_area_trialwise_pix',data=proc[key]['trialarea'][:])
                this_expt.create_dataset('pupil_area_trialwise_pct_eye_area',data=proc[key]['trialfracarea'][:])
            this_expt.create_dataset('decon',data=decon)
            this_expt.create_dataset('t_offset',data=t_offset)
            this_expt.create_dataset('stimulus_parameters',data=stimulus_parameters)
            this_expt['nbefore'] = nbefore
            this_expt['nafter'] = nafter
    return grouplist

def add_evan_data_struct_h5(filename, cell_type='PyrL23', keylist=None, frame_rate_dict=None, proc=None, nbefore=8, nafter=8,featurenames=['size','contrast','angle'],datasetnames=None,groupname='size_contrast'):
    if datasetnames is None:
        datasetnames = ['stimulus_'+name for name in featurenames]
    with ut.hdf5edit(filename) as data_struct:
        grouplist = [None]*len(keylist)
        for ikey,key in enumerate(keylist):
            dfof = proc[key]['dtrialwise'][:]
            decon = proc[key]['strialwise'][:] 
            trialwise = proc[key]['trialwise'][:] 
            trialrun = proc[key]['trialrun'][:] # 4 cm from disk ctr to estimated mouse location

            ufeature_list,stimulus_id = gen_stimulus_id(proc[key],featurenames)

            cell_id = np.arange(dfof.shape[0])
            session_id = key
            mouse_id = key.split('_')[1]
            
            if not session_id in data_struct.keys():
                this_session = data_struct.create_group(session_id)
                this_session['mouse_id'] = mouse_id
                this_session['cell_type'] = cell_type
                this_session.create_dataset('cell_id',data=cell_id)
                
                for field in ['cell_depth','cell_mask','cell_center','mean_red_channel','mean_green_channel']:
                    if field in proc[key]:
                        this_session.create_dataset(field,data=proc[key][field])
            else:
                this_session = data_struct[session_id]

            exptno = 0
            while groupname+'_'+str(exptno) in this_session.keys():
                exptno = exptno+1
            this_expt = this_session.create_group(groupname+'_'+str(exptno))

            grouplist[ikey] = session_id + '/' + groupname + '_' + str(exptno)

            this_expt.create_dataset('stimulus_id',data=stimulus_id)
            stimulus_parameters = [n.encode('ascii','ignore') for n in datasetnames]
            for name,ufeature in zip(datasetnames,ufeature_list):
                this_expt.create_dataset(name,data=ufeature)

            this_expt.create_dataset('trialrun',data=trialrun)
            this_expt.create_dataset('F',data=dfof)
#            this_expt.create_dataset('raw_trialwise',data=proc[key]['raw_trialwise'][:])
#            this_expt.create_dataset('neuropil_trialwise',data=proc[key]['neuropil_trialwise'][:])
            this_expt.create_dataset('decon',data=decon)
            this_expt.create_dataset('nondecon',data=trialwise)
            this_expt.create_dataset('stimulus_parameters',data=stimulus_parameters)
            this_expt['nbefore'] = nbefore
            this_expt['nafter'] = nafter
    return grouplist

def gen_stimulus_id(proc,featurenames):
    # inputs: proc, an hdf5 group proc[key]
    # featurenames, a list of either (1) strings, or (2) functions that take proc as argument and return:
    # i. a list of unique values this feature can take on
    # ii. a (# of trials,) array, where array==i iff feature==(list element i)
    stimulus_id_list = [None]*len(featurenames)
    ufeature_list = [None]*len(featurenames)
    for ifeat,feature in enumerate(featurenames):
        if type(feature) is str:
            ufeature_list[ifeat],stimulus_id_list[ifeat] = np.unique(proc[feature][:],return_inverse=True)
        else:
            ufeature_list[ifeat],stimulus_id_list[ifeat] = feature(proc) 
    stimulus_id = np.concatenate([ifeature[np.newaxis] for ifeature in stimulus_id_list],axis=0)
    return ufeature_list,stimulus_id    

def gen_session_id(foldname):
    session_id = 'session_'+foldname[:-1].replace('/','_')
    return session_id

def gen_datafiles(datafoldbase,foldname,filename,nplanes=4):
    datafold = datafoldbase+foldname+'ot/'
    numstrings = ['%03d' % i for i in range(nplanes)]
    datafiles = [filename+'_ot_'+number+'.rois' for number in numstrings]
    datafiles = [datafold+file for file in datafiles]
    datafiles = [x for x in datafiles if os.path.exists(x)]
    return datafiles

def gen_stimfile(stimfoldbase,foldname,filename):
    stimfile = stimfoldbase+foldname+filename+'.mat'
    return stimfile

def gen_ret_vars(retfile,stimfile):

    needed_ret_var_names = ['pval_ret','ret']
    needed_ret_var_vals = ut.loadmat(retfile,needed_ret_var_names)
    ret_vars = {key:val for key,val in zip(needed_ret_var_names,needed_ret_var_vals)}

    result = ut.loadmat(stimfile,'result')[()]
    ret_vars['position'] = result['position']

    paramdict_normal = ut.loadmat(retfile,'paramdict_normal')
    ret_vars['paramdict_normal'] = ut.matfile_to_dict(paramdict_normal)

    return ret_vars

def load_msk_ctr(filename):
    msk,ctr = ut.loadmat(filename,['msk','ctr'])
    return msk.astype('bool').transpose((2,0,1)), ctr.T

def compute_tuning(dsfile,exptname='retinotopy_0',run_cutoff=-np.inf,criterion_cutoff=0.4):
    with ut.hdf5read(dsfile) as f:
        keylist = [key for key in f.keys()]
        tuning = [None for i in range(len(keylist))]
        uparam = [None for i in range(len(keylist))]
        for ikey in range(len(keylist)):
            session = f[keylist[ikey]]
            if exptname in session:
                sc0 = session[exptname]
                data = sc0['decon'][:]
                stim_id = sc0['stimulus_id'][:]
                nbefore = sc0['nbefore'][()]
                nafter = sc0['nafter'][()]
                trialrun = np.nanmean(sc0['running_speed_cm_s'][:,nbefore:-nafter],-1)>run_cutoff
                if np.nanmean(trialrun)>criterion_cutoff:
                    tuning[ikey] = ut.compute_tuning(data,stim_id,trial_criteria=trialrun)[:]
                stim_params = sc0['stimulus_parameters']
                uparam[ikey] = [sc0[x][:] for x in stim_params]
        relevant_list = [key for key in keylist if exptname in f[key]]    
    return tuning,uparam,relevant_list

