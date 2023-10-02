#!/usr/bin/env python

import opto_utils
import numpy as np
import matplotlib.pyplot as plt
import pyute as ut
import sklearn

def use_sc_4_6(this_sc):
    return np.all(~np.isnan(np.nanmean(np.nanmean(this_sc[:,[0,2,3,5],:,:,:],3),0)))

def use_fg(this_fg):
    return np.all(~np.isnan(np.nanmean(np.nanmean(this_fg[:,:,:],2),0)))

def extract_with_classifier(dataset_info,keylist_dsi,rsdict,keylist_rs,session_dict,expttype,outline_frame_width=25,optotype=0,classifier_cutoff=0.5,dcutoff_fn=lambda x: x<10,to_use_in_avg_fn=use_sc_4_6,showy=True,ori_type_to_show='avg'):
    ncelltypes = 3
    rcelltype = [np.zeros((0,)+rsdict[expttype][0].shape[1:]) for icelltype in range(ncelltypes)]
    riexpt = [np.zeros((0,)) for icelltype in range(ncelltypes)]
    routlines = [np.array(()) for icelltype in range(ncelltypes)]
    frame_width = outline_frame_width
    for iexpt in range(len(dataset_info)):
        dii = dataset_info[iexpt]
        ituning = np.where(np.array([k==keylist_rs[iexpt] for k in keylist_dsi[optotype]]))[0]
        print(ituning)
        if ituning.size:
            ituning = ituning[0]
        if 'classifier' in dii and not dii['classifier'] is None and ituning.size:
            candidates = np.where(safe_predict_proba_(dii['classifier']['logreg'],dii['image_data']['features'])>classifier_cutoff)[0]

            this_tuning = rsdict[expttype][optotype][session_dict[expttype][optotype]==ituning]
            this_size_contrast = rsdict['size_contrast_opto_0'][optotype][session_dict[expttype][optotype]==ituning]
            this_ret = rsdict['retinotopy_0'][optotype][session_dict[expttype][optotype]==ituning]
            looks_ok = opto_utils.ret_aligned(this_ret,dcutoff_fn=dcutoff_fn)
            goes_up,pval = opto_utils.spont_effect(this_size_contrast[candidates])
            to_use_in_avg = to_use_in_avg_fn(this_tuning)
            for icelltype,select_on in enumerate([~goes_up,goes_up]):
                these_candidates = candidates[select_on & (pval < 0.05)]
                outlines = [None for icand in range(len(these_candidates))]
                sc_celltype = this_tuning[these_candidates]
                if sc_celltype.size: 
                    if expttype is 'size_contrast_opto_0':
                        sc_celltype_small = select_ori(sc_celltype,ori_type=ori_type_to_show,ori_axis=3)
                        sc_celltype_small = sc_celltype_small[:,~np.isnan(sc_celltype_small.sum(3).sum(2).sum(0))]
                        if sc_celltype_small.size:
                            ut.imshow_in_pairs(sc_celltype_small[:,:,:,0],sc_celltype_small[:,:,:,1])
                            to_show = np.nanmean(opto_utils.norm_to_mean_light_off(sc_celltype_small),0)
                            #usize = dii['tuning_data']['uparam'][0]
                            if showy:
                                plt.figure()
                                opto_utils.show_light_on_light_off(to_show)#,usize=usize)
                    elif expttype is 'figure_ground_opto_0':
                        sc_celltype_small = select_ori(sc_celltype,ori_type=ori_type_to_show,ori_axis=2)
                        if sc_celltype_small.size:
                            ut.imshow_in_pairs(sc_celltype_small[:,np.newaxis,:,0],sc_celltype_small[:,np.newaxis,:,1])
                            to_show = np.nanmean(opto_utils.norm_to_mean_light_off(sc_celltype_small),0)
                            #usize = dii['tuning_data']['uparam'][0]
                            if showy:
                                plt.figure()
                                opto_utils.show_light_on_light_off(to_show[:,np.newaxis])#,usize=usize)
                        
                    if showy:
                        plt.figure()
                        for iiroi in range(len(these_candidates)): #len(candidates)):
                            iroi = these_candidates[iiroi]
                            plt.subplot(10,10,iiroi+1)
                            img = dii['image_data']['r2'][dii['image_data']['depth'][iroi]]
                            outlines[iiroi] = opto_utils.show_roi_boundaries(np.minimum(1.5*img/img.max(),1),dii['image_data']['msk'][iroi],frame_width=frame_width)
                    if to_use_in_avg:
                        rcelltype[icelltype] = np.concatenate((rcelltype[icelltype],sc_celltype),axis=0)
                        riexpt[icelltype] = np.concatenate((riexpt[icelltype],iexpt*np.ones((sc_celltype.shape[0],))))
                        outlines = np.concatenate([this_outline[np.newaxis] for this_outline in outlines],axis=0)
                        if not routlines[icelltype].size:
                            routlines[icelltype] = outlines.copy()
                        else:
                            routlines[icelltype] = np.concatenate((routlines[icelltype],outlines),axis=0)
            if to_use_in_avg:
                icelltype = 2
                rcelltype[icelltype] = np.concatenate((rcelltype[icelltype],np.nanmean(this_tuning[looks_ok],0)[np.newaxis]),axis=0)
                riexpt[icelltype] = np.concatenate((riexpt[icelltype],iexpt*np.ones((1,))))
    return rcelltype,riexpt,routlines

def extract_outlines(image_data,candidates=None,frame_width=25,show_outline=True):
    if candidates is None:
        candidates = np.arange(image_data['msk'].shape[0])
    nroi = len(candidates)
    outlines = [None for icand in range(nroi)]
    for iiroi,iroi in enumerate(candidates):
        img = image_data['r2'][image_data['depth'][iroi]]
        outlines[iiroi] = opto_utils.show_roi_boundaries(np.minimum(1.5*img/img.max(),1),image_data['msk'][iroi],frame_width=frame_width,show_outline=show_outline)
    return outlines

def extract_outlines_red_green(image_data,candidates=None,frame_width=25):
    if candidates is None:
        candidates = np.arange(image_data['msk'].shape[0])
    nroi = len(candidates)
    outlines = [None for icand in range(nroi)]
    for iiroi,iroi in enumerate(candidates):
        img_red = image_data['r'][image_data['depth'][iroi]][:,:,np.newaxis]
        img_grn = image_data['g'][image_data['depth'][iroi]][:,:,np.newaxis]
        img_rgb = np.concatenate((img_red,img_grn,np.zeros_like(img_red)),axis=2)
        outlines[iiroi] = opto_utils.show_roi_boundaries(\
                np.minimum(1.5*img_rgb/img_rgb.max(),1),\
                image_data['msk'][iroi],frame_width=frame_width)
    return outlines

def select_ori(sc_celltype,ori_type='avg',ori_axis=3):
    slicer = [slice(None) for idim in range(len(sc_celltype.shape))]
    if ori_type=='avg':
        slicer[ori_axis] = slice(None)
    elif ori_type=='pref':
        slicer[ori_axis] = [0,4]
    elif ori_type=='orth':
        slicer[ori_axis] = [2,6]
    elif ori_type=='oblique':
        slicer[ori_axis] = [1,3,5,7]
    sc_celltype_small = np.nanmean(sc_celltype[tuple(slicer)],axis=ori_axis)
    return sc_celltype_small

def compute_spont_auroc(spont):
    nroi = spont.shape[0]
    spont_auroc = np.zeros((nroi,))
    for iroi in range(nroi):
        spont_auroc[iroi] = ut.compute_auroc(spont[iroi,:,0],spont[iroi,:,1])
    return spont_auroc
           
def gen_classifier(image_data,tuning_data,look_for = lambda x: x>np.percentile(x,95),red_image='r',feature_list=None,nbefore=8,nafter=8,summary_stat='slope_score'):
    tuning_data['r'] = tuning_data['tuning'][:,:,:,:,:,nbefore:-nafter].mean(-1)
    tuning_data['sc'] = tuning_data['r'].mean(-2) #.mean(-1)
    
    slope,score = opto_utils.roiwise_regression_slope(tuning_data['sc'])
    
    tuning_data['slope'] = slope
    tuning_data['score'] = score
    
    spont = opto_utils.compute_spont_light_on_light_off(tuning_data['r'])
    
    if summary_stat=='slope_score':
        decide_on = slope.copy()
    elif summary_stat=='spont_change':
        spont_auroc = compute_spont_auroc(spont)
        decide_on = spont_auroc.copy()
        
    tuning_data['spont'] = spont
    tuning_data['spont_auroc'] = spont_auroc
        
    
    feature_dict = opto_utils.compute_image_features(image_data[red_image],image_data['g'],image_data['msk'],image_data['depth'],summary_stat=np.median)
    
    def gen_features(feature_dict,feature_list=None):
        if feature_list is None:
            feature_list = [feature_dict[key] for key in ['rednucleus','redaround','greenlvl','redgreencorr','redmembrane','redcv']]
            feature_list = feature_list + [feature_dict['redlvl']/feature_dict['greenlvl']]
            feature_list = feature_list + [feature_dict['redlvl']/feature_dict['redaround']]
        features = np.concatenate([f[tuning_data['cell_criteria']][:,np.newaxis] for f in feature_list],axis=1)
        return features
    
    image_data['feature_dict'] = feature_dict
    image_data['features_fn'] = gen_features
    features = image_data['features_fn'](image_data['feature_dict'])
    image_data['features'] = features

    criterion = gen_criterion(tuning_data,image_data,summary_stat=summary_stat) #(score>0) & ~np.isnan(features.sum(1))

#     features = sst.zscore(features[criterion],axis=0)
#     these_features = features[criterion]

    train = (np.random.randn(features.shape[0])>0) & criterion
    if not np.all(np.isnan(decide_on[train])):
        outcome = look_for(decide_on[train])
        classifier = {}
        classifier['logreg'] = sklearn.linear_model.LogisticRegression()
        classifier['logreg'].fit(features[train],outcome)
        classifier['transform'] = (classifier['logreg'].coef_ * image_data['features']).sum(1)
        classifier['training_set'] = train
        classifier['prediction'] = safe_predict_(classifier['logreg'],features)
    else:
        classifier = None
    return image_data, tuning_data, classifier

def gen_criterion(tuning_data,image_data,summary_stat='slope_score'):
    criterion = ~np.isnan(image_data['features'].sum(1)) & np.all(np.abs(image_data['features'])<np.inf,axis=1)
    if summary_stat=='slope_score':
        criterion = criterion & (tuning_data['score']>0)
    return criterion

def safe_predict_(logreg,features):
    output = np.zeros((features.shape[0],),dtype='bool')
    lkat = ~np.isnan(features.sum(1)) & np.all(np.abs(features)<np.inf,axis=1)
    output[~lkat] = np.nan
    output[lkat] = logreg.predict(features[lkat])
    return output

def safe_predict_proba_(logreg,features):
    output = np.zeros((features.shape[0],))
    lkat = ~np.isnan(features.sum(1)) & np.all(np.abs(features)<np.inf,axis=1)
    output[~lkat] = np.nan
    output[lkat] = logreg.predict_proba(features[lkat])[:,1]
    return output

def extract_tuning_data(dsname,iexpt,running=False):
    ikey = iexpt
    with ut.hdf5read(dsname) as ds:
        keylist = list(ds.keys())
    tuning,cell_criteria,uparam = opto_utils.compute_tuning(dsname, running=False, dcutoff = np.inf,keylist=[keylist[ikey]])
    tuning_data = {}
    tuning_data['tuning'] = tuning[0]
    tuning_data['cell_criteria'] = cell_criteria[0]
    tuning_data['uparam'] = uparam[0]
    return tuning_data

def extract_image_data(dsname,keylist=None):
    with ut.hdf5read(dsname) as ds:
        if keylist is None:
            keylist = list(ds.keys())
        image_datas = {}
        for ikey in range(len(keylist)):
            print(keylist[ikey])
            image_data = {}
            source_keys = ['mean_green_channel','mean_red_channel','mean_green_channel_enhanced','mean_red_channel_corrected','cell_mask','cell_depth']
            target_keys = ['g','r','g2','r2','msk','depth']
            for skey,tkey in zip(source_keys,target_keys):
                if skey in ds[keylist[ikey]]:
                    if tkey=='depth':
                        image_data[tkey] = ds[keylist[ikey]][skey][:].astype('int')
                    else:
                        image_data[tkey] = ds[keylist[ikey]][skey][:]
                else:
                    print('%s not in dsfile'%skey)
            #image_data['g'] = ds[keylist[ikey]]['mean_green_channel'][:]
            #image_data['r'] = ds[keylist[ikey]]['mean_red_channel'][:]
            #image_data['g2'] = ds[keylist[ikey]]['mean_green_channel_enhanced'][:]
            #image_data['r2'] = ds[keylist[ikey]]['mean_red_channel_corrected'][:]
            #image_data['msk'] = ds[keylist[ikey]]['cell_mask'][:]
            #image_data['depth'] = ds[keylist[ikey]]['cell_depth'][:].astype('int')
            image_datas[keylist[ikey]] = image_data
    return image_datas

def extract_and_classify(dsname,iexpt,look_for = lambda x: x>np.percentile(x,95), running=False):
    tuning_data = extract_tuning_data(dsname,iexpt,running=False)
    image_data = extract_image_data(dsname,iexpt)
    if not np.any(np.isnan(image_data['r'])): # and not np.all(np.isnan(tuning_data['slope'])):
        image_data, tuning_data, classifier = gen_classifier(image_data,tuning_data,look_for=look_for)
    else:
        classifier = None
    return image_data,tuning_data,classifier
