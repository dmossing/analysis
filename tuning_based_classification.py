#!/usr/bin/env python

import numpy as np
import pyute as ut
import sklearn

def train_classifier(rsdict,between_which,roi_criterion_fn,stim_criterion_dict,dim_reduce_fn=None,transform_dict=None):
    # first take the desired cell types
    subdict = {key:[rsdict[key][this] for this in between_which] for key in rsdict}
    needed,orig_ind_dict = pare_down(subdict,roi_criterion_fn,stim_criterion_dict)
    if transform_dict is None:
#         temp = needed.copy()
        temp,transform_dict = dim_reduce_fn(needed.copy()) # done
#     print(needed['size_contrast_0'][0].shape)
    transformed = apply_transform_dict(needed,transform_dict)
    print(transformed['size_contrast_0'][0].shape)
#     for key in transform_dict:
#         tsc = this_stim_criterion[key].copy()
#         transform_dict[key] = lambda x: transform_dict[key](x[:,tsc])
    features,labels = gen_features_labels(transformed) # done
    orig_inds,_ = gen_features_labels(orig_ind_dict)
    train = np.random.randn(features.shape[0])>0
    rr = [features[these] for these in [train,~train]]
    yy = [labels[these] for these in [train,~train]]
    orig_inds = [orig_inds[these] for these in [train,~train]]
    classifier = {}
    classifier['logreg'] = sklearn.linear_model.LogisticRegression()
    classifier['logreg'].fit(features[train],labels[train])
    classifier['transform'] = (classifier['logreg'].coef_ * features).sum(1)
    classifier['training_set'] = train
    classifier['prediction'] = classifier['logreg'].predict(features)
    classifier['features'] = features
    classifier['labels'] = labels
    classifier['orig_inds'] = orig_inds
    classifier['transform_dict'] = transform_dict
    return classifier

def pare_down(subdict,roi_criterion_fn,stim_criterion_dict):
    this_subdict = listify(subdict.copy())
    right_rois,orig_ind_dict = apply_roi_criterion(this_subdict,roi_criterion_fn)
    needed = apply_needed_dict(right_rois,stim_criterion_dict)
    return needed,orig_ind_dict
    

def apply_roi_criterion(subdict,roi_criterion_fn):
    orig_ind_dict = {key:[np.arange(this.shape[0]) for this in subdict[key]] for key in subdict}
    this_roi_criterion = roi_criterion_fn(subdict)
    new_subdict = {key:[subdict[key][itype][rc] for itype,rc in enumerate(this_roi_criterion)] for key in subdict}
    new_orig_ind_dict = {key:[orig_ind_dict[key][itype][rc] for itype,rc in enumerate(this_roi_criterion)] for key in orig_ind_dict}
    return new_subdict,new_orig_ind_dict

def gen_features_labels(this_subdict):
    subdict = listify(this_subdict)
    keylist = list(subdict.keys())
    ntypes = len(subdict[keylist[0]])
    nkeys = len(keylist)
    nrois = [subdict[keylist[0]][itype].shape[0] for itype in range(ntypes)]
    
    labels = [itype*np.ones((nrois[itype],)) for itype in range(ntypes)]
    features = [[subdict[key][itype].reshape((nrois[itype],-1)) for itype in range(ntypes)] for key in subdict]
    features = [np.concatenate([features[ikey][itype] for ikey in range(nkeys)],axis=1) for itype in range(ntypes)]
    
    features = np.concatenate(features,axis=0)
    labels = np.concatenate(labels,axis=0)
    
    return features,labels

def listify(this_subdict):
    keylist = list(this_subdict.keys())
    nkeys = len(keylist)
    if isinstance(this_subdict[keylist[0]],list):
        ntypes = len(this_subdict[keylist[0]])
        subdict = this_subdict.copy()
    else: 
        ntypes = 1
        subdict = {key:[this_subdict[key].copy()] for key in this_subdict}
    return subdict

def reduce_to_ndim(arr,ndim=10):
    def flatten_and_norm_to_mean(arr0):
        arr_flat = arr0.reshape((arr0.shape[0],-1))
        arr_flat = arr_flat/arr_flat.mean(1)[:,np.newaxis]
        return arr_flat
    arr_flat = flatten_and_norm_to_mean(arr)
    u,s,v = np.linalg.svd(arr_flat,full_matrices=False)
    u_small = u[:,:ndim]
    s_small = np.diag(s[:ndim])
    v_small = v[:ndim,:]
    def transform_fn(arr1):
        arr1_flat = flatten_and_norm_to_mean(arr1)
        return arr1_flat @ v_small.T @ np.linalg.inv(s_small)
    return u_small,transform_fn

def dim_reduce_fn(subdict,fn_dict):
    
#     keylist = list(subdict.keys())
#     ntypes = len(subdict[keylist[0]])
#     nrois = [subdict[keylist[0]][itype].shape[0] for itype in range(ntypes)]
    nrois = get_nrois(subdict)
    
    reduced_dict = subdict.copy()
    transform_dict = {}
    a_reduced_dict = {}
    
    for key in fn_dict:
        fn = fn_dict[key]
        a_reduced_dict[key] = [None for itype in range(len(reduced_dict[key]))]
        for itype in range(len(reduced_dict[key])):
            a_reduced_dict[key][itype],transform_dict[key] = fn(reduced_dict[key][itype])
        
    return a_reduced_dict,transform_dict

def get_nrois(subdict):
    keylist = list(subdict.keys())
    ntypes = len(subdict[keylist[0]])
    nrois = [subdict[keylist[0]][itype].shape[0] for itype in range(ntypes)]
    return nrois

def empty_out(arr):
    return only_empty_out(arr),only_empty_out

def only_empty_out(arr):
    return arr[:,:0]

def keep_rois_ret_aligned(subdict,dcutoff=20):
    r0 = subdict['retinotopy_0']
    to_keep = [None for rtype in r0]
    for itype,rtype in enumerate(r0):
        sig = rtype[:,paramname_to_ind('rf_mapping_pval')]<0.05
        well_fit = rtype[:,paramname_to_ind('rf_sq_error')]<1
        near = rtype[:,paramname_to_ind('rf_distance_deg')]<dcutoff
        to_keep[itype] = ut.k_and(sig,well_fit,near) 
    return to_keep

def keep_rois_with_needed_stims(subdict,needed_dict):
    nrois = get_nrois(subdict)
    ntypes = len(nrois)
    to_keep = [np.ones((nroi,),dtype='bool') for nroi in nrois]
#     to_keep = np.ones((nrois,),dtype='bool')
    for key in subdict:
        for itype in range(ntypes):
            to_keep[itype] = np.logical_and(to_keep[itype],np.all(~np.isnan(subdict[key][itype][:,needed_dict[key]]),axis=1))
    return to_keep

def gen_needed_dict_from_rsdict(rsdict):
    needed_dict = {}
    for key in rsdict:
        needed_dict[key] = ~np.isnan(np.sum(rsdict[key],0))
    return needed_dict

def roi_criterion(subdict,needed_dict,dcutoff=20):
    # outputs a boolean array
    keep1 = keep_rois_ret_aligned(subdict,dcutoff=dcutoff)
    keep2 = keep_rois_with_needed_stims(subdict,needed_dict)
    return [(k1 & k2) for k1,k2 in zip(keep1,keep2)]

def paramname_to_ind(paramname):
    ind = np.where([x is paramname for x in ret_params])[0][0]
    return ind

ret_params = ['rf_mapping_pval','rf_sq_error','rf_sigma','rf_distance_deg']

def apply_transform_dict(a_subdict,transform_dict):
#     print(a_subdict['size_contrast_0'][0].shape)
    transformed = listify(a_subdict)
#     print(a_subdict['size_contrast_0'][0].shape)
    for key in transformed:
        transformed[key] = [transform_dict[key](tk) for tk in transformed[key]]
    return transformed

def apply_needed_dict(subdict,needed_dict):
    needed = listify(subdict)
    needed = {key:[sk[:,needed_dict[key]] for sk in needed[key]] for key in needed}
    return needed

def classify_new_data(to_classify,needed_dict,transform_dict,logreg):
    needed = apply_needed_dict(to_classify,needed_dict)
    transformed = apply_transform_dict(needed,transform_dict)
    features,labels = gen_features_labels(transformed)
    looks_ok = ~np.isnan(features.sum(1))
    good_ones = logreg.predict_proba(features[looks_ok])[:,1:]
    pred_proba = np.zeros((looks_ok.shape[0],good_ones.shape[1]))
    pred_proba[looks_ok] = good_ones
    pred_proba[~looks_ok] = np.nan
    return pred_proba
    
def apply_roc(new_vals,roc):
    xvals,fp,tp = roc
    dig = np.digitize(new_vals,xvals)
    return fp[dig],tp[dig]

class Classifier(object):
    needed_dict = None
    def __init__(self,to_classify=None,based_on=None,between_which=None,dim_reduce_fn_dict=None):
        self.to_classify = to_classify
        self.based_on = based_on
        self.between_which = between_which
        self.needed_dict = gen_needed_dict_from_rsdict(to_classify)
        self.needed_dict['size_contrast_0'][-1,:,:] = False
        print(self.needed_dict['retinotopy_0'])
        self.roi_criterion_fn = lambda x: roi_criterion(x,self.needed_dict,dcutoff=20)
#         self.stim_criterion_dict = self.needed_dict #lambda x: tuple([slice(None),self.needed_dict#{x[key][:,self.needed_dict[key]] for key in x}
        self.dim_reduce_fn = lambda x: dim_reduce_fn(x,dim_reduce_fn_dict)
        pared,_ = pare_down(to_classify,self.roi_criterion_fn,self.needed_dict)
        _,transform_dict = self.dim_reduce_fn(pared.copy())
        self.classifier = train_classifier(self.based_on,self.between_which,self.roi_criterion_fn,self.needed_dict,dim_reduce_fn=self.dim_reduce_fn,transform_dict=transform_dict)
        self.classifier['pred_proba_labeled'] = self.classifier['logreg'].predict_proba(self.classifier['features'])[:,1:]
        self.ntypes = len(between_which)
#         self.classifier['pred_proba_new'] = 
        self.classifier['pred_proba_new'] = classify_new_data(to_classify,self.needed_dict,self.classifier['transform_dict'],self.classifier['logreg'])
        this_to_classify = listify(to_classify)
        self.classifier['meeting_criterion_new'] = self.roi_criterion_fn(this_to_classify)
        x = self.classifier['pred_proba_labeled'][self.classifier['labels']==0][:,0]
        y = self.classifier['pred_proba_labeled'][self.classifier['labels']==1][:,0]
        self.classifier['roc'] = ut.compute_roc(x,y)
        self.classifier['false_positive_new'],self.classifier['true_positive_new'] = apply_roc(self.classifier['pred_proba_new'],self.classifier['roc'])        