#!/usr/bin/env python

import autograd.numpy as np
from autograd import elementwise_grad as egrad
import scipy.optimize as sop
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import pandas as pd
import pyute as ut
import matplotlib.pyplot as plt
import autograd.scipy.special as ssp
import scipy.stats as sst
import scipy.io as sio
from matplotlib.colors import ListedColormap
import von_mises_analysis as vma

# visual stim ordering convention
nub_locs = np.array([(0,0),(1,0),(0,1),(-1,0),(0,-1)])
# number of patches/nubs
nnub = nub_locs.shape[0]
# (32,5) array indicating which patches are active for each stimulus
nubs_active = np.concatenate([np.array([int(d) for d in '{0:b}'.format(x).zfill(5)])[np.newaxis] for x in range(32)],axis=0)

### not used in final analysis
and_gates = np.logical_and(nubs_active[:,1:],nubs_active[:,0:1]) # - 0.5*nubs_active[:,1:] - 0.5*nubs_active[:,0:1]
or_gates = np.logical_or(nubs_active[:,1:],nubs_active[:,0:1]) # - 0.5*nubs_active[:,1:] - 0.5*nubs_active[:,0:1]
xor_gates = np.logical_xor(nubs_active[:,1:],nubs_active[:,0:1]) # - 0.5*nubs_active[:,1:] - 0.5*nubs_active[:,0:1]
xnor_gates = ~np.logical_xor(nubs_active[:,1:],nubs_active[:,0:1]) # - 0.5*nubs_active[:,1:] - 0.5*nubs_active[:,0:1]
nubs_and = np.concatenate((nubs_active,and_gates),axis=1)
nubs_or = np.concatenate((nubs_active,or_gates),axis=1)
nubs_xor = np.concatenate((nubs_active,xor_gates),axis=1)
nubs_xnor = np.concatenate((nubs_active,xnor_gates),axis=1)
###

# similar to parula colormap, ported to python
parula_path = '/Users/dan/Documents/code/adesnal/'
parula_filename = parula_path+'matlab_parula_colormap.mat'
parula = ListedColormap(sio.loadmat(parula_filename)['cmap'][()])

###
# not used in final analysis

def f_miller_troyer(k,mu,s2):
    u = mu/np.sqrt(2*s2)
    A = 0.5*mu*(1+ssp.erf(u))
    B = np.sqrt(s2)/np.sqrt(2*np.pi)*np.exp(-u**2)
    return k*(A + B)

def f_mt(inp):
    return f_miller_troyer(1,inp,1)

def requ(inp):
    return inp**2*(inp>0)

def predict_output(s0,offset,fn=None,nub_var=nubs_active):
    inp = (s0[np.newaxis,:]*nub_var).sum(1) + offset
    return fn(inp)

def unpack_theta(theta):
    s0 = theta[:-1]
    offset = theta[-1]
    return s0,offset

def unpack_theta_amplitude(theta):
    s0 = theta[:-2]
    offset = theta[-2]
    amplitude = theta[-1]
    return s0,offset,amplitude

def unpack_gaussian_theta(theta):
    mu = theta[:2]
    sigma2 = theta[2]
    amplitudeI = theta[3]
    offset1 = theta[4]
    offset2 = theta[5]
    amplitudeO = theta[6]
    return mu,sigma2,amplitudeI,offset1,offset2,amplitudeO

def predict_output_theta(theta,fn=None,nub_var=nubs_active):
    s0,offset = unpack_theta(theta)
    ypred = predict_output(s0,offset,fn=fn,nub_var=nub_var)
    return ypred

def predict_output_theta_amplitude(theta,fn=None,nub_var=nubs_active):
    s0,offset,amplitude = unpack_theta_amplitude(theta)
    ypred = amplitude*predict_output(s0,offset,fn=fn,nub_var=nub_var)
    return ypred

def predict_output_gaussian_theta(theta,fn=None,nub_var=nubs_active):
    mu,sigma2,amplitudeI,offset1,offset2,amplitudeO = unpack_gaussian_theta(theta)
    rf = gaussian_fit_to_nub_rf(mu,sigma2,amplitudeI,offset1)
    ypred = amplitudeO*predict_output(rf,offset2,fn=fn,nub_var=nub_var)
    return ypred

def fit_output(this_response,fn=None,nub_var=nubs_active):
    def minusL(theta):
        ypred = predict_output_theta(theta,fn=fn,nub_var=nub_var)
        return ((this_response - ypred)**2).sum()
    def minusdLdtheta(theta):
        return egrad(minusL)(theta)
    theta0 = 1 + 2*np.concatenate((((nub_var-0.5)*this_response[:,np.newaxis]).sum(0),(1,)))
    #theta0 = 1+np.concatenate((this_response[[16,8,4,2,1]],(1,)))
    thetastar = sop.fmin_l_bfgs_b(minusL,theta0,fprime=minusdLdtheta)
    return thetastar

def initialize_theta_amplitude(this_response,nub_var=nubs_active):
    theta_dir = (this_response[:,np.newaxis]*(nub_var-0.5)).sum(0)
    amplitude = np.sqrt((this_response**2).sum())
    offset = (this_response/amplitude - (theta_dir[np.newaxis]*(nub_var-0.5)).sum(1)).mean()
    theta0 = np.concatenate((theta_dir,(offset,amplitude)))
    return theta0

def fit_output_amplitude(this_response,fn=None,theta0=None,lpoisson=0,nub_var=nubs_active,bounds=None,lam=0): # ,land=0
    scaleby = 1
    amplitude = np.sqrt((this_response**2).sum())/scaleby
    norm_response = this_response/amplitude
    fudge = 1e-4
    def minusL(theta):
        ypred = predict_output_theta_amplitude(theta,fn=fn,nub_var=nub_var)
        #if land:
        #    return ((norm_response - ypred)**2*(1+lpoisson/(ypred+norm_response+fudge))).sum() - land*np.abs(theta[5:9]/theta[-2]).sum()
        #else:
        if lam:
            l1_penalty = lam*np.abs(theta[nnub:-2]).sum()
        else:
            l1_penalty = 0
        return ((norm_response - ypred)**2*(1+lpoisson/(ypred+norm_response+fudge))).sum() + l1_penalty
    def minusdLdtheta(theta):
        return egrad(minusL)(theta)
    if theta0 is None:
        theta0 = initialize_theta_amplitude(norm_response,nub_var=nub_var)
        #ycorr = predict_output_theta_amplitude(theta0)
        #theta0 = 1 + 2*np.concatenate((((nubs_active-0.5)*this_response[:,np.newaxis]).sum(0),(1,),(0.1,)))
    #theta0 = 1+np.concatenate((this_response[[16,8,4,2,1]],(1,)))
    thetastar_temp = fit_output_amplitude_fixed_rf(this_response,fn=fn,theta0=theta0,nub_var=nub_var)
    if bounds is None:
        bounds = [(-np.inf,np.inf) for x in range(theta0.size-1)] + [(0,1)]
    #if nub_var.shape[1]==9:
    #    bounds[5:9] = [(0,0) for x in range(4)]
    thetastar = sop.fmin_l_bfgs_b(minusL,theta0,fprime=minusdLdtheta,bounds=bounds)
    thetastar[0][-1] = thetastar[0][-1]*amplitude
    return thetastar

def evaluate_output_amplitude(this_response,theta0,fn=None,lpoisson=0,nub_var=nubs_active):
    scaleby = 1
    amplitude = np.sqrt((this_response**2).sum())/scaleby
    norm_response = this_response/amplitude
    fudge = 1e-4
    theta = theta0.copy()
    theta[-1] = theta[-1]/amplitude
    def minusL(theta):
        ypred = predict_output_theta_amplitude(theta,fn=fn,nub_var=nub_var)
        return ((norm_response - ypred)**2*(1+lpoisson/(ypred+norm_response+fudge))).sum()
    return minusL(theta)

def fit_output_amplitude_offset(this_response,fn=None,theta0=None,lpoisson=0,nub_var=nubs_active,bounds=None,lam=0,allow_amplitude=True): # ,land=0
    scaleby = 1
    amplitude = np.sqrt((this_response**2).sum())/scaleby
    norm_response = this_response/amplitude
    fudge = 1e-4
    def minusL(theta):
        ypred = predict_output_theta_amplitude_offset(theta,fn=fn,nub_var=nub_var)
        #if land:
        #    return ((norm_response - ypred)**2*(1+lpoisson/(ypred+norm_response+fudge))).sum() - land*np.abs(theta[5:9]/theta[-2]).sum()
        #else:
        if lam:
            l1_penalty = lam*np.abs(theta[nnub:nub_var.shape[1]]).sum()
        else:
            l1_penalty = 0
        return ((norm_response - ypred)**2*(1+lpoisson/(ypred+norm_response+fudge))).sum() + l1_penalty
    def minusdLdtheta(theta):
        return egrad(minusL)(theta)
    if theta0 is None:
        theta0 = initialize_theta_amplitude(norm_response,nub_var=nub_var)
        theta0 = np.concatenate((theta0,(0.,)))
        #ycorr = predict_output_theta_amplitude(theta0)
        #theta0 = 1 + 2*np.concatenate((((nubs_active-0.5)*this_response[:,np.newaxis]).sum(0),(1,),(0.1,)))
    #theta0 = 1+np.concatenate((this_response[[16,8,4,2,1]],(1,)))
    #thetastar_temp = fit_output_amplitude_fixed_rf(this_response,fn=fn,theta0=theta0,nub_var=nub_var)
    if bounds is None:
        if allow_amplitude:
            bounds = [(-np.inf,np.inf) for x in range(theta0.size-2)] + [(0,1),(-np.inf,np.inf)]
        else:
            bounds = [(-np.inf,np.inf) for x in range(theta0.size-2)] + [(1,1),(-np.inf,np.inf)]
    #if nub_var.shape[1]==9:
    #    bounds[5:9] = [(0,0) for x in range(4)]
    thetastar = sop.fmin_l_bfgs_b(minusL,theta0,fprime=minusdLdtheta,bounds=bounds)
    #thetastar[0][nub_var.shape[1]+1] = thetastar[0][nub_var.shape[1]+1]*amplitude
    thetastar[0][nub_var.shape[1]+1:] = thetastar[0][nub_var.shape[1]+1:]*amplitude
    return thetastar

def evaluate_output_amplitude_offset(this_response,theta0,fn=None,lpoisson=0,nub_var=nubs_active):
    scaleby = 1
    amplitude = np.sqrt((this_response**2).sum())/scaleby
    norm_response = this_response/amplitude
    fudge = 1e-4
    theta = theta0.copy()
    theta[nub_var.shape[1]+1] = theta[nub_var.shape[1]+1]/amplitude
    def minusL(theta):
        ypred = predict_output_theta_amplitude_offset(theta,fn=fn,nub_var=nub_var)
        return ((norm_response - ypred)**2*(1+lpoisson/(ypred+norm_response+fudge))).sum()
    return minusL(theta)

def predict_output_theta_amplitude_offset(theta,fn=None,nub_var=nubs_active):
    s0,offsetI,amplitude,offsetO = unpack_theta_amplitude_offset(theta)
    ypred = amplitude*predict_output(s0,offsetI,fn=fn,nub_var=nub_var)+offsetO
    return ypred

def unpack_theta_amplitude_offset(theta):
    s0 = theta[:-3]
    offsetI = theta[-3]
    amplitude = theta[-2]
    offsetO = theta[-1]
    return s0,offsetI,amplitude,offsetO

def fit_output_amplitude_fixed_rf(this_response,fn=None,theta0=None,nub_var=nubs_active):
    scaleby = 1
    amplitude = np.sqrt((this_response**2).sum())/scaleby
    norm_response = this_response/amplitude
    def minusL(theta):
        ypred = predict_output_theta_amplitude(theta,fn=fn,nub_var=nub_var)
        return ((norm_response - ypred)**2).sum()
    def minusdLdtheta(theta):
        return egrad(minusL)(theta)
    if theta0 is None:
        theta0 = initialize_theta_amplitude(norm_response,nub_var=nub_var)
        #theta0 = 1 + 2*np.concatenate((((nubs_active-0.5)*this_response[:,np.newaxis]).sum(0),(1,),(0.1,)))
    #theta0 = 1+np.concatenate((this_response[[16,8,4,2,1]],(1,)))
    bounds = [(x,x) for x in theta0[:nub_var.shape[1]]] + [(-np.inf,1) for iparam in range(2)]
    thetastar = sop.fmin_l_bfgs_b(minusL,theta0,fprime=minusdLdtheta,bounds=bounds)
    thetastar[0][-1] = thetastar[0][-1]*amplitude
    return thetastar

def compute_overlap(mu,sigma2,diam=1):
    def compute_component(mu,sigma2):
        return 0.5*(ssp.erf((0.5*diam - mu)/np.sqrt(2*sigma2)) - ssp.erf((-0.5*diam - mu)/np.sqrt(2*sigma2)))
    components = [compute_component(mu[iaxis],sigma2) for iaxis in range(2)]
    return np.prod(components)

def gaussian_fit_to_nub_rf(mu,sigma2,amplitude,offset):
    nub_locs = np.array([(0,0),(1,0),(0,1),(-1,0),(0,-1)])
    rf = np.array([compute_overlap(mu - nub_locs[iloc],sigma2) for iloc in range(len(nub_locs))])
    rf = amplitude*(rf-offset)
    return rf

def fit_output_gaussian(this_response,fn=None):
    def minusL(theta):
        ypred = predict_output_gaussian_theta(theta,fn=fn)
        return ((this_response - ypred)**2).sum()
    def minusdLdtheta(theta):
        return egrad(minusL)(theta)
    single_patch_response = this_response[[16,8,4,2,1]]
    mu0 = nub_locs[np.argmax(single_patch_response)]
    A0 = np.sqrt(np.max(single_patch_response*(single_patch_response>0)))
    A1 = this_response.max()
    theta0 = np.concatenate((mu0,np.array((1,A0,0,0,A1))))
    bounds = [(-2,2),(-2,2),(1e-2,4),(-1,1),(-np.inf,np.inf),(-np.inf,np.inf),(0,np.inf)]
    thetastar = sop.fmin_l_bfgs_b(minusL,theta0,fprime=minusdLdtheta,bounds=bounds)
    return thetastar

def evaluate_gaussian(xx,yy,mu,sigma2,amplitude,offset):
    g = 1/np.sqrt(2*np.pi*sigma2)*np.exp(-0.5*((xx-mu[0])**2+(yy-mu[1])**2)/sigma2)
    return amplitude*(g-offset)

def evaluate_gaussian_theta(xx,yy,theta):
    mu,sigma2,amplitude,offset1,offset2 = unpack_gaussian_theta(theta)
    return evaluate_gaussian(xx,yy,mu,sigma2,amplitude,offset1)

def show_gaussian_fit(theta,bd=1.75,cbd=2):
    x = np.linspace(-bd,bd,100)
    y = np.linspace(bd,-bd,100)
    xx,yy = np.meshgrid(x,y)
    plt.imshow(evaluate_gaussian_theta(xx,yy,theta),extent=[-bd,bd,-bd,bd],cmap='bwr')
    plt.clim(-cbd,cbd)
    plt.scatter(nub_locs[:,0],nub_locs[:,1],c='g',marker='+')
    rects = []
    for inub in range(nub_locs.shape[0]):
        rect = Rectangle(nub_locs[inub]-np.array((0.5,0.5)),1,1)
        rects.append(rect)
    pc = PatchCollection(rects, facecolor='none', alpha=1, edgecolor='k')
    plt.gca().add_collection(pc)
    plt.xlim((-bd,bd))
    plt.ylim((-bd,bd))

def show_fit(theta,bd=1.75,cbd=1,nub_order=np.array([0,1,2,3,4]),add_1=False,nub_lbls=None):
    x = np.linspace(-bd,bd,100)
    y = np.linspace(bd,-bd,100)
    xx,yy = np.meshgrid(x,y)
    plt.imshow(theta[nub_locs.shape[0]+1]*np.ones_like(xx),extent=[-bd,bd,-bd,bd],cmap='bwr')
    plt.clim(-cbd,cbd)
    rects = []
    facecolors = []
    this_nub_locs = nub_locs[nub_order]
    for inub in range(nub_locs.shape[0]):
        facecolor = plt.cm.bwr((theta[inub]+cbd)/cbd/2)
        rect = Rectangle(this_nub_locs[inub]-np.array((0.5,0.5)),1,1,facecolor=facecolor)
        rects.append(rect)
        facecolors.append(facecolor)
    
    pc = PatchCollection(rects, alpha=1, facecolor=facecolors, edgecolor='k')
    plt.gca().add_collection(pc)
    #plt.scatter(nub_locs[:,0],nub_locs[:,1],c='g',marker='+')
    if nub_lbls is None:
        if add_1:
            nub_lbls = [str(n+1) for n in range(nub_locs.shape[0])]
        else:
            nub_lbls = [str(n) for n in range(nub_locs.shape[0])]
    for inub in range(this_nub_locs.shape[0]):
        plt.text(this_nub_locs[inub,0],this_nub_locs[inub,1],nub_lbls[inub],c='k',horizontalalignment='center',verticalalignment='center')
    #plt.text(nub_locs[:,0],nub_locs[:,1],nub_lbls,c='g')
    plt.xlim((-bd,bd))
    plt.ylim((-bd,bd))
    plt.xticks([])
    plt.yticks([])

###

def select_trials(trial_info,selector,training_frac,include_all=False,seed=0):
    # selector a dict saying what to do with each trial type. 
    # If a function, apply that function to the trial info column to 
    # obtain a boolean indexing variable
    # if 0, then the tuning output should be indexed by that variable
    # if 1, then that variable will be marginalized over in the tuning output
    def gen_train_test_exptwise(ti,training_frac):
        ntrials = ti[params[0]].size
        gd = np.ones((ntrials,),dtype='bool')
        for param in params:
            if callable(selector[param]): 
                # all the values of selector that are functions, ignore 
                # trials where that function evaluates to False
                # these trials will not be used in either training or 
                # test set
                exclude = ~selector[param](ti[param])
                print('excluding based on %s: %.02f'%(param,exclude.mean()))
                gd[exclude] = False
        condition_list = gen_condition_list(ti,selector) 
        # automatically, separated out such that each half of the data 
        # gets an equivalent fraction of trials with each condition type
        # where selector[condition] == 1
        # keep the non excluded trials
        condition_list = [c[gd] for c in condition_list]
        #in_training_set = np.zeros((ntrials,),dtype='bool')
        #in_test_set = np.zeros((ntrials,),dtype='bool')
        #to_keep = output_training_test(condition_list,training_frac,seed=seed)
        #in_training_set[gd] = to_keep
        #in_test_set[gd] = ~to_keep
        #if include_all:
        #    train_test = [in_training_set,in_test_set,np.logical_or(in_training_set,in_test_set)]
        #else:
        #    train_test = [in_training_set,in_test_set]
        #return train_test,ntrials
        if np.isscalar(training_frac):
            training_frac = [training_frac]
        nsplits = len(training_frac)
        in_training_split = [np.zeros((ntrials,),dtype='bool') for isplit in range(nsplits)]
        in_test_set = np.zeros((ntrials,),dtype='bool')
        to_keep = output_multi_split(condition_list,training_frac,seed=seed)
        for isplit in range(nsplits):
            in_training_split[isplit][gd] = to_keep[isplit]
        in_test_set[gd] = ~ut.k_or(*to_keep)
        if include_all:
            in_any = ut.k_or(*in_training_split,in_test_set)
            train_test = in_training_split + [in_test_set,in_any]
        else:
            train_test = in_training_split + [in_test_set]
        return train_test,ntrials
        
    params = list(selector.keys())
    keylist = list(trial_info.keys())
    if isinstance(trial_info[keylist[0]],dict):
        ntrials = {}
        train_test = {}
        for key in trial_info.keys():
            ti = trial_info[key]
            train_test[key],ntrials[key] = gen_train_test_exptwise(ti,training_frac)
    else:
        ti = trial_info
        train_test,ntrials = gen_train_test_exptwise(ti,training_frac)
        
    return train_test

#def output_training_test(condition_list,training_frac,seed=0):
#    np.random.seed(seed)
#    # output training and test sets balanced for conditions
#    # condition list, generated by gen_condition_list, has a row for each condition that should be equally assorted
#    if not isinstance(condition_list,list):
#        condition_list = [condition_list.copy()]
#    iconds,uconds = zip(*[pd.factorize(c,sort=True) for c in condition_list])
#    # extract uconds the unique elements of each array in list condition_list, iconds the array s.t.
#    # condition_list[i][iconds[i]==j] = uconds[i][j]
#    #uconds = [np.sort(u) for u in uconds]
#    nconds = np.array([u.size for u in uconds])
#    in_training_set = np.zeros(condition_list[0].shape,dtype='bool')
#    for iflat in range(np.prod(nconds)):
#        # iterate through combinations of values for the variables to balance
#        coords = np.unravel_index(iflat,tuple(nconds))
#        # find where the variables take on a given set of values
#        lkat = np.where(ut.k_and(*[iconds[ic] == coords[ic] for ic in range(len(condition_list))]))[0]
#        # reserve half of those for the training set at random
#        n_train = int(np.round(training_frac*len(lkat)))
#        to_train = np.random.choice(lkat,n_train,replace=False)
#        in_training_set[to_train] = True
#    #assert(True==False)
#    return in_training_set

def output_multi_split(condition_list,training_fracs,seed=0):
    np.random.seed(seed)
    # output training and test sets balanced for conditions
    # condition list, generated by gen_condition_list, has a row for each condition that should be equally assorted
    if not isinstance(condition_list,list):
        condition_list = [condition_list.copy()]
    iconds,uconds = zip(*[pd.factorize(c,sort=True) for c in condition_list])
    # extract uconds the unique elements of each array in list condition_list, iconds the array s.t.
    # condition_list[i][iconds[i]==j] = uconds[i][j]
    #uconds = [np.sort(u) for u in uconds]
    nconds = np.array([u.size for u in uconds])
    in_split = [np.zeros(condition_list[0].shape,dtype='bool') for tf in training_fracs]
    nsplits = len(in_split)
    for iflat in range(np.prod(nconds)):
        # iterate through combinations of values for the variables to balance
        coords = np.unravel_index(iflat,tuple(nconds))
        # find where the variables take on a given set of values
        lkat = np.where(ut.k_and(*[iconds[ic] == coords[ic] for ic in range(len(condition_list))]))[0]
        # reserve half of those for the training set at random
        n_trains = [int(np.round(tf*len(lkat))) for tf in training_fracs]
        to_train = np.random.choice(lkat,np.sum(n_trains),replace=False)
        tot = 0
        for isplit in range(nsplits):
            this_to_train = to_train[tot:tot+n_trains[isplit]]
            in_split[isplit][this_to_train] = True
            tot = tot+n_trains[isplit]
    #assert(True==False)
    return in_split

def gen_nub_selector_v1_all_sizes(run=False):
    # selector that creates separate indices for the three sizes
    selector = {}
    if run:
        selector['running'] = lambda x: x
    else:
        selector['running'] = lambda x: np.logical_not(x)
    selector['stimulus_size'] = 0
    selector['stimulus_direction_deg'] = 1
    selector['stimulus_nubs_active'] = 0
    return selector

def gen_nub_selector_v1_all_sizes_split_pupil(run=False,dilated=False,centered=False):
    # selector that creates separate indices for the three sizes
    selector = {}
    if run:
        selector['running'] = lambda x: x
    else:
        selector['running'] = lambda x: np.logical_not(x)
    selector['stimulus_size'] = 0
    selector['stimulus_direction_deg'] = 1
    selector['stimulus_nubs_active'] = 0
    selector['dilated'] = 1
    selector['centered'] = 1
    return selector

def gen_nub_selector_v1(run=False):
    # selector that ignores sizes other than 10 degrees
    selector = {}
    if run:
        selector['running'] = lambda x: x
    else:
        selector['running'] = lambda x: np.logical_not(x)
    selector['stimulus_size'] = lambda x: x==10
    selector['stimulus_direction_deg'] = 1
    selector['stimulus_nubs_active'] = 0
    return selector

def gen_nub_selector_v1_pupil(run=False,dilated=False):
    # selector that ignores (non-dilated/dilated) trials
    # and sizes other than 10 degrees
    selector = {}
    if run:
        selector['running'] = lambda x: x
    else:
        selector['running'] = lambda x: np.logical_not(x)
    if dilated:
        selector['dilated'] = lambda x: x
    else:
        selector['dilated'] = lambda x: np.logical_not(x)
    selector['stimulus_size'] = lambda x: x==10
    selector['stimulus_direction_deg'] = 1
    selector['stimulus_nubs_active'] = 0
    return selector

def gen_nub_selector_v1_split_pupil(run=False,dilated=False,centered=False):
    # selector that balances for pupil-centered and pupil-dilated trials
    selector = {}
    if run:
        selector['running'] = lambda x: x
    else:
        selector['running'] = lambda x: np.logical_not(x)
    selector['dilated'] = 1
    selector['centered'] = 1
    selector['stimulus_size'] = lambda x: x==10
    selector['stimulus_direction_deg'] = 1
    selector['stimulus_nubs_active'] = 0
    return selector

def gen_nub_selector_v1_split_pupil_15_degrees(run=False,dilated=False,centered=False):
    # selector that balances for pupil-centered and pupil-dilated trials
    selector = {}
    if run:
        selector['running'] = lambda x: x
    else:
        selector['running'] = lambda x: np.logical_not(x)
    selector['dilated'] = 1
    selector['centered'] = 1
    selector['stimulus_size'] = lambda x: x==15
    selector['stimulus_direction_deg'] = 1
    selector['stimulus_nubs_active'] = 0
    return selector

def gen_nub_selector_v1_split_pupil_after_blank(run=False,dilated=False,centered=False):
    # selector that balances for pupil-centered and pupil-dilated trials
    selector = {}
    if run:
        selector['running'] = lambda x: x
    else:
        selector['running'] = lambda x: np.logical_not(x)
    selector['dilated'] = 1
    selector['centered'] = 1
    selector['stimulus_size'] = lambda x: x==10
    selector['stimulus_direction_deg'] = 1
    selector['stimulus_nubs_active'] = 0
    selector['after_blank'] = lambda x: x
    return selector

def gen_nub_selector_v1_split_pupil_all_sizes(run=False,dilated=False,centered=False):
    # selector that balances for pupil-centered and pupil-dilated trials
    selector = {}
    if run:
        selector['running'] = lambda x: x
    else:
        selector['running'] = lambda x: np.logical_not(x)
    selector['dilated'] = 1
    selector['centered'] = 1
    selector['stimulus_size'] = 0
    selector['stimulus_direction_deg'] = 1
    selector['stimulus_nubs_active'] = 0
    return selector

def gen_nub_selector_v1_split_pupil_centered(run=False,dilated=False,centered=True):
    # selector that ignores (non-centered/centered) trials
    # and sizes other than 10 degrees, balances for dilation
    selector = {}
    if run:
        selector['running'] = lambda x: x
    else:
        selector['running'] = lambda x: np.logical_not(x)
    if centered:
        selector['centered'] = lambda x: x
    else:
        selector['centered'] = lambda x: np.logical_not(x)                                                 
    selector['dilated'] = 1
    selector['stimulus_size'] = lambda x: x==10
    selector['stimulus_direction_deg'] = 1
    selector['stimulus_nubs_active'] = 0
    return selector

def gen_nub_selector_v1_split_running_split_pupil_centered(run=None,dilated=None,centered=False):
    # selector that ignores (non-centered/centered trials
    # balances for running and dilation
    # ignores sizes other than 10 degrees
    selector = {}
    selector['running'] = 1
    if centered:
        selector['centered'] = lambda x: x
    else:
        selector['centered'] = lambda x: np.logical_not(x)                                                 
    selector['dilated'] = 1
    selector['stimulus_size'] = lambda x: x==10
    selector['stimulus_direction_deg'] = 1
    selector['stimulus_nubs_active'] = 0
    return selector

def gen_nub_selector_s1(run=True,dilated=0,centered=0):
    selector = {}
    if run:
        selector['running'] = lambda x: x
    else:
        selector['running'] = lambda x: np.logical_not(x)
    selector['stimulus_nubs_active'] = 0
    return selector

def gen_nub_selector_s1_after_blank(run=True,dilated=0,centered=0):
    selector = {}
    if run:
        selector['running'] = lambda x: x
    else:
        selector['running'] = lambda x: np.logical_not(x)
    selector['after_blank'] = lambda x: x
    selector['stimulus_nubs_active'] = 0
    return selector

def compute_tuning(df,trial_info,selector,include=None):
    # given a dataframe and trial_info dict as input, return trial-avged tuning 
    # curves for the running and non-running conditions. Trials are selected 
    # and balanced using selector
    # df is time-avged response per trial
    # include is a dict with a list of boolean arrays for each expt giving training
    # and test sets
    params = list(selector.keys())
    expts = list(trial_info.keys())
    nexpt = len(expts)
    tuning = [None for iexpt in range(nexpt)]
    if include is None:
        include = {expt:None for expt in expts}
    for iexpt,expt in enumerate(expts):
        in_this_expt = (df.session_id == expt)
        # reshape dataframe to roi x trial
        trialwise = df[in_this_expt].pivot(values='data',index='roi_index',columns='trial_index')
        nroi = trialwise.shape[0]
        ntrial = trialwise.shape[1]
        if include[expt] is None:
            include[expt] = np.ones((ntrial,),dtype='bool')
        if not isinstance(include[expt],list):
            include[expt] = [include[expt]]
        npart = len(include[expt])
        if isinstance(include[expt],list):
            tuning[iexpt] = [None for ipart in range(npart)]
        condition_list = []
        condition_list = gen_condition_list(trial_info[expt],selector,filter_selector=np.logical_not)
        iconds,uconds = zip(*[pd.factorize(c,sort=True) for c in condition_list])
        nconds = [len(u) for u in uconds]
        for ipart in range(npart):
            tuning[iexpt][ipart] = np.zeros((nroi,)+tuple(nconds))
            for iflat in range(np.prod(nconds)):
                coords = np.unravel_index(iflat,tuple(nconds))
                lkat = ut.k_and(include[expt][ipart],*[iconds[ic] == coords[ic] for ic in range(len(condition_list))])
                tuning[iexpt][ipart][(slice(None),)+coords] = np.nanmean(trialwise.iloc[:,lkat],-1)
    return tuning

def compute_tuning_df(df,trial_info,selector,include=None):
    params = list(selector.keys())
#     expts = list(trial_info.keys())
    expts = list(np.unique(df.session_id))
    nexpt = len(expts)
    tuning = pd.DataFrame()
    if include is None:
        include = {expt:None for expt in expts}
    for iexpt,expt in enumerate(expts):
        in_this_expt = (df.session_id == expt)
        trialwise = df.loc[in_this_expt].pivot(values='data',index='roi_index',columns='trial_index')
        nroi = trialwise.shape[0]
        ntrial = trialwise.shape[1]
        if include[expt] is None:
            include[expt] = np.ones((ntrial,),dtype='bool')
        if not isinstance(include[expt],list):
            include[expt] = [include[expt]]
        npart = len(include[expt])
#         if isinstance(include[expt],list):
#             tuning[iexpt] = [None for ipart in range(npart)]
        condition_list = []
        condition_list = gen_condition_list(trial_info[expt],selector,filter_selector=np.logical_not)
        iconds,uconds = zip(*[pd.factorize(c,sort=True) for c in condition_list])
        nconds = [len(u) for u in uconds]
        for ipart in range(npart):
            tip = np.zeros((nroi,)+tuple(nconds))
            for iflat in range(np.prod(nconds)):
                coords = np.unravel_index(iflat,tuple(nconds))
                lkat = ut.k_and(include[expt][ipart],*[iconds[ic] == coords[ic] for ic in range(len(condition_list))])
                tip[(slice(None),)+coords] = np.nanmean(trialwise.loc[:,lkat],-1)
            tip_df = pd.DataFrame(tip,index=np.arange(tip.shape[0]),columns=np.arange(tip.shape[1]))
            tip_df['partition'] = ipart
            tip_df['session_id'] = expt
            if 'area' in trial_info[expt]:
                tip_df['area'] = trial_info[expt]['area']
            else:
                tip_df['area'] = 'not given'
            tuning = tuning.append(tip_df)
    return tuning

def compute_pval_df(df,trial_info,selector,include=None):
    params = list(selector.keys())
#     expts = list(trial_info.keys())
    expts = list(np.unique(df.session_id))
    nexpt = len(expts)
    tuning = pd.DataFrame()
    if include is None:
        include = {expt:None for expt in expts}
    for iexpt,expt in enumerate(expts):
        in_this_expt = (df.session_id == expt)
        trialwise = df.loc[in_this_expt].pivot(values='data',index='roi_index',columns='trial_index')
        nroi = trialwise.shape[0]
        ntrial = trialwise.shape[1]
        if include[expt] is None:
            include[expt] = np.ones((ntrial,),dtype='bool')
        if not isinstance(include[expt],list):
            include[expt] = [include[expt]]
        npart = len(include[expt])
#         if isinstance(include[expt],list):
#             tuning[iexpt] = [None for ipart in range(npart)]
        condition_list = []
        condition_list = gen_condition_list(trial_info[expt],selector,filter_selector=np.logical_not)
        iconds,uconds = zip(*[pd.factorize(c,sort=True) for c in condition_list])
        nconds = [len(u) for u in uconds]
        lkat_no_stim = ut.k_and(include[expt][ipart],*[iconds[ic] == 0 for ic in range(len(condition_list))])
        for ipart in range(npart):
            tip = np.zeros((nroi,)+tuple(nconds))
            for iflat in range(np.prod(nconds)):
                coords = np.unravel_index(iflat,tuple(nconds))
                lkat = ut.k_and(include[expt][ipart],*[iconds[ic] == coords[ic] for ic in range(len(condition_list))])
                tip[(slice(None),)+coords] = np.nanmean(trialwise.loc[:,lkat],-1)
            tip_df = pd.DataFrame(tip,index=np.arange(tip.shape[0]),columns=np.arange(tip.shape[1]))
            tip_df['partition'] = ipart
            tip_df['session_id'] = expt
            tip_df['area'] = trial_info[expt]['area']
            tuning = tuning.append(tip_df)
    return tuning

def compute_bootstrap_error_faster(df,trial_info,selector,pct=(16,84),include=None):
    params = list(selector.keys())
    expts = list(trial_info.keys())
    nexpt = len(expts)
    bounds = [None for iexpt in range(nexpt)]
    npct = len(pct)
    if include is None:
        include = {expt:None for expt in expts}
    for iexpt,expt in enumerate(expts):
        in_this_expt = (df.session_id == expt)
        trialwise = df[in_this_expt].pivot(values='data',index='roi_index',columns='trial_index')
        nroi = trialwise.shape[0]
        ntrial = trialwise.shape[1]
        compress_flag = False
        if include[expt] is None:
            include[expt] = np.ones((ntrial,),dtype='bool')
        if not isinstance(include[expt],list):
            include[expt] = [include[expt]]
            compress_flag = True
        npart = len(include[expt])
        bounds[iexpt] = [None for ipart in range(npart)]
        condition_list = []
        condition_list = gen_condition_list(trial_info[expt],selector,filter_selector=np.logical_not)
        iconds,uconds = zip(*[pd.factorize(c,sort=True) for c in condition_list])
        nconds = [len(u) for u in uconds]
        for ipart in range(npart):
            bounds[iexpt][ipart] = [None for ipct in range(npct)]
            for ipct in range(npct):
                bounds[iexpt][ipart][ipct] = np.zeros((nroi,)+tuple(nconds))
            for iflat in range(np.prod(nconds)):
                coords = np.unravel_index(iflat,tuple(nconds))
                lkat = ut.k_and(include[expt][ipart],*[iconds[ic] == coords[ic] for ic in range(len(condition_list))])
                bds = ut.bootstrap(np.array(trialwise.loc[:,lkat]),np.nanmean,axis=1,pct=pct)
                for ipct in range(npct):
                    bounds[iexpt][ipart][ipct][(slice(None),)+coords] = bds[ipct]
        if compress_flag:
            bounds[iexpt] = bounds[iexpt][0]
    return bounds
                
def compute_bootstrap_error(df,trial_info,selector,pct=(16,84),include=None):
    params = list(selector.keys())
    expts = list(trial_info.keys())
    nexpt = len(expts)
    bounds = [None for iexpt in range(nexpt)]
    npct = len(pct)
    if include is None:
        include = {expt:None for expt in expts}
    for iexpt,expt in enumerate(expts):
        in_this_expt = (df.session_id == expt)
        trialwise = np.array(df[in_this_expt].pivot(values='data',index='roi_index',columns='trial_index'))
        nroi = trialwise.shape[0]
        ntrial = trialwise.shape[1]
        compress_flag = False
        if include[expt] is None:
            include[expt] = np.ones((ntrial,),dtype='bool')
        if not isinstance(include[expt],list):
            include[expt] = [include[expt]]
            compress_flag = True
        npart = len(include[expt])
        bounds[iexpt] = [None for ipart in range(npart)]
        condition_list = []
        condition_list = gen_condition_list(trial_info[expt],selector,filter_selector=np.logical_not)
        iconds,uconds = zip(*[pd.factorize(c,sort=True) for c in condition_list])
        nconds = [len(u) for u in uconds]
        for ipart in range(npart):
            bounds[iexpt][ipart] = [None for ipct in range(npct)]
            for ipct in range(npct):
                bounds[iexpt][ipart][ipct] = np.zeros((nroi,)+tuple(nconds))
            for iflat in range(np.prod(nconds)):
                coords = np.unravel_index(iflat,tuple(nconds))
                lkat = ut.k_and(include[expt][ipart],*[iconds[ic] == coords[ic] for ic in range(len(condition_list))])
                bds = ut.bootstrap(trialwise[:,lkat],np.nanmean,axis=1,pct=pct)
                for ipct in range(npct):
                    bounds[iexpt][ipart][ipct][(slice(None),)+coords] = bds[ipct]
        if compress_flag:
            bounds[iexpt] = bounds[iexpt][0]
    return bounds

def gen_condition_list(ti,selector,filter_selector=lambda x:True):
# ti: trial_info generated by ut.compute_tavg_dataframe
# selector: dict where each key is a param in ti.keys(), and each value 
# is either a callable returning a boolean, 
# to be applied to ti[param], or an input to the function filter_selector
# filter selector: if filter_selector(selector[param]), the tuning curve 
# will be separated into the unique elements of ti[param]. 
# return a list of arrays. The unique elements of the ith member of this list
# will ultimately be associated to the indices of the (i+1)th dimension of
# the output
    params = list(selector.keys())
    condition_list = []
    for param in params:
        if not callable(selector[param]) and filter_selector(selector[param]):
            condition_list.append(ti[param])
    return condition_list

def run_roiwise_fn(fn,*inp):
    outp = [None for iexpt in range(len(inp[0]))]
    for iexpt in range(len(inp[0])):
        nroi = len(inp[0][iexpt])
        iroi = 0
        temp = fn(inp[0][iexpt][iroi])
        outp[iexpt] = np.zeros((nroi,temp.shape[0]))
        for iroi in range(nroi):
            print(iroi)
            these_inps = [inp1[iexpt][iroi] for inp1 in inp]
            outp[iexpt][iroi] = fn(*these_inps)
    return outp

def reorder_stims(nub_ordering,flipped=None,nub_var=nubs_active):
    if flipped is None:
        flipped = np.zeros(nub_ordering.shape,dtype='bool')
    nnub = nub_var.shape[1]
    binary_vals = nub_var.copy()
    for inub in range(len(nub_ordering)):
        if flipped[inub]:
            binary_vals[:,inub] = ~binary_vals[:,inub]
    binary_vals = binary_vals[:,list(nub_ordering)]
    multby = np.array([2**inub for inub in range(nnub)][::-1])[np.newaxis,:]
    ordering = np.argsort((multby*binary_vals).sum(1))
    return ordering

def reorder_stims_weight(nub_weights,nub_var=nubs_active):
    ordering = np.argsort((nub_weights[np.newaxis]*nub_var).sum(1))
    return ordering

def show_combos(theta,this_combos,bd=1.75,cbd=1,nub_order=np.array([0,1,2,3,4])):
    combos = [(a,b) for a,b in zip(*np.where(this_combos))]
    ncombos = len(combos)
    for icombo,combo in enumerate(combos):
        show_combo(theta[nnub+icombo],combo,icombo,ncombos,bd=bd,cbd=cbd,nub_order=nub_order)

    plt.xlim((-bd,bd))
    plt.ylim((-bd,bd))
    plt.xticks([])
    plt.yticks([])
    
def show_combo(thetaval,combo,icombo,ncombos,bd=1.75,cbd=1,nub_order=np.array([0,1,2,3,4])):
    n_per_side = int(np.ceil(np.sqrt(ncombos)))
    scaleby = 1/n_per_side
    spacebetw = 2*bd*scaleby
    iloc = icombo // n_per_side
    jloc = icombo % n_per_side
    yloc = -spacebetw*(iloc - (n_per_side-1)/2)
    xloc = spacebetw*(jloc - (n_per_side-1)/2)
    rects = []
    facecolors = []
    facecolor = plt.cm.bwr((thetaval+cbd)/cbd/2)
    this_nub_locs = nub_locs[nub_order]
    for inub in range(this_nub_locs.shape[0]):
        ctr = (this_nub_locs[inub]-np.array((0.5,0.5)))*scaleby
        rect = Rectangle(ctr+np.array((xloc,yloc)),scaleby,scaleby,facecolor=facecolor)
        rects.append(rect)
        if inub in combo:
            facecolors.append(facecolor)
        else:
            facecolors.append('w')
        
    pc = PatchCollection(rects, alpha=1, facecolor=facecolors, edgecolor='k')
    plt.gca().add_collection(pc)

    combo_lbl = str(nnub+icombo)
    plt.text(xloc-scaleby,yloc+scaleby,combo_lbl,c='k',horizontalalignment='center',verticalalignment='center')

def show_reordered_plot_and_fit(this_bounds,this_theta,cbd=1,nub_var=nubs_active,axiswise_reordering=False,predict_fn=predict_output_theta_amplitude):
    this_prediction = predict_fn(this_theta,fn=f_mt,nub_var=nub_var)
    
    if axiswise_reordering:
        flipped = this_theta[:-2] < 0
        nub_ordering = np.argsort(np.abs(this_theta[:nub_var.shape[1]]))[::-1]
        stim_ordering = reorder_stims(nub_ordering,flipped=flipped,nub_var=nub_var)
    else:
        stim_ordering = reorder_stims_weight(this_theta[:nub_var.shape[1]],nub_var=nub_var)
    
    plt.clf()
    plt.subplot(1,2,1)
    plt.plot(np.arange(2**nnub),this_bounds[2][stim_ordering],label='data')
    plt.fill_between(np.arange(2**nnub),this_bounds[0][stim_ordering],this_bounds[1][stim_ordering],alpha=0.25)
    plt.xticks([])
    plt.yticks([])
    plt.plot(np.arange(2**nnub),this_prediction[stim_ordering],label='model')
    plt.legend()
    plt.title('Tuning curve')
    plt.ylabel('event rate (a.u.)')
    if axiswise_reordering:
        signs = []
        for inub in range(len(nub_ordering)):
            if flipped[nub_ordering[inub]]:
                signs.append('-')
            else:
                signs.append('+')
        markers = ','.join([s+str(n) for s,n in zip(signs,nub_ordering)])
        plt.xlabel('stimulus # (ordered [' + markers + '])')
    else:
        normwt = this_theta[:-2]/np.abs(this_theta[:nub_var.shape[1]]).sum()
        numbers = ','.join(['%d:%.2f' % (ix,x) for ix,x in enumerate(normwt)])
        plt.xlabel('stimulus # (ordered by weights [%s]' % numbers)
    plt.subplot(1,2,2)
    show_fit(this_theta,cbd=cbd)
    plt.title('GLM fit parameters')

def show_reordered_plot_and_nonlinear_fit(this_bounds,this_theta,this_combos,bd=1.75,cbd=1,nub_var=nubs_active,axiswise_reordering=False,nub_order=np.array([0,1,2,3,4])):
    this_prediction = predict_output_theta_amplitude(this_theta,fn=f_mt,nub_var=nub_var)
    
    if axiswise_reordering:
#         flipped = this_theta[:-2] < 0
#         nub_ordering = np.argsort(np.abs(this_theta[:-2]))[::-1]
        flipped = this_theta[:nnub] < 0
        nub_ordering = np.argsort(np.abs(this_theta[:nnub]))[::-1]
        stim_ordering = reorder_stims(nub_ordering,flipped=flipped,nub_var=nubs_active)
    else:
        stim_ordering = reorder_stims_weight(this_theta[:nub_var.shape[1]],nub_var=nub_var)
#         stim_ordering = reorder_stims_weight(this_theta[:nnub],nub_var=nubs_active)
    
    plt.clf()
    plt.subplot(1,3,1)
    plt.plot(np.arange(2**nnub),this_bounds[2][stim_ordering],label='data')
    plt.fill_between(np.arange(2**nnub),this_bounds[0][stim_ordering],this_bounds[1][stim_ordering],alpha=0.25)
    plt.xticks([])
    plt.yticks([])
    plt.plot(np.arange(2**nnub),this_prediction[stim_ordering],label='model')
    plt.legend()
    plt.title('Tuning curve')
    plt.ylabel('event rate (a.u.)')
    if axiswise_reordering:
        signs = []
        for inub in range(len(nub_ordering)):
            if flipped[nub_ordering[inub]]:
                signs.append('-')
            else:
                signs.append('+')
        markers = ','.join([s+str(n) for s,n in zip(signs,nub_ordering)])
        plt.xlabel('stimulus # (ordered [' + markers + '])')
    else:
        #normwt = this_theta[:-2]/np.abs(this_theta[:-2]).sum()
        #numbers = ','.join(['%d:%.2f' % (ix,x) for ix,x in enumerate(normwt)])
        ylim = plt.gca().get_ylim()
        scaley = np.diff(ylim)/5
        draw_stim_ordering(stim_ordering,scaley=scaley/nnub)
        plt.ylim((-scaley,ylim[-1]))
        plt.xlabel('stimulus #') # (ordered by weights [%s]' % numbers)

    plt.subplot(1,3,2)
    show_fit(this_theta,bd=bd,cbd=cbd,nub_order=nub_order)
    plt.title('GLM fit linear parameters')

    plt.subplot(1,3,3)
    show_combos(this_theta,this_combos,bd=bd,cbd=1,nub_order=nub_order)
    plt.title('GLM fit quadratic parameters')

def draw_stim(stim,xoffset=0,yoffset=0,scaley=1):
    locs = np.array((xoffset,yoffset))[np.newaxis,:] + np.array([(0,-i) for i in np.arange(nnub)])
    rects = []
    facecolors = []
    scale = np.array((1,scaley))
    for val,loc in zip(stim,locs):
#         print((val,loc))
        if val:
            facecolor = 'k'
        else:
            facecolor = 'w'
        rect = Rectangle(scale*(np.array((-0.5,-1))+loc),1,scaley,facecolor=facecolor)
        rects.append(rect)
        facecolors.append(facecolor)
        pc = PatchCollection(rects,alpha=1,facecolor=facecolors,edgecolor='k')
        plt.gca().add_collection(pc)
        
def draw_stim_ordering(stim_ordering,scaley=1,invert=False):
    if invert:
        this_ordering = np.logical_not(nubs_active[stim_ordering])
    else:
        this_ordering = nubs_active[stim_ordering]
    for istim,stim in enumerate(this_ordering):
        draw_stim(stim,xoffset=istim,yoffset=0,scaley=scaley)

def fdr_bh(pvals,fdr=0.05):
    sortind = np.argsort(pvals)
    M = pvals.size
    multby = M/(1+np.arange(M))
    pvals_corr = np.zeros_like(pvals)
    pvals_corr[sortind] = pvals[sortind]*multby
    sig = (pvals_corr < fdr)
    return sig

def test_sig_driven(df,roi_info,trial_info,pcutoff=0.05,dfof_cutoff=0.2,running=True):
    # inputs: time-averaged dataframe, roi_info and trial_info dicts from ut.compute_tavg_dataframe. 
    # Returns ROIs where evoked dF/F on at least one stim is different from spontaneous, 
    # Benjamini-Hochberg corrected pval < pcutoff, and averaged evoked dF/F across all stims > dfof_cutoff
    if dfof_cutoff is None:
        dfof_cutoff = 0.
    session_ids = list(roi_info.keys())
    for expt in session_ids:
        in_this_expt = (df.session_id == expt)
        # reshape df to ROI x trial for this expt.
        trialwise = df.loc[in_this_expt].pivot(values='data',index='roi_index',columns='trial_index')
        roilist = trialwise.index.unique()
        nroi = roilist.size
        roi_info[expt]['sig_driven'] = np.zeros((nroi,),dtype='bool')
        trialcond = trial_info[expt]['stimulus_nubs_active']
        trialrun = trial_info[expt]['running']
        if not running:
            trialrun = ~trialrun
        condlist = np.unique(trialcond)
        ncond = len(condlist)
        stim_driven = np.zeros((nroi,ncond-1))
        no_stim = (trialcond==0)&trialrun # spontaneous activity during (non)running trials
        yes_stim = (trialcond>0)&trialrun
        response_no_stim = np.array(trialwise.iloc[:,no_stim].T)
        roi_info[expt]['stim_pval'] = np.zeros((nroi,ncond-1))
        # on (non)running trials, the mean stim response - mean non-stim response
        mean_evoked_dfof = trialwise.iloc[:,yes_stim].mean(1) - trialwise.iloc[:,no_stim].mean(1)
        for icond,ucond in enumerate(condlist[1:]):
            this_stim = (trialcond==ucond)&trialrun
            response_this_stim = np.array(trialwise.iloc[:,this_stim].T)
            _,roi_info[expt]['stim_pval'][:,icond] = sst.ttest_ind(response_no_stim,response_this_stim)
        for iroi,uroi in enumerate(roilist):
            # compute B-H corrected significant response to any stim with false discovery rate pcutoff
            different_from_0 = np.any(fdr_bh(roi_info[expt]['stim_pval'][iroi],fdr=pcutoff))
            # sig. driven: significant response to at least one, and mean evoked dF/F > dfof_cutoff
            roi_info[expt]['sig_driven'][iroi] = (different_from_0 and mean_evoked_dfof[iroi]>dfof_cutoff)
        print('dF/F above cutoff: %d/%d'%((mean_evoked_dfof>dfof_cutoff).sum(),nroi))
        print('sig. driven: %d/%d'%(roi_info[expt]['sig_driven'].sum(),nroi))
        #if running:
        #    assert(True==False)
    return roi_info

def compute_tuning_all(df,trial_info):
    selector_s1 = gen_nub_selector_s1()
    selector_v1 = gen_nub_selector_v1()
    keylist = list(trial_info.keys())
    train_test = {}
    for key in keylist:
        if trial_info[key]['area'][:2]=='s1':
            selector = gen_nub_selector_s1()
        else:
            selector = gen_nub_selector_v1()
        train_test[key] = select_trials(trial_info[key],selector,0.5,include_all=True)
    tuning = pd.DataFrame()
    ttls = ['s1_l4','s1_l23','v1_l4','v1_l23']
    selectors = [selector_s1, selector_s1, selector_v1, selector_v1]
    for ttl,selector in zip(ttls,selectors):
        tuning = tuning.append(compute_tuning_df(df.loc[df.area==ttl],trial_info,selector,include=train_test))
    return tuning

def compute_tuning_many_partitionings(df,trial_info,npartitionings,training_frac=0.5,gen_nub_selector=None):
    selector_s1 = gen_nub_selector_s1()
    selector_v1 = gen_nub_selector_v1()
    keylist = list(trial_info.keys())
    train_test = {}
    for key in keylist:
        if gen_nub_selector is None and 'area' in trial_info[key]:
            if trial_info[key]['area'][:2]=='s1':
                selector = gen_nub_selector_s1()
            else:
                selector = gen_nub_selector_v1()
        else:
            selector = gen_nub_selector
        train_test[key] = [None for ipartitioning in range(npartitionings)]
        for ipartitioning in range(npartitionings):
            train_test[key][ipartitioning] = select_trials(trial_info[key],selector,training_frac)
    tuning = pd.DataFrame()
    ttls = ['s1_l4','s1_l23','v1_l4','v1_l23']
    selectors = [selector_s1, selector_s1, selector_v1, selector_v1]
    tt = [{k:v[ipartitioning] for k,v in zip(train_test.keys(),train_test.values())} for ipartitioning in range(npartitionings)]
    if not 'area' in df:
        ttls = [None]
        selectors = [selector]
    for ttl,selector in zip(ttls,selectors):
        for ipartitioning in range(npartitionings):
            if 'area' in df:
                new_tuning = compute_tuning_df(df.loc[df.area==ttl],trial_info,selector,include=tt[ipartitioning])
            else:
                new_tuning = compute_tuning_df(df,trial_info,selector,include=tt[ipartitioning])
            new_tuning.insert(new_tuning.shape[1],'partitioning',ipartitioning)
            #new_tuning['partitioning'] = ipartitioning
            tuning = tuning.append(new_tuning)
    return tuning

def compute_lin_subtracted(tuning,lkat=None):
    nexpt = len(tuning)
    concat = np.concatenate([tuning[iexpt][0] for iexpt in range(nexpt)],axis=0)
    if lkat is None:
        lkat = np.ones((concat.shape[0],),dtype='bool')
    tr = np.concatenate([tuning[iexpt][0] for iexpt in range(nexpt)],axis=0)[lkat] #np.array(tunings.loc[(tuning.partition==0) & (tunings.partitioning==0),list(np.arange(32))])
    te = np.concatenate([tuning[iexpt][1] for iexpt in range(nexpt)],axis=0)[lkat] #np.array(tunings.loc[(tunings.partition==1) & (tunings.partitioning==0),list(np.arange(32))])
    #te = np.concatenate([tuning[iexpt][0] for iexpt in range(nexpt)],axis=0)[lkat] #np.array(tunings.loc[(tunings.partition==1) & (tunings.partitioning==0),list(np.arange(32))]) # temporary for testing
    sortind,sorteach = do_sorting(tr)
    lin_subtracted,test_norm_response,linear_pred = subtract_lin(te)
    return lin_subtracted,sortind,sorteach

def compute_lin_subtracted_bounds(bounds):
    nexpt = len(bounds)
    lin_subtracted_bounds = [None for iexpt in range(nexpt)]
    for iexpt in range(len(bounds)):
        lin_subtracted_bounds[iexpt] = subtract_lin_bounds(bounds[iexpt])
    return lin_subtracted_bounds

def show_lin_subtracted(lin_subtracted,sortind,sorteach):
    lin_subtracted_sort = lin_subtracted.copy()
    for iroi in range(lin_subtracted.shape[0]):
        lin_subtracted_sort[iroi] = lin_subtracted[sortind][iroi][evan_order_actual][sorteach[iroi]]
    plt.figure(1)
    plt.imshow(lin_subtracted[sortind][:,evan_order_actual[6:]],extent=[0,1,0,1],cmap='bwr')
    plt.clim(-1.05,1.05)
    plt.figure(2)
    lb,ub = ut.bootstrap(lin_subtracted_sort[nub_no[sorteach[:,0]]>1],axis=0,pct=(16,84),fn=np.nanmean)
    plt.fill_between(np.arange(32),lb,ub)
    plt.axhline(0,linestyle='dotted',c='k')

def compute_lkat_roracle(t,rcutoff=0.5):
    # t a list of nexpt lists of two or more (one or more training + test) tuning curves, nroi x ncond arrays
    # returns a boolean array where expts are concatenated together
    roracle = [None for iexpt in range(len(t))]
    for iexpt in range(len(t)):
        nroi = t[iexpt][1].shape[0]
        roracle[iexpt] = np.zeros((nroi,))
        for iroi in range(nroi):
            roracle[iexpt][iroi] = np.corrcoef(t[iexpt][0][iroi].flatten(),t[iexpt][1][iroi].flatten())[0,1]
        print('roracle: %d/%d'%((roracle[iexpt]>rcutoff).sum(),nroi))
    roracle_lin = np.concatenate(roracle)
    lkat = roracle_lin>rcutoff
    return lkat

def subtract_lin(test_response):
    test_norm_response = test_response.copy() - test_response[:,0:1]
    mn = 0 #test_norm_response.min(1)[:,np.newaxis]
    mx = test_norm_response.max(1)[:,np.newaxis]
    test_norm_response = (test_norm_response-mn)#/(mx-mn) # need to put this back 4/8/2020
    linear_pred = np.zeros_like(test_norm_response)
    single_whisker_responses = test_norm_response[:,[2**n for n in range(5)][::-1]]
    for i in range(linear_pred.shape[1]):
        linear_pred[:,i] = np.sum(single_whisker_responses*utils.nubs_active[i][np.newaxis],1)
    lin_subtracted = (test_norm_response-linear_pred) #/(test_norm_response+linear_pred)
    return lin_subtracted,test_norm_response,linear_pred

def subtract_lin_bounds(test_response_bounds):
    # test_response bounds corresponds to 2.5,97.5,50 percentile
    sq_err_lo = (test_response_bounds[2] - test_response_bounds[0])**2
    sq_err_hi = (test_response_bounds[1] - test_response_bounds[2])**2
    # compute evoked response
    test_norm_response = test_response_bounds[2].copy() - test_response_bounds[2][:,0:1]
    # add error for evoked response
    err_lo,err_hi = [x+x[:,0:1] for x in [sq_err_lo,sq_err_hi]]
    mn = 0 #test_norm_response.min(1)[:,np.newaxis]
    # norm to max, no longer used!
    #mx = test_norm_response.max(1)[:,np.newaxis]
    #mx_ind = np.argmax(test_norm_response,axis=1)
    #mx_err_lo,mx_err_hi = [x[np.arange(x.shape[0]),mx_ind][:,np.newaxis] for x in [sq_err_lo,sq_err_hi]]
    #test_norm_response = (test_norm_response-mn)/(mx-mn)
    #sq_err_lo,sq_err_hi = [x/mx**2 + y*test_norm_response**2/mx**2 for x,y in zip([sq_err_lo,sq_err_hi],[mx_err_lo,mx_err_hi])]
    # compute aggregate error on linear difference
    linear_pred = np.zeros_like(test_norm_response)
    linear_pred_err_lo = np.zeros_like(test_norm_response)
    linear_pred_err_hi = np.zeros_like(test_norm_response)
    single_whisker_responses = test_norm_response[:,[2**n for n in range(5)][::-1]]
    single_whisker_err_lo = sq_err_lo[:,[2**n for n in range(5)][::-1]]
    single_whisker_err_hi = sq_err_hi[:,[2**n for n in range(5)][::-1]]
    for i in range(linear_pred.shape[1]):
        linear_pred[:,i] = np.sum(single_whisker_responses*nubs_active[i][np.newaxis],1)
        linear_pred_err_lo[:,i] = np.sum(single_whisker_err_lo*nubs_active[i][np.newaxis],1)
        linear_pred_err_hi[:,i] = np.sum(single_whisker_err_hi*nubs_active[i][np.newaxis],1)
    lin_subtracted = (test_norm_response-linear_pred) #/(test_norm_response+linear_pred)
    lin_subtracted_err_lo = linear_pred_err_lo + sq_err_lo
    lin_subtracted_err_hi = linear_pred_err_hi + sq_err_hi
    # take sqrt before adding to generate bounds
    lin_subtracted_lb = lin_subtracted-np.sqrt(lin_subtracted_err_lo)
    lin_subtracted_ub = lin_subtracted+np.sqrt(lin_subtracted_err_hi)
    lin_subtracted_bounds = [lin_subtracted_lb,lin_subtracted_ub,lin_subtracted]
    return lin_subtracted_bounds

def do_sorting(train_response):
    train_norm_response = train_response.copy() - train_response[:,0:1]
#     train_norm_response = train_norm_response
    sorteach = np.argsort(train_norm_response[:,evan_order_actual],1)[:,::-1]
    sortind = np.arange(train_norm_response.shape[0])
    for n in [0]: #[3,2,1,0]:
        new_indexing = np.argsort(sorteach[:,n],kind='mergesort')
        sortind = sortind[new_indexing]
        sorteach = sorteach[new_indexing]
    sortind = np.argsort(np.argmax(train_norm_response[:,evan_order_actual],axis=1))
    return sortind,sorteach

def test_validity_of_linear_pred(test_norm_response,linear_pred):
    (singles,) = np.where(nubs_active.sum(1)==1)
    which_single = np.argmax(nubs_active[singles],axis=1)
    singles = singles[np.argsort(which_single)]
    for whiskerno in range(1,6):
        (ntuples,) = np.where(nubs_active.sum(1)==whiskerno)
        for intuple,ntuple in enumerate(ntuples):
            (constituents,) = np.where(nubs_active[ntuple])
            assert(all(nubs_active[ntuple]==np.sum(nubs_active[singles[constituents]],axis=0)))
            x = np.sum(np.array([test_norm_response[:,singles[c]] for c in constituents]),axis=0)
            y = linear_pred[:,ntuple]
            assert(all((x==y)|np.isnan(y)))

def show_evan_style(train_response,test_response,ht=6,cmap=parula,line=True,draw_stim_ordering=True):
    sorteach = np.argsort(train_response[:,evan_order_actual],1)[:,::-1]
    sortind = np.arange(train_response.shape[0])
    for n in [0]:
        new_indexing = np.argsort(sorteach[:,n],kind='mergesort')
        sortind = sortind[new_indexing]
        sorteach = sorteach[new_indexing]
    img = plt.imshow(test_response[sortind][:,evan_order_actual]/test_response[sortind].max(1)[:,np.newaxis],extent=[-0.5,31.5,0,5*ht],cmap=cmap)
    if draw_stim_ordering:
        utils.draw_stim_ordering(evan_order_apparent,invert=True)
    
    nroi = test_response.shape[0]
    show_every = 300
    plt.yticks(5*ht-np.arange(0,5*ht,show_every/nroi*5*ht),['Neuron'] + list(np.arange(show_every,nroi,show_every)))
    for this_nub_no in range(2,6):
        first_ind = np.where(nub_no==this_nub_no)[0][0]
        plt.axvline(first_ind-0.45,c='w')
        plt.axhline(5*ht*(1-np.where(sorteach[:,0]==first_ind)[0][0]/nroi),c='w',linestyle='dotted')
    plt.ylabel('Neuron')
    plt.xlabel('Stimulus')
    if draw_stim_ordering:
        plt.ylim(-5,5*ht)
    else:
        plt.ylim(0,5*ht)
    plt.xlim(0.5,31.5)
    plt.xticks([])
    
    plt.text(-5,5*ht+1,'# patches: ')
    lbl_locs = [2.75,10.25,20.25,27.75,30.75]
    for inub,this_nub_no in enumerate(range(1,6)):
        plt.text(lbl_locs[inub],5*ht+1,this_nub_no)
    if line:
        plt.plot((0.5,31.5),(5*ht,0),c='k')
        plt.tight_layout(pad=7)

def ayaz_model(c,d,r0,rd,rs,sd,ss,m,n,delta):
    # return len(d) x len(c) array, corresponding to Ayaz et al. model-like outputs for each size and contrast
    D = np.array([compute_overlap(np.array((delta,0)),sd**2,diam=this_d) for this_d in d])[:,np.newaxis]
    S = np.array([compute_overlap(np.array((delta,0)),ss**2,diam=this_d) for this_d in d])[:,np.newaxis]
    return r0 + rd*(c[np.newaxis,:]**n)*(D**m)/(1+rs*(c[np.newaxis,:]**n)*(S**m))

def ayaz_like_theta(c,d,theta,fn=ayaz_model):
    return fn(c,d,*theta)

def ayaz_model_adapted(c,d,r0,rd,rs,sd,ss,n1,n2,delta):
    D = np.array([compute_overlap(np.array((delta,0)),sd**2,diam=this_d) for this_d in d])[:,np.newaxis]
    S = np.array([compute_overlap(np.array((delta,0)),ss**2,diam=this_d) for this_d in d])[:,np.newaxis]
    return r0 + rd*(c[np.newaxis,:]**n1)*D/(1+rs*(c[np.newaxis,:]**n2)*S)

def ayaz_model_two_n(c,d,r0,rd,rs,sd,ss,m,n1,n2,delta):
    D = np.array([compute_overlap(np.array((delta,0)),sd**2,diam=this_d) for this_d in d])[:,np.newaxis]
    S = np.array([compute_overlap(np.array((delta,0)),ss**2,diam=this_d) for this_d in d])[:,np.newaxis]
    return r0 + rd*(c[np.newaxis,:]**n1)*(D**m)/(1+rs*(c[np.newaxis,:]**n2)*(S**m))

def ayaz_model_two_n_offset(c,d,r0,a,rd,rs,sd,ss,m,n1,n2,delta):
    D = np.array([compute_overlap(np.array((delta,0)),sd**2,diam=this_d) for this_d in d])[:,np.newaxis]
    S = np.array([compute_overlap(np.array((delta,0)),ss**2,diam=this_d) for this_d in d])[:,np.newaxis]
    #print('D,S: '+str((D[d==0],S[d==0])))
    rate = r0 + (a+rd*(c[np.newaxis,:]**n1)*(D**m))/(1+rs*(c[np.newaxis,:]**n2)*(S**m))
    #print('rate, d=0: '+str(rate[d==0,:]))
    if not np.all(np.isnan(rate[d==0,:])) and not np.all(rate[d==0,:]==np.nanmean(rate[d==0,:])):
        print('r0,a,rd,rs,sd,ss,m,n1,n2,delta: '+str((r0,a,rd,rs,sd,ss,m,n1,n2,delta)))
    return rate

def ayaz_model_one_n_offset(c,d,r0,a,rd,rs,sd,ss,m,n,delta):
    D = np.array([compute_overlap(np.array((delta,0)),sd**2,diam=this_d) for this_d in d])[:,np.newaxis]
    S = np.array([compute_overlap(np.array((delta,0)),ss**2,diam=this_d) for this_d in d])[:,np.newaxis]
    #print('D,S: '+str((D[d==0],S[d==0])))
    rate = r0 + (a+rd*(c[np.newaxis,:]**n)*(D**m))/(1+rs*(c[np.newaxis,:]**n)*(S**m))
    #print('rate, d=0: '+str(rate[d==0,:]))
    if not np.all(np.isnan(rate[d==0,:])) and not np.all(rate[d==0,:]==np.nanmean(rate[d==0,:])):
        print('r0,a,rd,rs,sd,ss,m,n,delta: '+str((r0,a,rd,rs,sd,ss,m,n,delta)))
    return rate

def ayaz_model_sub_one_n_offset(c,d,r0,a,rd,rs,rm,sd,ss,sm,m,n,delta):
    D = np.array([compute_overlap(np.array((delta,0)),sd**2,diam=this_d) for this_d in d])[:,np.newaxis]
    S = np.array([compute_overlap(np.array((delta,0)),ss**2,diam=this_d) for this_d in d])[:,np.newaxis]
    M = np.array([compute_overlap(np.array((delta,0)),sm**2,diam=this_d) for this_d in d])[:,np.newaxis]
    #print('D,S: '+str((D[d==0],S[d==0])))
    rate = r0 + (a+rd*(c[np.newaxis,:]**n)*(D**m)-rm*(c[np.newaxis,:]**n)*(M**m))/(1+rs*(c[np.newaxis,:]**n)*(S**m))
    #print('rate, d=0: '+str(rate[d==0,:]))
    #if not np.all(np.isnan(rate[d==0,:])) and not np.all(rate[d==0,:]==np.nanmean(rate[d==0,:])):
    #    print('r0,a,rd,rs,sd,ss,m,n,delta: '+str((r0,a,rd,rs,sd,ss,m,n,delta)))
    return rate

def ayaz_model_sub_three_n_offset(c,d,r0,a,rd,rs,rm,sd,ss,sm,m,nd,ns,nm,delta):
    D = np.array([compute_overlap(np.array((delta,0)),sd**2,diam=this_d) for this_d in d])[:,np.newaxis]
    S = np.array([compute_overlap(np.array((delta,0)),ss**2,diam=this_d) for this_d in d])[:,np.newaxis]
    M = np.array([compute_overlap(np.array((delta,0)),sm**2,diam=this_d) for this_d in d])[:,np.newaxis]
    #print('D,S: '+str((D[d==0],S[d==0])))
    rate = r0 + (a+rd*(c[np.newaxis,:]**nd)*(D**m)-rm*(c[np.newaxis,:]**nm)*(M**m))/(1+rs*(c[np.newaxis,:]**ns)*(S**m))
    #print('rate, d=0: '+str(rate[d==0,:]))
    #if not np.all(np.isnan(rate[d==0,:])) and not np.all(rate[d==0,:]==np.nanmean(rate[d==0,:])):
    #    print('r0,a,rd,rs,sd,ss,m,n,delta: '+str((r0,a,rd,rs,sd,ss,m,n,delta)))
    return rate

#def ayaz_model_sub_ind_n_offset(c,d,r0,rdp,rsp,rdm,rsm,sdp,ssp,sdm,ssm,m,npp,nmm,delta):
#    Dp = np.array([compute_overlap(np.array((delta,0)),sdp**2,diam=this_d) for this_d in d])[:,np.newaxis]
#    Sp = np.array([compute_overlap(np.array((delta,0)),ssp**2,diam=this_d) for this_d in d])[:,np.newaxis]
#    Dm = np.array([compute_overlap(np.array((delta,0)),sdm**2,diam=this_d) for this_d in d])[:,np.newaxis]
#    Sm = np.array([compute_overlap(np.array((delta,0)),ssm**2,diam=this_d) for this_d in d])[:,np.newaxis]
#    rate = r0 + rdp*(c[np.newaxis,:]**npp)*(Dp**m)/(1+rsp*(c[np.newaxis,:]**npp)*(Sp**m)) - rdm*(c[np.newaxis,:]**nmm)*(Dm**m)/(1+rsm*(c[np.newaxis,:]**nmm)*(Sm**m))
#    return rate

def ayaz_model_sub_ind_n_offset(c,d,r0,ap,rdp,rsp,am,rdm,rsm,sdp,ssp,sdm,ssm,m,npp,nmm,delta):
    #ap,am = 0,0
    rate = r0 + compute_norm_model(c,d,ap,rdp,rsp,sdp,ssp,m,npp,npp,delta) - compute_norm_model(c,d,am,rdm,rsm,sdm,ssm,m,nmm,nmm,delta)
    return rate

def ayaz_model_sub_ind_n_one_ss_offset(c,d,r0,ap,rdp,rsp,am,rdm,rsm,sdp,ssp,sdm,ssm,m,npp,nmm,delta):
    #am = 0
    #rate = r0 + compute_norm_model(c,d,ap,rdp,rsp,sdp,ssp,m,npp,npp,delta) - compute_norm_model(c,d,0,rdm,rsp,sdm,ssp,m,nmm,npp,delta)
    rate = r0 + compute_norm_model(c,d,ap,rdp,rsp,sdp,ssp,m,npp,nmm,delta) - compute_norm_model(c,d,0,rdm,rsp,sdm,ssp,m,nmm,nmm,delta)
    return rate

def compute_norm_model(c,d,a,rd,rs,sd,ss,m,nd,ns,delta):
    D = np.array([compute_overlap(np.array((delta,0)),sd**2,diam=this_d) for this_d in d])[:,np.newaxis]
    S = np.array([compute_overlap(np.array((delta,0)),ss**2,diam=this_d) for this_d in d])[:,np.newaxis]
    return (a+rd*(c[np.newaxis,:]**nd)*(D**m))/(1+rs*(c[np.newaxis,:]**ns)*(S**m))

def ayaz_model_div_ind_n_offset(c,d,r0,ap,rdp,rsp,am,rdm,rsm,sdp,ssp,sdm,ssm,m,npp,nmm,delta):
    #ap,am = 0,0
    pterm = compute_norm_model(c,d,ap,rdp,rsp,sdp,ssp,m,npp,npp,delta)
    #dterm = compute_norm_model(c,d,am,rdm,rsm,sdm,ssm,m,nmm,nmm,delta)
    dterm = compute_norm_model(c,d,am,rdm,rsm,sdm,ssm,m,npp,npp,delta)
    #pterm = compute_norm_model(c,d,ap,rdp,rsp,sdp,sdm,m,npp,npp,delta)
    #dterm = compute_norm_model(c,d,am,rdm,rsm,sdm,sdm,m,npp,npp,delta)
    rate = r0 + pterm/(1 + dterm)
    return rate

def ayaz_model_div_ind_n_one_ss_offset(c,d,r0,ap,rdp,rsp,am,rdm,rsm,sdp,ssp,sdm,ssm,m,npp,nmm,delta):
    #ap,am = 0,0
    #pterm = compute_norm_model(c,d,ap,rdp,rsp,sdp,ssp,m,npp,npp,delta)
    ##dterm = compute_norm_model(c,d,am,rdm,rsm,sdm,ssm,m,nmm,nmm,delta)
    #dterm = compute_norm_model(c,d,am,rdm,rsm,sdm,ssm,m,npp,npp,delta)
    pterm = compute_norm_model(c,d,ap,rdp,rsp,sdp,sdm,m,npp,npp,delta)
    dterm = compute_norm_model(c,d,am,rdm,rsm,sdm,sdm,m,npp,npp,delta)
    rate = r0 + pterm/(1 + dterm)
    return rate

def ayaz_model_div_ind_n_pm_offset(c,d,r0,ap,rdp,rsp,am,rdm,rsm,sdp,ssp,sdm,ssm,m,npp,nmm,delta):
    #ap,am = 0,0
    pterm = compute_norm_model(c,d,ap,rdp,rsp,sdp,ssp,m,npp,npp,delta)
    #dterm = compute_norm_model(c,d,am,rdm,rsm,sdm,ssm,m,nmm,nmm,delta)
    dterm = compute_norm_model(c,d,am,rdm,rsm,sdm,ssm,m,nmm,nmm,delta)
    #pterm = compute_norm_model(c,d,ap,rdp,rsp,sdp,sdm,m,npp,npp,delta)
    #dterm = compute_norm_model(c,d,am,rdm,rsm,sdm,sdm,m,npp,npp,delta)
    rate = r0 + pterm/(1 + dterm)
    return rate

def ayaz_model_div_ind_n_offset_with_vm(c,d,r0,ap,rdp,rsp,am,rdm,rsm,sdp,ssp,sdm,ssm,m,npp,nmm,delta):
    pterm = compute_norm_model(c,d,ap,rdp,rsp,sdp,ssp,m,npp,npp,delta)
    dterm = compute_norm_model(c,d,am,rdm,rsm,sdm,ssm,m,npp,npp,delta)
    rate = r0 + pterm/(1 + dterm)
    return rate
