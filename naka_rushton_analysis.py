#!/usr/bin/env python
import numpy as np
import scipy.optimize as sop
import matplotlib.pyplot as plt

# fit models of the form (a*(c/c50)^n + b)/((c/c50)^n + 1), where subsets of a, b, and c50 are allowed to vary for each size

#def fit_opt_params(c,r):
#    a_0 = r.max()
#    b_0 = 0
#    c50_0 = 50
#    n_0 = 1
#    params_0 = np.array((a_0,b_0,c50_0,n_0))
#    params_opt = sop.least_squares(lambda params: r-naka_rushton(c,params),params_0,bounds=((0,0,0,0),(np.inf,np.inf,200,5)))
#    return params_opt['x']

def fit_opt_params(c,R,Rsem=None):
    nsizes = R.shape[0]
    a_0 = R.max(1)
    b_0 = 0
    c50_0 = c[1]#c.max()/2*np.ones(nsizes)
    n_0 = 2
    bds_a = [(0,np.inf) for a in a_0]
    bds_b = [(0,np.inf)]
    bds_c50 = [(0,np.max(c)) for isize in range(nsizes)]
    bds_n = [(0,4) for ivar in range(1)]
    bds = zip_pairs(bds_a + bds_b + bds_c50 + bds_n)
    params_0 = np.concatenate((a_0,(b_0,),c50_0,(n_0,)))
    if Rsem is None:
        Rsem = np.ones_like(R)
    else:
        Rsem[Rsem==0] = np.min(Rsem[Rsem>0])
    def compute_this_cost(params):
        return (R-naka_rushton(c,params,nsizes))/Rsem
    params_opt = sop.least_squares(lambda params: compute_this_cost(params).flatten(),params_0,bounds=bds)
    return params_opt['x']

def fit_opt_params_offset(c,R,Rsem=None):
    nsizes = R.shape[0]
    a_0 = R.max(1)
    b_0 = 0
    c50_0 = c.max()/2*np.ones(nsizes)
    n_0 = 2
    bds_a = [(0,np.inf) for a in a_0]
    bds_b = [(0,np.inf)]
    bds_c50 = [(0,np.max(c)) for isize in range(nsizes)]
    bds_n = [(0,4) for ivar in range(1)]
    bds = zip_pairs(bds_a + bds_b + bds_c50 + bds_n)
    params_0 = np.concatenate((a_0,(b_0,),c50_0,(n_0,)))
    if Rsem is None:
        Rsem = np.ones_like(R)
    else:
        Rsem[Rsem==0] = np.min(Rsem[Rsem>0])
    def compute_this_cost(params):
        return (R-naka_rushton_offset(c,params,nsizes))/Rsem
    params_opt = sop.least_squares(lambda params: compute_this_cost(params).flatten(),params_0,bounds=bds)
    return params_opt['x']

def fit_opt_params_monotonic(c,R,Rsem=None):
    nsizes = R.shape[0]
    cmax = (np.argmax(R,axis=1)).astype('int')
    for isize in range(R.shape[0]):
        R[isize,cmax[isize]+2:] = -1
    a_0 = R.max(1)
    b_0 = 0
    c50_0 = c[cmax]/2
    n_0 = 2
    bds_a = [(0,np.inf) for a in a_0]
    bds_b = [(0,np.inf)]
    bds_c50 = [(0,cc+1) for cc in c[cmax]]
    bds_n = [(0,4) for ivar in range(1)]
    bds = zip_pairs(bds_a + bds_b + bds_c50 + bds_n)
    params_0 = np.concatenate((a_0,(b_0,),c50_0,(n_0,)))
    if Rsem is None:
        Rsem = np.ones_like(R)
    else:
        Rsem[Rsem==0] = np.min(Rsem[Rsem>0])
    def compute_this_cost(params):
        return (R>0)*(R-naka_rushton(c,params,nsizes))/Rsem
    params_opt = sop.least_squares(lambda params: compute_this_cost(params).flatten(),params_0,bounds=bds)
    return params_opt['x']

def fit_opt_params_two_n_monotonic(c,R,Rsem=None):
    nsizes = R.shape[0]
    cmax = (np.argmax(R,axis=1)).astype('int')
    for isize in range(R.shape[0]):
        R[isize,cmax[isize]+2:] = -1
    a_0 = R.max(1)
    b_0 = 0
    c50_0 = c[cmax]/2
    n_top_0 = 2
    n_bottom_0 = 2
    bds_a = [(0,np.inf) for a in a_0]
    bds_b = [(0,np.inf)]
    bds_c50 = [(0,cc+1) for cc in c[cmax]]
    bds_n = [(0,4) for ivar in range(2)]
    bds = zip_pairs(bds_a + bds_b + bds_c50 + bds_n)
    params_0 = np.concatenate((a_0,(b_0,),c50_0,(n_top_0,n_bottom_0)))
    if Rsem is None:
        Rsem = np.ones_like(R)
    else:
        Rsem[Rsem==0] = np.min(Rsem[Rsem>0])
    def compute_this_cost(params):
        return (R>0)*(R-naka_rushton_two_n(c,params,nsizes))/Rsem
    params_opt = sop.least_squares(lambda params: compute_this_cost(params).flatten(),params_0,bounds=bds)
    return params_opt['x']

#def fit_opt_params_tv(c,r):
#    a_0 = r.max()
#    b_0 = 0
#    c50_0 = 50
#    n_0 = 2
#    params_0 = np.array((a_0,b_0,c50_0,n_0))
#    params_opt = sop.least_squares(lambda params: r-naka_rushton(c,params),params_0,bounds=(0,200))
#    params_0 = params_opt['x']
#    cost_0 = params_opt['cost']
#    cinterp = np.arange((100))
#    np.abs(np.diff(naka_rushton_all

#def naka_rushton(c,params):
#    a = params[0]
#    b = params[1]
#    c50 = params[2]
#    n = params[3]
#    return (a*(c/c50)**n + b)/(1+(c/c50)**n)

def naka_rushton(c,params,ncells):
    a = params[:ncells]
    b = params[ncells]
    c50 = params[ncells+1:-1]
    n = params[-1]
    aux = (c[np.newaxis]/c50[:,np.newaxis])
    return (a[:,np.newaxis]*aux**n + b)/(1+aux**n)

def naka_rushton_offset(c,params,ncells):
    a = params[:ncells]
    b = params[ncells]
    c50 = params[ncells+1:-1]
    n = params[-1]
    aux = (c[np.newaxis]/c50[:,np.newaxis])
    return (a[:,np.newaxis]*aux**n)/(1+aux**n) + b

def fit_opt_params_only_a(c,R):
    nsizes = R.shape[0]
    a_0 = R.max(1)
    b_0 = 0
    c50_0 = 50
    n_0 = 2
    params_0 = np.concatenate((a_0,(b_0,),(c50_0,),(n_0,)))
    params_opt = sop.least_squares(lambda params: (R-naka_rushton_only_a(c,params,nsizes)).flatten(),params_0,bounds=(0,np.inf))
    return params_opt['x']

def naka_rushton_only_a(c,params,ncells):
    a = params[:ncells]
    b = params[ncells]
    c50 = params[ncells+1]
    n = params[-1]
    aux = (c[np.newaxis,:]/c50)**n
    return (a[:,np.newaxis]*aux + b)/(1+aux)


def fit_opt_params_all(c,R):
    nsizes = R.shape[0]
    a_0 = R.max(1)
    b_0 = 0
    c50_0 = 50*np.ones(nsizes)
    n_0 = 1
    params_0 = np.concatenate((a_0,(b_0,),c50_0,(n_0,)))
    params_opt = sop.least_squares(lambda params: (R-naka_rushton_all(c,params,nsizes)).flatten(),params_0,bounds=(0,np.inf))
    return params_opt['x']

def naka_rushton_all(c,params,ncells):
    a = params[:ncells]
    b = params[ncells]
    c50 = params[ncells+1:-1]
    n = params[-1]
    aux = (c[np.newaxis]/c50[:,np.newaxis])**n
    return (a[:,np.newaxis]*aux + b)/(1+aux)

def zip_pairs(list_of_pairs):
    lb = [lp[0] for lp in list_of_pairs]
    ub = [lp[1] for lp in list_of_pairs]
    return (lb,ub)

def fit_opt_params_two_n(c,R,Rsem=None):
    nsizes = R.shape[0]
    a_0 = R.max(1)
    b_0 = 0
    c50_0 = c.max()/2*np.ones(nsizes)
    n_top_0 = 2
    n_bottom_0 = 2
    bds_a = [(0,np.inf) for a in a_0]
    bds_b = [(0,np.inf)]
    bds_c50 = [(0,np.max(c)) for isize in range(nsizes)]
    bds_n = [(0,4) for ivar in range(2)]
    bds = zip_pairs(bds_a + bds_b + bds_c50 + bds_n)
    params_0 = np.concatenate((a_0,(b_0,),c50_0,(n_top_0,),(n_bottom_0,)))
    if Rsem is None:
        Rsem = np.ones_like(R)
    else:
        Rsem[Rsem==0] = np.min(Rsem[Rsem>0])
    def compute_this_cost(params):
        return (R-naka_rushton_two_n(c,params,nsizes))/Rsem
    params_opt = sop.least_squares(lambda params: compute_this_cost(params).flatten(),params_0,bounds=bds)
    return params_opt['x']

def naka_rushton_two_n(c,params,ncells):
    a = params[:ncells]
    b = params[ncells]
    c50 = params[ncells+1:-2]
    n_top = params[-2]
    n_bottom = params[-1]
    aux = (c[np.newaxis]/c50[:,np.newaxis])
    return (a[:,np.newaxis]*aux**n_top + b)/(1+aux**n_bottom)

def fit_opt_params_weibull(c,R,Rsem=None):
    nsizes = R.shape[0]
    a0_0 = R.max(1)
    a1_0 = np.zeros(nsizes)
    lam_0 = c.max()/2*np.ones(nsizes)
    k_0 = np.ones(nsizes)
    bds_a0 = [(0,np.inf) for _ in a0_0]
    bds_a1 = [(0,np.inf) for _ in a1_0]
    bds_lam = [(0,2*np.max(c)) for _ in lam_0]
    bds_k = [(0,10) for _ in k_0]
    bds = zip_pairs(bds_a0 + bds_a1 + bds_lam + bds_k)
    params_0 = np.concatenate((a0_0,a1_0,lam_0,k_0))
    if Rsem is None:
        Rsem = np.ones_like(R)
    else:
        Rsem[Rsem==0] = np.min(Rsem[Rsem>0])

    #def compute_this_cost(params):
    #    return (R-weibull(c,params,nsizes))/Rsem
    #params_opt = sop.least_squares(lambda params: compute_this_cost(params).flatten(),params_0,bounds=bds)
    #try:
    #    cov = np.linalg.inv(params_opt['jac'].T @ params_opt['jac']) 
    #except:
    #    cov = np.nan*np.ones((params_opt['jac'].shape[1],params_opt['jac'].shape[1]))
    #return params_opt['x'],cov

    nvar = 4
    popt = np.zeros((nvar*nsizes,))
    pcov = np.zeros((nvar*nsizes,nvar*nsizes))
    for isize in range(nsizes):
        slc = slice(isize,nsizes*nvar,nsizes)
        popt[slc],pcov[slc][:,slc] = sop.curve_fit(weibull_one_size,c,R[isize],p0=params_0[slc],bounds=[b[slc] for b in bds])
    return popt,pcov

def weibull(c,params,ncells):
    a0 = params[:ncells][:,np.newaxis]
    a1 = params[ncells:2*ncells][:,np.newaxis]
    lam = params[2*ncells:3*ncells][:,np.newaxis]
    k = params[3*ncells:4*ncells][:,np.newaxis]
    aux = (c[np.newaxis,:]/lam)
    return a0 + (1-a0-a1)*(1-np.exp(-aux**k))

def weibull_one_size(c,*params):
    return weibull(c,np.array(params),1)[0]

def two_asymptote_fn(x,*params):
    x0,a1,b1,a2,b2,lam = params
    as1 = a1*x + b1
    as2 = a2*x + b2
    factor = 1/(1+np.exp((x-x0)/lam))
    return factor*as1 + (1-factor)*as2

def fit_opt_params_two_asymptote_fn(x,R):
    nsizes = R.shape[0]
    x0_0 = np.nanmean(x)*np.ones((nsizes,))
    a1_0 = (R[:,1]-R[:,0])/(x[1]-x[0])
    b1_0 = R[:,0] - a1_0[:]*x[0]
    a2_0 = (R[:,-1]-R[:,-2])/(x[-1]-x[-2])
    b2_0 = R[:,-1] - a2_0[:]*x[-1]
    lam_0 = np.ones((nsizes,))
    bds = [(-np.inf,np.inf) for _ in range(5)] + [(0,np.inf)]
    bds = zip_pairs(bds)
    params_0 = np.concatenate([z[:,np.newaxis] for z in (x0_0,a1_0,b1_0,a2_0,b2_0,lam_0)],axis=1)

    nvar = 6
    popt = np.nan*np.ones((nsizes,nvar))
    pcov = np.nan*np.ones((nsizes,nvar,nvar))
    for isize in range(nsizes):
        try:
            popt[isize],pcov[isize] = sop.curve_fit(two_asymptote_fn,x,R[isize],p0=params_0[isize],bounds=bds)
        except:
            print('did not work for %d'%isize)
    return popt,pcov

def plot_model_comparison(c,mn,lb,ub,fit_fn=naka_rushton_only_a,popt=None,rowlen=10):
    ncells = mn.shape[0]
    nsizes = mn.shape[1]
    nrows = int(np.ceil(ncells/rowlen))
    for k in range(ncells):
    	plt.subplot(nrows,rowlen,k+1)
    	r = mn[k]
    	colors = plt.cm.viridis(np.linspace(0,1,nsizes))
    	sim = fit_fn(c,popt[k],ncells)
    	for s in range(r.shape[0]):
    	    plot_errorbar(c,r[s],lb[k,s],ub[k,s],color=colors[s])
    	    plt.plot(c,sim[s],c=colors[s],linestyle='dashed')
    	    plt.axis('off')
