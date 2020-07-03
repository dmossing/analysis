#!/usr/bin/env python
import numpy as np
import scipy.optimize as sop
import matplotlib.pyplot as plt

# fit models of the form (a*(c/c50)^n + b)/((c/c50)^n + 1), where subsets of a, b, and c50 are allowed to vary for each size

def fit_opt_params(c,r):
    a_0 = r.max()
    b_0 = 0
    c50_0 = 50
    n_0 = 1
    params_0 = np.array((a_0,b_0,c50_0,n_0))
    params_opt = sop.least_squares(lambda params: r-naka_rushton(c,params),params_0,bounds=((0,0,0,0),(np.inf,np.inf,200,5)))
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

def naka_rushton(c,params):
    a = params[0]
    b = params[1]
    c50 = params[2]
    n = params[3]
    return (a*(c/c50)**n + b)/(1+(c/c50)**n)

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
    c50_0 = 50*np.ones(nsizes)
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
