#!/usr/bin/env python
import autograd.numpy as np
from autograd import grad,elementwise_grad,jacobian
import scipy.optimize as sop
import matplotlib.pyplot as plt
import pyute as ut
import nub_utils
import sim_utils
import size_contrast_analysis as sca
from numpy import maximum as npmaximum

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

def fit_opt_params_monotonic(c,R,Rsem=None,clip_decreasing=False,clip_after=1):
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
    if clip_decreasing:
        for isize in range(nsizes):
            #decreasing = (np.diff(R[isize])<0)
            decreasing = (R[isize] < npmaximum.accumulate(R[isize]))
            if np.any(decreasing):
                R[isize,np.where(decreasing)[0][0]+1+clip_after:] = -1
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

class nr_model(object):
    # a Naka-Rushton-like model fit to data
    def __init__(self,c,R,nr_fn,bd_fn,init_fn,Rsem=None):
        self.c = c
        self.non_nan_rows = ~np.all(np.isnan(R),axis=1)
        self.data = R[self.non_nan_rows]
        self.base_fn = nr_fn
        self.nsizes = self.data.shape[0]
        self.bds = bd_fn(self.c,self.data)
        self.params_0 = init_fn(self.c,self.data)
        if Rsem is None:
            self.datasem = np.ones_like(self.data)
        else:
            self.datasem[self.datasem==0] = np.min(self.datasem[self.datasem>0])
        if not np.all(np.isnan(self.data)):
            self.fit()
        else:
            self.params_opt = np.nan*np.ones_like(self.params_0)
    def compute_cost(self,params):
        return (self.data-self.base_fn(self.c,params,self.nsizes))/self.datasem
    def fit(self):
        params_opt = sop.least_squares(lambda params: self.compute_cost(params).flatten(),self.params_0,bounds=self.bds)
        self.params_opt = params_opt['x']
        self.fn = lambda c,d: self.base_fn(c,self.params_opt,self.nsizes)

class nr_two_n_model(nr_model):
    def __init__(self,c,R):
        super().__init__(c,R,naka_rushton_two_n,self.bd_fn,self.init_fn)
    def bd_fn(self,c,R):
        nsizes = R.shape[0]
        bds_a = [(0,np.inf) for isize in range(nsizes)]
        bds_b = [(0,np.inf)]
        bds_c50 = [(0,np.max(c)) for isize in range(nsizes)]
        bds_n = [(0,4) for ivar in range(2)]
        bds = zip_pairs(bds_a + bds_b + bds_c50 + bds_n)
        return bds
    def init_fn(self,c,R):
        nsizes = R.shape[0]
        a_0 = R.max(1)
        b_0 = 0
        c50_0 = c.max()/2*np.ones(nsizes)
        n_top_0 = 2
        n_bottom_0 = 2
        params_0 = np.concatenate((a_0,(b_0,),c50_0,(n_top_0,),(n_bottom_0,)))
        return params_0

class nr_two_c50_model(nr_model):
    def __init__(self,c,R):
        super().__init__(c,R,naka_rushton_two_c50,self.bd_fn,self.init_fn)
    def bd_fn(self,c,R):
        nsizes = R.shape[0]
        bds_a = [(0,np.inf) for isize in range(nsizes)]
        bds_b = [(0,np.inf)]
        bds_c50 = [(0,np.max(c)) for isize in range(nsizes)]
        bds_n = [(0,4) for ivar in range(1)]
        bds = zip_pairs(bds_a + bds_b + bds_c50 + bds_c50 + bds_n)
        return bds
    def init_fn(self,c,R):
        nsizes = R.shape[0]
        a_0 = R.max(1)
        b_0 = 0
        c50_0 = c.max()/2*np.ones(nsizes)
        n_0 = 2
        params_0 = np.concatenate((a_0,(b_0,),c50_0,c50_0,(n_0,)))
        return params_0

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
    # return a ncells x len(c) array of naka rushton function outputs as a function of contrast, for each "cell".
    a = params[:ncells]
    b = params[ncells]
    c50 = params[ncells+1:-2]
    n_top = params[-2]
    n_bottom = params[-1]
    aux = (c[np.newaxis]/c50[:,np.newaxis])
    return (a[:,np.newaxis]*aux**n_top + b)/(1+aux**n_bottom)

def naka_rushton_two_c50(c,params,ncells):
    # return a ncells x len(c) array of naka rushton function outputs as a function of contrast, for each "cell".
    a = params[:ncells]
    b = params[ncells]
    c50top = params[ncells+1:2*ncells+1]
    c50bottom = params[2*ncells+1:3*ncells+1]
    n = params[-2]
    r0 = params[-1]
    aux_top = (c[np.newaxis]/c50top[:,np.newaxis])
    aux_bottom = (c[np.newaxis]/c50bottom[:,np.newaxis])
    return (a[:,np.newaxis]*aux_top**n + b)/(1+aux_top**n)/(1+aux_bottom**n) + r0

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
    lam_0 = 0.25*(x[-1]-x[0])*np.ones((nsizes,))
    bds = [(-np.inf,np.inf) for _ in range(5)] + [(0,np.inf)]
    #bds = zip_pairs(bds)
    params_0 = np.concatenate([z[:,np.newaxis] for z in (x0_0,a1_0,b1_0,a2_0,b2_0,lam_0)],axis=1)

    nvar = 6
    popt = np.nan*np.ones((nsizes,nvar))
    pcov = np.nan*np.ones((nsizes,nvar,nvar))
    def f(*params):
        return two_asymptote_fn(x,*params) 
    def fprime(x,*params):
        return grad(f)(x,*params)
    for isize in range(nsizes):
        def cost(params):
            return np.sum((R[isize] - f(*params))**2)
        def costprime(params):
            return grad(cost)(params)
        res = sop.minimize(cost,x0=params_0[isize],bounds=bds,method='L-BFGS-B',jac=costprime)
        popt[isize] = res.x
        pcov[isize] = res.jac
        if not res.success:
            print('did not work for %d'%isize)
        #popt[isize],pcov[isize] = sop.curve_fit(two_asymptote_fn,x,R[isize],p0=params_0[isize],bounds=bds)
        #try:
        #    popt[isize],pcov[isize] = sop.curve_fit(two_asymptote_fn,x,R[isize],p0=params_0[isize],bounds=bds)
        #except:
        #    print('did not work for %d'%isize)
        #    popt[isize] = params_0[isize]
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

class ayaz_model(object):
    # runs fitting if theta is None
    def __init__(self,data,usize=np.array((5,8,13,22,36,60)),ucontrast=np.array((0,6,12,25,50,100))/100,norm_top=False,norm_bottom=False,theta0=None,theta=None,delta0=0,two_n=False,sub=False):
        self.data = data
        self.usize = usize        
        self.ucontrast = ucontrast
        self.norm_top = norm_top
        self.norm_bottom = norm_bottom
        if not two_n:
            self.base_fn = nub_utils.ayaz_model_div_ind_n_offset
        elif sub:
            self.base_fn = nub_utils.ayaz_model_sub_ind_n_offset
        else:
            self.base_fn = nub_utils.ayaz_model_div_ind_n_pm_offset
        if len(data.shape)==2:
            self.data = self.data[np.newaxis]
        assert(self.data.shape[1]==len(usize))
        assert(self.data.shape[2]==len(ucontrast))
        if theta is None:
            self.fit(theta0=theta0,delta0=delta0)
        else:
            self.theta = theta
            # len(d) x len(c) output
            self.fn = lambda c,d: nub_utils.ayaz_like_theta(c,d,self.theta,fn=self.base_fn)
    def fit(self,theta0=None,delta0=0):
        r0 = np.nanmin(self.data)
        rd = 100
        rs = 100
        sd = 10
        ss = 60
        m = 1
        n = 1#2
        delta = delta0
        mmin = 0.5
        mmax = 5
        smax = 180
            
        def cost(theta):
            modeled = nub_utils.ayaz_like_theta(self.ucontrast,self.usize,theta,fn=self.base_fn)
            non_nan = ~np.isnan(self.data)
            return np.sum((modeled[np.newaxis] - self.data)[non_nan]**2)
        
        if theta0 is None:
            #theta0 = np.array((r0,0,rd,rs,0,rd,rs,sd,ss,sd,ss,m,1,2,delta))
            theta0 = np.array((r0,0,rd,rs,0,rd,rs,sd,ss,sd,ss,m,n,n,delta))
        bds = sop.Bounds(np.zeros_like(theta0),np.inf*np.ones_like(theta0))
#         bds.ub[4] = 0
        bds.lb[-4:-1] = mmin
        bds.ub[-4:-1] = mmax
        bds.ub[-8:-4] = smax
        bds.ub[4] = 0
        
        if not self.norm_top:
            bds.ub[3] = 0
            theta0[3] = 0
        
        if not self.norm_bottom:
            bds.ub[6] = 0
            theta0[6] = 0
            
        res = sop.minimize(cost,theta0,method='L-BFGS-B',bounds=bds,jac=grad(cost))
        if res.success:
            self.theta = res.x
            self.cost = res.fun
            self.modeled = nub_utils.ayaz_like_theta(self.ucontrast,self.usize,self.theta,fn=self.base_fn)
            self.fn = lambda c,d: nub_utils.ayaz_like_theta(c,d,self.theta,fn=self.base_fn)
            self.c50 = self.theta[5]**(-1/self.theta[-3])
            self.c50top = self.theta[3]**(-1/self.theta[-3])
            self.c50bottom = self.theta[6]**(-1/self.theta[-3])
        else:
            print(str((itype,iexpt))+' unsuccessful')

class ayaz_ori_model(ayaz_model):
    def __init__(self,data,usize=np.array((5,8,13,22,36,60)),ucontrast=np.array((0,6,12,25,50,100))/100,uangle=np.arange(0,360,45),norm_top=False,norm_bottom=False,theta0=None,theta=None,delta0=0,two_n=False):
        super().__init__(data,usize=usize,ucontrast=ucontrast,norm_top=norm_top,norm_bottom=norm_bottom,theta0=theta0,theta=theta,delta0=delta0,two_n=two_n)
        self.uangle = uangle

def plot_interp_contrast_tuning(ams=None,data=None,theta=None,these_sizes=[0,2,4],ninterp=101,usize=np.array((5,8,13,22,36,60)),ucontrast=np.array((0,6,12,25,50,100))/100,colors=None,deriv=False,deriv_axis=2):
    ucontrast_interp = np.linspace(0,1,ninterp)
    this_nsize = len(these_sizes)
    this_ucontrast = ucontrast_interp

    if ams is None:
        #assert(not data is None and not theta is None)
        assert(not data is None)
        ams = ayaz_model(data,usize=usize,ucontrast=ucontrast,theta=theta)
    this_usize = np.array([ams.usize[i] for i in these_sizes])
    this_theta = ams.theta
    fn = ams.fn

    this_data = ams.data
    ind0 = 2
    cinds = np.concatenate(((5+np.log2(ucontrast_interp[ind0]),),np.arange(1,6)))
    if deriv:
        if deriv_axis==1:
            this_data = sca.compute_slope_avg(usize,this_data,axis=deriv_axis)
        elif deriv_axis==2:
            this_data = sca.compute_slope_avg(ucontrast,this_data,axis=deriv_axis)
        this_modeled = np.zeros((this_nsize,ninterp))
        for isize in range(this_nsize):
            if deriv_axis==2:
                crf = lambda c: ams.fn(np.array((c,)),np.array((this_usize[isize],)))[0,0]
                cslope = np.array([grad(crf)(cc) for cc in this_ucontrast])
                this_modeled[isize] = cslope
                #print(cslope)
            elif deriv_axis==1:
                for icontrast,cc in enumerate(this_ucontrast):
                    srf = lambda d: ams.fn(np.array((cc,)),np.array((d,)))[0,0]
                    sslope = grad(srf)(this_usize[isize])
                    this_modeled[isize,icontrast] = sslope
    else:
        #this_modeled = nub_utils.ayaz_like_theta(this_ucontrast,this_usize,this_theta,fn=fn)
        this_modeled = ams.fn(this_ucontrast,this_usize)
    ut.plot_bootstrapped_errorbars_hillel(cinds,this_data[:,these_sizes,:].transpose((0,1,2)),linewidth=0,colors=colors)
    for isize in range(this_nsize):
        plt.plot(5+np.log2(ucontrast_interp[ind0:]),this_modeled[isize,ind0:],c=colors[isize])
    plt.xticks(cinds,(100*ucontrast).astype('int'))
    ut.erase_top_right()
    plt.xlabel('contrast (%)')
    plt.ylabel('event rate/mean')
    if not deriv:
        plt.gca().set_ylim(bottom=0)
    plt.tight_layout()

def plot_interp_size_tuning(ams=None,data=None,theta=None,these_contrasts=[1,3,5],ninterp=101,usize=np.array((5,8,13,22,36,60)),ucontrast=np.array((0,6,12,25,50,100))/100,colors=None,error_type='bs',deriv=False,deriv_axis=1,sub=False,two_n=False):
    usize_interp = np.linspace(0,usize[-1],ninterp)
    this_ncontrast = len(these_contrasts)
    this_usize = usize_interp
    this_ucontrast = ucontrast

    if ams is None:
        #assert(not data is None and not theta is None)
        assert(not data is None)
        ams = ayaz_model(data,usize=usize,ucontrast=ucontrast,theta=theta,sub=sub,two_n=two_n)
    this_ucontrast = np.array([ams.ucontrast[i] for i in these_contrasts])
    this_theta = ams.theta
    fn = ams.fn

    usize0 = np.concatenate(((0,),usize))
    this_data = sim_utils.gen_size_tuning(ams.data)

    if deriv:
        if deriv_axis==1:
            this_data = sca.compute_slope_avg(usize0,this_data,axis=deriv_axis)
        elif deriv_axis==2:
            this_data = sca.compute_slope_avg(ucontrast,this_data,axis=deriv_axis)
        this_modeled = np.zeros((ninterp,this_ncontrast))
        for icontrast in range(this_ncontrast):
            if deriv_axis==1:
                srf = lambda d: ams.fn(np.array((this_ucontrast[icontrast],)),np.array((d,)))[0,0]
                sslope = np.array([grad(srf)(ss) for ss in this_usize])
                this_modeled[:,icontrast] = sslope
            elif deriv_axis==2:
                for isize,ss in enumerate(this_usize):
                    crf = lambda c: ams.fn(np.array((c,)),np.array((ss,)))[0,0]
                    sslope = grad(crf)(this_ucontrast[icontrast])
                    this_modeled[isize,icontrast] = sslope
    else:
        this_modeled = ams.fn(this_ucontrast,this_usize)

    if True:
        if error_type=='bs':
            ut.plot_bootstrapped_errorbars_hillel(usize0,this_data[:,:,these_contrasts].transpose((0,2,1)),linewidth=0,colors=colors)
        elif error_type=='pct':
            ut.plot_pct_errorbars_hillel(usize0,this_data[:,:,these_contrasts].transpose((0,2,1)),linewidth=0,colors=colors,pct=(16,84))
        for icontrast in range(this_ncontrast):
            plt.plot(usize_interp,this_modeled[:,icontrast],c=colors[icontrast])
        #plt.xticks(cinds,(100*ucontrast).astype('int'))
    ut.erase_top_right()
    plt.xlabel('size ($^o$)')
    plt.ylabel('event rate/mean')
    if not deriv:
        plt.gca().set_ylim(bottom=0)
    plt.tight_layout()

def slope_c50_fn(crf,cinterp,thresh=0.5):
    cslope = np.array([grad(crf)(cc) for cc in cinterp])
    cmin = np.nanargmin(cslope)
    cmax = np.nanargmax(cslope)
    cinfl = cmax+np.where((cslope[cmax:])<slope_thresh*(cslope[cmax]))[0]
    if cinfl.size:
        cinfl = cinfl[0]
        pseudo_c50 = cinterp[cinfl]
    else:
        print('could not do!')
        print(cslope[cmax])
    return pseudo_c50

def numeric_c50_fn(crf,cinterp,thresh=0.5):
    cval = np.array([crf(cc) for cc in cinterp])
    decreasing = (np.diff(cval)<0)
    if np.any(decreasing):
        cval = cval[:np.where(decreasing)[0][0]+1]
    cval_max = np.nanmax(cval)
    cval_min = cval[0]
    cinfl = np.where(cval-cval_min>=thresh*(cval_max-cval_min))[0]
    if cinfl.size:
        cinfl = cinfl[0]
        pseudo_c50 = cinterp[cinfl]
    else:
        print('could not do!')
        print(cval_max)
    return pseudo_c50

def compute_pseudo_c50_fn_size_x_contrast(ams,thresh=0.5,resolution=501,pseudo_c50_fn=slope_c50_fn):
    if hasattr(ams,'ucontrast'):
        max_contrast = ams.ucontrast.max()
    else:
        max_contrast = ams.c.max()
    cinterp = np.linspace(0,max_contrast,resolution)
    if hasattr(ams,'non_nan_rows'):
        nsize = len(ams.non_nan_rows)
        non_nan_rows = ams.non_nan_rows
    else:
        nsize = ams.data.shape[0] 
        non_nan_rows = np.ones((nsize,),dtype='bool')
    pseudo_c50 = np.nan*np.ones((nsize,))
    if hasattr(ams,'usize'):
        usize = ams.usize
    else:
        usize = np.ones((nsize,))
    for isize in range(nsize):
        if non_nan_rows[isize]:
            crf = lambda c: ams.fn(np.array((c,)),usize[isize:isize+1])[0,0]
            pseudo_c50[isize] = pseudo_c50_fn(crf,cinterp,thresh=thresh)
            #cslope = np.array([grad(crf)(cc) for cc in cinterp])
            #cmin = np.nanargmin(cslope)
            #cmax = np.nanargmax(cslope)
            #cinfl = cmax+np.where((cslope[cmax:])<slope_thresh*(cslope[cmax]))[0]
            #if cinfl.size:
            #    cinfl = cinfl[0]
            #    pseudo_c50[isize] = cinterp[cinfl]
            #else:
            #    print('could not do: ')
            #    print((isize,cmax))
            #    print(cslope[cmax])
        else:
            pseudo_c50[isize] = np.nan
    return pseudo_c50

def compute_pseudo_c50(ams_sst,slope_thresh=0.5):
    cinterp = np.linspace(0,1,501)
    nexpt = len(ams_sst)
    nlight = len(ams_sst[0])
    nsize = len(ams_sst[0][0].usize)
    usize = ams_sst[0][0].usize
    pseudo_c50sst = np.nan*np.ones((nexpt,nsize,nlight))
    for iexpt in range(nexpt):
        if not ams_sst[iexpt][0] is None:
            #plt.figure(figsize=(5,2.5))
            for ilight in range(nlight):
                #plt.subplot(1,2,ilight+1)
                for isize in range(nsize):
                    crf = lambda c: ams_sst[iexpt][ilight].fn(np.array((c,)),np.array((usize[isize],)))[0,0]
                    cslope = np.array([grad(crf)(cc) for cc in cinterp])
                    cmin = np.nanargmin(cslope)
                    cmax = np.nanargmax(cslope)
        #             cinfl = cmax+np.where((cslope[cmax:]-cslope[cmin])<0.5*(cslope[cmax]-cslope[cmin]))[0]
                    cinfl = cmax+np.where((cslope[cmax:])<slope_thresh*(cslope[cmax]))[0]
                    if cinfl.size:
                        cinfl = cinfl[0]
                        pseudo_c50sst[iexpt,isize,ilight] = cinterp[cinfl]
#                         plt.scatter(100*cinterp[cinfl],cslope[cinfl],c=csize[isize])
                    else:
                        print('could not do: ')
                        print((iexpt,ilight,isize,cmax))
                        print(cslope[cmax])
                    if False: # True:
                        plt.plot(100*cinterp,cslope,c=csize[isize])
                        plt.axhline(0.5*cslope[cmax],c=csize[isize],linestyle='dashed')
                        if cinfl.size:
                            plt.scatter(100*cinterp[cinfl],cslope[cinfl],c=csize[isize])
    return pseudo_c50sst
