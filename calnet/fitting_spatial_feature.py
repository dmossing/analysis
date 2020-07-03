#!/usr/bin/env python
import autograd.numpy as np
import scipy.optimize as sop
from autograd import elementwise_grad as egrad
from autograd import grad
from autograd import jacobian
from autograd import hessian
import matplotlib.pyplot as plt

def print_labeled(lbl,var):
    print(lbl+': '+str(var))

def combine_inside_out(item,combine_fn,niter=0):
    combined_item = item.copy()
    for iiter in range(niter):
        for iel in range(len(combined_item)):
            combined_item[iel] = combine_fn(item[iel])
    return combined_item

def access_nested_list_element(this_list,imulti):
    current_el = this_list.copy()
    for ithis in imulti:
        current_el = current_el[ithis]
    return current_el

def flatten_nested_list_of_2d_arrays(this_list):
    current_el = this_list.copy()
    shp = tuple(())
    while isinstance(current_el,list):
        shp = shp + (len(current_el),)
        current_el = current_el[0]
    nflat = np.prod(shp)
    n0,n1 = current_el.shape
    to_return = np.zeros((n0,n1*nflat))
    for iflat in range(nflat):
        imulti = np.unravel_index(iflat,shp)
        to_return[:,iflat*n1:(iflat+1)*n1] = access_nested_list_element(this_list,imulti)
    return to_return
        

def add_up_lists(lst):
    to_return = []
    for item in lst:
        to_return.append(item)
    return to_return

def fit_W_sim(Xhat,Xpc_list,Yhat,Ypc_list,pop_rate_fn=None,pop_deriv_fn=None,neuron_rate_fn=None,W0list=None,bounds=None,dt=1e-1,perturbation_size=5e-2,niter=1,wt_dict=None,eta=0.1,compute_hessian=False,l2_penalty=1.0):
    # X is (N,P), y is (N,Q). Finds wZx, wZy: (P,Q) + (Q,Q) weight matrices to explain Y as Y = f(Xwmx + Ywmy,Xwsx + Ywsy)
    # f is a static nonlinearity, given as a function
    
    #Xpc_list = combine_inside_out(Xpc_list,add_up_lists,niter=2)
    #Ypc_list = combine_inside_out(Ypc_list,add_up_lists,niter=2)
    
    #for iiter in range(2): 
    #    for ixp in range(len(Xpc_list)):
    #        Xpc_list[ixp] = Xpc_list[ixp][0].copy()+Xpc_list[ixp][1].copy()
    #for iiter in range(2): 
    #    for iyp in range(len(Ypc_list)):
    #        Ypc_list[iyp] = Ypc_list[iyp][0].copy()+Ypc_list[iyp][1].copy() # list of (npixels*nQ) elements
    
    factr=1e7
    epsilon=1e-8
    pgtol=1e-5
    fudge = 1e-4
    noise = 1
    
    fprime_m = pop_deriv_fn #utils.fprime_miller_troyer #egrad(pop_rate_fn,0)
    
    #combine_fn = lambda x: np.concatenate(x,axis=1)
    #YYhat = combine_inside_out(Yhat,combine_fn,niter=2)
    #XXhat = combine_inside_out(Xhat,combine_fn,niter=2)
    YYhat = flatten_nested_list_of_2d_arrays(Yhat)
    XXhat = flatten_nested_list_of_2d_arrays(Xhat)
    
    nS = len(Yhat)
    nT = len(Yhat[0])
    assert(nS==len(Xhat))
    assert(nT==len(Xhat[0]))
    nN,nP = Xhat[0][0].shape
    nQ = Yhat[0][0].shape[1]
    assert(nN==Yhat[0][0].shape[0])
    
    def add_key_val(d,key,val):
        if not key in d:
            d[key] = val
    
    if wt_dict is None:
        wt_dict = {}
    add_key_val(wt_dict,'celltypes',np.ones((1,nT*nS*nQ)))
    add_key_val(wt_dict,'inputs',np.concatenate([np.array((1,0)) for i in range(nT*nS)],axis=0)[np.newaxis,:])
    add_key_val(wt_dict,'stims',np.ones((nN,1)))
    add_key_val(wt_dict,'X',1)
    add_key_val(wt_dict,'Y',1)
    add_key_val(wt_dict,'Eta',1)
    add_key_val(wt_dict,'Xi',1)
    add_key_val(wt_dict,'barrier',1)
    
    wtCell = wt_dict['celltypes']
    wtInp = wt_dict['inputs']
    wtStim = wt_dict['stims']
    wtX = wt_dict['X']
    wtY = wt_dict['Y']
    wtEta = wt_dict['Eta']
    wtXi = wt_dict['Xi']
    barrier_wt = wt_dict['barrier']
    
    first = True
    

        
    # Yhat is all measured tuning curves, Y is the averages of the model tuning curves
    def parse_W(W):
        # Wmx,Wmy,Wsx,Wsy,s02,K,kappa,T,XX,XXp,YY,Eta,Xi
        shapes = [(nP,nQ),(nQ,nQ),(nP,nQ),(nQ,nQ),(nQ,),(nQ,),(1,),(nQ,),(nN,nT*nS*nP),(nN,nT*nS*nP),(nN,nT*nS*nQ),(nN,nT*nS*nQ)]
        Ws = parse_thing(W,shapes)
        return Ws
    
    def parse_thing(V,shapes):
        for shape in shapes:
            if type(shape) is int:
                shape = (shape,)
        sizes = [np.prod(shape) for shape in shapes]
        sofar = 0
        outputs = []
        for size,shape in zip(sizes,shapes):
            if size > 1:
                new_element = V[sofar:sofar+size].reshape(shape)
            else:
                new_element = V[sofar] # if just a float
            outputs.append(new_element)
            sofar = sofar + size
        return outputs
    
    def unparse_W(*Ws):
        return np.concatenate([ww.flatten() for ww in Ws])
    
    def normalize(arr):
        arrsum = arr.sum(1)
        well_behaved = (arrsum>0)[:,np.newaxis]
        arrnorm = well_behaved*arr/arrsum[:,np.newaxis] + (~well_behaved)*np.ones_like(arr)/arr.shape[1]
        return arrnorm
    
    def gen_Weight(W,K,kappa,T):
        #Wpartlist = [W*(K[np.newaxis,:]**np.abs(iS)) for iS in range(-nS+1,nS)]
        #WWlist = [np.concatenate(Wpartlist[nS-iS-1:2*nS-iS-1],axis=1) for iS in range(nS)]
#        WW0 = np.concatenate((W,W*K[np.newaxis,:],W*K[np.newaxis,:]**2),axis=1)
#        WW1 = np.concatenate((W*K[np.newaxis,:]*kappa,W),axis=1)
        #WW = np.concatenate(WWlist,axis=0)
        #print('T: '+str(T.shape))
        #WT = circulate(W,T,nT,mu=None)
        #MuT = np.array((1,2,1))
        MuT = np.array((1,1))
        MuK = np.array((1,kappa))
        WT = circulate(W,T,nT,Mu=MuT)
        #print('WT: '+str(WT.shape))
        #KKlist = [np.concatenate([K[np.newaxis,:] for iZ in range(WT.shape[0])],axis=0) for iS in range(nS)]
        #KKlist = [np.concatenate([K[np.newaxis,:] for iZ in range(WT.shape[0])],axis=0) for iT in range(nT)]
        KKlist = [K for iT in range(nT)]
        KK = np.concatenate(KKlist,axis=0)
        #print('KK: '+str(KK.shape))
        WW = circulate(WT,KK,nS,Mu=MuK)
        #print('WW: '+str(WW.shape))
        #WW = circulate(W,K,nS,mu=kappa)
        #TTlist = [np.concatenate([T[np.newaxis,:] for iZ in range(WW.shape[0])],axis=0) for iS in range(nS)]
        #TT = np.concatenate(TTlist,axis=1)
        #WT = circulate(WW,TT,nT)
        return WW
    
   # def circulate(V,M,nZ,mu=None):
   #     Vpartlist = [V*(M[np.newaxis,:]**np.abs(iZ)) for iZ in range(-nZ+1,nZ)]
   #     if not mu is None:
   #         if not mu.shape:
   #             Mu = np.array(((1,1),(mu,mu)))
   #         else:
   #             Mu = np.ones((nZ,nZ),dtype=mu.dtype)
   #             for iZ in range(1,nZ):
   #                 these = list(np.arange(iZ)) + list(np.arange(iZ+1,nZ))
   #                 Mu[iZ,these] = mu[iZ-1]
   #     else:
   #         Mu = np.ones((nZ,nZ))
   #     VVlist = [np.concatenate([m*v for m,v in zip(Mu[iZ],Vpartlist[nZ-iZ-1:2*nZ-iZ-1])],axis=1) for iZ in range(nZ)]
   #     VV = np.concatenate(VVlist,axis=0)
   #     return VV
                       
    def circulate(V,M,nZ,Mu=None):
        Vpartlist = [V*(M[np.newaxis,:]**np.abs(iZ)) for iZ in range(-nZ+1,nZ)]
        if Mu is None:
            Mu = np.ones((nZ,))
        #VVlist = [np.concatenate([m*v for m,v in zip(Mu,Vpartlist[nZ-iZ-1:2*nZ-iZ-1])],axis=1) for iZ in range(nZ)]
        VVlist = [np.concatenate(Vpartlist[nZ-iZ-1:2*nZ-iZ-1],axis=1) for iZ in range(nZ)]
        #VV = np.concatenate(VVlist,axis=0)
        VV = np.concatenate([m*v for m,v in zip(Mu,VVlist)],axis=0)
        return VV
    
    def inner_product_(a,b):
        return np.sum(a*b,0)
    
    def compute_tr_siginv2_sig1(stim_deriv,noise,pc_list):
            # pc_list a list of PCs of the covariance matrix, each a tuple of (sigma,normed vector)
        tot = 0
        this_ncelltypes = len(pc_list[0][0])
        for iel in range(stim_deriv.shape[1]):
            iS,iT,icelltype = np.unravel_index(iel,(nS,nT,this_ncelltypes))
            sigma2 = np.sum(stim_deriv[:,iel]**2)
            #print_labeled('sigma2',sigma2)
            if sigma2>0 and not pc_list[iS][iT][icelltype] is None:
                inner_prod = np.sum([pc[0]**2*np.sqrt(sigma2)*inner_product_(stim_deriv[:,iel],pc[1]) for pc in pc_list[iS][iT][icelltype]])
            else:
                inner_prod = 0 
            tot = tot - 1/noise/(noise + sigma2)*inner_prod #.sum()
        return tot
        
    def compute_log_det_sig2(stim_deriv,noise):
        sigma2 = inner_product_(stim_deriv,stim_deriv) #np.sum(stim_deriv**2,0)
        return np.sum([np.log(s2+noise) for s2 in sigma2])
        
    def compute_mahalanobis_dist(stim_deriv,noise,mu_data,mu_model):
            # in the case where stim_deriv = 0 (no variability model) only the noise (sqerror) term contributes
        mu_dist = mu_model-mu_data
        inner_prod = inner_product_(stim_deriv,mu_dist) #np.einsum(stim_deriv,mu_dist,'ik,jk->ij')
        sigma2 = inner_product_(stim_deriv,stim_deriv) #np.sum(stim_deriv**2,0)
        noise_term = 1/noise*np.sum(inner_product_(mu_dist,mu_dist)) #np.inner(mu_dist,mu_dist)
        cov_term = np.sum([-1/noise/(noise+s2)*np.sum(ip**2) for s2,ip in zip(sigma2,inner_prod)])
        return noise_term + cov_term
        
    def compute_kl_divergence(stim_deriv,noise,mu_data,mu_model,pc_list):
            # omitting a few terms: - d - log(sig1) # where d is the dimensionality
            # in the case where stim_deriv = 0 (no variability model) only the noise (sqerror) 
            # term in mahalanobis_dist contributes
        log_det = compute_log_det_sig2(stim_deriv,noise)
        tr_sig_quotient = compute_tr_siginv2_sig1(stim_deriv,noise,pc_list)
        maha_dist = compute_mahalanobis_dist(stim_deriv,noise,mu_data,mu_model)
        lbls = ['log_det','tr_sig_quotient','maha_dist']
        vars = [log_det,tr_sig_quotient,maha_dist]
        #for lbl,var in zip(lbls,vars):
        #    print_labeled(lbl,var)
        return 0.5*(log_det + tr_sig_quotient + maha_dist)
    
    def optimize(W0,compute_hessian=False):
        
        def compute_fprime_(Eta,Xi,s02):
#             Wmx,Wmy,Wsx,Wsy,s02,k,kappa,XX,YY,Eta,Xi = parse_W(W)
#             WWx,WWy = [gen_Weight(W,k,kappa) for W in [Wx,Wy]]
            return fprime_m(Eta,Xi**2+np.concatenate([s02 for ipixel in range(nS*nT)]))*Xi

        def compute_f_(Eta,Xi,s02):
            return pop_rate_fn(Eta,Xi**2+np.concatenate([s02 for ipixel in range(nS*nT)],axis=0))
        
        def compute_f_fprime_t_(W,perturbation):
            Wmx,Wmy,Wsx,Wsy,s02,k,kappa,T,XX,XXp,Eta,Xi = parse_W(W)
            fval = compute_f_(Eta,Xi,s02)
            resEta = Eta - u_fn(XX,fval,Wmx,Wmy,k,kappa,T)
            resXi  = Xi - u_fn(XX,fval,Wsx,Wsy,k,kappa)
            YY = fval + perturbation
            def dYYdt(YY,Eta1,Xi1):
                return -YY + compute_f_(Eta1,Xi1,s02)
            for t in range(niter):
                Eta1 = resEta + u_fn(XX,YY,Wmx,Wmy,k,kappa)
                Xi1 = resXi + u_fn(XX,YY,Wmx,Wmy,k,kappa)
                YY = YY + dt*dYYdt(YY,Eta1,Xi1)
                
            YYprime = compute_fprime_(Eta1,Xi1,s02)
            
            return YY,YYprime
        
        def compute_f_fprime_t_avg_(W,perturbation,burn_in=0.5):
            Wmx,Wmy,Wsx,Wsy,s02,K,kappa,T,XX,XXp,Eta,Xi = parse_W(W)
            fval = compute_f_(Eta,Xi,s02)
            resEta = Eta - u_fn(XX,fval,Wmx,Wmy,K,kappa,T)
            resXi  = Xi - u_fn(XX,fval,Wsx,Wsy,K,kappa,T)
            YY = fval + perturbation
            YYmean = np.zeros_like(Eta)
            YYprimemean = np.zeros_like(Eta)
            def dYYdt(YY,Eta1,Xi1):
                return -YY + compute_f_(Eta1,Xi1,s02)
            for t in range(niter):
                Eta1 = resEta + u_fn(XX,YY,Wmx,Wmy,K,kappa,T)
                Xi1 = resXi + u_fn(XX,YY,Wmx,Wmy,K,kappa,T)
                YY = YY + dt*dYYdt(YY,Eta1,Xi1)
                if t>niter*burn_in:
                    YYprime = compute_fprime_(Eta1,Xi1,s02)
                    YYmean = YYmean + 1/niter/burn_in*YY
                    YYprimemean = YYprimemean + 1/niter/burn_in*YYprime
                
            return YYmean,YYprimemean

        def u_fn(XX,YY,Wx,Wy,K,kappa,T):
            WWx,WWy = [gen_Weight(W,K,kappa,T) for W in [Wx,Wy]]
            #print(WWx.shape)
            #print(WWy.shape)
            #print_labeled('WWx',WWx)
            #print_labeled('WWy',WWy)
            #plt.figure(1)
            #plt.imshow(WWy)
            #plt.savefig('WWy.jpg',dpi=300)
            return XX @ WWx + YY @ WWy
                        
        def minusLW(W):
            
            def compute_sq_error(a,b,wt):
                return np.sum(wt*(a-b)**2)
            
            def compute_kl_error(mu_data,pc_list,mu_model,fprimeval,wt):
                # how to model variability in X?
                kl = compute_kl_divergence(fprimeval,noise,mu_data,mu_model,pc_list)
                return kl #wt*kl
                # principled way would be to use 1/wt for noise term. Should add later.
            
            Wmx,Wmy,Wsx,Wsy,s02,K,kappa,T,XX,XXp,Eta,Xi = parse_W(W)

            #print_labeled('T',T)
            #print_labeled('K',K)
            #print_labeled('Wmy',Wmy)
            
            perturbation = perturbation_size*np.random.randn(*Eta.shape)
            
#             fval,fprimeval = compute_f_fprime_t_(W,perturbation) # Eta the mean input per cell, Xi the stdev. input per cell, s02 the baseline variability in input
            fval,fprimeval = compute_f_fprime_t_avg_(W,perturbation) # Eta the mean input per cell, Xi the stdev. input per cell, s02 the baseline variability in input
            #print_labeled('fval',fval)
            
            Xterm = compute_kl_error(XXhat,Xpc_list,XX,XXp,wtStim*wtInp) # XX the modeled input layer (L4)
            Yterm = compute_kl_error(YYhat,Ypc_list,fval,fprimeval,wtStim*wtCell) # fval the modeled output layer (L2/3)
            Etaterm = compute_sq_error(Eta,u_fn(XX,fval,Wmx,Wmy,K,kappa,T),wtStim*wtCell) # magnitude of fudge factor in mean input
            Xiterm = compute_sq_error(Xi,u_fn(XX,fval,Wsx,Wsy,K,kappa,T),wtStim*wtCell) # magnitude of fudge factor in input variability
            # returns value float
            cost = wtX*Xterm + wtY*Yterm + wtEta*Etaterm + wtXi*Xiterm
            #lbls = ['Yterm']
            #vars = [Yterm]
            lbls = ['cost']
            vars = [cost]
            for lbl,var in zip(lbls,vars):
                print_labeled(lbl,var)
            return cost

        def minusdLdW(W): 
            # returns value (R,)
            # sum in first dimension: (N,1) times (N,1) times (N,P)
#             return jacobian(minusLW)(W)
            return grad(minusLW)(W)
        
        def fix_violations(w,bounds):
            lb = np.array([b[0] for b in bounds])
            ub = np.array([b[1] for b in bounds])
            lb_violation = w<lb
            ub_violation = w>ub
            w[lb_violation] = lb[lb_violation]
            w[ub_violation] = ub[ub_violation]
            return w,lb_violation,ub_violation
        
        def sorted_r_eigs(w):
            drW,prW = np.linalg.eig(w)
            srtinds = np.argsort(drW)
            return drW[srtinds],prW[:,srtinds]
    
        def compute_eig_penalty_(Wmy,K0,kappa,T0):
            # still need to finish! Hopefully won't need
            # need to fix this to reflect addition of kappa argument
            Wsquig = gen_Weight(Wmy,K0,kappa,T0)
            drW,prW = sorted_r_eigs(Wsquig - np.eye(nQ*nS*nT))
            plW = np.linalg.inv(prW)
            eig_outer_all = [np.real(np.outer(plW[:,k],prW[k,:])) for k in range(nS*nQ*nT)]
            eig_penalty_size_all = [barrier_wt/np.abs(np.real(drW[k])) for k in range(nS*nQ*nT)]
            eig_penalty_dir_w = [eig_penalty_size*((eig_outer[:nQ,:nQ] + eig_outer[nQ:,nQ:]) + K0[np.newaxis,:]*(eig_outer[:nQ,nQ:] + kappa*eig_outer[nQ:,:nQ])) for eig_outer,eig_penalty_size in zip(eig_outer_all,eig_penalty_size_all)]
            eig_penalty_dir_k = [eig_penalty_size*((eig_outer[:nQ,nQ:] + eig_outer[nQ:,:nQ]*kappa)*W0my).sum(0) for eig_outer,eig_penalty_size in zip(eig_outer_all,eig_penalty_size_all)]
            eig_penalty_dir_kappa = [eig_penalty_size*(eig_outer[nQ:,:nQ]*k0[np.newaxis,:]*W0my).sum().reshape((1,)) for eig_outer,eig_penalty_size in zip(eig_outer_all,eig_penalty_size_all)]
            eig_penalty_dir_w = np.array(eig_penalty_dir_w).sum(0)
            eig_penalty_dir_k = np.array(eig_penalty_dir_k).sum(0)
            eig_penalty_dir_kappa = np.array(eig_penalty_dir_kappa).sum(0)
            return eig_penalty_dir_w,eig_penalty_dir_k,eig_penalty_dir_kappa
        
        def compute_eig_penalty(W):
            # still need to finish! Hopefully won't need
            W0mx,W0my,W0sx,W0sy,s020,K0,kappa0,T0,XX0,XXp0,Eta0,Xi0 = parse_W(W)
            eig_penalty_dir_w,eig_penalty_dir_k,eig_penalty_dir_kappa = compute_eig_penalty_(W0my,k0,kappa0)
            eig_penalty_W = unparse_W(np.zeros_like(W0mx),eig_penalty_dir_w,np.zeros_like(W0sx),np.zeros_like(W0sy),np.zeros_like(s020),eig_penalty_dir_k,eig_penalty_dir_kappa,np.zeros_like(XX0),np.zeros_like(XXp0),np.zeros_like(Eta0),np.zeros_like(Xi0))
#             assert(True==False)
            return eig_penalty_W

        allhot = np.zeros(W0.shape)
        allhot[:nP*nQ+nQ**2] = 1
        W_l2_reg = lambda W: np.sum((W*allhot)**2)
        f = lambda W: minusLW(W) + l2_penalty*W_l2_reg(W)
        fprime = lambda W: minusdLdW(W) + 2*l2_penalty*W*allhot
        
        W1,loss,result = sop.fmin_l_bfgs_b(f,W0,fprime=fprime,bounds=bounds,factr=1e5,maxiter=int(1e3),epsilon=3e-9)
        if compute_hessian:
            gr = grad(minusLW)(W1)
            hess = hessian(minusLW)(W1)
        else:
            gr = None
            hess = None
        
#         W0mx,W0my,W0sx,W0sy,s020,k0,kappa0,XX0,XXp0,Eta0,Xi0 = parse_W(W1)
        
        return W1,loss,gr,hess,result
    
    W0 = unparse_W(*W0list)

    W1,loss,gr,hess,result = optimize(W0,compute_hessian=compute_hessian)
        
    Wt = parse_W(W1) #[Wmx,Wmy,Wsx,Wsy,s02,k,kappa,XX,XXp,Eta,Xi]
    
    return Wt,loss,gr,hess,result
