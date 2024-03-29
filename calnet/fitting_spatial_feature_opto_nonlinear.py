#!/usr/bin/env python
import autograd.numpy as np
import scipy.optimize as sop
from autograd import elementwise_grad as egrad
from autograd import grad
from autograd import jacobian
from autograd import hessian
import matplotlib.pyplot as plt
from calnet import utils


def fit_W_sim(Xhat,Xpc_list,Yhat,Ypc_list,dYY,pop_rate_fn=None,pop_deriv_fn=None,neuron_rate_fn=None,W0list=None,bounds=None,dt=1e-1,perturbation_size=5e-2,niter=1,wt_dict=None,eta=0.1,compute_hessian=False,l2_penalty=1.0,constrain_isn=False,opto_mask=None,nsize=6,ncontrast=6,coupling_constraints=[(1,0,-1)],tv=False,topo_stims=np.arange(36),topo_shape=(6,6),use_opto_transforms=False,opto_transform1=None,opto_transform2=None):
    # coupling constraints: (i,j,sgn) --> coupling term i->j is constrained to be > 0 (sgn=1) or < 0 (sgn=-1)
    
    fudge = 1e-4
    noise = 1
    big_val = 1e6
    
    fprime_m = pop_deriv_fn #utils.fprime_miller_troyer #egrad(pop_rate_fn,0)
    
    YYhat = utils.flatten_nested_list_of_2d_arrays(Yhat)
    XXhat = utils.flatten_nested_list_of_2d_arrays(Xhat)
    
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
    add_key_val(wt_dict,'opto',100.)
    add_key_val(wt_dict,'isn',0.01)
    add_key_val(wt_dict,'tv',0.01)
    add_key_val(wt_dict,'celltypesOpto',np.ones((1,nT*nS*nQ)))
    add_key_val(wt_dict,'stimsOpto',np.ones((nN,1)))
    add_key_val(wt_dict,'dirOpto',np.ones((2,)))
    
    wtCell = wt_dict['celltypes']
    wtInp = wt_dict['inputs']
    wtStim = wt_dict['stims']
    wtX = wt_dict['X']
    wtY = wt_dict['Y']
    wtEta = wt_dict['Eta']
    wtXi = wt_dict['Xi']
    barrier_wt = wt_dict['barrier']
    wtOpto = wt_dict['opto']
    wtISN = wt_dict['isn']
    wtdYY = wt_dict['dYY']
    wtEta12 = wt_dict['Eta12']
    #wtEtaTV = wt_dict['EtaTV']
    wtTV = wt_dict['tv']
    wtCoupling = wt_dict['coupling']
    wtCellOpto = wt_dict['celltypesOpto']
    wtStimOpto = wt_dict['stimsOpto']
    wtDirOpto = wt_dict['dirOpto']

    #if wtEtaTV > 0:
    #    assert(nsize*ncontrast==nN)

    if wtCoupling > 0:
        assert(not coupling_constraints is None)
        constrain_coupling = True
    else:
        constrain_coupling = False

    # Wmx,Wmy,Wsx,Wsy,s02,K,kappa,T,XX,XXp,YY,Eta,Xi,h1,h2,Eta1,Eta2
    shapes = [(nP,nQ),(nQ,nQ),(nP,nQ),(nQ,nQ),(nQ,),(nQ*(nS-1),),(1,),(nQ*(nT-1),),(nN,nT*nS*nP),(nN,nT*nS*nP),(nN,nT*nS*nQ),(nN,nT*nS*nQ),(1,),(1,),(nN,nT*nS*nQ),(nN,nT*nS*nQ)]

    first = True
        
    # Yhat is all measured tuning curves, Y is the averages of the model tuning curves
    def parse_W(W):
        Ws = utils.parse_thing(W,shapes)
        return Ws
    
    def unparse_W(*Ws):
        return np.concatenate([ww.flatten() for ww in Ws])
    
    def normalize(arr):
        arrsum = arr.sum(1)
        well_behaved = (arrsum>0)[:,np.newaxis]
        arrnorm = well_behaved*arr/arrsum[:,np.newaxis] + (~well_behaved)*np.ones_like(arr)/arr.shape[1]
        return arrnorm
    
    def gen_Weight(W,K,kappa,T):
        return utils.gen_Weight_k_kappa_t(W,K,kappa,T,nS=nS,nT=nT) 
        
    def compute_kl_divergence(stim_deriv,noise,mu_data,mu_model,pc_list):
        return utils.compute_kl_divergence(stim_deriv,noise,mu_data,mu_model,pc_list,nS=nS,nT=nT)

    def compute_var(Xi,s02):
        return fudge+Xi**2+np.concatenate([s02 for ipixel in range(nS*nT)],axis=0)
    
    def optimize(W0,compute_hessian=False):
        
        def compute_fprime_(Eta,Xi,s02):
            return fprime_m(Eta,compute_var(Xi,s02))*Xi

        def compute_f_(Eta,Xi,s02):
            return pop_rate_fn(Eta,compute_var(Xi,s02))

        def compute_f_fprime_t_(W,perturbation,max_dist=1): # max dist added 10/14/20
            Wmx,Wmy,Wsx,Wsy,s02,k,kappa,T,XX,XXp,Eta,Xi,h1,h2,Eta1,Eta2 = parse_W(W)
            fval = compute_f_(Eta,Xi,s02)
            fprimeval = compute_fprime_(Eta,Xi,s02)
            resEta = Eta - u_fn(XX,fval,Wmx,Wmy,k,kappa,T)
            resXi  = Xi - u_fn(XX,fval,Wsx,Wsy,k,kappa)
            YY = fval + perturbation
            YYp = fprimeval
            def dYYdt(YY,Eta1,Xi1):
                return -YY + compute_f_(Eta1,Xi1,s02)
            def dYYpdt(YYp,Eta1,Xi1):
                return -YYp + compute_fprime_(Eta1,Xi1,s02)
            for t in range(niter):
                if np.mean(np.abs(YY-fval)) < max_dist:
                    Eta1 = resEta + u_fn(XX,YY,Wmx,Wmy,k,kappa,T)
                    Xi1 = resXi + u_fn(XX,YY,Wmx,Wmy,k,kappa,T)
                    YY = YY + dt*dYYdt(YY,Eta1,Xi1)
                    YYp = YYp + dt*dYYpdt(YYp,Eta1,Xi1)
                elif np.remainder(t,500)==0:
                    print('unstable fixed point?')
                
            #YYp = compute_fprime_(Eta1,Xi1,s02)
            
            return YY,YYp
        
        def compute_f_fprime_t_avg_(W,perturbation,burn_in=0.5,max_dist=1):
            Wmx,Wmy,Wsx,Wsy,s02,K,kappa,T,XX,XXp,Eta,Xi,h1,h2,Eta1,Eta2 = parse_W(W)
            fval = compute_f_(Eta,Xi,s02)
            fprimeval = compute_fprime_(Eta,Xi,s02)
            resEta = Eta - u_fn(XX,fval,Wmx,Wmy,K,kappa,T)
            resXi  = Xi - u_fn(XX,fval,Wsx,Wsy,K,kappa,T)
            YY = fval + perturbation
            YYp = fprimeval
            YYmean = np.zeros_like(Eta)
            YYprimemean = np.zeros_like(Eta)
            def dYYdt(YY,Eta1,Xi1):
                return -YY + compute_f_(Eta1,Xi1,s02)
            def dYYpdt(YYp,Eta1,Xi1):
                return -YYp + compute_fprime_(Eta1,Xi1,s02)
            for t in range(niter):
                if np.mean(np.abs(YY-fval)) < max_dist:
                    Eta1 = resEta + u_fn(XX,YY,Wmx,Wmy,K,kappa,T)
                    Xi1 = resXi + u_fn(XX,YY,Wsx,Wsy,K,kappa,T)
                    YY = YY + dt*dYYdt(YY,Eta1,Xi1)
                    YYp = YYp + dt*dYYpdt(YYp,Eta1,Xi1)
                else:
                    print('unstable fixed point?')
                #Eta1 = resEta + u_fn(XX,YY,Wmx,Wmy,K,kappa,T)
                #Xi1 = resXi + u_fn(XX,YY,Wsx,Wsy,K,kappa,T)
                #YY = YY + dt*dYYdt(YY,Eta1,Xi1)
                if t>niter*burn_in:
                    #YYp = compute_fprime_(Eta1,Xi1,s02)
                    YYmean = YYmean + 1/niter/burn_in*YY
                    YYprimemean = YYprimemean + 1/niter/burn_in*YYp
                
            return YYmean,YYprimemean

        def u_fn(XX,YY,Wx,Wy,K,kappa,T):
            WWx,WWy = [gen_Weight(W,K,kappa,T) for W in [Wx,Wy]]
            return XX @ WWx + YY @ WWy
                        
        def minusLW(W):
            
            def compute_sq_error(a,b,wt):
                return np.sum(wt*(a-b)**2)
            
            def compute_kl_error(mu_data,pc_list,mu_model,fprimeval,wt):
                # how to model variability in X?
                kl = compute_kl_divergence(fprimeval,noise,mu_data,mu_model,pc_list)
                return kl #wt*kl
                # principled way would be to use 1/wt for noise term. Should add later.

            def compute_opto_error_nonlinear(W,wt=None):
                if wt is None:
                    wt = np.ones((2*nN,nQ*nS*nT))
                Wmx,Wmy,Wsx,Wsy,s02,K,kappa,T,XX,XXp,Eta,Xi,h1,h2,Eta1,Eta2 = parse_W(W)
                Eta12 = np.concatenate((Eta1,Eta2),axis=0)
                Xi12 = np.concatenate((Xi,Xi),axis=0)
                XX12 = np.concatenate((XX,XX),axis=0)
                fval12 = compute_f_(Eta12,Xi12,s02)

                fval = compute_f_(Eta,Xi,s02)
                dYY12 = fval12 - np.concatenate((fval,fval),axis=0)
                dYYterm = np.sum(wt[opto_mask]*(dYY12[opto_mask] - dYY[opto_mask])**2)

                dHH = np.zeros((nN,nQ*nS*nT))
                dHH[:,np.arange(2,nQ*nS*nT,nQ)] = 1
                dHH = np.concatenate((dHH*h1,dHH*h2),axis=0)
                Eta12perf = u_fn(XX12,fval12,Wmx,Wmy,K,kappa,T) + dHH
                Eta12term = np.sum(wt*(Eta12perf - Eta12)**2)

                #cost = wtdYY*dYYterm + wtEta12*Eta12term
                return dYYterm,Eta12term #cost

            def compute_opto_error_nonlinear_transform(W,wt=None):
                if wt is None:
                    wt = np.ones((2*nN,nQ*nS*nT))
                Wmx,Wmy,Wsx,Wsy,s02,K,kappa,T,XX,XXp,Eta,Xi,h1,h2,Eta1,Eta2 = parse_W(W)
                Eta12 = np.concatenate((Eta1,Eta2),axis=0)
                Xi12 = np.concatenate((Xi,Xi),axis=0)
                XX12 = np.concatenate((XX,XX),axis=0)
                fval12 = compute_f_(Eta12,Xi12,s02)

                fval = compute_f_(Eta,Xi,s02)
                #fvalrep = np.concatenate((fval,fval),axis=0)
                #dYY12 = fval12 - fvalrep
                fval12target = np.concatenate((opto_transform1.transform(fval),opto_transform2.transform(fval)),axis=0)
                #this_dYY = fval12target - fvalrep

                dYYterm = np.sum(wt[opto_mask]*(fval12[opto_mask] - fval12target[opto_mask])**2)

                dHH = np.zeros((nN,nQ*nS*nT))
                dHH[:,np.arange(2,nQ*nS*nT,nQ)] = 1
                dHH = np.concatenate((dHH*h1,dHH*h2),axis=0)
                Eta12perf = u_fn(XX12,fval12,Wmx,Wmy,K,kappa,T) + dHH
                Eta12term = np.sum(wt*(Eta12perf - Eta12)**2)

                #cost = wtdYY*dYYterm + wtEta12*Eta12term
                return dYYterm,Eta12term #cost

            def compute_coupling(W):
                Wmx,Wmy,Wsx,Wsy,s02,K,kappa,T,XX,XXp,Eta,Xi,h1,h2,Eta1,Eta2 = parse_W(W)
                WWy = gen_Weight(Wmy,K,kappa,T)
                Phi = fprime_m(Eta,compute_var(Xi,s02))
                Phi = np.concatenate((Phi,Phi),axis=0)
                Phi1 = np.array([np.diag(phi) for phi in Phi])
                coupling = np.array([phi1 @ np.linalg.inv(np.eye(nQ*nS*nT) - WWy @ phi1) for phi1 in Phi1])
                return coupling

            def compute_coupling_error(W,i,j,sgn=-1):
                # constrain coupling term i,j to have a specified sign, 
                # -1 for negative or +1 for positive
                coupling = compute_coupling(W)
                log_arg = sgn*coupling[:,i,j]
                cost = utils.minus_sum_log_ceil(log_arg,big_val/nN)
                return cost

            #def compute_eta_tv(this_Eta):
            #    Etar = this_Eta.reshape((nsize,ncontrast,nQ*nS*nT))
            #    diff_size = np.sum(np.abs(np.diff(Etar,axis=0)))
            #    diff_contrast = np.sum(np.abs(np.diff(Etar,axis=1)))
            #    return diff_size + diff_contrast

            def compute_isn_error(W):
                Wmx,Wmy,Wsx,Wsy,s02,K,kappa,T,XX,XXp,Eta,Xi,h1,h2,Eta1,Eta2 = parse_W(W)
                Phi = fprime_m(Eta,compute_var(Xi,s02))
                #print('min Eta: %f'%np.min(Eta[:,0]))
                #print('WEE: %f'%Wmy[0,0])
                #print('min phiE*WEE: %f'%np.min(Phi[:,0]*Wmy[0,0]))
                log_arg = Phi[:,0]*Wmy[0,0]-1
                cost = utils.minus_sum_log_ceil(log_arg,big_val/nN)
                #print('ISN cost: %f'%cost)
                return cost
            
            def compute_tv_error(W):
                # sq l2 norm for tv error
                Wmx,Wmy,Wsx,Wsy,s02,K,kappa,T,XX,XXp,Eta,Xi,h1,h2,Eta1,Eta2 = parse_W(W)
                topo_var_list = [arr.reshape(topo_shape+(-1,)) for arr in \
                        [XX,XXp,Eta,Xi,Eta1,Eta2]]
                sqdiffy = [np.sum(np.abs(np.diff(top,axis=0))**2) for top in topo_var_list]
                sqdiffx = [np.sum(np.abs(np.diff(top,axis=1))**2) for top in topo_var_list]
                cost = np.sum(sqdiffy+sqdiffx)
                return cost

            Wmx,Wmy,Wsx,Wsy,s02,K,kappa,T,XX,XXp,Eta,Xi,h1,h2,Eta1,Eta2 = parse_W(W)

            #utils.print_labeled('T',T)
            #utils.print_labeled('K',K)
            #utils.print_labeled('Wmy',Wmy)
            
            perturbation = perturbation_size*np.random.randn(*Eta.shape)
            
#             fval,fprimeval = compute_f_fprime_t_(W,perturbation) # Eta the mean input per cell, Xi the stdev. input per cell, s02 the baseline variability in input
            fval,fprimeval = compute_f_fprime_t_avg_(W,perturbation) # Eta the mean input per cell, Xi the stdev. input per cell, s02 the baseline variability in input
            #utils.print_labeled('fval',fval)
            
            Xterm = compute_kl_error(XXhat,Xpc_list,XX,XXp,wtStim*wtInp) # XX the modeled input layer (L4)
            Yterm = compute_kl_error(YYhat,Ypc_list,fval,fprimeval,wtStim*wtCell) # fval the modeled output layer (L2/3)
            Etaterm = compute_sq_error(Eta,u_fn(XX,fval,Wmx,Wmy,K,kappa,T),wtStim*wtCell) # magnitude of fudge factor in mean input
            Xiterm = compute_sq_error(Xi,u_fn(XX,fval,Wsx,Wsy,K,kappa,T),wtStim*wtCell) # magnitude of fudge factor in input variability
            # returns value float
            #Optoterm = compute_opto_error_nonlinear(W) #testing out 8/20/20
            opto_wt = np.concatenate([wtStimOpto*wtCellOpto*w for w in wtDirOpto],axis=0)
            if use_opto_transforms:
                dYYterm,Eta12term = compute_opto_error_nonlinear_transform(W,opto_wt)
            else:
                dYYterm,Eta12term = compute_opto_error_nonlinear(W,opto_wt)
            Optoterm = wtdYY*dYYterm + wtEta12*Eta12term
            #EtaTVterm = 0
            #for this_Eta in [Eta,Eta1,Eta2]:
            #    EtaTVterm = EtaTVterm + compute_eta_tv(this_Eta)
            cost = wtX*Xterm + wtY*Yterm + wtEta*Etaterm + wtXi*Xiterm + wtOpto*Optoterm# + wtEtaTV*EtaTVterm 
            if constrain_isn:
                ISNterm = compute_isn_error(W)
                cost = cost + wtISN*ISNterm
            if constrain_coupling:
                Couplingterm = 0
                for el in coupling_constraints:
                    i,j,sgn = el
                    Couplingterm = Couplingterm + compute_coupling_error(W,i,j,sgn)
                cost = cost + wtCoupling*Couplingterm
            if tv:
                TVterm = compute_tv_error(W)
                cost = cost + wtTV*TVterm
                
            if isinstance(Xterm,float):
                print('X:%f'%(wtX*Xterm))
                print('Y:%f'%(wtY*Yterm))
                print('Eta:%f'%(wtEta*Etaterm))
                print('Xi:%f'%(wtXi*Xiterm))
                print('Opto dYY:%f'%(wtOpto*wtdYY*dYYterm))
                print('Opto Eta:%f'%(wtOpto*wtEta12*Eta12term))
                #print('TV:%f'%(wtEtaTV*EtaTVterm))
                print('TV:%f'%(wtTV*TVterm))
                if constrain_isn:
                    print('ISN:%f'%(wtISN*ISNterm))
                if constrain_coupling:
                    print('coupling:%f'%(wtCoupling*Couplingterm))

            #lbls = ['Yterm']
            #vars = [Yterm]
            lbls = ['cost']
            vars = [cost]
            for lbl,var in zip(lbls,vars):
                utils.print_labeled(lbl,var)
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
            W0mx,W0my,W0sx,W0sy,s020,K0,kappa0,T0,XX0,XXp0,Eta0,Xi0,h10,h20,Eta10,Eta20 = parse_W(W)
            eig_penalty_dir_w,eig_penalty_dir_k,eig_penalty_dir_kappa = compute_eig_penalty_(W0my,k0,kappa0)
            eig_penalty_W = unparse_W(np.zeros_like(W0mx),eig_penalty_dir_w,np.zeros_like(W0sx),np.zeros_like(W0sy),np.zeros_like(s020),eig_penalty_dir_k,eig_penalty_dir_kappa,np.zeros_like(XX0),np.zeros_like(XXp0),np.zeros_like(Eta0),np.zeros_like(Xi0))
#             assert(True==False)
            return eig_penalty_W

        allhot = np.zeros(W0.shape)
        allhot[:nP*nQ+nQ**2] = 1
        W_l2_reg = lambda W: np.sum((W*allhot)**2)
        f = lambda W: minusLW(W) + l2_penalty*W_l2_reg(W)
        fprime = lambda W: minusdLdW(W) + 2*l2_penalty*W*allhot

        fix_violations(W0,bounds)
        
        W1,loss,result = sop.fmin_l_bfgs_b(f,W0,fprime=fprime,bounds=bounds,factr=1e2,maxiter=int(1e3),maxls=40)
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
