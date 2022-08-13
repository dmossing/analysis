import oasis.functions as ofun
import numpy as np
import scipy.ndimage.filters as sfi
#from oasis.functions import deconvolve
import re
import pyute as ut

def process_ca_traces(to_add,ds=10,blspan=3000,blcutoff=1,frm=None,nbefore=4,nafter=4,b_nonneg=True,g0=(None,),reestimate_noise=False,normalize_tavg=False):
    # convert neuropil-corrected calcium traces to df/f. compute baseline as
    # blcutoff percentile filter, over a moving window of blspan frame, down
    # sampled by a factor of ds. Deconvolve using OASIS, and trialize
    # b_nonneg: whether to constrain baseline to be nonnegative
    # g0: if not none, pre-defined AR(1) parameter
    
    to_add[np.isnan(to_add)] = np.nanmin(to_add) #0
    if to_add.max():
        baseline = sfi.percentile_filter(to_add[:,::ds],blcutoff,(1,int(blspan/ds)))
        topline = sfi.percentile_filter(to_add[:,::ds],99,(1,int(blspan/ds))) # dan added 18/10/30
        baseline = np.maximum(baseline,topline/10) # dan added 18/10/30
        baseline = np.repeat(baseline,ds,axis=1)
        if baseline.shape[1]>to_add.shape[1]:
            baseline = baseline[:,:to_add.shape[1]]
        c = np.zeros_like(to_add)
        s = np.zeros_like(to_add)
        b = np.zeros((to_add.shape[0],))
        g = np.zeros((to_add.shape[0],))
        this_dfof = np.zeros_like(to_add)
        for i in range(c.shape[0]):
            this_dfof[i] = (to_add[i]-baseline[i,:])/baseline[i,:]
            this_dfof[i][np.isnan(this_dfof[i])] = 0
            y = this_dfof[i].astype(np.float64)
            c[i],s[i],b[i],g[i],_  = ofun.deconvolve(y,penalty=1,b_nonneg=b_nonneg,g=g0)
            if reestimate_noise:
                sn = ofun.GetSn(y-c[i])
                c[i],s[i],b[i],g[i],_  = ofun.deconvolve(y,penalty=1,b_nonneg=b_nonneg,g=(g[i],),sn=sn)
            if normalize_tavg:
                s[i] = s[i]/(1-g[i])

    else:
        this_dfof = np.zeros_like(to_add)
        c = np.zeros_like(to_add)
        s = np.zeros_like(to_add)
        b = np.zeros((to_add.shape[0],))
        g = np.zeros((to_add.shape[0],2))
    to_add = ut.trialize(to_add,frm,nbefore,nafter)
    c = ut.trialize(c,frm,nbefore,nafter)
    s = ut.trialize(s,frm,nbefore,nafter)
    d = ut.trialize(this_dfof,frm,nbefore,nafter)
    return to_add,c,s,d,b,g #,this_dfof #(non-trialized)

def gen_precise_trialwise(datafiles,nbefore=4,nafter=8,blcutoff=1,blspan=3000,ds=10,rg=None,frame_adjust=None):
    
    def tack_on(to_add,existing):
        try:
            existing = np.concatenate((existing,to_add),axis=0)
        except:
            existing = to_add.copy()
        return existing
    
    def process(to_add,uncorrected,neuropil,roilines):

        # import oasis.functions as ofun

        to_add_copy = to_add.copy()
        to_add = ut.interp_nans(to_add,axis=-1)
        to_add[np.isnan(to_add)] = np.minimum(np.nanmin(to_add),0)
#        to_add[to_add<0] = 0
        baseline = sfi.percentile_filter(to_add[:,::ds],blcutoff,(1,int(blspan/ds)))
        baseline = np.repeat(baseline,ds,axis=1)
        if baseline.shape[1]>to_add.shape[1]:
            baseline = baseline[:,:to_add.shape[1]]
        #to_correct = to_add<0 # commented out 19/2/5
        #to_correct = baseline<0 # changed 19/2/4 # commented out 19/2/5
        #to_add[to_correct] = to_add[to_correct] - baseline[to_correct] # commented out 19/2/5
        #baseline[to_correct] = 0 # commented out 19/2/5
        c = np.zeros_like(to_add)
        s = np.zeros_like(to_add)
        this_dfof = np.zeros_like(to_add)
        for i in range(c.shape[0]):
            #try:
            fudge = 5e-2*np.nanmax(to_add[i])
            if to_add[i].max()>0:
                this_dfof[i] = (to_add[i]-baseline[i,:])/np.maximum(fudge,baseline[i,:])
            else:
                print('roi '+str(i)+' all zeros')
            c[i],s[i],_,_,_  = ofun.deconvolve(this_dfof[i].astype(np.float64),penalty=1)
            #except:
            #    print("couldn't do "+str(i))
        #to_add = precise_trialize(to_add,frm,line,roilines,nbefore=nbefore,nafter=nafter)
        #cc = precise_trialize(c,frm,line,roilines,nbefore=nbefore,nafter=nafter)
        #ss = precise_trialize(s,frm,line,roilines,nbefore=nbefore,nafter=nafter)
        #dd = precise_trialize(this_dfof.astype(np.float64),frm,line,roilines,nbefore=nbefore,nafter=nafter)
        to_add,trialwise_t_offset = ut.precise_trialize_no_interp(to_add,frm,line,roilines,nbefore=nbefore,nafter=nafter,nplanes=len(datafiles))
        raw_traces,_ = ut.precise_trialize_no_interp(uncorrected,frm,line,roilines,nbefore=nbefore,nafter=nafter,nplanes=len(datafiles))
        neuropil,_ = ut.precise_trialize_no_interp(neuropil,frm,line,roilines,nbefore=nbefore,nafter=nafter,nplanes=len(datafiles))
        cc,_ = ut.precise_trialize_no_interp(c,frm,line,roilines,nbefore=nbefore,nafter=nafter,nplanes=len(datafiles))
        ss,_ = ut.precise_trialize_no_interp(s,frm,line,roilines,nbefore=nbefore,nafter=nafter,nplanes=len(datafiles))
        dd,_ = ut.precise_trialize_no_interp(this_dfof.astype(np.float64),frm,line,roilines,nbefore=nbefore,nafter=nafter,nplanes=len(datafiles))
        return to_add,cc,ss,this_dfof,s,dd,trialwise_t_offset,raw_traces,neuropil
        
    trialwise = np.array(())
    ctrialwise = np.array(())
    strialwise = np.array(())
    dtrialwise = np.array(())
    dfof = np.array(())
    straces = np.array(())
    trialwise_t_offset = np.array(())
    proc = {}
    proc['raw_trialwise'] = np.array(())
    proc['neuropil_trialwise'] = np.array(())
    proc['trialwise_t_offset'] = np.array(())
    for datafile in datafiles:
        thisdepth = int(datafile.split('_ot_')[-1].split('.rois')[0])
        info = ut.loadmat(re.sub('_ot_[0-9]*.rois','.mat',datafile),'info')
        frm = info['frame'][()]
        line = info['line'][()]
        event_id = info['event_id'][()]
        ignore_first = 0
        ignore_last = 0
        while event_id[0]==2:
            event_id = event_id[1:]
            frm = frm[1:]
            line = line[1:]
            ignore_first = ignore_first+1
        while event_id[-1]==2:
            event_id = event_id[:-1]
            frm = frm[:-1]
            line = line[:-1]
            ignore_last = ignore_last+1
        if not rg is None:
            thisrg = (rg[0]-ignore_first,rg[1]+ignore_last)
            print(thisrg)
            frm = frm[thisrg[0]:frm.size+thisrg[1]]
            line = line[thisrg[0]:line.size+thisrg[1]]
        else:
            frm = frm[event_id==1]
            line = line[event_id==1]
        if not frame_adjust is None:
            frm = frame_adjust(frm)
            line = frame_adjust(line)
        (to_add,ctr,uncorrected,neuropil) = ut.loadmat(datafile,('corrected','ctr','Data','Neuropil'))
        print(datafile)
        print(to_add.shape)
        nlines = ut.loadmat(datafile,'msk').shape[0]
        roilines = ctr[0] + nlines*thisdepth
        #to_add,c,s,this_dfof,this_straces,dtr = process(to_add,roilines)
        to_add,c,s,this_dfof,this_straces,dtr,tt,uncorrected,neuropil = process(to_add,uncorrected,neuropil,roilines)
        trialwise = tack_on(to_add,trialwise)
        ctrialwise = tack_on(c,ctrialwise)
        strialwise = tack_on(s,strialwise)
        dtrialwise = tack_on(dtr,dtrialwise)
        dfof = tack_on(this_dfof,dfof)
        straces = tack_on(this_straces,straces)
        #trialwise_t_offset = tack_on(tt,trialwise_t_offset)
        proc['raw_trialwise'] = tack_on(uncorrected,proc['raw_trialwise'])
        proc['neuropil_trialwise'] = tack_on(neuropil,proc['neuropil_trialwise'])
        proc['trialwise_t_offset'] = tack_on(tt,proc['trialwise_t_offset'])

    #return trialwise,ctrialwise,strialwise,dfof,straces,dtrialwise
    return trialwise,ctrialwise,strialwise,dfof,straces,dtrialwise,proc # trialwise_t_offset