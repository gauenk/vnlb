

import svnlb
import torch as th
import numpy as np

import inspect
from easydict import EasyDict as edict

from .utils import optional

def not_here():

    # -- place on cuda --
    device = gpuid
    if not(th.is_tensor(noisy)):
        noisy = th.FloatTensor(noisy).to(device)
        zero_basic = th.zeros_like(noisy)
        basic = zero_basic if basic is None else basic
        basic = basic.to(device)
    if not(clean is None):
        clean = th.FloatTensor(clean).to(device)

    # -- init outputs --
    shape = noisy.shape
    t,c,h,w = noisy.shape
    deno = th.zeros_like(noisy)
    # flows = edict({k:th.FloatTensor(v).to(device) for k,v in flows.items()})

    # -- to device flow --
    # flows = edict({k:th.FloatTensor(v).to(device) for k,v in flows.items()})
    zflow = torch.zeros((t,2,h,w)).to(device)
    fflow = optional(flows,'fflow',zflow)
    bflow = optional(flows,'bflow',zflow)

def default_params(sigma):
    params = edict()
    params.aggreBoost = [True,True]
    params.beta = [1.0,1.0]
    params.bsize = [128,128]
    params.c = [3,3]
    params.coupleChannels = [False,False]
    params.device = ['cpu','cpu']
    params.flatAreas = [False,True]
    params.gamma = [0.95, 0.95]
    params.isFirstStep = [True,False]
    params.mod_sel = ["clipped","clipped"]
    params.nParts = [-1,-1]
    params.nThreads = [-1,-1]
    params.nSimilarPatches = [100,60]
    params.nkeep = [-1,-1]
    params.nstreams = [8,18]
    params.offset = [2*(sigma/255.)**2,0.]
    params.onlyFrame = [-1,-1]
    params.procStep = [3,3]
    params.rank = [39,39]
    params.sigma = [sigma,sigma]
    params.sigmaBasic = [0,0]
    params.sizePatch  = [7,7]
    params.sizePatchTime  = [2,2]
    params.sizeSearchTimeBwd = [6,6]
    params.sizeSearchTimeFwd = [6,6]
    params.sizeSearchWindow = [27,27]
    params.step = [0,1]
    params.tau = [0,400.]
    params.testing = [False,False]
    params.use_imread = [False,False]
    params.var_mode = [0,0]
    params.variThres = [0.7,0.7]
    params.verbose = [False,False]
    return params

def get_params(sigma):
    params = default_params(sigma)
    params['nSimilarPatches'][0] = 100
    params['nSimilarPatches'][1] = 60
    # params['gamma'][1] = 1.00
    params['sizePatch'] = [7,7]
    # params['useWeights'] = [False,False]
    # params['simPatchRefineKeep'] = [100,100]
    # params['cleanSearch'] = [True,True]
    # params['cleanSearch'] = [False,False]
    # params['variThres'] = [0.,0.]
    # params['useWeights'] = [False,False]
    # params['nfilter'] = [-1,-1]
    return params

def get_args(params,c,step,device):
    """

    A Python implementation for one step of the NLBayes code

    """

    class VnlbArgs(dict):
        """
        An indexed EasyDict
        """

        def __init__(self, params, step):

            # -- list properties --
            def isprop(v):
                return isinstance(v, property)
            propnames = [name for (name, value) in inspect.getmembers(VnlbArgs, isprop)]

            # -- set params --
            for k, v in params.items():
                if not(k in propnames):
                    setattr(self, k, v[step])

            # -- Class attributes --
            for k in self.__class__.__dict__.keys():
                set_bool = not (k.startswith('__') and k.endswith('__'))
                set_bool = set_bool and (not k in ('update', 'pop'))
                set_bool = set_bool and (not(k in propnames))
                if set_bool: setattr(self, k, getattr(self, k))

        def __setattr__(self, name, value):
            if isinstance(value, (list, tuple)):
                value = [self.__class__(x)
                         if isinstance(x, dict) else x for x in value]
            elif isinstance(value, dict) and not isinstance(value, self.__class__):
                value = self.__class__(value)
            super(VnlbArgs, self).__setattr__(name, value)
            super(VnlbArgs, self).__setitem__(name, value)

        __setitem__ = __setattr__

        def update(self, e=None, **f):
            d = e or dict()
            d.update(f)
            for k in d:
                setattr(self, k, d[k])

        def pop(self, k, d=None):
            delattr(self, k)
            return super(VnlbArgs, self).pop(k, d)

        """
        Shortcuts
        """

        @property
        def ps(self): return self.sizePatch
        @property
        def ps_t(self): return self.sizePatchTime
        @property
        def npatches(self): return self.nSimilarPatches
        @property
        def w_s(self): return self.sizeSearchWindow
        @property
        def nWt_f(self): return self.sizeSearchTimeFwd
        @property
        def nWt_b(self): return self.sizeSearchTimeBwd
        @property
        def couple_ch(self): return self.coupleChannels
        @property
        def group_chnls(self):
            return 1 if self.couple_ch else c
        @property
        def group_chnls(self):
            return 1 if self.couple_ch else c
        @property
        def sigma2(self): return self.sigma**2
        @property
        def sigmaBasic2(self): return self.sigmaBasic**2
        @property
        def sigmab2(self): return self.sigmaBasic**2
        @property
        def thresh(self): return self.variThres
        @property
        def flat_areas(self): return self.flatAreas
        @property
        def step_s(self): return self.procStep
        @property
        def patch_shape(self):
            return get_patch_shape(self)
        @property
        def bufs_shape(self):
            return get_bufs_shape(self)

    # -- optional --
    offsets = [2*(params.sigma[0]/255.)**2,0.]
    params.step = [0,1]
    params.c = [c,c]
    params.nstreams = [int(x) for x in optional(params,'nstreams',[1,18])]
    params.nkeep = [int(x) for x in optional(params,'simPatchRefineKeep',[-1,-1])]
    params.offset = [float(x) for x in optional(params,'offset',offsets)]
    params.bsize = [int(x) for x in optional(params,'bsize_s',[128,128])]
    params.nfilter = [int(x) for x in optional(params,'nfilter',[-1,-1])]
    params.mod_sel = ["clipped","clipped"]
    params.device = [device,device]

    # -- args --
    args = VnlbArgs(params,step)

    return args

def get_args_old(params,c,step):

    # -- unpack --
    args = edict() # used for call
    args.c = c
    args.step = step
    args.ps = params['sizePatch'][step]
    # args.sizePatch = args.ps
    args.ps_t = params['sizePatchTime'][step]
    # args.sizePatchTime = ps_t
    args.npatches = params['nSimilarPatches'][step]
    # args.nSimilarPatches = npatches
    args.w_s = params['sizeSearchWindow'][step]

    args.nWt_f = params['sizeSearchTimeFwd'][step]
    args.nWt_b = params['sizeSearchTimeBwd'][step]
    args.couple_ch = params['coupleChannels'][step]
    args.group_chnls = 1 if args.couple_ch else c
    args.step1 = step == 0
    check_steps(args.step1,step)
    args.sigma = params['sigma'][step]
    args.sigma2 = params['sigma'][step]**2
    args.beta = params['beta'][step]
    args.sigmaBasic2 = params['sigmaBasic'][step]**2
    args.sigmab2 = args.beta * args.sigmaBasic2 if step==1 else args.sigma**2
    args.rank =  params['rank'][step]
    args.thresh =  params['variThres'][step]
    args.flat_areas = params['flatAreas'][step]
    args.gamma = params['gamma'][step]
    args.procStep = 1
    args.step_s = 1#params['procStep'][step]

    # -- optional --
    args.nstreams = int(optional(params,'nstreams',[1,12])[step])
    args.nkeep = int(optional(params,'simPatchRefineKeep',[-1,-1])[step])
    args.offset = float(optional(params,'offset',[2*(args.sigma/255.)**2,0.])[step])
    args.bsize = int(optional(params,'bsize_s',[128,128])[step])
    args.nfilter = int(optional(params,'nfilter',[-1,-1])[step])

    # -- ints to bool --
    use_weights = int(optional(params,'useWeights',[False,False])[step])
    clean_srch = int(optional(params,'cleanSearch',[False,False])[step])
    args.use_weights = True if use_weights == 1 else False
    args.clean_srch = True if clean_srch == 1 else False

    # -- final derived values  --
    args.patch_shape = get_patch_shape(args)
    args.bufs_shape = get_bufs_shape(args)

    return args


def check_steps(step1,step):
    is_step_1 = (step1 == True) and (step == 0)
    is_not_step_1 = (step1 == False) and (step == 1)
    assert is_step_1 or is_not_step_1

def get_patch_shape(args):
    tsize = args.nstreams * args.bsize
    npa = args.npatches
    ps_t,c,ps = args.ps_t,args.c,args.ps
    shape = (tsize,npa,ps_t,c,ps,ps)
    return shape

def get_bufs_shape(args):
    tsize = args.nstreams*args.bsize
    npa = args.npatches
    shape = (tsize,npa)
    return shape
