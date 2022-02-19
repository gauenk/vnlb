

import torch as th
import numpy as np

import inspect
from easydict import EasyDict as edict

from .utils import optional

def default_params(sigma,verbose=False):
    params = edict()
    params.aggreBoost = [True,True]
    params.beta = [1.0,1.0]
    params.bsize = [128,128]
    params.c = [3,3]
    params.coupleChannels = [False,False]
    params.device = ['cpu','cpu']
    params.flatAreas = [False,True]
    params.gamma = [0.95, 0.2]
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
    params.sigmaBasic = [sigma,0]
    params.sizePatch  = [7,7]
    params.sizePatchTime  = [2,2]
    params.sizeSearchTimeBwd = [6,6]
    params.sizeSearchTimeFwd = [6,6]
    params.sizeSearchWindow = [27,27]
    params.step = [0,1]
    params.tau = [0,400.]
    params.testing = [False,False]
    params.use_imread = [False,False]
    params.stype = ["l2","l2"]
    params.var_mode = [0,0]
    params.variThres = [2.7,0.7] # 0.7
    params.verbose = [verbose,verbose]
    return params

def get_params(sigma,verbose=False):
    params = default_params(sigma,verbose)
    # version = "default"
    # version = "exp"
    version = "sss" # smart-search-space
    # version = "sss_v2" # smart-search-space
    version = "iphone"
    print("version: ",version)
    if version == "exp":
        params['nSimilarPatches'][0] = 100
        params['nSimilarPatches'][1] = 60
        params['sizePatch'] = [7,7]
        params['sizePatchTime'] = [2,2]
    elif version == "sss":
        params['nSimilarPatches'][0] = 100
        params['nSimilarPatches'][1] = 60
        params['sizePatch'] = [7,7]
        params['sizePatchTime'] = [2,2]
        params['stype'] = ["l2","l2"]
        params.sizeSearchTimeBwd = [10,10]
        params.sizeSearchTimeFwd = [10,10]
        params.sizeSearchWindow = [15,15]
    elif version == "sss_v2":
        params['nSimilarPatches'][0] = 100
        params['nSimilarPatches'][1] = 60
        params['sizePatch'] = [7,7]
        params['sizePatchTime'] = [1,2] # pt = 1
        params['stype'] = ["l2","l2"]
        params.sizeSearchTimeBwd = [10,10]
        params.sizeSearchTimeFwd = [10,10]
        params.sizeSearchWindow = [15,15]
    elif version == "iphone":
        params['nSimilarPatches'][0] = 100
        params['nSimilarPatches'][1] = 60
        params['sizePatch'] = [7,7]
        params['sizePatchTime'] = [1,2]
        params['stype'] = ["needle","l2"]
        params.sizeSearchTimeBwd = [10,10]
        params.sizeSearchTimeFwd = [10,10]
        params.sizeSearchWindow = [15,15]
    # params['gamma'][1] = 1.00
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
        def pt(self): return self.sizePatchTime
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
    params.bsize = [int(x) for x in optional(params,'bsize',[8,8])]
    params.nfilter = [int(x) for x in optional(params,'nfilter',[-1,-1])]
    params.mod_sel = ["clipped","clipped"]
    params.device = [device,device]

    # -- args --
    args = VnlbArgs(params,step)

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
