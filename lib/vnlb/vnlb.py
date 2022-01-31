
# -- python deps --
import copy
import numpy as np
import torch
import torch as th
from PIL import Image
from einops import rearrange
from easydict import EasyDict as edict

# -- warnings --
import warnings
from numba import NumbaPerformanceWarning
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)

# -- local imports --
# from .sim_search import runSimSearch
# from .bayes_est import runBayesEstimate
# from .comp_agg import computeAggregation
# from .proc_nlb import processNLBayes

# -- project imports --
import svnlb # for params only
# from svnlb.gpu import processNLBayes as proc_nlb
from .proc_nn import proc_nn
from .proc_nlb import processNLBayes as proc_nlb
from .utils import Timer
# from vnlb.utils import groups2patches,patches2groups,compute_psnrs
# from vnlb.testing.file_io import save_images,save_image

def get_params(shape,sigma):
    params = svnlb.swig.setVnlbParams(shape,sigma)
    params['nSimilarPatches'][0] = 100
    params['nSimilarPatches'][1] = 60
    params['gamma'][1] = 1.00
    # params['useWeights'] = [False,False]
    # params['simPatchRefineKeep'] = [100,100]
    # params['cleanSearch'] = [True,True]
    # params['cleanSearch'] = [False,False]
    # params['variThres'] = [0.,0.]
    # params['useWeights'] = [False,False]
    # params['nfilter'] = [-1,-1]
    return params

def get_flows(shape,device):
    t,c,h,w = shape
    fflow = th.zeros(t,2,h,w).to(device)
    bflow = th.zeros(t,2,h,w).to(device)
    flows = edict({"fflow":fflow,"bflow":bflow})
    return flows

def denoise(noisy, sigma, clipped_noise, gpuid, silent, model=None, islice=None):

    # -- timing --
    clock = Timer()
    clock.tic()

    # -- load nn --
    # model = load_nn_model(model,sigma,gpuid)

    # -- denoise with model --
    basic = proc_nn(noisy,sigma,model)
    basic = basic.to(noisy.device)

    # -- optional slice --
    if not(islice is None):
        basic = basic[islice.t,:,islice.h,islice.w]

    # -- format v basic --
    vbasic = basic.clone()
    vbasic = vbasic*255
    # vbasic = vbasic.clamp(0,255).type(th.uint8)
    vbasic = vbasic.type(th.float)

    # -- denoise with vnlb --
    params = get_params(noisy.shape,sigma)
    flows = get_flows(noisy.shape,noisy.device)
    vnlb_results = proc_nlb(noisy*255.,vbasic,sigma,1,flows,params,gpuid=gpuid)
    deno = vnlb_results['denoised']/255.
    # processNLBayes(noisy,basic,sigma,step,flows,params,gpuid=0,clean=None)

    # -- alpha ave --
    alpha = 0.25
    deno = alpha * deno + (1 - alpha) * basic

    # -- timeit --
    tdelta = clock.toc()

    return deno,basic,tdelta
