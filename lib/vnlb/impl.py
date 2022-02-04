
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

# -- old exec import --
import svnlb
from svnlb.gpu import processNLBayes

# -- project imports --
import vnlb.alloc as alloc
from .proc_nn import proc_nn
from .proc_nl import proc_nl
from .params import get_args,get_params
from .utils import Timer

def denoise(noisy, sigma, alpha, vid_name, clipped_noise, gpuid, silent,
            vid_set="set8", deno_model="pacnet", islice=None, clean=None):

    # -- timing --
    clock = Timer()
    clock.tic()

    #
    # -- proc nn --
    #

    # -- denoise with model --
    deno_nn = proc_nn(deno_model,vid_set,vid_name,sigma)
    deno_nn = deno_nn.to(noisy.device)

    # -- optional slice --
    if not(islice is None):
        deno_nn = deno_nn[islice.t,:,islice.h,islice.w]

    #
    # -- proc nl --
    #

    # -- format v basic --
    basic = deno_nn.clone()
    basic = basic*255
    basic = basic.type(th.float)

    # -- setup vnlb inputs --
    c = noisy.shape[1]
    params = get_params(sigma)
    args = get_args(params,c,1,noisy.device)
    flows = alloc.allocate_flows(noisy.shape,noisy.device)
    images = alloc.allocate_images(noisy*255.,basic,clean)

    # -- exec vnlb --
    proc_nl(images,flows,args)
    deno_nl = images['deno']/255.

    # -- [old-gpu] exec vnlb --
    # params = svnlb.swig.setVnlbParams(noisy.shape,sigma)
    # flows = {k:v.cpu().numpy() for k,v in flows.items()}
    # py_results = processNLBayes(noisy,basic,sigma,1,flows,params)
    # deno_nl = py_results['denoised']

    # -- alpha ave --
    deno_final = alpha * deno_nl + (1 - alpha) * deno_nn

    # -- timeit --
    tdelta = clock.toc()

    return deno_final,deno_nl,deno_nn,tdelta
