
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

# -- project imports --
import vnlb.alloc as alloc
from .proc_nn import proc_nn
from .proc_nl import proc_nl
from .params import get_args,get_params
from .utils import Timer


def denoise(noisy, sigma, gpuid=0, clean=None):
    """
    Video Non-Local Bayes (VNLB)

    """

    # -- timing --
    clock = Timer()
    clock.tic()

    # -- get device --
    use_gpu = torch.cuda.is_available() and gpuid >= 0
    device = 'cuda:%d' % gpuid if use_gpu else 'cpu'

    # -- to tensor --
    if not th.is_tensor(noisy):
        noisy = th.from_numpy(noisy).to(device)

    # -- setup vnlb inputs --
    c = noisy.shape[1]
    params = get_params(sigma)
    flows = alloc.allocate_flows(noisy.shape,noisy.device)

    # -- [step 1] --
    images = alloc.allocate_images(noisy,None,clean)
    args = get_args(params,c,0,noisy.device)
    proc_nl(images,flows,args)
    basic = images['deno'].clone()
    print("basic.max(): ",basic.max())

    # -- [step 2] --
    images = alloc.allocate_images(noisy,basic,clean)
    args = get_args(params,c,1,noisy.device)
    proc_nl(images,flows,args)
    deno = images['deno']

    # -- timeit --
    tdelta = clock.toc()

    return deno,basic,tdelta


def deno_nnnl(noisy, sigma, alpha, vid_name, clipped_noise, gpuid, silent,
              vid_set="set8", deno_model="pacnet", islice=None, clean=None):
    """
    Method submitted to ECCV 2022

    """

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

    # -- alpha ave --
    deno_final = alpha * deno_nl + (1 - alpha) * deno_nn

    # -- timeit --
    tdelta = clock.toc()

    return deno_final,deno_nl,deno_nn,tdelta
