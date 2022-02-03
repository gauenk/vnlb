
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


def denoise(noisy, sigma, vid_name, clipped_noise, gpuid, silent,
            vid_set="set8", deno_model="pacnet", islice=None, clean=None):

    # -- timing --
    clock = Timer()
    clock.tic()

    #
    # -- proc nn --
    #

    # model = load_nn_model(model,sigma,gpuid)

    # -- denoise with model --
    basic = proc_nn(noisy,sigma,vid_name,vid_set,deno_model)
    basic = basic.to(noisy.device)

    # -- optional slice --
    if not(islice is None):
        basic = basic[islice.t,:,islice.h,islice.w]

    #
    # -- proc nl --
    #

    # -- format v basic --
    vbasic = basic.clone()
    vbasic = vbasic*255
    # vbasic = vbasic.clamp(0,255).type(th.uint8)
    vbasic = vbasic.type(th.float)

    # -- setup vnlb inputs --
    c = noisy.shape[1]
    params = get_params(noisy.shape,sigma)
    args = get_args(params,c,1,noisy.device)
    flows = alloc.allocate_flows(noisy.shape,noisy.device)
    images = alloc.allocate_images(noisy*255.,vbasic,clean)

    # -- exec vnlb --
    print(args.nstreams,args.bsize)
    proc_nl(images,flows,args)
    deno = images['deno']/255.

    # -- alpha ave --
    alpha = 0.25
    deno = alpha * deno + (1 - alpha) * basic

    # -- timeit --
    tdelta = clock.toc()

    return deno,basic,tdelta
