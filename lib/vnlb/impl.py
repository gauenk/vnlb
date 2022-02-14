
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


def denoise(noisy, sigma, gpuid=0, clean=None, verbose=True):
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
    params = get_params(sigma,verbose)
    flows = alloc.allocate_flows(noisy.shape,noisy.device)

    # -- [step 1] --
    images = alloc.allocate_images(noisy,None,clean)
    args = get_args(params,c,0,noisy.device)
    proc_nl(images,flows,args)
    basic = images['deno'].clone()

    # -- [step 2] --
    images = alloc.allocate_images(noisy,basic,clean)
    args = get_args(params,c,1,noisy.device)
    proc_nl(images,flows,args)
    deno = images['deno']

    # -- timeit --
    tdelta = clock.toc()

    return deno,basic,tdelta
