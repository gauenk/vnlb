
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


def denoise(noisy, sigma, flows=None, gpuid=0, clean=None, verbose=True):
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
    flows = alloc.allocate_flows(flows,noisy.shape,noisy.device)
    # params.srch_img = ["clean","clean"]

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

def denoise_mod(noisy, sigma, flows=None, gpuid=0, clean=None, verbose=True):
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
    flows = alloc.allocate_flows(flows,noisy.shape,noisy.device)

    # -- [step 1] --
    params['bsize'][0] = 512
    params['nSimilarPatches'][0] = 10
    params['cpatches'][0] = "noisy"
    params['srch_img'][0] = "noisy"
    params['deno'][0] = "ave"
    params['nkeep'][0] = 1
    params['sizePatchTime'] = [1,2]
    params['stype'] = ["needle","l2"]
    params['stype'] = ["l2","l2"]
    params.sizeSearchTimeBwd[0] = 6
    params.sizeSearchTimeFwd[0] = 6
    params.sizeSearchWindow[0] = 27

    images = alloc.allocate_images(noisy,None,clean)
    args = get_args(params,c,0,noisy.device)
    print("HI")
    proc_nl(images,flows,args)
    basic = images['deno'].clone()

    # -- [step 2] --
    alpha = 1.
    nsteps = 1
    for i in range(nsteps):
        basic = alpha * basic + (1 - alpha) * noisy
        params['nSimilarPatches'][0] = 10
        params['cpatches'][0] = "noisy"
        params['srch_img'][0] = "basic"
        params['deno'][0] = "ave"
        params['nkeep'][0] = min(2+2*(i+1),5)
        params.sizeSearchTimeBwd[0] = 6
        params.sizeSearchTimeFwd[0] = 6
        params.sizeSearchWindow[0] = 27
        # params.sizeSearchTimeBwd[0] = 30
        # params.sizeSearchTimeFwd[0] = 30
        # params.sizeSearchWindow[0] = 10
        images = alloc.allocate_images(noisy,basic,clean)
        args = get_args(params,c,0,noisy.device)
        proc_nl(images,flows,args)
        basic = images['deno'].clone()

    params.sizeSearchTimeBwd[0] = 6
    params.sizeSearchTimeFwd[0] = 6
    params.sizeSearchWindow[0] = 27
    params['nSimilarPatches'][0] = 100
    params['cpatches'][0] = "noisy"
    params['srch_img'][0] = "basic"
    params['deno'][0] = "bayes"
    params['nkeep'][0] = -1
    params['stype'] = ["l2","l2"]
    images = alloc.allocate_images(noisy,basic,clean)
    args = get_args(params,c,0,noisy.device)
    proc_nl(images,flows,args)
    basic = images['deno'].clone()

    # -- [step 3] --
    # params['stype'] = ["needle","l2"]
    params.sizeSearchTimeBwd[1] = 6
    params.sizeSearchTimeFwd[1] = 6
    params.sizeSearchWindow[1] = 27
    params['nSimilarPatches'][1] = 60
    params['gamma'][1] = 0.2
    params['cpatches'][1] = "basic"
    images = alloc.allocate_images(noisy,basic,clean)
    args = get_args(params,c,1,noisy.device)
    proc_nl(images,flows,args)
    deno = images['deno']

    # -- timeit --
    tdelta = clock.toc()

    return deno,basic,tdelta

