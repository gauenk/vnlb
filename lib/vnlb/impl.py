
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
from . import alloc as alloc
from .proc_nn import proc_nn
from .proc_nl import proc_nl
from .params import get_args,get_params
from .utils import Timer
from .stn import stn_basic_est,stn_denoise

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
    params = get_params(sigma,verbose,"default")
    flows = alloc.allocate_flows(flows,noisy.shape,noisy.device)
    if not(clean is None):
        params.srch_img = ["clean","clean"]

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

def denoise_npc(noisy, sigma, npc_img, gpuid=0, clean=None, verbose=True,
                deno="bayes", eigh_method="torch"):
    """
    Video Non-Local Bayes (VNLB)
    using our search image

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
    params = get_params(sigma,verbose,"default")
    flows = alloc.allocate_flows(None,noisy.shape,noisy.device)

    # -- use the npc --
    params['deno'] = [deno,deno]
    params['srch_img'][0] = "search"
    params['srch_img'][1] = "search"
    params['cpatches'][0] = "noisy"
    params['eigh_method'] = [eigh_method,eigh_method]

    # -- [step 1] --
    images = alloc.allocate_images(noisy,None,clean,npc_img)
    args = get_args(params,c,0,noisy.device)
    proc_nl(images,flows,args)
    basic = images['deno'].clone()

    # -- [step 2] --
    images = alloc.allocate_images(noisy,basic,clean,npc_img)
    args = get_args(params,c,1,noisy.device)
    proc_nl(images,flows,args)
    deno = images['deno']

    # -- timeit --
    tdelta = clock.toc()

    return deno,basic,tdelta

def denoise_stn(noisy, sigma, flows=None, gpuid=0, clean=None, verbose=True):
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
    params = get_params(sigma,verbose,"default")
    flows = alloc.allocate_flows(flows,noisy.shape,noisy.device)
    # params.srch_img = ["clean","clean"]
    params['cpatches'][0] = "noisy"
    params['srch_img'][0] = "basic"

    # -- crerate basic estimate --
    images = alloc.allocate_images(noisy,None,clean)
    args = get_args(params,c,0,noisy.device)
    basic = stn_denoise(images,args)
    deno = basic

    return deno,basic,0.


def denoise_mod_v2(noisy, sigma, flows=None, gpuid=0, clean=None, verbose=True):
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
    params['cpatches'][0] = "noisy"
    params['srch_img'][0] = "basic"

    # -- crerate basic estimate --
    images = alloc.allocate_images(noisy,None,clean)
    args = get_args(params,c,0,noisy.device)
    basic = stn_basic_est(images,args)

    # -- [step 1] --
    # images = alloc.allocate_images(noisy,None,clean)
    # args = get_args(params,c,0,noisy.device)
    # proc_nl(images,flows,args)
    # basic = images['deno'].clone()
    images = alloc.allocate_images(noisy,basic,clean)
    args = get_args(params,c,1,noisy.device)
    proc_nl(images,flows,args)
    deno = images['deno']


    # -- [step 2] --
    images = alloc.allocate_images(noisy,basic,clean)
    args = get_args(params,c,1,noisy.device)
    proc_nl(images,flows,args)
    deno = images['deno']

    # -- timeit --
    tdelta = clock.toc()

    return deno,basic,tdelta

def denoise_mod_v1(noisy, sigma, flows=None, gpuid=0, clean=None, verbose=True):
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
    params['nSimilarPatches'][0] = 2
    params['cpatches'][0] = "noisy"
    params['srch_img'][0] = "noisy"
    params['deno'][0] = "bayes"
    params['nkeep'][0] = 1
    params['sizePatchTime'] = [2,2]
    params['stype'] = ["l2","l2"]
    params['stype'] = ["l2","l2"]
    params.sizeSearchTimeBwd[0] = 6
    params.sizeSearchTimeFwd[0] = 6
    params.sizeSearchWindow[0] = 10

    images = alloc.allocate_images(noisy,None,clean)
    args = get_args(params,c,0,noisy.device)
    proc_nl(images,flows,args)
    basic = images['deno'].clone()

    # -- [step 2] --
    alpha = 0.5
    basic = alpha * basic + (1 - alpha) * noisy
    nsteps = 1
    for i in range(nsteps):
        params['nSimilarPatches'][0] = 100
        params['cpatches'][0] = "noisy"
        params['srch_img'][0] = "basic"
        params['deno'][0] = "bayes"
        params['nkeep'][0] = 10#min(2+2*(i+1),5)
        params.sizeSearchTimeBwd[0] = 6
        params.sizeSearchTimeFwd[0] = 6
        params.sizeSearchWindow[0] = 10
        # params.sizeSearchTimeBwd[0] = 30
        # params.sizeSearchTimeFwd[0] = 30
        # params.sizeSearchWindow[0] = 10
        images = alloc.allocate_images(noisy,basic,clean)
        args = get_args(params,c,0,noisy.device)
        proc_nl(images,flows,args)
        basic = images['deno'].clone()
        basic = alpha * basic + (1 - alpha) * noisy

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


def denoise_niters(noisy, sigma, niters = 3, flows=None, gpuid=0, clean=None, verbose=True):
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
    params['nSimilarPatches'][0] = 2
    params['cpatches'][0] = "noisy"
    params['srch_img'][0] = "noisy"
    params['deno'][0] = "bayes"
    params['nkeep'][0] = 1
    params['sizePatchTime'] = [2,2]
    params['stype'] = ["l2","l2"]
    params['stype'] = ["l2","l2"]
    params.sizeSearchTimeBwd[0] = 6
    params.sizeSearchTimeFwd[0] = 6
    params.sizeSearchWindow[0] = 10

    images = alloc.allocate_images(noisy,None,clean)
    args = get_args(params,c,0,noisy.device)
    proc_nl(images,flows,args)
    basic = images['deno'].clone()

    # -- [step 2] --
    alpha = 0.5
    basic = alpha * basic + (1 - alpha) * noisy
    for i in range(nsteps):
        params['nSimilarPatches'][0] = 100
        params['cpatches'][0] = "noisy"
        params['srch_img'][0] = "basic"
        params['deno'][0] = "bayes"
        params['nkeep'][0] = 1#min(2+2*(i+1),5)
        params.sizeSearchTimeBwd[0] = 6
        params.sizeSearchTimeFwd[0] = 6
        params.sizeSearchWindow[0] = 10
        # params.sizeSearchTimeBwd[0] = 30
        # params.sizeSearchTimeFwd[0] = 30
        # params.sizeSearchWindow[0] = 10
        images = alloc.allocate_images(noisy,basic,clean)
        args = get_args(params,c,0,noisy.device)
        proc_nl(images,flows,args)
        basic = images['deno'].clone()
        basic = alpha * basic + (1 - alpha) * noisy

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

