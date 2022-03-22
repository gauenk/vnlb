
# -- python deps --
from tqdm import tqdm
import copy,math
import torch
import torch as th
import numpy as np
from einops import rearrange,repeat
from easydict import EasyDict as edict

import torchvision.utils as tvu

# -- package --
import vnlb
from vnlb.testing.file_io import save_images

# -- imports --
import vnlb.agg as agg
import vnlb.utils as utils
import vnlb.alloc as alloc
import vnlb.search_mask as search_mask
import vnlb.search as search
import vnlb.deno as deno
import vnlb.utils as utils
from vnlb.utils import update_flat_patch
from vnlb.utils import idx2coords,coords2idx,patches2groups,groups2patches

# -- project imports --
from vnlb.utils.gpu_utils import apply_color_xform_cpp,yuv2rgb_cpp
from vnlb.utils import groups2patches,patches2groups,optional,divUp,save_burst
from vnlb.utils.video_io import read_nl_sequence
from vnlb.testing import save_images
from vnlb.utils.streams import init_streams,wait_streams
from vnlb.params import get_args,get_params

import pprint
pp = pprint.PrettyPrinter(indent=4)

def global_search_default(noisy,sigma,clock,ps=7,nsim=100,pt=1,pfill=True,bstride=1):

    # -- init --
    verbose = False
    c = int(noisy.shape[-3])
    images = alloc.allocate_images(noisy,None,None)
    flows = alloc.allocate_flows(None,noisy.shape,noisy.device)
    params = get_params(sigma,verbose,"default")
    params['bsize'][0] = 4096*5
    params['sizePatch'][0] = ps
    params['nSimilarPatches'][0] = nsim
    params['sizePatchTime'][0] = pt
    params['nstreams'][0] = 1
    args = get_args(params,c,0,noisy.device)
    args.aggreBoost = False
    args.pfill = pfill
    args.bstride = bstride

    # -- exec search [to be timed] --
    if not(clock is None): clock.tic()
    bufs = global_search(images,flows,args)
    th.cuda.synchronize()
    if not(clock is None): clock.toc()
    return bufs

def init_rtn_bufs(nelems,args):

    # -- get shapes --
    numQueries = (nelems - 1)//(args.bstride) + 1
    K = args.nSimilarPatches
    pshape = list(args.patch_shape)
    pshape[0] = numQueries

    # -- types --
    device = args.device
    tf32 = th.float32
    ti32 = th.int32

    # -- return vals --
    rtn_bufs = edict()
    rtn_bufs.dists = float("inf") * th.ones((numQueries,K),dtype=tf32,device=device)
    rtn_bufs.inds = -th.ones((numQueries,K),dtype=ti32,device=device)
    rtn_bufs.patches = -th.ones(pshape,dtype=tf32,device=device)

    return rtn_bufs

def global_search(images,flows,args):

    """
    Use for testing against the "global tiling" method
    """

    # -- batching params --
    h,w = images.shape[-2:]
    nelems = t*h*w
    nbatches = (nelems - 1) // (args.bstride*args.bsize*args.nstreams) + 1
    # print("nbatches: ",nbatches)

    # -- allocate memory --
    patches = alloc.allocate_patches_fast(args.patch_shape,images.clean,args.device)

    t = images.shape[0]
    # bufs = alloc.allocate_bufs_fast(args,t,args.bufs_shape,args.device)
    bufs = alloc.allocate_bufs(args.bufs_shape,args.device)

    # -- create return shells --
    # rtn_bufs = init_rtn_bufs(nelems,args)

    # -- color xform --
    # utils.rgb2yuv_images(images)

    # -- logging --
    # if args.verbose: print(f"Processing VNLB [step {args.step}]")

    # -- over batches --
    if args.verbose: pbar = tqdm(total=nelems)
    for batch in range(nbatches):

        # -- exec search --
        done = search.exec_search_fast(patches,images,flows,None,bufs,args)

        # -- copy results --
        # tsize = bufs.vals.shape[0]
        # bslice = slice(batch*tsize,(batch+1)*tsize)
        # print(batch*tsize,(batch+1)*tsize,tsize)
        # print(bufs.vals.shape)
        # rtn_bufs.dists[bslice,...] = bufs.vals[...]
        # rtn_bufs.inds[bslice,...] = bufs.inds[...]
        # rtn_bufs.patches[bslice,...] = patches.noisy[...]


    return rtn_bufs
