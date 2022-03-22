
"""
Allocate memory once for many subroutines
"""

import torch as th
from easydict import EasyDict as edict


def allocate_patches(shape,clean,device):

    # -- unpack shapes --
    tsize,npa,ps_t,c,ps,ps = shape
    tf32 = th.float32
    shape = (tsize,npa,ps_t,c,ps,ps)

    # -- alloc mem --
    patches = edict()
    patches.noisy = th.zeros(shape,dtype=tf32,device=device)
    patches.basic = th.zeros(shape,dtype=tf32,device=device)
    patches.clean = None
    if not(clean is None):
        patches.clean = th.zeros(shape,dtype=tf32,device=device)
    patches.flat = th.zeros((tsize),dtype=th.bool,device=device)
    patches.shape = shape

    # -- names --
    patches.images = ["noisy","basic","clean"]
    patches.tensors = ["noisy","basic","clean","flat"]

    return patches

def allocate_patches_fast(shape,clean,device):

    # -- unpack shapes --
    tsize,npa,ps_t,c,ps,ps = shape
    tf32 = th.float32

    # -- alloc mem --
    patches = edict()
    patches.noisy = th.zeros((tsize,npa,ps_t,c,ps,ps),dtype=tf32,device=device)
    patches.basic = None
    patches.clean = None
    patches.flat = th.zeros((tsize),dtype=th.bool,device=device)
    patches.shape = shape

    # -- names --
    patches.images = ["noisy"]
    patches.tensors = ["noisy","flat"]

    return patches


def allocate_images(noisy,basic,clean,search=None):

    # -- create images --
    imgs = edict()
    imgs.noisy = noisy
    imgs.shape = noisy.shape
    imgs.device = noisy.device

    # -- unpack params --
    dtype = noisy.dtype
    device = noisy.device
    t,c,h,w = noisy.shape

    # -- basic --
    imgs.basic = basic
    if basic is None:
        imgs.basic = th.zeros((t,c,h,w),dtype=dtype,device=device)

    # -- clean --
    imgs.clean = clean
    if not(clean is None) and not(th.is_tensor(clean)):
        imgs.clean = th.from_numpy(imgs.clean).to(device)

    # -- deno & agg weights --
    imgs.deno = th.zeros((t,c,h,w),dtype=dtype).to(device)
    imgs.weights = th.zeros((t,h,w),dtype=dtype).to(device)
    imgs.vals = th.zeros((t,h,w),dtype=dtype).to(device)
    imgs.search = search

    # -- names --
    imgs.patch_images = ["noisy","basic","clean"]
    imgs.ikeys = ["noisy","basic","clean","deno","search"]

    return imgs

def allocate_flows(flows,shape,device):
    t,c,h,w = shape
    if flows is None:
        flows = edict()
        zflow = th.zeros((t,2,h,w)).to(device)
        flows.fflow = zflow
        flows.bflow = zflow.clone()
    else:
        flows = edict({k:v.to(device) for k,v in flows.items()})
    return flows

def allocate_bufs(shape,device):

    # -- unpack shapes --
    tsize,npa = shape
    tf32 = th.float32
    tfl = th.long

    # -- alloc mem --
    l2bufs = edict()
    l2bufs.vals = th.zeros((tsize,npa)).type(tf32).to(device)
    l2bufs.inds = -th.ones((tsize,npa)).type(tfl).to(device)
    l2bufs.shape = shape
    l2bufs.fast = False

    return l2bufs


def allocate_bufs_fast(args,t,shape,device):

    # -- unpack shapes --
    tsize,npa = shape
    tf32 = th.float32
    tfl = th.long
    ti32 = th.int32

    # -- alloc mem --
    l2bufs = edict()

    # -- standard allocs --
    l2bufs.vals = th.zeros((tsize,npa),dtype=tf32,device=device)
    l2bufs.inds = -th.ones((tsize,npa),dtype=tfl,device=device)

    # -- create shapes --
    ps = args.ps
    w_s = args.w_s
    nWt_f = args.nWt_f
    nWt_b = args.nWt_b
    ps_t = args.ps_t
    w_t = min(nWt_f + nWt_b + 1,t-ps_t+1)

    # -- fast allocs --
    shape = (tsize,w_t,w_s,w_s)
    l2bufs.srch_dists = float("inf") * th.ones(shape,dtype=tf32,device=device)
    l2bufs.srch_locs = -th.ones(shape,dtype=tfl,device=device)
    shape = (tsize,3,w_t,w_s,w_s)
    l2bufs.srch_bufs = -th.ones(shape,dtype=ti32,device=device)

    # -- misc --
    l2bufs.shape = shape
    l2bufs.fast = True

    return l2bufs



