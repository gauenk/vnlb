
"""
Allocate memory once for many subroutines
"""

import torch as th
from easydict import EasyDict as edict


def allocate_patches(shape,clean,device):

    # -- unpack shapes --
    tsize,npa,ps_t,c,ps,ps = shape
    tf32 = th.float32

    # -- alloc mem --
    patches = edict()
    patches.noisy = th.zeros((tsize,npa,ps_t,c,ps,ps)).type(tf32).to(device)
    patches.basic = th.zeros((tsize,npa,ps_t,c,ps,ps)).type(tf32).to(device)
    patches.clean = None
    if not(clean is None):
        patches.clean = th.zeros((tsize,npa,ps_t,c,ps,ps)).type(tf32).to(device)
    patches.flat = th.zeros((tsize)).type(th.bool).to(device)
    patches.shape = shape

    # -- names --
    patches.images = ["noisy","basic","clean"]

    return patches

def allocate_images(noisy,basic,clean):

    # -- create images --
    imgs = edict()
    imgs.noisy = noisy
    imgs.shape = noisy.shape
    imgs.device = noisy.device

    # -- unpack params --
    device = noisy.device
    t,c,h,w = noisy.shape

    # -- basic --
    imgs.basic = basic
    if basic is None:
        imgs.basic = th.zeros((t,c,h,w)).to(device)

    # -- clean --
    imgs.clean = clean

    # -- deno & agg weights --
    imgs.deno = th.zeros((t,c,h,w)).to(device)
    imgs.weights = th.zeros((t,h,w)).to(device)

    # -- names --
    imgs.patch_images = ["noisy","basic","clean"]

    return imgs

def allocate_flows(shape,device):
    t,c,h,w = shape
    flows = edict()
    zflow = th.zeros((t,2,h,w)).to(device)
    flows.fflow = zflow
    flows.bflow = zflow.clone()
    return flows

def allocate_bufs(shape,device):

    # -- unpack shapes --
    tsize,npa = shape
    tf32 = th.float32
    tfl = th.long

    # -- alloc mem --
    l2bufs = edict()
    l2bufs.vals = th.zeros((tsize,npa)).type(tf32).to(device)
    l2bufs.inds = th.zeros((tsize,npa)).type(tfl).to(device)
    l2bufs.shape = shape

    return l2bufs


