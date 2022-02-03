

import torch
import numpy as np
from einops import rearrange
from easydict import EasyDict as edict

def update_flat_patch(patches,args):
    # -- unpack --
    gamma = args.gamma
    sigma2 = args.sigma2
    if args.step == 1:
        patches.flat[...] = 0
        exec_flat_areas(patches.flat,patches.noisy,gamma,sigma2)

def exec_flat_areas(flat_patch,patches,gamma,sigma2):
    """
    Decide if the region's area is "flat"
    """

    # -- shapes --
    bsize,num,ps_t,c,ps,ps = patches.shape
    pflat = rearrange(patches,'b n pt c ph pw -> b c (n pt ph pw)')

    # -- compute var --
    B,C,Z = pflat.shape
    psum = torch.sum(pflat,dim=2)
    psum2 = torch.sum(pflat**2,dim=2)
    var = (psum2 - (psum*psum/Z)) / (Z-1)
    var = torch.mean(var,dim=1)

    # -- compute thresh --
    thresh = gamma*sigma2
    flat_patch[...] = var < thresh
