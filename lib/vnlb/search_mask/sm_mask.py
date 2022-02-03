
# -- python deps --
import torch
import numpy as np
from einops import rearrange
from easydict import EasyDict as edict

# -- numba --
from numba import jit,njit,prange,cuda

def sm_mask(basic,mask,thresh):

    # -- estimate smooth basic --
    mask = mask.clone()
    smooth = comp_smooth_pix(basic,mask,thresh)

    # -- parse inputs --
    mask = th.logical_and(mask,smooth)

    return mask

def comp_smooth_pix(basic,mask,thresh):

    # -- init tensor --
    t,c,h,w = basic.shape
    smooth = np.zeros((t,h,w),dtype=np.int8)

    # -- exec over batch --
    for batch in range(nbatches):

        # -- get inds to compute --
        access = get_inds(mask)
        if access.shape[0] == 0:
            break

        # -- get similar patches --
        patches = get_sim(basic,access)

        # -- compute if flat region --
        edges = apply_sobel_to_patches(patches,pshape)

        # -- fill --
        view(smooth)[...] = 1-edges

    return smooth
