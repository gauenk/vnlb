
import torch as th
from easydict import EasyDict as edict

import vnlb.alloc as alloc
from vnlb.params import get_args,get_params
from vnlb.search import exec_search
import vnlb.search_mask as search_mask
import vnlb.utils as utils

def exec_search_testing(noisy,sigma,pidx,params,step,clean=None):

    # -- to torch --
    device = 'cuda:0'
    if not th.is_tensor(noisy):
        noisy = th.from_numpy(noisy)
        noisy = noisy.to(device)
    noisy = noisy.clone()

    # -- params --
    c = noisy.shape[1]
    params2 = get_params(sigma)
    args2 = get_args(params2,c,step,noisy.device)
    args = params
    args.patch_shape = args2.patch_shape
    args.bufs_shape = args2.bufs_shape
    args.device = args2.device

    # -- alloc tensors --
    flows = alloc.allocate_flows(noisy.shape,noisy.device)
    images = alloc.allocate_images(noisy,noisy,clean)
    patches = alloc.allocate_patches(args.patch_shape,images.clean,args.device)
    bufs = alloc.allocate_bufs(args.bufs_shape,args.device)

    # -- color xform --
    utils.rgb2yuv_images(images)

    # -- create access mask --
    mask,ngroups = search_mask.init_mask(images.shape,args2)
    mask[...] = 0

    t,c,h,w = noisy.shape
    tchw = t*c*h*w
    chw = t*c*h*w
    hw = h*w
    ti = pidx // chw
    hi = (pidx % hw) // w
    wi = pidx % w
    mask[ti,hi,wi] = 1

    # -- exech search --
    exec_search(patches,images,flows,mask,bufs,args2)
    noisy = patches.noisy

    # -- returns --
    results = edict()
    results.patches = patches.noisy
    results.indices = bufs.inds
    results.values = bufs.vals

    # -- compute top k --
    vals,inds = bufs.vals,bufs.inds
    args = th.where(inds == 1411)
    vals_b = vals[args]

    # -- convert inds (with color -> no color) --
    # inds = results.indices
    # aug_inds = search_mask.expand_inds(inds[[0]],t,c,h,w)
    # inds = aug_inds[:,0] * hw + aug_inds[:,1] * w + aug_inds[:,2]
    # results.indices = inds[None,:]

    return results
