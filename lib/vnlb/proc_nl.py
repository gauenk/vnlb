
# -- python deps --
from tqdm import tqdm
import copy,math
import torch
import torch as th
import numpy as np
from einops import rearrange,repeat
from easydict import EasyDict as edict

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
import pprint
pp = pprint.PrettyPrinter(indent=4)


def proc_nl(images,flows,args):

    # -- init --
    # pp.pprint(args)

    # -- create access mask --
    mask,ngroups = search_mask.init_mask(images.shape,args)
    mask_r = repeat(mask,'t h w -> t c h w',c=3)
    save_burst(mask_r,"output/mask/","mask")

    # -- allocate memory --
    patches = alloc.allocate_patches(args.patch_shape,images.clean,args.device)
    bufs = alloc.allocate_bufs(args.bufs_shape,args.device)

    # -- batching params --
    nelems,nbatches = utils.batch_params(mask,args.bsize,args.nstreams)
    cmasked_prev = nelems

    # -- color xform --
    utils.rgb2yuv_images(images)

    # -- logging --
    if args.verbose: print(f"Processing VNLB [step {args.step}]")

    # -- over batches --
    if args.verbose: pbar = tqdm(total=nelems)
    for batch in range(nbatches):

        # -- exec search --
        done = search.exec_search(patches,images,flows,mask,bufs,args)

        # -- refinemenent the searching --
        search.exec_refinement(patches,bufs,args.sigma)

        # -- flat patches --
        update_flat_patch(patches,args)

        # -- valid patches --
        vpatches = get_valid_patches(patches,bufs)

        # -- denoise patches --
        deno.denoise(vpatches,args)

        # -- fill valid --
        fill_valid_patches(vpatches,patches,bufs)

        # -- aggregate patches --
        agg.agg_patches(patches,images,bufs,args)

        # -- misc --
        torch.cuda.empty_cache()

        # -- loop update --
        cmasked = mask.sum().item()
        delta = cmasked_prev - cmasked
        cmasked_prev = cmasked
        nmasked  = nelems - cmasked
        msg = "[Pixels %d/%d]: %d" % (nmasked,nelems,delta)
        if args.verbose:
            tqdm.write(msg)
            pbar.update(delta)

        # -- logging --
        # print("sum weights: ",torch.sum(images.weights).item())
        # print("sum deno: ",torch.sum(images.deno).item())
        # print("sum basic: ",torch.sum(images.basic).item())

        # - terminate --
        if done: break

    # -- reweight vals --
    # reweight_vals(images)
    # images.weights[th.where(images.weights<5)]=0

    # -- reweight deno --
    weights = repeat(images.weights,'t h w -> t c h w',c=args.c)
    index = torch.nonzero(weights,as_tuple=True)
    images.deno[index] /= weights[index]

    # -- fill zeros with basic --
    fill_img = images.basic if args.step==1 else images.noisy
    index = torch.nonzero(weights==0,as_tuple=True)
    images.deno[index] = fill_img[index]

    # -- color xform --
    utils.yuv2rgb_images(images)

    # -- synch --
    torch.cuda.synchronize()

def reweight_vals(images):
    nmask_before = images.weights.sum().item()
    index = torch.nonzero(images.weights,as_tuple=True)
    images.vals[index] /= images.weights[index]
    irav = images.vals.ravel().cpu().numpy()
    print(np.quantile(irav,[0.1,0.2,0.5,0.8,0.9]))
    # thresh = 0.00014
    thresh = 1e-3
    nz = th.sum(images.vals < thresh).item()
    noupdate = th.nonzero(images.vals > thresh,as_tuple=True)
    images.weights[noupdate] = 0
    th.cuda.synchronize()
    nmask_after = images.weights.sum().item()
    delta_nmask = nmask_before - nmask_after
    print("tozero: [%d/%d]" % (nmask_after,nmask_before))


def fill_valid_patches(vpatches,patches,bufs):
    valid = th.nonzero(th.all(bufs.inds!=-1,1),as_tuple=True)
    for key in patches:
        if (key in patches.tensors) and not(patches[key] is None):
            patches[key][valid] = vpatches[key]

def get_valid_patches(patches,bufs):
    valid = th.nonzero(th.all(bufs.inds!=-1,1),as_tuple=True)
    nv = len(valid)
    vpatches = edict()
    for key in patches:
        if (key in patches.tensors) and not(patches[key] is None):
            vpatches[key] = patches[key][valid]
        else:
            vpatches[key] = patches[key]
    vpatches.shape[0] = nv

    return vpatches

def proc_nl_cache(vid_set,vid_name,sigma):
    return read_nl_sequence(vid_set,vid_name,sigma)


