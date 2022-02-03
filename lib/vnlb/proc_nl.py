
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
from vnlb.utils import groups2patches,patches2groups,optional,divUp
from vnlb.testing import save_images
from vnlb.utils.streams import init_streams,wait_streams
import pprint
pp = pprint.PrettyPrinter(indent=4)


def proc_nl(images,flows,args):

    # -- init --
    # pp.pprint(args)

    # -- create access mask --
    mask,ngroups = search_mask.init_mask(images.shape,args)

    # -- allocate memory --
    patches = alloc.allocate_patches(args.patch_shape,images.clean,args.device)
    bufs = alloc.allocate_bufs(args.bufs_shape,args.device)

    # -- batching params --
    nelems,nbatches = utils.batch_params(mask,args.bsize,args.nstreams)
    cmasked_prev = nelems

    # -- color xform --
    utils.rgb2yuv_images(images)

    # -- over batches --
    pbar = tqdm(total=nelems)
    for batch in range(nbatches):

        # -- exec search --
        done = search.exec_search(patches,images,flows,mask,bufs,args)

        # -- flat patches --
        update_flat_patch(patches,args)

        # -- denoise patches --
        deno.denoise(patches,args)

        # -- aggregate patches --
        agg.agg_patches(patches,images,bufs,args)

        # -- misc --
        torch.cuda.empty_cache()
        if done: break

        # -- loop update --
        cmasked = mask.sum().item()
        delta = cmasked_prev - cmasked
        cmasked_prev = cmasked
        nmasked  = nelems - cmasked
        msg = "[%d/%d]: %d" % (nmasked,nelems,delta)
        tqdm.write(msg)
        pbar.update(delta)

        # -- logging --
        # print("sum weights: ",torch.sum(images.weights).item())

    # -- reweight --
    weights = repeat(images.weights,'t h w -> t c h w',c=args.c)
    index = torch.nonzero(weights,as_tuple=True)
    images.deno[index] /= weights[index]

    # -- fill zeros with basic --
    index = torch.nonzero(weights==0,as_tuple=True)
    images.deno[index] = images.basic[index]

    # -- color xform --
    utils.yuv2rgb_images(images)

    # -- synch --
    torch.cuda.synchronize()

