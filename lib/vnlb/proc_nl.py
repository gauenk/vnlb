
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
from vnlb.testing import save_images
from vnlb.utils.streams import init_streams,wait_streams
import pprint
pp = pprint.PrettyPrinter(indent=4)


def proc_nl(images,flows,args,full_args=None):

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

    # -- over batches --
    pbar = tqdm(total=nelems)
    for batch in range(nbatches):

        # -- exec search --
        done = search.exec_search(patches,images,flows,mask,bufs,args)

        # -- flat patches --
        update_flat_patch(patches,args)

        # -- denoise patches --
        # run_swig_bayes(patches,images,args,full_args)
        # from svnlb.gpu import bayes_estimate_batch
        # cs_ptr = th.cuda.default_stream().cuda_stream
        # rank_var = bayes_estimate_batch(patches.noisy,patches.basic,None,
        #                                 args.sigma2,args.sigmab2,args.rank,
        #                                 args.group_chnls,args.thresh,
        #                                 args.step==1,patches.flat,0,cs_ptr)
        deno.denoise(patches,args)

        # -- aggregate patches --
        agg.agg_patches(patches,images,bufs,args)

        # -- misc --
        torch.cuda.empty_cache()

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
        # print("sum deno: ",torch.sum(images.deno).item())
        # print("sum basic: ",torch.sum(images.basic).item())

        # - terminate --
        if done: break

    # -- reweight --
    weights = repeat(images.weights,'t h w -> t c h w',c=args.c)
    index = torch.nonzero(weights,as_tuple=True)
    images.deno[index] /= weights[index]

    print(weights[0,0,:3,:3])
    print(weights[0,0,-3:,-3:])

    weights /= weights.max().item()
    print(weights[0,0,:3,:3])
    print(weights[0,0,-3:,-3:])
    vnlb.utils.save_burst(weights, "output/example/", "mask")

    # -- fill zeros with basic --
    fill_img = images.basic if args.step==1 else images.noisy
    index = torch.nonzero(weights==0,as_tuple=True)
    images.deno[index] = fill_img[index]

    # -- color xform --
    utils.yuv2rgb_images(images)

    # -- synch --
    torch.cuda.synchronize()


def run_swig_bayes(patches,images,args,full_args):
    from svnlb.swig import computeBayesEstimate
    device = patches.noisy.device

    # -- unpack tensors --
    noisy = patches.noisy.cpu().numpy()
    basic = patches.basic.cpu().numpy()

    # -- rearrange --
    noisy = rearrange(noisy,'b n t c h w -> b n c (t h w)')
    basic = rearrange(basic,'b n t c h w -> b n c (t h w)')

    # -- fake full --
    full_args = edict({k:[v,v] for k,v in args.items()})
    full_args.nParts = [0,0]

    # -- unroll --
    N = noisy.shape[1]
    nshell = np.zeros((9477,3,98),dtype=np.float32)
    bshell = np.zeros((9477,3,98),dtype=np.float32)
    sfloat = np.array([args.sigma,args.sigma],np.float32)
    B = noisy.shape[0]
    for b in range(B):
        print("noisy[b].shape: ",noisy[b].shape)
        nshell[:N] = noisy[b]
        bshell[:N] = basic[b]
        results = computeBayesEstimate(nshell,bshell,args.rank,args.npatches,
                                       images.shape,{'sigma':sfloat},step=args.step)
        noisy[b] = results['groupNoisy'].copy()[:N]
        basic[b] = results['groupBasic'].copy()[:N]

    # -- rearrange --
    noisy = rearrange(noisy,'b n c (t h w) -> b n t c h w',t=2,h=7)
    basic = rearrange(basic,'b n c (t h w) -> b n t c h w',t=2,h=7)
    patches.noisy[...] = th.from_numpy(noisy).to(device)
    patches.basic[...] = th.from_numpy(basic).to(device)


