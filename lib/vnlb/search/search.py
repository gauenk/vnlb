"""
Search for similar patches across batches
"""



# -- python imports --
import torch
import torch as th
from einops import rearrange,repeat
from easydict import EasyDict as edict

# -- [a required package] --
import vpss

# -- local package --
import vnlb.search_mask as search_mask

# -- utils --
from vnlb.utils.batching import view_batch
from vnlb.utils.logger import vprint
from vnlb.utils import divUp
from vnlb.utils import Timer

# -- searching --
@Timer("exec_search")
def exec_search(patches,imgs,flows,mask,bufs,args):

    # -- setup --
    bsize = args.bsize
    cs = th.cuda.default_stream()
    cs_ptr = th.cuda.default_stream().cuda_stream
    done = False

    # --reset values --
    bufs.inds[...] = -1
    bufs.vals[...] = float("inf")

    # -- smaller batch sizes impact quality --
    for index in range(args.nstreams):

        # -- grab access --
        srch_inds = search_mask.mask2inds(mask,bsize)
        if srch_inds.shape[0] == 0:
            done = True
            break

        # -- grab batch --
        vbufs = edict()
        for key in bufs.keys():
            if key in ["fast"]:
                vbufs[key] = bufs[key]
            else:
                vbufs[key] = view_batch(bufs[key],bsize,index)

        vpatches = edict()
        for key in patches.keys():
            vpatches[key] = view_batch(patches[key],bsize,index)

        # -- exec search --
        search_and_fill(imgs,vpatches,vbufs,srch_inds,flows,args)

        # -- update mask naccess --
        before = mask.sum().item()
        search_mask.update_mask_inds(mask,vbufs.inds,args.c,
                                     nkeep=args.nkeep,
                                     boost=args.aggreBoost)
        after = mask.sum().item()

        # -- wait for all streams --
        # torch.cuda.synchronize()

    # -- update term. condition --
    done = done or (mask.sum().item() == 0)

    return done

def search_and_fill(imgs,patches,bufs,srch_inds,flows,args,cs=None):

    # -- select search image --
    if args.srch_img == "noisy":
        srch_img = imgs.noisy
    elif args.srch_img == "basic":
        srch_img = imgs.basic
    elif args.srch_img == "clean":
        srch_img = imgs.clean
    elif args.srch_img == "search":
        srch_img = imgs.search
    else:
        raise ValueError(f"uknown search image [{srch_img}]")
    # srch_img = imgs.noisy if args.step == 0 else imgs.basic
    # srch_img = srch_img if (imgs.clean is None) else imgs.clean

    # -- sim search block --
    bufs.inds[...] = -1
    bufs.vals[...] = float("inf")
    if bufs.fast:
        vpss.exec_sim_search_burst_l2_fast(srch_img,srch_inds,bufs.vals,bufs.inds,
                                           bufs.srch_dists,bufs.srch_locs,
                                           bufs.srch_bufs,flows,args.sigma,args)
    else:
        vpss.exec_sim_search_burst(srch_img,srch_inds,bufs.vals,
                                   bufs.inds,flows,args.sigma,args)


    # -- fill patches --
    for key in imgs.patch_images:

        # -- skip --
        pass_key = (imgs[key] is None) or (patches[key] is None)
        if pass_key: continue

        # -- fill --
        vpss.fill_patches(patches[key],imgs[key],bufs.inds)

# -- searching --
@Timer("exec_search_fast")
def exec_search_fast(patches,imgs,flows,mask,bufs,args):

    # -- init streams --
    device = args.device
    nstreams = args.nstreams
    stream_id = 0
    streams = [th.cuda.default_stream()]
    streams +=[th.cuda.Stream(device=device,priority=0) for s in range(nstreams-1)]

    # -- setup --
    bsize = args.bsize
    bstride = args.bstride
    cs = th.cuda.default_stream()
    cs_ptr = th.cuda.default_stream().cuda_stream
    done = False

    # --reset values --
    t,chnls,h,w = imgs.shape
    # bufs.inds[...] = -1
    # bufs.vals[...] = float("inf")

    # -- smaller batch sizes impact quality --
    for index in range(args.nstreams):

        # -- grab access --
        # th.cuda.set_stream(streams[stream_id])
        # th.cuda.StreamContext(streams[stream_id])
        args.cs_ptr = streams[stream_id].cuda_stream

        # -- create access pattern --
        start = index * bsize * bstride
        stop = ( index + 1 ) * bsize * bstride
        srch_inds = th.arange(start,stop,bstride,device=device)[:,None]
        print(srch_inds)
        srch_inds = search_mask.get_3d_inds(srch_inds,1,h,w)

        # -- grab batch --
        vbufs = edict()
        for key in bufs.keys():
            if key in ["fast"]:
                vbufs[key] = bufs[key]
            else:
                vbufs[key] = view_batch(bufs[key],bsize,index)

        vpatches = edict()
        for key in patches.keys():
            vpatches[key] = view_batch(patches[key],bsize,index)

        # -- exec search --
        search_and_fill(imgs,vpatches,vbufs,srch_inds,flows,args)

        # -- update stream --
        stream_id = (stream_id + 1) % nstreams
        if stream_id == 0:
            torch.cuda.synchronize()


    # -- update term. condition --
    torch.cuda.synchronize()

    return done

