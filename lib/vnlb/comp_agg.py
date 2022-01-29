
# -- python deps --
import torch
import scipy
import numpy as np
from einops import rearrange

# -- numba --
from numba import njit,cuda

# -- package --
from vnlb.utils import groups2patches

def computeAggregation(deno,group,indices,weights,mask,nSimP,params=None,step=0):

    # # -- create python-params for parser --
    # params,swig_params,_,_ = parse_args(deno,0.,None,params)
    # params = edict({k:v[0] for k,v in params.items()})

    # -- extract info for explicit call --
    ps = params['sizePatch'][step]
    ps_t = params['sizePatchTime'][step]
    onlyFrame = params['onlyFrame'][step]
    aggreBoost =  params['aggreBoost'][step]

    # -- convert groups to patches  --
    t,c,h,w = deno.shape
    nSimP = len(indices)
    patches = groups2patches(group,c,ps,ps_t,nSimP)

    # -- exec search --
    deno_clone = deno.copy()
    nmasked = exec_aggregation(deno,patches,indices,weights,mask,
                               ps,ps_t,onlyFrame,aggreBoost)

    # -- pack results --
    results = {}
    results['deno'] = deno
    results['weights'] = weights
    results['mask'] = mask
    results['nmasked'] = nmasked
    results['psX'] = ps
    results['psT'] = ps_t

    return results

def compute_agg_batch(deno,patches,inds,weights,ps,ps_t,cs_ptr):

    # -- numbify the torch tensors --
    deno_nba = cuda.as_cuda_array(deno)
    patches_nba = cuda.as_cuda_array(patches)
    inds_nba = cuda.as_cuda_array(inds)
    weights_nba = cuda.as_cuda_array(weights)
    cs_nba = cuda.external_stream(cs_ptr)

    # -- launch params --
    # num = patches.shape[0]
    # print("inds.shape: ",inds.shape)
    # print("patches.shape: ",patches.shape)
    bsize,num = inds.shape
    # bsize,bsize = patches.shape[-2:]
    threads = num
    blocks = bsize

    # -- launch kernel --
    # print(deno.shape,weights.shape)
    # print("[agg] deno: ",deno.min().item(),deno.max().item())
    # print("[agg] weights: ",weights.min().item(),weights.max().item())
    # exec_agg[blocks,threads,cs_nba](deno_nba,patches_nba,inds_nba,
    #                                 weights_nba,ps,ps_t)
    exec_agg_simple(deno,patches,inds,weights,ps,ps_t)
    # print("[agg (post)] deno: ",deno.min().item(),deno.max().item())
    # print("[agg (post)] weights: ",weights.min().item(),weights.max().item())


def exec_agg_simple(deno,patches,inds,weights,ps,ps_t):

    # -- numbify --
    device = deno.device
    deno_nba = deno.cpu().numpy()
    patches_nba = patches.cpu().numpy()
    inds_nba = inds.cpu().numpy()
    weights_nba = weights.cpu().numpy()
    # print("patches.shape: ",patches.shape)

    # -- exec numba --
    exec_agg_simple_numba(deno_nba,patches_nba,inds_nba,weights_nba,ps,ps_t)

    # -- back pack --
    deno_nba = torch.FloatTensor(deno_nba).to(device)
    deno[...] = deno_nba
    weights_nba = torch.FloatTensor(weights_nba).to(device)
    weights[...] = weights_nba


@njit
def exec_agg_simple_numba(deno,patches,inds,weights,ps,ps_t):

    # -- shape --
    nframes,color,height,width = deno.shape
    chw = color*height*width
    hw = height*width
    bsize,npatches = inds.shape

    for bi in range(bsize):
        for ni in range(npatches):
            ind = inds[bi,ni]
            if ind == -1: continue
            t0 = ind // chw
            h0 = (ind % hw) // width
            w0 = ind % width

            # print(t0,h0,w0)
            for pt in range(ps_t):
                for pi in range(ps):
                    for pj in range(ps):
                        for ci in range(color):
                            gval = patches[bi,ni,pt,ci,pi,pj]
                            deno[t0+pt,ci,h0+pi,w0+pj] += gval
                        weights[t0+pt,h0+pi,w0+pj] += 1.


@cuda.jit(max_registers=64)
def exec_agg(deno,patches,inds,weights,ps,ps_t):

    # -- shape --
    nframes,color,height,width = deno.shape
    chw = color*height*width
    hw = height*width
    t_bsize = inds.shape[1]

    # -- access with blocks and threads --
    hi = cuda.blockIdx.x
    wi = cuda.blockIdx.y

    # -- cuda threads --
    pindex = cuda.threadIdx.x

    # -> race condition across "batches [t,h,w]"
    # -- we want enough work per thread, so we keep the "t" loop --
    for ti in range(t_bsize):

        # -- unpack ind --
        ind = inds[pindex,ti,hi,wi]
        t0 = ind // chw
        c0 = (ind % chw) // hw
        h0 = (ind % hw) // width
        w0 = ind % width

        # -- set using patch info --
        for pt in range(ps_t):
            for pi in range(ps):
                for pj in range(ps):
                    # for ci in range(color):
                    #     gval = 1#patches[pindex,ti,pt,ci,pi,pj,hi,wi]
                    #     deno[t0+pt,ci,h0+pi,w0+pj] += gval
                    # deno[t0+pt,0,h0+pi,w0+pj] += 1.
                    # deno[t0+pt,1,h0+pi,w0+pj] += 1.
                    # deno[t0+pt,2,h0+pi,w0+pj] += 1.

                    deno[0,0,0,0] += 1.
                    weights[0,0,0] += 1.

                    # deno[t0,0,h0,w0] += 1.
                    # weights[t0,h0,w0] += 1.
                    # weights[t0+pt,h0+pi,w0+pj] += 1.


# @njit
# def exec_aggregation_batch(deno,patches,indices,weights,mask,
#                            ps,ps_t,onlyFrame,aggreBoost):


@njit
def exec_aggregation(deno,patches,indices,weights,mask,
                     ps,ps_t,onlyFrame,aggreBoost):

    # -- def functions --
    def idx2coords(idx,width,height,color):

        # -- get shapes --
        whc = width*height*color
        wh = width*height

        # -- compute coords --
        t = (idx      ) // whc
        c = (idx % whc) // wh
        y = (idx % wh ) // width
        x = idx % width

        return t,c,y,x

    def pixRmColor(ind,c,w,h):
        whc = w*h*c
        wh = w*h
        ind1 = (ind // whc) * wh + ind % wh;
        return ind1

    # -- init --
    nmasked = 0
    t,c,h,w = deno.shape
    nSimP = len(indices)

    # -- update [deno,weights,mask] --
    for n in range(indices.shape[0]):

        # -- get the sim locaion --
        ind = indices[n]
        ind1 = pixRmColor(ind,c,h,w)
        t0,c0,h0,w0 = idx2coords(ind,w,h,c)
        t1,c1,h1,w1 = idx2coords(ind1,w,h,1)

        # -- handle "only frame" case --
        if onlyFrame >= 0 and onlyFrame != t0:
            continue

        # -- set using patch info --
        for pt in range(ps_t):
            for pi in range(ps):
                for pj in range(ps):
                    for ci in range(c):
                        ij = ind + ci*w*h
                        gval = patches[n,pt,ci,pi,pj]
                        deno[t0+pt,ci,h0+pi,w0+pj] += gval
                    weights[t1+pt,h1+pi,w1+pj] += 1

        # -- apply paste trick --
        if (mask[t1,h1,w1] == 1): nmasked += 1
        mask[t1,h1,w1] = False

        if (aggreBoost):
            if ( (h1 > 2*ps) and (mask[t1,h1-1,w1]==1) ): nmasked += 1
            if ( (h1 < (h - 2*ps)) and (mask[t1,h1+1,w1]==1) ): nmasked += 1
            if ( (w1 > 2*ps) and (mask[t1,h1,w1-1]==1) ): nmasked += 1
            if ( (w1 < (w - 2*ps)) and (mask[t1,h1,w1+1]==1) ): nmasked += 1

            if (h1 > 2*ps):  mask[t1,h1-1,w1] = False
            if (h1 < (h - 2*ps)): mask[t1,h1+1,w1] = False
            if (w1 > 2*ps):  mask[t1,h1,w1-1] = False
            if (w1 < (w - 2*ps)): mask[t1,h1,w1+1] = False

    return nmasked

