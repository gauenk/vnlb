

# -- python deps --
import torch
import numpy as np
from einops import rearrange
from easydict import EasyDict as edict

# -- numba --
from numba import jit,njit,prange,cuda

# -- parser for cpp --
from svnlb.swig.vnlb.mask_parser import mask_parser

def mask2inds(mask,bsize,rand=True,order=None):
    index = torch.nonzero(mask)
    if index.shape[0] == 0: return index

    if rand:
        # -- randomly shuffly --
        mlen = max(len(index),bsize)
        if order is None or mlen == bsize:
            order = torch.randperm(index.shape[0])
        index = index[order[:bsize]]
        return index
    else:
        # -- index in order --
        return index[:bsize]

def update_mask(mask,access,val=0):
    assert access.shape[1] == 3
    mask[access[:,0],access[:,1],access[:,2]] = val

def update_mask_inds(mask,inds,chnls,cs_ptr,boost=True,val=0,nkeep=-1):

    # -- shape --
    t,h,w = mask.shape
    bsize,num = inds.shape
    hw,chw = h*w,chnls*h*w

    # -- keep only to "nkeep" --
    if nkeep != -1:
        inds = inds[:,:nkeep]
    bsize,num = inds.shape

    # -- rm "-1" inds --
    if inds.shape[0] == 0: return
    args = torch.where(torch.all(inds != -1,1))
    # print("A",inds.shape)
    inds = inds[args]
    # print("B",inds.shape)
    f_bsize,_ = inds.shape
    if inds.shape[0] == 0: return

    # -- augment inds --
    aug_inds = torch.zeros((3,f_bsize,num),dtype=torch.int64)
    aug_inds = aug_inds.to(inds.device)

    # -- (one #) -> (three #s) --
    tdiv = torch.div
    tmod = torch.remainder
    aug_inds[0,...] = tdiv(inds,chw,rounding_mode='floor') # inds // chw
    aug_inds[1,...] = tdiv(tmod(inds,hw),w,rounding_mode='floor') # (inds % hw) // w
    aug_inds[2,...] = tmod(inds,w)
    aug_inds = rearrange(aug_inds,'three b n -> (b n) three')

    # if f_bsize < 10:
    #     print("-- inds --")
    #     print(inds[:,0])
    #     print("-- aug inds --")
    #     aug_inds_p = rearrange(aug_inds,'(b n) three -> three b n',b=f_bsize)
    #     print(aug_inds_p[:,:,0])

    # -- aggregate boost --
    if boost:
        aug_inds = agg_boost(aug_inds,t,chnls,h,w,cs_ptr)

    # -- assign mask info --
    update_mask(mask,aug_inds,val)

def agg_boost(inds,t,c,h,w,cs_ptr):

    # include neighbor pixels as "masked"

    # -- deltas --
    deltas = [[0,0,0],[0,0,-1],[0,0,1],[0,1,0],[0,-1,0]]
    deltas = torch.IntTensor(deltas).to(inds.device)

    # -- create var --
    aggMult = len(deltas)
    B,three = inds.shape
    agg = -torch.ones(B,aggMult,3,dtype=torch.int64).to(inds.device)

    # -- launch --
    print("agg.shape: ",agg.shape)
    print("inds.shape: ",inds.shape,t,c,h,w)
    agg_boost_launcher(agg,inds,deltas,t,c,h,w,cs_ptr)

    # -- remove "-1" --
    agg = rearrange(agg,'b four three -> (b four) three')
    check = torch.all(agg != -1,1)
    args = torch.where(check)[0]
    agg = agg[args]

    return agg

def agg_boost_launcher(agg,inds,deltas,t,c,h,w,cs_ptr):

    # -- numba-fy --
    agg_nba = cuda.as_cuda_array(agg)
    inds_nba = cuda.as_cuda_array(inds)
    deltas_nba = cuda.as_cuda_array(deltas)
    cs_nba = cuda.external_stream(cs_ptr)

    # -- launch --
    B,three = inds.shape
    work_per_thread = 1
    threads = 512
    blocks = divUp(B,threads*work_per_thread)
    # print(blocks,threads,cs_nba)
    agg_boost_cuda[blocks,threads,cs_nba](agg_nba,inds_nba,deltas_nba,
                                          work_per_thread,t,c,h,w)


@cuda.jit(max_registers=64)
def agg_boost_cuda(agg,inds,deltas,wpt,t,c,h,w):

    # -- access with blocks and threads --
    ndeltas = len(deltas)
    bdimX = cuda.blockDim.x
    tIdx = cuda.threadIdx.x
    bIdx = cuda.blockIdx.x
    start_idx = tIdx*wpt + bIdx*bdimX*wpt
    for work_idx in range(wpt):
        idx = start_idx + work_idx
        if idx > inds.shape[0]: continue

        ti = inds[idx,0]
        hi = inds[idx,1]
        wi = inds[idx,2]

        # -- valid ind --
        valid_t = (0 <= ti) and (ti < t)
        valid_h = (0 <= hi) and (hi < h)
        valid_w = (0 <= wi) and (wi < w)
        valid_ind = valid_t and valid_h and valid_w

        for d in range(ndeltas):
            delta = deltas[d]

            # -- modify values --
            mT = ti+delta[0]
            mH = hi+delta[1]
            mW = wi+delta[2]

            # -- valid change --
            valid_t = (0 <= mT) and (mT < t)
            valid_h = (0 <= mH) and (mH < h)
            valid_w = (0 <= mW) and (mW < w)
            valid_prop = valid_t and valid_h and valid_w

            # -- fill data --
            valid = valid_ind and valid_prop
            agg[idx,d,0] = mT if valid else -1
            agg[idx,d,1] = mH if valid else -1
            agg[idx,d,2] = mW if valid else -1


def initMask(shape,vnlb_params,step=0,info=None):

    # -- parse inputs --
    t,c,h,w = shape
    mask = np.zeros((t,h,w),dtype=np.int8)
    vnlb_params = {k:v[step] for k,v in vnlb_params.items()}
    mask_params = mask_parser(mask,vnlb_params,info)
    params = comp_params(mask_params,t,h,w)

    # -- exec --
    ngroups = fill_mask_launcher(mask,params)

    # -- format results --
    results = edict()
    results.mask = mask
    results.ngroups = ngroups

    return results

def comp_params(mask_params,t,h,w):

    # -- init --
    params = edict()
    sPx = mask_params.ps
    sPt = mask_params.ps_t
    sWx = mask_params.sWx
    sWt = mask_params.sWt

    # -- borders --
    params.border_w0 = mask_params.origin_w > 0
    params.border_h0 = mask_params.origin_h > 0
    params.border_t0 = mask_params.origin_t > 0
    params.border_w1 = mask_params.ending_w < w
    params.border_h1 = mask_params.ending_h < h
    params.border_t1 = mask_params.ending_t < t

    # -- origins --
    border_s = sPx-1 + sWx//2
    border_t = sPt-1 + sWt//2
    params.ori_w = border_s if params.border_w0 else 0
    params.ori_h = border_s if params.border_h0 else 0
    params.ori_t = border_t if params.border_t0 else 0
    params.end_w = (w - border_s) if params.border_w1 else (w-sPx+1)
    params.end_h = (h - border_s) if params.border_h1 else (h-sPx+1)
    params.end_t = (t - border_t) if params.border_t1 else (t-sPt+1)

    # -- copy over misc. --
    params.sPx = mask_params.ps
    params.sPt = mask_params.ps_t
    params.sWx = mask_params.sWx
    params.sWt = mask_params.sWt
    params.step_t = mask_params.step_t
    params.step_h = mask_params.step_h
    params.step_w = mask_params.step_w

    return params


def fill_mask_launcher(mask,params):
    # -- unpack --
    step_t = params.step_t
    step_h = params.step_h
    step_w = params.step_w
    border_t0 = params.border_t0
    border_t1 = params.border_t1
    border_h0 = params.border_h0
    border_h1 = params.border_h1
    border_w0 = params.border_w0
    border_w1 = params.border_w1
    ori_t = params.ori_t
    ori_h = params.ori_h
    ori_w = params.ori_w
    end_t = params.end_t
    end_h = params.end_h
    end_w = params.end_w
    ngroups = 0
    ngroups = fill_mask(mask,ngroups,step_t,step_h,step_w,
                        border_t0,border_t1,border_h0,
                        border_h1,border_w0,border_w1,
                        ori_t,ori_h,ori_w,end_t,end_h,end_w)
    return ngroups

@njit
def fill_mask(mask,ngroups,step_t,step_h,step_w,
              border_t0,border_t1,border_h0,
              border_h1,border_w0,border_w1,
              ori_t,ori_h,ori_w,end_t,end_h,end_w):

    # -- init --
    t_size = end_t - ori_t
    h_size = end_h - ori_h
    w_size = end_w - ori_w

    # -- fill it up! --
    ngroups = 0
    for dt in prange(t_size):
        for dh in prange(h_size):
            for dw in prange(w_size):

                # -- unpack --
                ti = ori_t + dt
                hi = ori_h + dh
                wi = ori_w + dw

                # -- bools --
                take_t_step = dt % step_t == 0
                last_t = ti == (end_t-1) and not(border_t1)
                if take_t_step or last_t:

                    phase_h = 0 if last_t else ti//step_t
                    take_h_step = dh % step_h == phase_h % step_h
                    first_h = not(border_h0) and hi == ori_h
                    last_h = not(border_h1) and hi == (end_h-1)

                    if (take_h_step or first_h or last_h):

                        phase_w = 0 if last_h else phase_h + hi//step_h
                        take_w_step = dw % step_w == phase_w % step_w
                        first_w = not(border_w0) and wi == 0
                        last_w = not(border_w1) and wi == (end_w-1)

                        if (take_w_step or first_w or last_w):
                            mask[ti,hi,wi] = True
                            ngroups+=1

    return ngroups


def divUp(a,b): return (a-1)//b+1

