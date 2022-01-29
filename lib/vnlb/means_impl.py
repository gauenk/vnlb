
import torch
import scipy
from scipy import linalg as scipy_linalg
import numpy as np
from einops import rearrange,repeat
import vnlb

# from .cov_mat import computeCovMat
from vnlb.utils import groups2patches,patches2groups
from vnlb.testing import save_images

def centering(patches,center=None):
    if center is None:
        center = patches.mean(dim=2,keepdim=True)
    patches[...] -= center
    return center

def centering_patches(patchesNoisy,patchesBasic,patchesClean,
                      step2,valid_clean,flat_patch):

    # -- center basic --
    centerBasic,centerClean = None,None
    if step2:
        centerBasic = centering(patchesBasic)
    if valid_clean:
        centerClean = centering(patchesClean)

    # -- choose center for noisy --
    centerNoisy = patchesNoisy.mean(dim=2,keepdim=True)
    if step2:
        flat_inds = torch.where(flat_patch == 1)[0]
        centerNoisy[flat_inds] = centerBasic[flat_inds]

    # -- center noisy --
    centerNoisy = centering(patchesNoisy,centerNoisy)

    return centerNoisy,centerBasic

def compute_weights(patches,sigma2,h=2.):

    # -- unpack shape --
    bsize,chnls,num,pdim = patches.shape
    offset = 2*sigma2/(255.**2)
    # print("offset: ",offset)

    # -- delta --
    patches /= 255.
    # delta = patches[:,:,None,] - patches[:,:,:,None]
    delta = (patches[:,:,[0],] - patches)**2
    # print("[1] delta.shape: ",delta.shape)
    delta = torch.mean(delta,dim=-1,keepdim=True)
    # print("[1.5] delta.shape: ",delta.shape)
    delta = delta[:,[0]]
    # delta = torch.mean(delta,dim=1,keepdim=True)
    # print(delta)
    # print(delta)
    # print(delta[0])
    # print(delta[-1])

    # -- offset --
    delta -= offset
    # print("[2] delta.shape: ",delta.shape)

    # -- remove large deltas --
    # cutoff = 5./255.**2
    # cutoff = 100./(255.**2)
    # print(cutoff)
    # args = torch.where(delta > cutoff)
    # delta[args] = float("inf")

    # -- max pool --
    # print(delta.min(),delta.max(),offset)
    zero = torch.FloatTensor([0.]).view(1,1,1).to(patches.device)
    delta = torch.maximum(delta,zero)
    # print(delta.min(),delta.max(),offset)

    # print(delta[0])
    # if delta.shape[0] > 8:
    #     print(delta[10])
    # print(delta[-1])

    # -- weights --
    weights = torch.exp(-delta/h**2)
    # print(weights)

    return weights

def filter_patches(patches,weights):
    # print("filter: ",patches.shape,weights.shape)
    C = torch.sum(weights,dim=2)
    # print("C: ",C.shape)
    wsum = torch.sum(patches * weights,dim=2)
    # print("wsum: ",wsum.shape)
    patches[:,:,0] = wsum/C

def means_estimate_batch(in_pNoisy,pBasic,pClean,vals,sigma2,sigmab2,
                         rank,group_chnls,thresh,step2,valid_clean,
                         flat_patch,cs,cs_ptr,mod_sel="clipped"):

    # -- shaping --
    pNoisy = in_pNoisy
    shape = list(pNoisy.shape)
    # shape[1],shape[-1] = 1,nSimP
    # print("pNoisy.shape: ",pNoisy.shape)
    bsize,num,ps_t,chnls,ps,ps = pNoisy.shape
    pdim = ps*ps*ps_t

    # -- reshape for centering --
    shape_str = "b n pt c ph pw -> b c n (pt ph pw)"
    pNoisy = rearrange(pNoisy,shape_str)
    pBasic = rearrange(pBasic,shape_str)
    pClean = rearrange(pClean,shape_str)

    # -- group noisy --
    centerNoisy,centerBasic = centering_patches(pNoisy,pBasic,pClean,
                                                valid_clean,step2,flat_patch)

    # -- reshape for processing --
    # shape_str = "b c n p -> (b c) n p"
    # pNoisy = rearrange(pNoisy,shape_str)
    # if step2: pBasic = rearrange(pBasic,shape_str)

    # shape_str = "b c 1 p -> (b c) 1 p"
    # centerNoisy = rearrange(centerNoisy,shape_str)
    # if step2: centerBasic = rearrange(centerBasic,shape_str)

    # -- denoise! --
    h = .4*np.sqrt(sigma2)/255.
    pInput = pBasic if step2 else pNoisy
    # pInput = pClean if valid_clean else pInput
    weights = compute_weights(pInput,sigma2,h=h)
    filter_patches(pNoisy,weights)

    # -- add back center --
    pNoisy[...] += centerNoisy

    # -- reshape --
    # shape_str = '(b c) n (pt ph pw) -> b n pt c ph pw'
    shape_str = 'b c n (pt ph pw) -> b n pt c ph pw'
    kwargs = {"pt":ps_t,"ph":ps,"pw":ps,"b":bsize}
    pNoisy = rearrange(pNoisy,shape_str,**kwargs)
    if step2:
        pBasic = rearrange(pBasic,shape_str,**kwargs)

    # -- fill data --
    in_pNoisy[...] = pNoisy


    return 0.
