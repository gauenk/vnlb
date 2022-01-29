
import torch
import scipy
from scipy import linalg as scipy_linalg
import numpy as np
from einops import rearrange,repeat
import vnlb

# from .cov_mat import computeCovMat
from vnlb.utils import groups2patches,patches2groups

def centering(patches,center=None):
    if center is None:
        center = patches.mean(dim=2,keepdim=True)
    patches[...] -= center
    return center

def centering_patches(patchesNoisy,patchesBasic,step2,flat_patch):

    # -- center basic --
    centerBasic = None
    if step2:
        centerBasic = centering(patchesBasic)

    # -- choose center for noisy --
    centerNoisy = patchesNoisy.mean(dim=2,keepdim=True)
    if step2:
        flat_inds = torch.where(flat_patch == 1)[0]
        centerNoisy[flat_inds] = centerBasic[flat_inds]

    # -- center noisy --
    centerNoisy = centering(patchesNoisy,centerNoisy)

    return centerNoisy,centerBasic

def compute_qr(patches,sigma2,rank):
    with torch.no_grad():

        # -- cov mat --
        bsize,num,pdim = patches.shape
        covMat = torch.matmul(patches.transpose(2,1),patches)
        covMat /= num

        # -- Cov + \sigma I --
        idiag = torch.arange(covMat.shape[-1])
        covMat[:,idiag,idiag] += sigma2

        # -- QR --
        Q,R = torch.linalg.qr(covMat)

        # -- convert back --
        covMat[:,idiag,idiag] -= sigma2

    return covMat,Q,R

def filter_patches(patches,covMat,Q,R,sigma2,rank):

    # ---------------------------------
    #      hX' = X' * U * (W * U')
    # ---------------------------------

    # print("patches.shape: ",patches.shape)
    # print("covMat.shape: ",covMat.shape)
    # print("Q.shape: ",Q.shape)
    # print("R.shape: ",R.shape)

    # -- inv 1 --
    # Z = torch.matmul(Q.transpose(2,1),patches.transpose(2,1))
    # print(Z.shape,Q.shape)
    # covInv = torch.triangular_solve(Z,Q).solution

    # print("patches.shape: ",patches.shape)
    # print("Q.shape: ",Q.shape)
    # print("R.shape: ",R.shape)

    # -- inv 2 --
    # Z = torch.triangular_solve(patches.transpose(2,1),R).solution
    Pt = patches.transpose(2,1)
    Qb = torch.matmul(Q,Pt)
    # Qb = torch.matmul(Q.transpose(2,1),Pt)
    # Qb = Q @ Pt
    # Rt = R.transpose(2,1)
    Rt = R.transpose(2,1)
    # covInv = torch.triangular_solve(Qb,Rt,upper=False).solution
    covInv = torch.triangular_solve(Qb,R,upper=True).solution

    # Z = torch.triangular_solve(Qb,Rt,upper=False).solution
    # covInv = torch.matmul(Q.transpose(2,1),Z)
    # covInv = torch.matmul(Q,Z)
    # print("covInv: ",covInv.shape)
    # diag_idx = torch.arange(covMat.shape[-1])
    # print("covdiag: ",covMat[:,diag_idx,diag_idx].shape)
    # print(covMat[0])
    # covMat[:,diag_idx,diag_idx] -= sigma2
    # print(covMat[0])
    # covSig = covMat + sigma2*torch.eye(covMat.shape)
    # print(covMat.shape)
    tmp = torch.matmul(covMat,covInv)
    print("tmp.shape: ",tmp.shape)

    # -- fill --
    patches[...] = tmp.transpose(2,1)


def qr_estimate_batch(in_patchesNoisy,patchesBasic,sigma2,
                      sigmab2,rank,group_chnls,thresh,
                      step2,flat_patch,cs,cs_ptr,mod_sel="clipped"):

    # -- shaping --
    patchesNoisy = in_patchesNoisy
    shape = list(patchesNoisy.shape)
    # shape[1],shape[-1] = 1,nSimP
    # print("patchesNoisy.shape: ",patchesNoisy.shape)
    bsize,num,ps_t,chnls,ps,ps = patchesNoisy.shape
    pdim = ps*ps*ps_t

    # -- reshape for centering --
    shape_str = "b n pt c ph pw -> b c n (pt ph pw)"
    patchesNoisy = rearrange(patchesNoisy,shape_str)
    patchesBasic = rearrange(patchesBasic,shape_str)

    # -- group noisy --
    centerNoisy,centerBasic = centering_patches(patchesNoisy,patchesBasic,
                                                step2,flat_patch)

    # -- reshape for processing --
    shape_str = "b c n p -> (b c) n p"
    patchesNoisy = rearrange(patchesNoisy,shape_str)
    if step2: patchesBasic = rearrange(patchesBasic,shape_str)

    shape_str = "b c 1 p -> (b c) 1 p"
    centerNoisy = rearrange(centerNoisy,shape_str)
    if step2: centerBasic = rearrange(centerBasic,shape_str)

    # if step2: patchesBasic = rearrange(patchesBasic,shape_str)
    # shape = (b c n p)

    # -- denoising! --
    rank_var = 0.

    import time

    # -- compute eig stuff --
    start = time.perf_counter()
    patchesInput = patchesNoisy if not(step2) else patchesBasic
    covMat,Q,R = compute_qr(patchesInput,sigma2,rank)
    end = time.perf_counter() - start
    print("QR Time: ",end)

    # -- modify eigenvals --
    # start = time.perf_counter()
    # eigVals_rs = rearrange(eigVals,'(b c) p -> b c p',b=bsize)
    # rank_var = torch.mean(torch.sum(eigVals_rs,dim=2),dim=1)
    # denoise_eigvals(R,sigmab2,mod_sel,rank)
    # bayes_filter_coeff(eigVals,sigma2,thresh)
    # end = time.perf_counter() - start
    # print("DN + Filter Time: ",end)

    # -- denoise! --
    filter_patches(patchesNoisy,covMat,Q,R,sigma2,rank)

    # -- add back center --
    patchesNoisy[...] += centerNoisy

    # -- reshape --
    # shape_str = '(bt bh bw) c n (pt px py) -> n bt pt c px py bh bw'
    # kwargs = {"pt":ps_t,"px":ps,"bh":h_bsize,"bw":w_bsize}
    shape_str = '(b c) n (pt ph pw) -> b n pt c ph pw'
    kwargs = {"pt":ps_t,"ph":ps,"pw":ps,"b":bsize}
    patchesNoisy = rearrange(patchesNoisy,shape_str,**kwargs)
    if step2:
        patchesBasic = rearrange(patchesBasic,shape_str,**kwargs)
    in_patchesNoisy[...] = patchesNoisy

    # -- pack results --
    results = {}
    # results['patchesNoisy'] = patchesNoisy
    # results['patchesBasic'] = patchesBasic
    # results['patches'] = patchesNoisy
    # results['center'] = centerNoisy
    # results['covMat'] = covMat
    # results['covEigVecs'] = eigVecs
    # results['covEigVals'] = eigVals
    # results['rank_var'] = rank_var
    # return results

    return rank_var#,patchesNoisy
