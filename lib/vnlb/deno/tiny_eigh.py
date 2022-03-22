# -- python --
import torch as th
import torch,math
import scipy
from scipy import linalg as scipy_linalg
import numpy as np
from einops import rearrange,repeat
import vnlb

from easydict import EasyDict as edict

# from .cov_mat import computeCovMat
from vnlb.utils import groups2patches,patches2groups
from vnlb.utils.gpu_utils import apply_yuv2rgb
# from vnlb.gpu.patch_subset import exec_patch_subset
# from vnlb.gpu.patch_utils import yuv2rgb_patches,patches_psnrs
from .bayes_est import flat_pdim,center_patches,flat_bdim,compute_cov_mat

def prepare_patches(pnoisy,pbasic,step,c):

    # -- optional basic --
    if pbasic is None:
        pbasic = pnoisy.clone()

    # -- copy original patches --
    bsize = pnoisy.shape[0]

    # -- flatten dims --
    patches = edict({'noisy':pnoisy,'basic':pbasic,'images':['noisy','basic']})
    flat_pdim(patches)

    # -- center patches --
    flat = th.zeros(bsize,dtype=th.bool,device=pnoisy.device)
    cnoisy,cbasic = center_patches(patches.noisy,patches.basic,flat,step==1,c)

    # -- flatten across color-channel and batch --
    flat_bdim(patches)
    pnoisy,pbasic = patches.noisy,patches.basic

    return pnoisy,pbasic

def get_covmat(patches):
    with th.no_grad():
        # -- cov mat --
        bsize,num,pdim = patches.shape
        covMat = torch.matmul(patches.transpose(2,1),patches)
        covMat /= num
        print("covMat.shape: ",covMat.shape)
    return covMat

def cov2eigs(covMat):
    with th.no_grad():
        # -- eigen stuff --
        eigVals,eigVecs = torch.linalg.eigh(covMat)
        eigVals= torch.flip(eigVals,dims=(1,))
        eigVecs = torch.flip(eigVecs,dims=(2,))
    return eigVals,eigVecs

def patches2cov(pnoisy,pbasic=None,c=3,cpatches="noisy",step=0):
    # -- copy original patches --
    pnoisy,pbasic = prepare_patches(pnoisy,pbasic,step,c)

    # -- cov --
    pinput = pnoisy if cpatches == "noisy" else pbasic
    covMat = get_covmat(pinput)
    return covMat

def run_tiny_eigh(pnoisy,rank=39,pbasic=None,c=3,cpatches="noisy",step=0):

    covMat = patches2cov(pnoisy,pbasic,c,cpatches,step)
    eigVals,eigVecs = cov2eigs(covMat)
    eigVecs = eigVecs[...,:rank]

    return covMat,eigVals,eigVecs
