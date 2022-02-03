
# -- python --
import torch,math
import scipy
from scipy import linalg as scipy_linalg
import numpy as np
from einops import rearrange,repeat
import vnlb

# from .cov_mat import computeCovMat
from vnlb.utils import groups2patches,patches2groups
from vnlb.utils.gpu_utils import apply_yuv2rgb
# from vnlb.gpu.patch_subset import exec_patch_subset
# from vnlb.gpu.patch_utils import yuv2rgb_patches,patches_psnrs


def check_steps(step1,step):
    is_step_1 = (step1 == True) and (step == 0)
    is_not_step_1 = (step1 == False) and (step == 1)
    assert is_step_1 or is_not_step_1

def runBayesEstimate(patchesNoisy,patchesBasic,rank_var,nSimP,shape,params,
                     step=0,flatPatch=False):

    # -- create python-params for parser --
    # params,swig_params,_,_ = parse_args(deno,0.,None,params)
    # params = edict({k:v[0] for k,v in params.items()})

    # -- init outputs --
    t,c,h,w = noisy.shape
    zero_basic = th.zeros_like(noisy)
    deno = th.zeros_like(noisy)
    basic = optional(tensors,'basic',zero_basic)
    nstreams = optional(params,'nstreams',1)
    flows = tensors
    deno = th.zeros_like(noisy)
    fflow = flows['fflow']
    bflow = flows['bflow']

    # -- unpack --
    ps = params['sizePatch'][step]
    ps_t = params['sizePatchTime'][step]
    npatches = params['nSimilarPatches'][step]
    w_s = params['sizeSearchWindow'][step]
    nWt_f = params['sizeSearchTimeFwd'][step]
    nWt_b = params['sizeSearchTimeBwd'][step]
    couple_ch = params['coupleChannels'][step]
    step1 = params['isFirstStep'][step]
    check_steps(step1,step)
    sigma2 = params['sigma'][step]**2
    beta = params['beta'][step]
    sigmaBasic2 = params['sigmaBasic'][step]**2
    sigmab2 = beta * sigmaBasic2 if step==1 else sigma**2
    rank =  params['rank'][step]
    thresh =  params['variThres'][step]
    t,chnls,h,w = shape
    group_chnls = 1 if couple_ch else c

    # -- exec python version --
    results = exec_bayes_estimate(patchesNoisy,patchesBasic,sigma2,sigmab2,rank,
                                  nSimP,chnls,group_chnls,thresh,step==1,flatPatch)

    rank_var = results['rank_var']
    return rank_var

def centering(patches,center=None):
    if center is None:
        center = patches.mean(dim=2,keepdim=True)
    patches[...] -= center
    return center

def centering_patches(patchesNoisy,patchesBasic,step2,flat_patch):

    # print("centering patches: ",patchesNoisy.shape,patchesBasic.shape)
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

def compute_cov_mat(patches,rank):

    import time

    with torch.no_grad():
        # -- cov mat --
        bsize,num,pdim = patches.shape
        covMat = torch.matmul(patches.transpose(2,1),patches)
        covMat /= num

        # -- eigen stuff --

        # start = time.perf_counter()
        # eigVals,eigVecs = torch.linalg.eig(covMat)
        # end = time.perf_counter() - start

        # print("Eig Lin Time: ",end)
        # print("a: ",eigVals.shape,eigVecs.shape)

        # eigVals = torch.real(eigVals)#[...,:rank])
        # eigVecs = torch.real(eigVecs[...,:rank])
        # eigVals[...,rank:] = 0
        # print(eigVals[0])
        # print(eigVecs[0,0,:])

        # -- svd stuff --
        eigVals,eigVecs = torch.linalg.eigh(covMat)
        # print(eigVals[0])
        eigVals= torch.flip(eigVals,dims=(1,))
        # print(eigVals[0])
        # print(eigVecs[0,0,:])
        eigVecs = torch.flip(eigVecs,dims=(2,))[...,:rank]
        # print(eigVals[0])
        # print(eigVecs[0,0,:])

        # print("eigh: ",end)
        # print("b: ",eigVals.shape,eigVecs.shape,bsize)

        # -- qr stuff --
        # start = time.perf_counter()
        # Q,R = torch.linalg.qr(covMat)
        # end = time.perf_counter() - start
        # print("qr: ",end)
        # print("QR: ",Q.shape,R.shape)

        # -- chol stuff --
        # start = time.perf_counter()
        # L,info = torch.linalg.cholesky_ex(covMat)
        # end = time.perf_counter() - start
        # print("chol: ",end)
        # print("chol: ",L.shape)

        # -- eig --
        # eigVals = rearrange(eigVals,'(b c) p -> b c p',b=bsize)
        # eigVecs = rearrange(eigVecs,'(b c) p p2 -> b c p p2',b=bsize)

        # -- rank --

    return covMat,eigVals,eigVecs


def denoise_eigvals(eigVals,sigmab2,mod_sel,rank):
    if mod_sel == "clipped":
        th_sigmab2 = torch.FloatTensor([sigmab2]).reshape(1,1,1)
        th_sigmab2 = th_sigmab2.to(eigVals.device)
        emin = torch.min(eigVals[...,:rank],th_sigmab2)
        eigVals[...,:rank] -= emin
    elif mod_sel == "paul_var":
        # bsize,num,pdim = eigVals.shape
        # gamma = pdim/num
        # const = sigmab2 * (1 + math.sqrt(gamma))**2
        # args = torch.where(eigVals > const)
        # eigVals[args] =
	# float tmp, gamma = (float)pdim/(float)nSimP;
	# if (mat.covEigVals[i] > sigmab2 * (tmp = (1 + sqrtf(gamma)))*tmp){
	#   tmp = mat.covEigVals[i] - sigmab2 * (1 + gamma);
	#   mat.covEigVals[i] = tmp * 0.5
	#     * (1. + sqrtf(std::max(0., 1. - 4.*gamma*sigmab2*sigmab2/tmp/tmp)));
	# }else{
	#   mat.covEigVals[i] = 0;
	# }

        pass
    else:
        raise ValueError(f"Uknown eigen-stuff modifier: [{mod_sel}]")

def bayes_filter_coeff(eigVals,sigma2,thresh):
    geq = torch.where(eigVals > (thresh*sigma2))
    leq = torch.where(eigVals <= (thresh*sigma2))
    eigVals[geq] = 1. / (1. + sigma2 / eigVals[geq])
    eigVals[leq] = 0.

def filter_patches_noisy(patches_noisy,patches,covMat,sigma2,eigVals,eigVecs,rank):
    bsize = patches.shape[0]
    Z = torch.matmul(patches,eigVecs)
    R = eigVecs * eigVals[:,None,:rank]
    tmp = torch.matmul(Z,R.transpose(2,1))
    patches_noisy[...] = tmp

def filter_patches(patches,covMat,sigma2,eigVals,eigVecs,rank):

    # reshape
    bsize = patches.shape[0]
    # print(patches.shape,eigVals.shape,eigVecs.shape)
    # patches_rs = rearrange(patches,'b c n p -> (b c) n p')
    # eigVals = rearrange(eigVals,'b c p -> (b c) p')
    # eigVecs = rearrange(eigVecs,'b c p r -> (b c) p r')

    # ---------------------------------
    #      hX' = X' * U * (W * U')
    # ---------------------------------

    # Z = X'*U; (n x p) x (p x r) = (n x r)
    Z = torch.matmul(patches,eigVecs)

    # R = U*W; p x r
    R = eigVecs * eigVals[:,None,:rank]
    # print("R")
    # print(R[0,:2,:2])
    # print("Ratio")
    # print(R[0,:2,:2]/eigVecs[0,:2,:2])
    # print(eigVals[0,:2])

    # hX' = Z'*R' = (X'*U)'*(U*W)'; (n x r) x (r x p) = (n x p)
    tmp = torch.matmul(Z,R.transpose(2,1))
    # tmp = rearrange(tmp,'(b c) n p -> b c n p',b=bsize)
    # print("tmp.shape: ",tmp.shape)
    patches[...] = tmp


# ----------------------------------
#
#   Primary Function in this File
#
# ----------------------------------

def bayes_estimate_batch(in_patchesNoisy,patchesBasic,patchesClean,
                         sigma2,sigmab2,rank,group_chnls,
                         thresh,step2,flat_patch,cs,cs_ptr,mod_sel="clipped",
                         use_weights=False,inds=None):


    # print("rank: ",rank)
    # print("thresh: ",thresh)
    # print("sigmab2: ",sigmab2)
    # print("group_chnls: ",group_chnls)

    # -- shaping --
    patchesNoisy = in_patchesNoisy
    shape = list(patchesNoisy.shape)
    # shape[1],shape[-1] = 1,nSimP
    # print("patchesNoisy.shape: ",patchesNoisy.shape)
    bsize,num,ps_t,chnls,ps,ps = patchesNoisy.shape
    pdim = ps*ps*ps_t
    sigma = math.sqrt(sigma2)

    # -- reshape for centering --
    shape_str = "b n pt c ph pw -> b c n (pt ph pw)"
    patchesNoisy = rearrange(patchesNoisy,shape_str)
    patchesBasic = rearrange(patchesBasic,shape_str)
    if not(patchesClean is None):
        patchesClean = rearrange(patchesClean,shape_str)

    # -- group noisy --
    centerNoisy,centerBasic = centering_patches(patchesNoisy,patchesBasic,
                                                step2,flat_patch)
    # print("centerNoisy.shape: ",centerBasic.shape)

    # -- reshape for processing --
    shape_str = "b c n p -> (b c) n p"
    patchesNoisy = rearrange(patchesNoisy,shape_str)
    if step2: patchesBasic = rearrange(patchesBasic,shape_str)
    # print(patchesNoisy.mean(dim=2))
    # print(patchesBasic.mean(dim=2))

    shape_str = "b c x p -> (b c) x p"
    centerNoisy = rearrange(centerNoisy,shape_str)
    if step2: centerBasic = rearrange(centerBasic,shape_str)

    # -- denoising! --
    rank_var = 0.
    patchesInput = patchesNoisy if not(step2) else patchesBasic
    covMat,eigVals,eigVecs = compute_cov_mat(patchesInput,rank)

    # -- modify eigenvals --
    eigVals_rs = rearrange(eigVals,'(b c) p -> b c p',b=bsize)
    rank_var = torch.mean(torch.sum(eigVals_rs,dim=2),dim=1)
    denoise_eigvals(eigVals_rs,sigmab2,mod_sel,rank)
    bayes_filter_coeff(eigVals,sigma2,thresh)

    # -- denoise! --
    filter_patches(patchesNoisy,covMat,sigma2,eigVals,eigVecs,rank)
    # filter_patches_noisy(patchesNoisy,patchesBasic,covMat,sigma2,eigVals,eigVecs,rank)

    # -- add back center --
    patchesNoisy[...] += centerNoisy
    # patchesNoisy[...] += centerBasic

    # -- reshape --
    # shape_str = '(bt bh bw) c n (pt px py) -> n bt pt c px py bh bw'
    # kwargs = {"pt":ps_t,"px":ps,"bh":h_bsize,"bw":w_bsize}
    shape_str = '(b c) n (pt ph pw) -> b n pt c ph pw'
    kwargs = {"pt":ps_t,"ph":ps,"pw":ps,"b":bsize}
    # print("patchesNoisy.shape: ",patchesNoisy.shape)
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
