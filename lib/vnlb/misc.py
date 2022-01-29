
# -- python deps --
import copy,torch,tqdm,math
import torch as th
from einops import rearrange,repeat
from easydict import EasyDict as edict

# -- package --
import vnlb

# -- local imports --
from .init_mask import initMask,mask2inds,update_mask,update_mask_inds
from .sim_search import sim_search_batch
from .bayes_est import compute_cov_mat,cov_to_psnrs
from .patch_subset import exec_patch_subset

# -- other --
import torch
import torch as th
import scipy
from scipy import linalg as scipy_linalg
import numpy as np
from einops import rearrange,repeat
import vnlb

# -- project imports --
from vnlb.utils.gpu_utils import apply_color_xform_cpp,yuv2rgb_cpp

# -- project imports --
from vnlb.utils import groups2patches,patches2groups,optional,divUp
from vnlb.testing import save_images

# -- plotting --
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


def patch_est_plot(noisy,clean,sigma,flows,params,gpuid=0):
    """

    Explore error decay for iid v.s. noisy patches when est cov

    """

    # -- place on cuda --
    device = gpuid
    if not(th.is_tensor(noisy)):
        noisy = th.FloatTensor(noisy).to(device)
        clean = th.FloatTensor(clean).to(device)

    # -- init outputs --
    step = 0
    shape = noisy.shape
    t,c,h,w = noisy.shape
    nstreams = int(optional(params,'nstreams',[1,1])[step])

    # -- to device flow --
    flows = edict({k:th.FloatTensor(v).to(device) for k,v in flows.items()})
    zflow = torch.zeros((t,2,h,w)).to(device)
    fflow = optional(flows,'fflow',zflow)
    bflow = optional(flows,'bflow',zflow)

    # -- unpack --
    ps = 7#params['sizePatch'][step]
    ps_t = params['sizePatchTime'][step]
    npatches = 700 #params['nSimilarPatches'][step]
    w_s = params['sizeSearchWindow'][step]
    nWt_f = params['sizeSearchTimeFwd'][step]
    nWt_b = params['sizeSearchTimeBwd'][step]
    couple_ch = params['coupleChannels'][step]
    step1 = params['isFirstStep'][step]
    sigma = params['sigma'][step]
    sigma2 = params['sigma'][step]**2
    beta = params['beta'][step]
    sigmaBasic2 = params['sigmaBasic'][step]**2
    sigmab2 = beta * sigmaBasic2 if step==1 else sigma**2
    rank =  params['rank'][step]
    thresh =  params['variThres'][step]
    flat_areas = params['flatAreas'][step]
    gamma = params['gamma'][step]
    step_s = params['procStep'][step] # spatial step
    t,c,h,w = shape
    group_chnls = 1 if couple_ch else c
    step_s = 1
    # print("step_s: ",step_s)
    offset = 2*(sigma/255.)**2
    bsize_s = int(optional(params,'bsize_s',[256,256])[step])
    bsize_s = 10

    # -- create mask --
    mask = initMask(noisy.shape,params,step)['mask']
    mask = torch.ByteTensor(mask).to(device)

    # -- run the step --
    exec_patch_est_plot(noisy,clean,mask,fflow,bflow,sigma2,sigmab2,rank,ps,
                        ps_t,npatches,step_s,w_s,nWt_f,nWt_b,group_chnls,couple_ch,
                        thresh,flat_areas,gamma,offset,step,bsize_s,nstreams)


def exec_patch_est_plot(noisy,clean,mask,fflow,bflow,sigma2,sigmab2,rank,
                        ps,ps_t,npatches,step_s,w_s,nWt_f,nWt_b,group_chnls,couple_ch,
                        thresh,flat_areas,gamma,offset,step,bsize_s,nstreams):

    # -- unpack info --
    use_imread = False
    device = noisy.device
    shape = noisy.shape
    nframes,chnls,height,width = noisy.shape
    is_yuv = True

    # -- color xform --
    if is_yuv:
        noisy_yuv = apply_color_xform_cpp(noisy)
        clean_yuv = apply_color_xform_cpp(clean)
    else:
        noisy_yuv = noisy
        clean_yuv = clean

    # -- search region aliases --
    w_t = min(nWt_f + nWt_b + 1,nframes-1)
    nsearch = w_s * w_s * w_t
    print(nsearch)

    # -- batching height and width --
    bsize,ssize = bsize_s*nstreams,1
    # bsize,ssize = bsize_s,1
    nelems = torch.sum(mask).item()
    nbatches = divUp(divUp(nelems,nstreams),bsize)
    nmasked = 0

    # -- create shell --
    ns,np,t,c = nstreams,npatches,nframes,chnls
    tf32,ti32 = torch.float32,torch.int32
    patchesNoisy = torch.zeros(bsize,np,ps_t,c,ps,ps).type(tf32).to(device)
    patchesVoid_a = torch.zeros(bsize,np,ps_t,c,ps,ps).type(tf32).to(device)
    patchesVoid_b = torch.zeros(bsize,np,ps_t,c,ps,ps).type(tf32).to(device)
    patchesNoisyCs = torch.zeros(bsize,np,ps_t,c,ps,ps).type(tf32).to(device)
    patchesCleanNs = torch.zeros(bsize,np,ps_t,c,ps,ps).type(tf32).to(device)
    patchesCleanCs = torch.zeros(bsize,np,ps_t,c,ps,ps).type(tf32).to(device)
    inds = -torch.ones(bsize,np).type(torch.int32).to(device)
    vals = torch.ones(bsize,np).type(tf32).to(device)*float("inf")
    weights = torch.zeros(nframes,height,width).type(tf32).to(device)
    flat_patches = torch.zeros(bsize).type(ti32).to(device)

    # -- get indies from mask --
    cs = 0 # current stream
    cs_ptr = torch.cuda.default_stream().cuda_stream
    access = mask2inds(mask,bsize)
    update_mask(mask,access)
    access = mask2inds(mask,bsize)
    # update_mask_inds(mask,inds_s,chnls,cs_ptr,nkeep=nkeep)


    # -- search using noisy patches [step == 0] --
    # s_offset = offset
    # sim_search_batch(noisy_yuv,clean_yuv,None,
    #                  patchesNoisy,patchesVoid,None,
    #                  access,vals,inds,fflow,bflow,step_s,bsize,
    #                  ps,ps_t,w_s,nWt_f,nWt_b,True,s_offset,cs,cs_ptr)

    # -- search noisy and fill clean [step == 0] --
    inds[...] = -1
    vals[...] = 0
    s_offset = offset/math.sqrt(2)
    print(s_offset)
    print(patchesNoisy.shape)
    sim_search_batch(clean_yuv,clean_yuv,noisy_yuv,
                     patchesCleanNs,patchesVoid_a,patchesNoisy,
                     access,vals,inds,fflow,bflow,step_s,bsize,
                     ps,ps_t,w_s,nWt_f,nWt_b,True,s_offset,cs,cs_ptr)
    torch.cuda.synchronize()

    # -- search clean and fill noisy [step == 0] --
    inds[...] = -1
    vals[...] = 0
    s_offset = 0
    print(noisy_yuv.shape,clean_yuv.shape)
    sim_search_batch(noisy_yuv,clean_yuv,clean_yuv,
                     patchesNoisyCs,patchesVoid_b,patchesCleanCs,
                     access,vals,inds,fflow,bflow,step_s,bsize,
                     ps,ps_t,w_s,nWt_f,nWt_b,True,s_offset,cs,cs_ptr)

    # -- search clean and fill clean [step == 1] --
    # inds[...] = -1
    # vals[...] = 0
    # s_offset = 0
    # sim_search_batch(clean_yuv,clean_yuv,None,
    #                  patchesCleanCs,patchesVoid,None,
    #                  access,vals,inds,fflow,bflow,step_s,bsize,
    #                  ps,ps_t,w_s,nWt_f,nWt_b,True,s_offset,cs,cs_ptr)
    torch.cuda.synchronize()


    # -- bayes denoising --
    compute_errors(patchesNoisy,patchesNoisyCs,patchesCleanNs,
                   patchesCleanCs,sigma2,rank,thresh,is_yuv)

def compute_errors(pNoisy,pNoisyCs,pCleanNoiseSearch,pClean,sigma2,rank,thresh,is_yuv):

    # -- normalize all --
    pgroups = [pNoisy,pNoisyCs,pCleanNoiseSearch,pClean]
    # for idx,pgroup in enumerate(pgroups):
    #     pgroups[idx] /= 255.

    # -- rename --
    pCleanNs = pCleanNoiseSearch
    pCleanCs = pClean

    # -- unpack shape --
    device = pNoisy.device
    b,n,pt,c,ph,pw = pNoisy.shape
    pNoisy = rearrange(pNoisy,'b n pt c ph pw -> (b c) n (pt ph pw)')
    pNoisyCs = rearrange(pNoisyCs,'b n pt c ph pw -> (b c) n (pt ph pw)')
    pCleanNs = rearrange(pCleanNs,'b n pt c ph pw -> (b c) n (pt ph pw)')
    pCleanCs = rearrange(pCleanCs,'b n pt c ph pw -> (b c) n (pt ph pw)')
    pClean = rearrange(pClean,'b n pt c ph pw -> (b c) n (pt ph pw)')

    # -- reshape shape --
    bsize,num,pdim = pNoisy.shape

    # -- reference noise eye covariance --
    covNoise = torch.eye(pdim,device=device) * sigma2

    # -- sigma2 -> sigma --
    sigma = int(np.sqrt(np.array([sigma2])).item())
    print("sigma2,sigma: ",sigma2,sigma)

    # -- reorder noisy patches by true bias  --
    delta = torch.sum((pCleanNs - pCleanCs[:,[0]])**2,dim=2)
    delta = repeat(delta,'b n -> b n r',r=pdim)
    order = torch.argsort(delta,dim=1)
    pOrdered = torch.gather(pNoisy,1,order)

    # -- compute deltas for noisy patches searched with clean --
    delta = torch.sum((pNoisyCs - pCleanCs[:,[0]])**2,dim=2)
    delta = repeat(delta,'b n -> b n r',r=pdim)
    order = torch.argsort(delta,dim=1)
    pNoisyCs = torch.gather(pNoisyCs,1,order)

    # -- shell of proposed method --
    # pGradSort,_,_ = exec_patch_subset(pNoisy,sigma)
    pGradSort,_,wGradSort,_ = exec_patch_subset(pNoisy,sigma,
                                                ref_patches=pNoisy)#,clean=pClean)
    #,clean=pClean)
    print(wGradSort[0,:])
    print("pNoisy.shape: ",pNoisy.shape)
    print("pGradSort.shape: ",pGradSort.shape)

    # -- get clean patch --
    # covMat_clean,_,_ = compute_cov(pClean[:,[0]],rank)
    covMat_clean,_,_ = compute_cov(pCleanCs[:,[0]],rank)
    covMat_clean = covMat_clean + covNoise
    # covMat_clean,_,_ = compute_cov(pNoisy,rank,clean=pCleanCs[:,[0]])

    # -- shell of iid to fill --
    iid_patches = th.zeros(bsize,num,pdim).to(device)
    for sim_patch in range(num):
        iid_patches[:,sim_patch] = iid_noisy(pCleanCs[:,0],sigma)

    # -- shell of iid to fill --
    iid_patches_1 = th.zeros(bsize,num,pdim).to(device)
    for sim_patch in range(num):
        iid_patches_1[:,sim_patch] = iid_noisy(pCleanCs[:,1],sigma)

    # -- shell of iid to fill --
    iid_patches_10 = th.zeros(bsize,num,pdim).to(device)
    for sim_patch in range(num):
        iid_patches_10[:,sim_patch] = iid_noisy(pCleanCs[:,9],sigma)

    # -- compute error --
    npoints = 10
    num = 500
    skip = divUp(num,npoints)
    print("skip: ",skip)
    # prange = [1,] + [i*skip for i in range(1,npoints+1)]
    # prange = [i*skip for i in range(1,npoints+1)]
    # prange = [i for i in range(1,skip)]
    prange = []
    prange = prange + [i*skip for i in range(1,npoints+1)]
    prange = [100] + prange[-2:]
    print("prange: ",prange)

    # -- errors --
    noisy_error,iid_error = [],[]
    iid_error_1,iid_error_10 = [],[]
    cn_error,nc_error,sorted_error = [],[],[]
    gsorted_error = []
    cc_error,iid_error_noisy = [],[]

    # -- psnrs --
    noisy_psnrs,iid_psnrs = [],[]
    iid_psnrs_1,iid_psnrs_10 = [],[]
    cn_psnrs,nc_psnrs,sorted_psnrs = [],[],[]
    gsorted_psnrs = []
    cc_psnrs,iid_psnrs_noisy = [],[]

    for npatches_i in tqdm.tqdm(prange):

        # -- compute noisy error --
        patchesInput = pNoisy[:,:npatches_i]
        pNoisyAtPi = pNoisy[:,:npatches_i]
        covMat,_,_ = compute_cov(patchesInput,rank)
        noisy_error.append(cov_error(covMat,covMat_clean))
        noisy_psnrs.append(cov_to_psnrs(covMat,pClean,patchesInput,
                                        sigma2,rank,thresh,is_yuv))

        # -- compute noisy error search with ordering --
        patchesInput = pOrdered[:,:npatches_i]
        covMat,_,_ = compute_cov(patchesInput,rank)
        sorted_error.append(cov_error(covMat,covMat_clean))
        # sorted_psnrs.append(cov_to_psnrs(covMat,pClean,patchesInput,
        #                                  sigma2,rank,thresh,is_yuv))
        sorted_psnrs.append(cov_to_psnrs(covMat,pClean,pNoisyAtPi,
                                         sigma2,rank,thresh,is_yuv))

        # -- compute "grad search" method --
        patchesInput = pGradSort[:,:npatches_i]
        patchesWeights = wGradSort[:,:npatches_i]
        covMat,_,_ = compute_cov(patchesInput,rank,patchesWeights)
        gsorted_error.append(cov_error(covMat,covMat_clean))
        # gsorted_psnrs.append(cov_to_psnrs(covMat,pClean,patchesInput,
        #                                   sigma2,rank,thresh,is_yuv))
        gsorted_psnrs.append(cov_to_psnrs(covMat,pClean,pNoisyAtPi,
                                          sigma2,rank,thresh,is_yuv))

        # -- compute noisy with clean search --
        patchesInput = pNoisyCs[:,:npatches_i]
        covMat,_,_ = compute_cov(patchesInput,rank)
        nc_error.append(cov_error(covMat,covMat_clean))
        nc_psnrs.append(cov_to_psnrs(covMat,pClean,pNoisyAtPi,
                                     sigma2,rank,thresh,is_yuv))
        # nc_psnrs.append(cov_to_psnrs(covMat,pClean,patchesInput,
        #                              sigma2,rank,thresh,is_yuv))


        # -- compute noisy with clean search --
        patchesInput = pCleanNs[:,:npatches_i]
        covMat,_,_ = compute_cov(patchesInput,rank)
        cn_error.append(cov_error(covMat,covMat_clean))
        cn_psnrs.append(cov_to_psnrs(covMat,pClean,pNoisyAtPi,
                                     sigma2,rank,thresh,is_yuv))

        # -- compute clean patches with clean search --
        # -- [NOT ABOVE!] just using the GT cov --
        patchesInput = pCleanCs[:,:npatches_i]
        covMat,_,_ = compute_cov(patchesInput,rank)
        cc_error.append(cov_error(covMat_clean,covMat_clean))
        cc_psnrs.append(cov_to_psnrs(covMat_clean,pClean,pNoisyAtPi,
                                     sigma2,rank,thresh,is_yuv))
        # cc_psnrs.append(cov_to_psnrs(covMat,pClean,patchesInput,
        #                              sigma2,rank,thresh,is_yuv))

        # -- compute iid error --
        patchesInput = iid_patches[:,:npatches_i]
        covMat,_,_ = compute_cov(patchesInput,rank)
        iid_error.append(cov_error(covMat,covMat_clean))
        iid_psnrs.append(cov_to_psnrs(covMat,pClean,pNoisyAtPi,
                                      sigma2,rank,thresh,is_yuv,True))
        # iid_psnrs.append(cov_to_psnrs(covMat,pClean,patchesInput,
        #                               sigma2,rank,thresh,is_yuv))

        # -- compute iid error --
        patchesInput = iid_patches_1[:,:npatches_i]
        covMat,_,_ = compute_cov(patchesInput,rank)
        iid_error_1.append(cov_error(covMat,covMat_clean))
        iid_psnrs_1.append(cov_to_psnrs(covMat,pClean,pNoisyAtPi,
                                        sigma2,rank,thresh,is_yuv))
        # iid_psnrs_1.append(cov_to_psnrs(covMat,pClean,patchesInput,
        #                                 sigma2,rank,thresh,is_yuv))

        # -- compute iid error --
        patchesInput = iid_patches_10[:,:npatches_i]
        covMat,_,_ = compute_cov(patchesInput,rank)
        iid_error_10.append(cov_error(covMat,covMat_clean))
        iid_psnrs_10.append(cov_to_psnrs(covMat,pClean,pNoisyAtPi,
                                         sigma2,rank,thresh,is_yuv))
        # iid_psnrs_10.append(cov_to_psnrs(covMat,pClean,patchesInput,
        #                                  sigma2,rank,thresh,is_yuv))

        # -- compute iid error for noisy images --
        patchesInput = iid_patches[:,:npatches_i]
        covMat,_,_ = compute_cov(patchesInput,rank)
        iid_error_noisy.append(cov_error(covMat,covMat_clean))
        iid_psnrs_noisy.append(cov_to_psnrs(covMat,pClean,pNoisyAtPi,
                                            sigma2,rank,thresh,is_yuv))

    # -- print message --
    print("noisy: ",noisy_psnrs)
    print("sorted: ",sorted_psnrs)
    print("gsorted: ",gsorted_psnrs)
    print("gt-cov: ",cc_psnrs)
    print("noisy-clean: ",nc_psnrs)
    print("iid: ",iid_psnrs)

    # -- summary stats --
    n_means,n_stds = error_stats(noisy_error)
    s_means,s_stds = error_stats(sorted_error)
    gs_means,gs_stds = error_stats(gsorted_error)
    nc_means,nc_stds = error_stats(nc_error)
    cn_means,cn_stds = error_stats(cn_error)
    cc_means,cc_stds = error_stats(cc_error)
    i_means,i_stds = error_stats(iid_error)
    i1_means,i1_stds = error_stats(iid_error_1)
    i10_means,i10_stds = error_stats(iid_error_10)
    in_means,in_stds = error_stats(iid_error_noisy)

    # -- summary stats --
    noisy_psnrs = psnr_stats(noisy_psnrs)
    sorted_psnrs = psnr_stats(sorted_psnrs)
    nc_psnrs = psnr_stats(nc_psnrs)
    cc_psnrs = psnr_stats(cc_psnrs)
    gsorted_psnrs = psnr_stats(gsorted_psnrs)
    iid_psnrs = psnr_stats(iid_psnrs)
    iid_psnrs_1 = psnr_stats(iid_psnrs_1)

    # -- postfix_string --
    yuv_str = "yuv" if is_yuv else "rgb"
    postfix = f"{yuv_str}"

    # -- plot the error --
    fig,ax = plt.subplots()
    ax.errorbar(prange,n_means,yerr=1.96*n_stds,fmt='x-',label='noisy')
    ax.errorbar(prange,s_means,yerr=1.96*s_stds,fmt='o-',label='sorted')
    ax.errorbar(prange,gs_means,yerr=1.96*gs_stds,fmt='o-',label='gsorted')
    ax.errorbar(prange,nc_means,yerr=1.96*nc_stds,fmt='o-',label='noisy-clean')
    ax.errorbar(prange,cn_means,yerr=1.96*cn_stds,fmt='o-',label='clean-noisy')
    ax.errorbar(prange,cc_means,yerr=1.96*cc_stds,fmt='o-',label='clean-clean')
    ax.errorbar(prange,i_means,yerr=1.96*i_stds,fmt='+-',label='iid')
    ax.errorbar(prange,i1_means,yerr=1.96*i1_stds,fmt='+-',label='iid_1')
    ax.errorbar(prange,i10_means,yerr=1.96*i10_stds,fmt='+-',label='iid_10')
    ax.legend()
    plt.savefig(f"cov_error_{postfix}.png",transparent=True,bbox_inches='tight')
    plt.clf()
    plt.cla()
    plt.close("all")

    # -- plot psnrs v.s. error --
    fig,ax = plt.subplots()
    ax.errorbar(sorted_psnrs,s_means,yerr=1.96*s_stds,
                fmt='o-',label='sorted')
    ax.errorbar(nc_psnrs,nc_means,yerr=1.96*nc_stds,
                fmt='o-',label='noisy-clean')
    ax.errorbar(gsorted_psnrs,gs_means,yerr=1.96*nc_stds,
                fmt='o-',label='grad-sorted')
    # ax.errorbar(cn_psnrs,cn_means,yerr=1.96*cn_stds,
    #             fmt='o-',label='clean-noisy')
    # ax.errorbar(cc_psnrs,cc_means,yerr=1.96*cc_stds,
    #             fmt='o-',label='gt-cov')
    ax.errorbar(iid_psnrs,i_means,yerr=1.96*i_stds,
                fmt='+-',label='iid')
    ax.errorbar(iid_psnrs_1,i1_means,yerr=1.96*i1_stds,
                fmt='+-',label='iid_1')
    # ax.errorbar(iid_psnrs_10,i10_means,yerr=1.96*i10_stds,
    #             fmt='+-',label='iid_10')
    # ax.errorbar(iid_psnrs_noisy,in_means,yerr=1.96*in_stds,
    #             fmt='+-',label='iid_noisy')
    ax.errorbar(noisy_psnrs,n_means,yerr=1.96*n_stds,
                fmt='x-',label='noisy')
    ax.legend()
    plt.savefig(f"coverror_v_psnr_{postfix}.png",transparent=True,bbox_inches='tight')
    plt.clf()
    plt.cla()
    plt.close("all")

def compute_cov(patches,rank,weights = None):
    if not(weights is None):
        # weights = weights / weights[:,[0]]
        print(weights.shape,patches.shape)
        patches = patches * weights[:,:,None]

    # -- zm method 1 --
    # mpatch = patches.mean(dim=(1,2),keepdim=True)
    # mpatch = patches.mean(dim=1,keepdim=True)
    # mpatch = patches.mean(dim=2,keepdim=True)
    # patches_zm = patches - mpatch
    # patches_zm = patches_zm - patches_zm.mean(dim=2,keepdim=True)

    # -- zm method 2 --
    bsize,num,pdim = patches.shape
    mpatch1 = patches.mean(dim=1,keepdim=True)
    if num == 1: mpatch1 = torch.zeros_like(mpatch1)
    mpatch1 = torch.zeros_like(mpatch1)
    mpatch2 = (patches - mpatch1).mean(dim=2,keepdim=True)
    if pdim == 1: mpatch2 = torch.zeros_like(mpatch2)
    # mpatch2 = torch.zeros_like(mpatch2)
    patches_zm = patches - mpatch1 - mpatch2

    # -- compute cov mat --
    rscale = 1.
    covMat,a,b = compute_cov_mat(patches_zm/1.,rank)

    return covMat,a,b

def psnr_stats(errors):
    errors = torch.stack(errors)
    return errors[:,0]

def error_stats(errors):
    errors = torch.stack(errors)
    sqrt_size = np.sqrt(errors.shape[1])
    means = torch.mean(errors,dim=1).cpu().numpy()
    stderrs = torch.std(errors,dim=1).cpu().numpy()/sqrt_size
    return means,stderrs

def cov_error(covMat,covMat_clean):
    error = torch.mean(torch.abs(covMat-covMat_clean),dim=(1,2))
    return error

def iid_noisy(pclean,sigma):
    noise = sigma*torch.randn(*pclean.shape)
    return pclean + noise.to(pclean.device)
