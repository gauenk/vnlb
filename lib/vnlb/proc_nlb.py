
# -- python deps --
import copy,math
import torch
import torch as th
import numpy as np
from einops import rearrange,repeat
from easydict import EasyDict as edict

# -- package --
# import hids
import vnlb
from vnlb.testing.file_io import save_images

# -- local imports --
from .init_mask import initMask,mask2inds,update_mask,update_mask_inds
from .flat_areas import run_flat_areas
from .search import sim_search_batch
from .bayes_est import bayes_estimate_batch#,patches_psnrs,yuv2rgb_patches
from .comp_agg import compute_agg_batch
from .qr_est import qr_estimate_batch
from .trim_sims import trim_sims
from .means_impl import means_estimate_batch
from .explore_gp import explore_gp

# -- wrapped functions --
from .wrapped import weightedAggregation,computeAgg
from .wrapped import computeBayesEstimate,estimateSimPatches

from vnlb.utils import idx2coords,coords2idx,patches2groups,groups2patches
# from vnlb.utils import apply_color_xform_cpp,numpy_div0,yuv2rgb_cpp

# -- project imports --
from vnlb.utils.gpu_utils import apply_color_xform_cpp,yuv2rgb_cpp

# -- project imports --
from vnlb.utils import groups2patches,patches2groups,optional,divUp
from vnlb.testing import save_images

# -- streams
from .streams import init_streams,wait_streams,get_hw_batches
# from vnlb.gpu.streams import view_batch,vprint,get_nbatches


def view_batch(tensor,bsize,bidx):
    index = slice(bsize*bidx,bsize*(bidx+1))
    view = tensor[index]
    return view

def processNLBayes(noisy,basic,sigma,step,flows,params,gpuid=0,clean=None):
    """

    A Python implementation for one step of the NLBayes code

    """

    # -- place on cuda --
    device = gpuid
    if not(th.is_tensor(noisy)):
        noisy = th.FloatTensor(noisy).to(device)
        zero_basic = th.zeros_like(noisy)
        basic = zero_basic if basic is None else basic
        basic = basic.to(device)
    if not(clean is None):
        clean = th.FloatTensor(clean).to(device)

    # -- init outputs --
    shape = noisy.shape
    t,c,h,w = noisy.shape
    deno = th.zeros_like(noisy)
    nstreams = int(optional(params,'nstreams',[1,1])[step])
    # flows = edict({k:th.FloatTensor(v).to(device) for k,v in flows.items()})

    # -- to device flow --
    # flows = edict({k:th.FloatTensor(v).to(device) for k,v in flows.items()})
    zflow = torch.zeros((t,2,h,w)).to(device)
    fflow = optional(flows,'fflow',zflow)
    bflow = optional(flows,'bflow',zflow)

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
    sigma = params['sigma'][step]
    sigma2 = params['sigma'][step]**2
    beta = params['beta'][step]
    sigmaBasic2 = params['sigmaBasic'][step]**2
    sigmab2 = beta * sigmaBasic2 if step==1 else sigma**2
    rank =  params['rank'][step]
    thresh =  params['variThres'][step]
    flat_areas = params['flatAreas'][step]
    gamma = params['gamma'][step]
    step_s = params['procStep'][step]
    t,c,h,w = shape
    group_chnls = 1 if couple_ch else c
    step_s = 1
    # print("step_s: ",step_s)
    nkeep = int(optional(params,'simPatchRefineKeep',[-1,-1])[step])
    use_weights = int(optional(params,'useWeights',[False,False])[step])
    clean_srch = int(optional(params,'cleanSearch',[False,False])[step])
    offset = float(optional(params,'offset',[2*(sigma/255.)**2,0.])[step])
    # offset = float(optional(params,'offset',[0.,0.])[step])
    # bsize = int(optional(params,'bsize_s',[256,256])[step])
    bsize = int(optional(params,'bsize_s',[128,128])[step])
    nfilter = int(optional(params,'nfilter',[-1,-1])[step])

    # -- ints to bool --
    use_weights = True if use_weights == 1 else False
    clean_srch = True if clean_srch == 1 else False

    # -- create mask --
    mask = initMask(noisy.shape,params,step)['mask']
    mask = torch.ByteTensor(mask).to(device)

    # -- run the step --
    exec_step(noisy,basic,clean,deno,mask,fflow,bflow,sigma2,sigmab2,rank,ps,
              ps_t,npatches,step_s,w_s,nWt_f,nWt_b,group_chnls,couple_ch,
              thresh,flat_areas,gamma,offset,step,bsize,nstreams,nkeep,
              use_weights,clean_srch,nfilter)

    # -- format outputs --
    results = edict()
    results.basic = basic
    results.denoised = deno
    results.ngroups = npatches

    return results

def exec_step(noisy,basic,clean,deno,mask,fflow,bflow,sigma2,sigmab2,rank,
              ps,ps_t,npatches,step_s,w_s,nWt_f,nWt_b,group_chnls,couple_ch,
              thresh,flat_areas,gamma,offset,step,bsize,nstreams,nkeep,
              use_weights,clean_srch,nfilter):

    """
    ** Our "simsearch" is not the same as "vnlb" **

    1. the concurrency of using multiple cuda-streams creates io issues
       for using the mask
    2. if with no concurrency, the details of using an "argwhere" each batch
       seems strange
    3. it is unclear if we will want this functionality for future uses
       of this code chunk
    """

    # -- unpack info --
    use_imread = False
    device = noisy.device
    shape = noisy.shape
    nframes,chnls,height,width = noisy.shape
    sigma = math.sqrt(sigma2)
    sigmab = math.sqrt(sigmab2)

    # -- using the "patch subset" weighted values --
    # if nkeep != -1: use_weights = True
    # else: use_weights = False
    # print("noisy.shape: ",noisy.shape)

    # -- init tensors --
    if step == 0:
        deno = basic
        deno[...] = 0
    # weights = th.zeros((nframes,height,width),dtype=th.float32)

    # -- color xform --
    noisy_yuv = apply_color_xform_cpp(noisy)
    basic_yuv = apply_color_xform_cpp(basic)
    if not(clean is None):
        clean_yuv = apply_color_xform_cpp(clean)
    else: clean_yuv = None
    # print("clean is None: ",clean is None)

    # -- search region aliases --
    w_t = min(nWt_f + nWt_b + 1,nframes-1)
    nsearch = w_s * w_s * w_t

    # -- batching height and width --
    nstreams = 1
    tsize = bsize*nstreams
    nelems = torch.sum(mask).item()
    nbatches = divUp(divUp(nelems,nstreams),bsize)
    nmasked = 0

    # -- synch before start --
    curr_stream = 0
    torch.cuda.synchronize()
    bufs,streams = init_streams(curr_stream,nstreams,device)

    # -- create shell --
    # print("npatches: ",npatches)
    ns,npa,t,c = nstreams,npatches,nframes,chnls
    tf32,ti32 = torch.float32,torch.int32
    patchesNoisy = torch.zeros(tsize,npa,ps_t,c,ps,ps).type(tf32).to(device)
    patchesBasic = torch.zeros(tsize,npa,ps_t,c,ps,ps).type(tf32).to(device)
    patchesClean = torch.zeros(tsize,npa,ps_t,c,ps,ps).type(tf32).to(device)
    inds = -torch.ones(tsize,npa).type(torch.int32).to(device)
    vals = torch.zeros(tsize,npa).type(tf32).to(device)
    weights = torch.zeros(nframes,height,width).type(tf32).to(device)
    flat_patches = torch.zeros(tsize).type(ti32).to(device)
    access_breaks = [False,]*nstreams
    print("[proc] sigma: ",sigma)

    # -- over batches --
    for batch in range(nbatches):

        # -- break if complete --
        if any(access_breaks): break

        # ----------------------------------------------
        #
        #           -> similarity search <-
        #
        # ----------------------------------------------
        # print("batch: [%d/%d]" % (batch+1,nbatches))

        # -- reset --
        flat_patches[...] = 0

        # -- exec search --
        access_breaks = [False,]*nstreams
        for curr_stream in range(nstreams):

            # -- assign to stream --
            cs = curr_stream
            torch.cuda.set_stream(streams[cs])
            cs_ptr = streams[cs].cuda_stream

            # -- get indies from mask --
            access = mask2inds(mask,bsize)
            # print("access.shape: ",access.shape)
            if access.shape[0] == 0:
                access_breaks[cs] = True
                continue

            # -- select data for stream --
            patchesNoisy_s = view_batch(patchesNoisy,bsize,cs)
            patchesBasic_s = view_batch(patchesBasic,bsize,cs)
            patchesClean_s = view_batch(patchesClean,bsize,cs)
            vals_s = view_batch(vals,bsize,cs)
            inds_s = view_batch(inds,bsize,cs)

            # -- sim_search_block --
            inds_s[...] = -1
            # exec_patch_search(noisy_yuv,sigma,access,npatches,ps,
            #                   patches=patchesNoisy_s)
            sim_search_batch(noisy_yuv,basic_yuv,clean_yuv,sigma,sigmab,
                             patchesNoisy_s,patchesBasic_s,patchesClean_s,
                             access,vals_s,inds_s,fflow,bflow,step_s,bsize,
                             ps,ps_t,w_s,nWt_f,nWt_b,step==0,offset,cs,cs_ptr,
                             clean_srch=clean_srch,nfilter=nfilter)

            # -- update mask --
            prev_masked = mask.sum().item()
            update_mask_inds(mask,inds_s,chnls,cs_ptr,nkeep=nkeep)
            curr_masked = mask.sum().item()
            delta = prev_masked - curr_masked
            nmasked += prev_masked - curr_masked
            print("[%d/%d]: %d" % (nmasked,nelems,delta))

        # ----------------------------------------------
        #
        # -> trim some repetition from "sim searches" <-
        #
        # ----------------------------------------------

        # -- synch --
        torch.cuda.synchronize()
        trim_breaks = [False,]*nstreams
        pGroups = [patchesNoisy,patchesBasic]
        # trim_sims(inds,mask,pGroups,trim_breaks,bsize)
        trim_breaks = [False,]*nstreams
        # print("access_breaks: ",access_breaks)

        # -- update access_breaks --
        # for bidx in range(nstreams):
        #     if trim_breaks[bidx]:
        #         access_breaks[bidx] = False

        # -- verbose --
        # print(trim_breaks)
        # print(access_breaks)

        # ----------------------------------
        #
        #           Denoise
        #
        # ----------------------------------

        # -- compute valid bool --
        ivalid = torch.where(torch.all(inds!=-1,1))[0]
        print("ivalid.shape: ",ivalid.shape)
        vals_v = vals[ivalid]
        inds_v = inds[ivalid]
        patchesNoisy_v = patchesNoisy[ivalid]
        patchesBasic_v = patchesBasic[ivalid]
        patchesClean_v = patchesClean[ivalid]
        if inds_v.shape[0] == 0:
            break

        # -- optional flat patch --
        # if flat_areas:
        #     run_flat_areas(flat_patches,patchesNoisy,gamma,sigma2)

        # -- TODO: remove me!! [only use if clean case] --
        flat_patches_v = flat_patches[ivalid]
        # flat_patches_v[...] = 1

        # -- bayes denoising --
        # delta = patchesNoisy_v - patchesBasic_v
        # delta = torch.abs(delta)
        # delta = torch.sum(delta).item()
        # print("delta: %2.3f" % delta)
        inds_i = inds_v if use_weights else None
        # patchesNoisy_v_og = patchesNoisy_v.clone()

        rank_var = bayes_estimate_batch(patchesNoisy_v,patchesBasic_v,None,
                                        sigma2,sigmab2,rank,group_chnls,thresh,
                                        step==1,flat_patches_v,cs,cs_ptr,
                                        use_weights=use_weights,inds=inds_i)

        # -- aggregate results --
        compute_agg_batch(deno,patchesNoisy_v,inds_v,weights,ps,ps_t,cs_ptr)

        # ----------------------------------
        #            MISC
        # ----------------------------------

        # -- wait for all streams --
        torch.cuda.synchronize()

        # -- break if complete --
        if any(access_breaks): break
        # if (nelems - nmasked) < 50:
        #     print("breaking early: the last 50 pixels don't matter. :P")
        #     break

    # -- reweight --
    weights = repeat(weights,'t h w -> t c h w',c=chnls)
    index = torch.nonzero(weights,as_tuple=True)
    deno[index] /= weights[index]

    # -- inspect --
    # weights = weights.ravel().cpu()
    # print(torch.histogram(weights,50))

    # -- yuv 2 rgb --
    if not(use_imread):
        yuv2rgb_cpp(deno)
    torch.cuda.synchronize()


def check_steps(step1,step):
    is_step_1 = (step1 == True) and (step == 0)
    is_not_step_1 = (step1 == False) and (step == 1)
    assert is_step_1 or is_not_step_1
