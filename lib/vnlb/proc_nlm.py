
# -- python deps --
import copy,torch
import torch as th
from einops import rearrange,repeat
from easydict import EasyDict as edict

# -- package --
import vnlb

# -- local imports --
from .init_mask import initMask,mask2inds,update_mask,update_mask_inds
from .flat_areas import run_flat_areas
from .sim_search import sim_search_batch
from .bayes_est import bayes_estimate_batch
from .comp_agg import compute_agg_batch
from .qr_est import qr_estimate_batch
from .trim_sims import trim_sims
from .means_impl import means_estimate_batch

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
from vnlb.gpu.sim_search.streams import init_streams,wait_streams,get_hw_batches
from vnlb.gpu.sim_search.streams import view_batch,vprint,get_nbatches


def processNLMeans(noisy,basic,sigma,step,flows,params,gpuid=0,clean=None):
    """

    A Python implementation for one step of the NLBayes code

    """

    # -- place on cuda --
    device = gpuid
    if not(th.is_tensor(noisy)):
        noisy = th.FloatTensor(noisy).to(device)
        zero_basic = th.zeros_like(noisy)
        basic = optional(basic,'basic',zero_basic)
        basic = basic.to(device)

    # -- init outputs --
    shape = noisy.shape
    t,c,h,w = noisy.shape
    deno = th.zeros_like(noisy)
    nstreams = int(optional(params,'nstreams',[1,1])[step])
    flows = edict({k:th.FloatTensor(v).to(device) for k,v in flows.items()})

    # -- to device flow --
    flows = edict({k:th.FloatTensor(v).to(device) for k,v in flows.items()})
    zflow = torch.zeros((t,2,h,w)).to(device)
    fflow = optional(flows,'fflow',zflow)
    bflow = optional(flows,'bflow',zflow)

    # -- move clean to cuda --
    if not(clean is None):
        if not(torch.is_tensor(clean)):
            clean = torch.FloatTensor(clean).to(device)

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
    offset = 2*(sigma/255.)**2
    # offset = 0.

    # -- new args --
    clean_srch = int(optional(params,'cleanSearch',[False,False])[step])
    nfilter = int(optional(params,'nfilter',[-1,-1])[step])

    # -- create mask --
    mask = initMask(noisy.shape,params,step)['mask']
    mask = torch.ByteTensor(mask).to(device)

    # -- run the step --
    npatches = 20
    exec_step(noisy,basic,deno,mask,fflow,bflow,sigma2,sigmab2,rank,ps,
              ps_t,npatches,step_s,w_s,nWt_f,nWt_b,group_chnls,couple_ch,
              thresh,flat_areas,gamma,offset,step,nstreams,clean,clean_srch,nfilter)

    # -- format outputs --
    results = edict()
    results.basic = basic
    results.denoised = deno
    results.ngroups = npatches

    return results

def exec_step(noisy,basic,deno,mask,fflow,bflow,sigma2,sigmab2,rank,ps,ps_t,npatches,
              step_s,w_s,nWt_f,nWt_b,group_chnls,couple_ch,thresh,flat_areas,gamma,
              offset,step,nstreams,clean,clean_srch,nfilter):

    """
    ** Our "simsearch" is not the same as "vnlm" **

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

    # -- init tensors --
    deno = basic if step == 0 else deno
    # weights = th.zeros((nframes,height,width),dtype=th.float32)

    # -- color xform --
    noisy_yuv = apply_color_xform_cpp(noisy)
    basic_yuv = apply_color_xform_cpp(basic)
    if not(clean is None):
        clean_yuv = apply_color_xform_cpp(clean)
    else:
        clean_yuv = None

    # -- search region aliases --
    w_t = min(nWt_f + nWt_b + 1,nframes-1)
    nsearch = w_s * w_s * w_t

    # -- batching height and width --
    bsize,ssize = 256*nstreams,1
    nelems = torch.sum(mask).item()
    nbatches = divUp(divUp(nelems,nstreams),bsize)
    nmasked = 0
    valid_clean = not(clean is None)

    # -- synch before start --
    curr_stream = 0
    torch.cuda.synchronize()
    bufs,streams = init_streams(curr_stream,nstreams,device)

    # -- create shell --
    ns,np,t,c = nstreams,npatches,nframes,chnls
    tf32,ti32 = torch.float32,torch.int32
    patchesNoisy = torch.zeros(nstreams,bsize,np,ps_t,c,ps,ps).type(tf32).to(device)
    patchesBasic = torch.zeros(nstreams,bsize,np,ps_t,c,ps,ps).type(tf32).to(device)
    patchesClean = torch.zeros(nstreams,bsize,np,ps_t,c,ps,ps).type(tf32).to(device)
    inds = -torch.ones(nstreams,bsize,np).type(torch.int32).to(device)
    vals = torch.zeros(nstreams,bsize,np).type(tf32).to(device)
    weights = torch.zeros(nframes,height,width).type(tf32).to(device)
    flat_patches = torch.zeros(nstreams,bsize).type(ti32).to(device)

    # -- print statements --
    # mins = noisy.min(0).values.min(1).values.min(1).values
    # maxs = noisy.max(0).values.max(1).values.max(1).values
    # print("noisy: ",mins,maxs)
    # mins = noisy_yuv.min(0).values.min(1).values.min(1).values
    # maxs = noisy_yuv.max(0).values.max(1).values.max(1).values
    # print("noisy_yuv: ",mins,maxs)
    access_breaks = [False,]*nstreams

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
        inds[...] = -1
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
            if access.shape[0] == 0:
                access_breaks[cs] = True
                continue

            # -- select data for stream --
            patchesNoisy_s = patchesNoisy[cs]
            patchesBasic_s = patchesBasic[cs]
            patchesClean_s = patchesClean[cs]
            vals_s = vals[cs]
            inds_s = inds[cs]

            # import time
            # start = time.perf_counter()
            # -- sim_search_block --
            sim_search_batch(noisy_yuv,basic_yuv,clean_yuv,patchesNoisy_s,
                             patchesBasic_s,patchesClean_s,access,
                             vals_s,inds_s,fflow,bflow,step_s,bsize,ps,
                             ps_t,w_s,nWt_f,nWt_b,step==0,offset,cs,cs_ptr,
                             clean_srch=clean_srch,nfilter=nfilter)
            # nzeros = torch.sum(patchesNoisy_s==0).item()
            # end = time.perf_counter() - start
            # print("simsearch: ",end)

            # -- update mask --
            # inds_rs = rearrange(inds,'s b n -> (s b) n')
            # print("cs_ptr: ",cs_ptr)
            prev_masked = mask.sum().item()
            inds_s = inds_s[:,:1]
            update_mask_inds(mask,inds_s,chnls,cs_ptr,boost=False)
            curr_masked = mask.sum().item()
            delta = prev_masked - curr_masked
            nmasked += prev_masked - curr_masked
            # print("[%d/%d]: %d" % (nmasked,nelems,delta))

        # ----------------------------------------------
        #
        # -> trim some repetition from "sim searches" <-
        #
        # ----------------------------------------------

        # -- synch --
        torch.cuda.synchronize()
        trim_breaks = [False,]*nstreams
        # trim_sims(inds,mask,patchesNoisy,patchesBasic,trim_breaks)
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

        # -- shape entire batch --
        vals_rs = rearrange(vals,'s b n -> (s b) n')
        inds_rs = rearrange(inds,'s b n -> (s b) n')
        shape_str = 's b n pt c ph pw -> (s b) n pt c ph pw'
        patchesNoisy_rs = rearrange(patchesNoisy,shape_str)
        patchesBasic_rs = rearrange(patchesBasic,shape_str)
        patchesClean_rs = rearrange(patchesClean,shape_str)
        flat_patch_rs = rearrange(flat_patches,'s b -> (s b)')

        # -- compute valid bool --
        ivalid = torch.where(torch.all(inds_rs!=-1,1))[0]
        vals_rs = vals_rs[ivalid]
        inds_rs = inds_rs[ivalid]
        patchesNoisy_rs = patchesNoisy_rs[ivalid]
        patchesBasic_rs = patchesBasic_rs[ivalid]
        patchesClean_rs = patchesClean_rs[ivalid]
        if inds_rs.shape[0] == 0:
            break
        # print("[proc_nlb] nvalid: ",inds_rs.shape)
        # print("patchesNoisy_rs.shape: ",patchesNoisy_rs.shape)
        # print("patchesBasic_rs.shape: ",patchesBasic_rs.shape)

        # -- optional flat patch --
        # if flat_areas:
        #     run_flat_areas(flat_patch_rs,patchesNoisy_s,gamma,sigma2)

        # -- bayes denoising --
        rank_var = means_estimate_batch(patchesNoisy_rs,patchesBasic_rs,
                                        patchesClean_rs,vals_rs,
                                        sigma2,sigmab2,rank,group_chnls,thresh,
                                        step==1,valid_clean,flat_patch_rs,cs,cs_ptr)

        # -- aggregate results --
        inds_rs = inds_rs[:,:1]
        patchesNoisy_rs = patchesNoisy_rs[:,:1]
        compute_agg_batch(deno,patchesNoisy_rs,inds_rs,weights,ps,ps_t,cs_ptr)

        # ----------------------------------
        #            MISC
        # ----------------------------------

        # -- aggregate results --
        # shape_str = 's b n pt c ph pw -> (s b) n pt c ph pw'
        # patchesNoisy_rs = rearrange(patchesNoisy,shape_str)
        # shape_str = 's b n -> (s b) n '
        # inds_rs = rearrange(inds,shape_str)
        # compute_agg_batch(deno,patchesNoisy_rs,inds_rs,weights,ps,ps_t,cs_ptr)

        # -- wait for all streams --
        torch.cuda.synchronize()

        # -- break if complete --
        if any(access_breaks): break

        # -- aggregate across streams --
        # inds_rs = rearrange(inds,'s b n -> (s b) n')
        # prev_masked = mask.sum().item()
        # update_mask_inds(mask,inds_rs,nframes,chnls,height,width)
        # curr_masked = mask.sum().item()
        # delta = prev_masked - curr_masked
        # nmasked += prev_masked - curr_masked
        # print("[%d/%d]: %d" % (nmasked,nelems,delta))
        # if delta < bsize:
        #     print(mask2inds(mask,bsize))


    # -- reweight --
    # print("noisy: ",noisy.min(),noisy.max(),noisy.shape)
    # print("noisy_yuv: ",noisy_yuv.min(),noisy_yuv.max(),noisy_yuv.shape)
    # print("all: ",torch.all(weights>0))
    # print("weights: ",weights.min(),weights.max(),weights.shape)
    # print("deno: ",deno.min(),deno.max(),deno.shape)
    # wmax = weights.max().item()
    # save_images("gpu_weights.png",weights.cpu().numpy()[:,None],imax=wmax)
    # wmax = weights.max().item()
    # weights = repeat(weights,'t h w -> t c h w',c=chnls)
    # print(weights[0,0,:3,:3])
    # print(weights[0,0,8:10,8:10])
    # print("wmax: ",wmax)

    # -- reweight --
    weights = repeat(weights,'t h w -> t c h w',c=chnls)
    index = torch.nonzero(weights,as_tuple=True)
    deno[index] /= weights[index]

    # -- yuv 2 rgb --
    if not(use_imread):
        yuv2rgb_cpp(deno)
    # print("[post-2] deno: ",deno.min(),deno.max())
    torch.cuda.synchronize()


def check_steps(step1,step):
    is_step_1 = (step1 == True) and (step == 0)
    is_not_step_1 = (step1 == False) and (step == 1)
    assert is_step_1 or is_not_step_1
