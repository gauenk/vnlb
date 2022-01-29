
# -- python imports --
import torch
import torch as th
from einops import rearrange,repeat
from easydict import EasyDict as edict

# -- package imports --
from vnlb.utils import get_patch_shapes_from_params,optional,groups2patches,check_flows,check_and_expand_flows
from vnlb.utils.gpu_utils import apply_color_xform_cpp
from vnlb.testing import save_images

# -- import hids subsetting --
# import hids
from sim_search import compute_l2norm_cuda,fill_patches,fill_patches_img

# -- local imports --
from .init_mask import initMask,mask2inds,update_mask
from .streams import init_streams,wait_streams,get_hw_batches,view_batch,vprint
# from .subave_impl import compute_subset_ave
# from .l2norm_impl import compute_l2norm_cuda
# from .fill_patches import fill_patches,fill_patches_img
# from vnlb.gpu.patch_utils import yuv2rgb_patches,save_patches

#
# -- exec across concurrent streams --
#

def runSimSearch(noisy,sigma,tensors,params,step=0,gpuid=0):

    # -- move to device --
    noisy = th.FloatTensor(noisy).to(gpuid)
    device = noisy.device

    # -- extract info for explicit call --
    t,c,h,w = noisy.shape
    ps = params['sizePatch'][step]
    ps_t = params['sizePatchTime'][step]
    npatches = params['nSimilarPatches'][step]
    nwindow_xy = params['sizeSearchWindow'][step]
    nfwd = params['sizeSearchTimeFwd'][step]
    nbwd = params['sizeSearchTimeBwd'][step]
    nwindow_t = nfwd + nbwd + 1
    couple_ch = params['coupleChannels'][step]
    step1 = params['isFirstStep'][step]
    use_imread = params['use_imread'][step] # use rgb for patches or yuv?
    step_s = params['procStep'][step]
    basic = optional(tensors,'basic',th.zeros_like(noisy))
    assert ps_t == 2,"patchsize for time must be 2."
    nstreams = optional(params,'nstreams',1)
    rand_mask = optional(params,'rand_mask',True)
    offset = 2*(sigma/255.)**2


    # -- format flows for c++ (t-1 -> t) --
    if check_flows(tensors):
        check_and_expand_flows(tensors,t)

    # -- extract tensors --
    zflow = th.zeros((t,2,h,w),dtype=th.float32).to(gpuid)
    fflow = optional(tensors,'fflow',zflow.clone()).to(gpuid)
    bflow = optional(tensors,'bflow',zflow.clone()).to(gpuid)

    # -- color transform --
    noisy_yuv = apply_color_xform_cpp(noisy)
    basic_yuv = apply_color_xform_cpp(basic)
    clean_yuv = None

    # -- create mask --
    nframes,chnls,height,width = noisy.shape
    # mask = torch.zeros(nframes,height,width).type(torch.int8).to(device)
    mask = torch.ones(nframes,height,width).type(torch.int8).to(device)
    # mask = torch.ByteTensor(mask).to(device)

    # -- find the best patches using c++ logic --
    srch_img = noisy_yuv if step1 else basic_yuv
    results = exec_sim_search(srch_img,basic_yuv,clean_yuv,fflow,bflow,mask,
                              sigma,ps,ps_t,npatches,step_s,nwindow_xy,
                              nfwd,nbwd,couple_ch,step1,offset,nstreams,
                              rand_mask)
    patchesNoisy,patchesBasic,dists,indices = results
    dists = rearrange(dists,'nb b p -> (nb b) p')
    indices = rearrange(indices,'nb b p -> (nb b) p')

    # -- group the values and indices --
    img_noisy = noisy if use_imread else noisy_yuv
    img_basic = basic if use_imread else basic_yuv
    # patchesNoisy = fill_patches_img(img_noisy,indices,ps,ps_t)
    # patchesBasic = fill_patches_img(img_basic,indices,ps,ps_t)

    # -- groups from patches --
    # patchesNoisy = groups2patches(groupNoisy)
    # patchesBasic = groups2patches(groupBasic)
    # groupNoisy = groups2patches(patchesNoisy.cpu().numpy())
    # groupBasic = groups2patches(patchesBasic.cpu().numpy())
    groupNoisy = None
    groupBasic = None

    # -- extract some params info --
    i_params = edict({k:v[step] if isinstance(v,list) else v for k,v in params.items()})
    pinfo = get_patch_shapes_from_params(i_params,c)
    patch_num,patch_dim,patch_chnls = pinfo
    nsearch = nwindow_xy * nwindow_xy * (nfwd + nbwd + 1)

    # -- pack results --
    results = edict()
    results.patches = patchesNoisy
    results.patchesNoisy = patchesNoisy
    results.patchesBasic = patchesBasic
    results.groupNoisy = groupNoisy
    results.groupBasic = groupBasic
    results.indices = indices
    results.nSimP = len(indices)
    results.nflat = results.nSimP * ps * ps * ps_t * c
    results.values = dists
    results.nsearch = nsearch
    results.ngroups = patch_num
    results.npatches = len(patchesNoisy)
    results.ps = ps
    results.ps_t = ps_t
    results.psX = ps
    results.psT = ps_t
    results.access = None

    return results

def exec_sim_search(noisy,basic,clean,fflow,bflow,mask,sigma,ps,ps_t,
                    npatches,step_s,w_s,nWt_f,nWt_b,couple_ch,step1,
                    offset,nstreams,rand_mask=False):
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
    device = noisy.device
    t,c,h,w = noisy.shape

    # -- search region aliases --
    w_t = min(nWt_f + nWt_b + 1,t-1)
    nsearch = w_s * w_s * w_t

    # -- batching height and width --
    nelems = torch.sum(mask).item()
    bsize = min(4096,nelems)
    nbatches = divUp(nelems,bsize)

    # -- synch before start --
    curr_stream = 0
    torch.cuda.synchronize()
    bufs,streams = init_streams(curr_stream,nstreams,device)

    # -- create shell --
    ns,np = nstreams,npatches
    patchesNoisy = torch.zeros(nbatches,bsize,npatches,ps_t,c,ps,ps).to(device)
    patchesBasic = torch.zeros(nbatches,bsize,npatches,ps_t,c,ps,ps).to(device)
    vals = torch.zeros(nbatches,bsize,npatches).type(torch.float32).to(device)
    inds = -torch.ones(nbatches,bsize,npatches).type(torch.int32).to(device)
    # vals = torch.zeros(npatches,t,h,w).type(torch.float32).to(device)
    # inds = torch.zeros(npatches,t,h,w).type(torch.int32).to(device)

    # -- exec search --
    # print("nbatches: ",nbatches)
    # print("nelems: ",nelems)
    for batch in range(nbatches):

        # print("batch: ",batch)
        # -- assign to stream --
        cs = curr_stream
        torch.cuda.set_stream(streams[cs])
        cs_ptr = streams[cs].cuda_stream

        # -- grab access --
        access = mask2inds(mask,bsize,rand_mask)
        if access.shape[0] == 0: break

        # -- grab data for current stream --
        vals_s = vals[batch]
        inds_s = inds[batch]
        patchesNoisy_s = patchesNoisy[batch]
        patchesBasic_s = patchesBasic[batch]
        patchesClean_s = patchesBasic_s

        # -- sim search block --
        sim_search_batch(noisy,basic,clean,patchesNoisy_s,patchesBasic_s,
                         patchesClean_s,access,vals_s,inds_s,fflow,bflow,
                         step_s,bsize,ps,ps_t,w_s,nWt_f,nWt_b,
                         step1,offset,cs,cs_ptr)

        # -- update mask naccess --
        update_mask(mask,access)

        # -- change stream --
        if nstreams > 0: curr_stream = (curr_stream + 1) % nstreams

    # -- wait for all streams --
    torch.cuda.synchronize()

    return patchesNoisy,patchesBasic,vals,inds

def sim_search_batch(noisy,basic,clean,sigma,sigmab,patchesNoisy,patchesBasic,
                     patchesClean,access,vals,inds,fflow,bflow,step_s,bsize,
                     ps,ps_t,w_s,nWt_f,nWt_b,step1,offset,cs,cs_ptr,
                     clean_srch=True,nfilter=-1):


    # print("sim search [step1]: ",step1)
    # -- compute difference --
    srch_img = noisy if step1 else basic
    srch_img_str = "noisy" if step1 else "basic"
    if not(clean is None) and (clean_srch is True):
        srch_img_str = "clean"
        srch_img = clean
    # print(access)
    # print("noisy.shape: ",noisy.shape)
    # print("basic.shape: ",basic.shape)
    l2_vals,l2_inds = compute_l2norm_cuda(srch_img,fflow,bflow,access,step_s,ps,
                                           ps_t,w_s,nWt_f,nWt_b,step1,offset,cs_ptr)
    # -- get inds info --
    # nzero = torch.sum(l2_inds==0).item()
    # size = l2_inds.numel()
    # print("[sim_search: l2_inds] perc zero: %2.3f" % (nzero / size * 100))

    # nzero = torch.sum(l2_inds==-1).item()
    # size = l2_inds.numel()
    # print("[sim_search: l2_inds] perc invalid: %2.3f" % (nzero / size * 100))

    # -- get inds info --
    # nzero = torch.sum(inds==0).item()
    # size = inds.numel()
    # print("[sim_search: inds] perc zero: %2.3f" % (nzero / size * 100))

    # nzero = torch.sum(inds==-1).item()
    # size = inds.numel()
    # print("[sim_search: inds] perc invalid: %2.3f" % (nzero / size * 100))

    # -- filter down if we have a positive search num --
    # -- compute topk --
    get_topk(l2_vals,l2_inds,vals,inds)
    rpatches = None

    # -- compare --
    # ivalid = torch.where(torch.all(inds!=-1,1))[0]
    # if nfilter > 0 and len(ivalid) == 128 and not(clean is None):
    #     misc = [fflow,bflow,access,step_s,ps,ps_t,w_s,nWt_f,nWt_b,step1,offset,cs_ptr]
    #     compute_clean_inds_error(clean,l2_vals,l2_inds,inds,misc)

    # -- fill noisy patches --
    fill_patches(patchesNoisy,noisy,inds,cs_ptr)

    # -- fill basic patches --
    if not(step1): fill_patches(patchesBasic,basic,inds,cs_ptr)

    # -- fill clean patches --
    valid_clean = not(clean is None)
    valid_clean = valid_clean and not(patchesClean is None)
    if valid_clean: fill_patches(patchesClean,clean,inds,cs_ptr)

    # if not(rpatches is None):
    #     b = rpatches.shape[0]
    #     delta = torch.sum(torch.abs(patchesNoisy[:b]-rpatches)).sum()
    #     print("[noisy] delta: ",delta.item())
    #     delta = torch.sum(torch.abs(patchesClean[:b]-cpatches)).sum()
    #     print("[clean] delta: ",delta.item())

    # -- checking --
    # args = torch.where(torch.all(inds!=-1,1))[0]
    # if len(args) > 0 and len(args) != 36:
    #     inds_v = inds[args]
    #     access_v = access[args]
    #     idx_v = access_v[:,0] * 256*256*3 + access_v[:,1] * 256 + access_v[:,2]
    #     print(idx_v[:3])
    #     print(inds_v[:3,0])
    #     delta = torch.sum(torch.abs(inds_v[:,0]-idx_v)).item()
    #     print("delta: ",delta)
    #     if delta > 1.:
    #         diff = torch.abs(inds_v[:,0]-idx_v)
    #         args = torch.where(diff>1)[0]
    #         print(args)
    #         print(access_v[args])
    #         print(idx_v[args])
    #         print(inds_v[args,0])
    #     assert delta < 1.

# def compute_clean_inds_error(clean,l2_vals,l2_inds,h_inds,misc):

#     # -- clean search --
#     nkeep = h_inds.shape[1]
#     fflow,bflow,access,step_s,ps,ps_t,w_s,nWt_f,nWt_b,step1,offset,cs_ptr = misc
#     gt_vals,gt_inds = compute_l2norm_cuda(clean,fflow,bflow,access,step_s,ps,
#                                           ps_t,w_s,nWt_f,nWt_b,step1,0.,cs_ptr)

#     # -- compute top k --
#     print("nkeep: ",nkeep)
#     gt_vals,gt_inds = get_topk_pair(gt_vals,gt_inds,nkeep)
#     _,l2_inds = get_topk_pair(l2_vals,l2_inds,nkeep)
#     print("gt_inds.shape: ",gt_inds.shape)
#     print("l2_inds.shape: ",l2_inds.shape)

#     # -- inspect vals --
#     print("top: ",gt_vals[0])
#     print("gt_inds[0]: ",th.sort(gt_inds[0]).values)
#     print("l2_inds[0]: ",th.sort(l2_inds[0]).values)
#     print("h_inds[0]: ",th.sort(h_inds[0]).values)

#     # -- compare results --
#     B = gt_inds.shape[0]
#     for b in range(B):
#         l2_cmp = hids.compare_inds(gt_inds[[b]],l2_inds[[b]])
#         hids_cmp = hids.compare_inds(gt_inds[[b]],h_inds[[b]])
#         print("l2_cmp: %2.2f" % l2_cmp)
#         print("hids_cmp: %2.2f" % hids_cmp)
#     # h_inds[...] = gt_inds[...]
#     exit(0)

def get_topk_pair(vals_srch,inds_srch,k):
    device,b = vals_srch.device,vals_srch.shape[0]
    vals = torch.FloatTensor(b,k).to(device)
    inds = torch.IntTensor(b,k).to(device)
    get_topk(vals_srch,inds_srch,vals,inds)
    return vals,inds


def get_topk(l2_vals,l2_inds,vals,inds):

    # -- shape info --
    b,_ = l2_vals.shape
    _,k = vals.shape

    # -- take mins --
    # order = torch.topk(-l2_vals,k,dim=1).indices
    order = torch.argsort(l2_vals,dim=1,descending=False)
    # -- get top k --
    vals[:b,:] = torch.gather(l2_vals,1,order[:,:k])
    inds[:b,:] = torch.gather(l2_inds,1,order[:,:k])

# ------------------------------
#
#      Swap Tensor Dims
#
# ------------------------------

# -- swap dim --
def swap_2d_dim(tensor,dim):
    tensor = tensor.clone()
    tmp = tensor[0].clone()
    tensor[0] = tensor[1].clone()
    tensor[1] = tmp
    return tensor

def divUp(a,b): return (a-1)//b+1


