"""
Search for similar patches across batches
"""



# -- python imports --
import torch
import torch as th
from einops import rearrange,repeat
from easydict import EasyDict as edict

# -- [a required package] --
# from sim_search import compute_l2norm_cuda,fill_patches,fill_patches_img
# from sim_search import exec_search#compute_l2norm_cuda,fill_patches,fill_patches_img
import sim_search

# -- local package --
import vnlb.search_mask as search_mask

# -- utils --
from vnlb.utils.batching import view_batch
from vnlb.utils.logger import vprint
from vnlb.utils import divUp

# -- searching --
def exec_search(patches,imgs,flows,mask,bufs,args):

    # -- setup --
    bsize = args.bsize
    cs = th.cuda.default_stream()
    cs_ptr = th.cuda.default_stream().cuda_stream
    done,delta = False,0

    # -- smaller batch sizes impact quality --
    for index in range(args.nstreams):

        # -- grab access --
        srch_inds = search_mask.mask2inds(mask,bsize)
        if srch_inds.shape[0] == 0:
            done = True
            break

        # -- grab batch --
        vbufs = edict()
        for key in bufs.keys():
            vbufs[key] = view_batch(bufs[key],bsize,index)

        vpatches = edict()
        for key in patches.keys():
            vpatches[key] = view_batch(patches[key],bsize,index)

        # -- exec search --
        search_and_fill(imgs,vpatches,vbufs,srch_inds,flows,args)

        # -- update mask naccess --
        before = mask.sum().item()
        search_mask.update_mask_inds(mask,vbufs.inds,args.c)
        after = mask.sum().item()
        delta += before - after

        # -- wait for all streams --
        torch.cuda.synchronize()

    return done,delta

def search_and_fill(imgs,patches,bufs,srch_inds,flows,args):

    # -- select search image --
    srch_img = imgs.noisy if args.step == 0 else imgs.basic
    srch_img = srch_img if (imgs.clean is None) else imgs.clean

    # -- sim search block --
    sim_search.exec_sim_search_burst(srch_img,srch_inds,bufs.vals,
                                     bufs.inds,flows,args.sigma,args)
    # -- fill patches --
    for key in imgs.patch_images:

        # -- skip --
        pass_key = (imgs[key] is None) or (patches[key] is None)
        if pass_key: continue

        # -- fill --
        sim_search.fill_patches(patches[key],imgs[key],bufs.inds)

def exec_search_v2(noisy,basic,clean,fflow,bflow,mask,sigma,ps,ps_t,
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

def select_search_img(imgs,args):
    # -- compute difference --
    srch_img = imgs.noisy if args.step1 else imgs.basic
    srch_img_str = "noisy" if args.step1 else "basic"
    if not(imgs.clean is None) and (args.clean_srch is True):
        srch_img_str = "clean"
        srch_img = imgs.clean
    return srch_img,srch_img_str

# def sim_search(images,patches,l2_tensors,mask,params):


#     bsize = l2_tensors.inds.shape[0] # num to search at once
#     tsize = patches.noisy.shape[0] # num to search each call
#     nbatches = divUp(tsize,bsize)
#     for index in range(nbatches):

#         # -- access from mask --
#         access = select_mask_batch(mask)

#         # -- verify access --
#         if access.shape[0] == 0:
#             break

#         # -- search --
#         patches_i = view_tdict(patches)
#         l2_tensors_i = view_tdict(l2_tensors)
#         search_access(access,images,patches_i,l2_tensors_i,params)


# def search_access(access,imgs,patches,l2_tensors,params):

#     # noisy,basic,clean,sigma,sigmab,patchesNoisy,patchesBasic,
#     #                       patchesClean,access,vals,inds,fflow,bflow,step_s,bsize,
#     #                       ps,ps_t,w_s,nWt_f,nWt_b,step1,offset,cs,cs_ptr,
#     #                       clean_srch=True,nfilter=-1):

#     # -- compute l2 search --
#     srch_img,srch_img_str = select_search_img(imgs,params)
#     l2_vals,l2_inds = compute_l2norm_cuda(srch_img,flows.f,flows.b,access,
#                                           params.step_s,params.ps,params.ps_t,
#                                           params.w_s,params.nWt_f,params.nWt_b,
#                                           params.step1,params.offset,params.cs_ptr)
#     # -- compute topk --
#     vals = l2_tensors.vals
#     inds = l2_tensors.inds
#     get_topk(l2_vals,l2_inds,vals,inds)

#     # -- fill noisy patches --
#     fill_patches(patches.noisy,imgs.noisy,inds,cs_ptr)

#     # -- fill basic patches --
#     if not(step1): fill_patches(patches.basic,imgs.basic,inds,cs_ptr)

#     # -- fill clean patches --
#     valid_clean = not(clean is None)
#     valid_clean = valid_clean and not(patches.clean is None)
#     if valid_clean: fill_patches(patches.clean,imgs.clean,inds,cs_ptr)

# def get_topk_pair(vals_srch,inds_srch,k):
#     device,b = vals_srch.device,vals_srch.shape[0]
#     vals = torch.FloatTensor(b,k).to(device)
#     inds = torch.IntTensor(b,k).to(device)
#     get_topk(vals_srch,inds_srch,vals,inds)
#     return vals,inds

# def get_topk(l2_vals,l2_inds,vals,inds):

#     # -- shape info --
#     b,_ = l2_vals.shape
#     _,k = vals.shape

#     # -- take mins --
#     # order = torch.topk(-l2_vals,k,dim=1).indices
#     order = torch.argsort(l2_vals,dim=1,descending=False)
#     # -- get top k --
#     vals[:b,:] = torch.gather(l2_vals,1,order[:,:k])
#     inds[:b,:] = torch.gather(l2_inds,1,order[:,:k])


