
import torch
from einops import rearrange,repeat

# -- local imports --
from .init_mask import update_mask_inds

def trim_sims(inds,mask,pGroups,trim_breaks,bsize):

    # -- shapes --
    nstreams,bsize,npatches = inds.shape
    pNoisy = pGroups[0]
    tsize,npatches,ps_t,c,ps,ps = pNoisy.shape
    nstreams = tsize//bsize
    # tsize,npatches,ps_t,c,ps,ps = pBasic.shape
    # tsize,npatches,ps_t,c,ps,ps = pClean.shape

    # -- marks --
    g_labels = torch.zeros((tsize),dtype=torch.int32)
    rm_labels = torch.zeros((tsize),dtype=torch.int32)
    mark_same(g_labels,inds)
    mark_remove(rm_labels,g_labels)
    remove_inds(rm_labels,inds)

    # -- modify inds --
    nvalid = reorder_invalids(inds,pGroups)
    print("nvalid: ",nvalid)
    print("tsize: ",tsize)
    nbatches = compute_nbatches(inds,nvalid,nstreams,bsize,thresh=.5)
    # set_trim_breaks(trim_breaks,nstreams,nbatches)

    # -- correct mask for batching --
    inds = rearrange(inds,'(s b) n -> s b n',b=bsize)
    # reset_skipped_mask(inds,mask,trim_breaks)


def mark_same(labels,inds):
    # mark_same_simple(labels,inds)
    mark_same_topk(labels,inds,k=100)

def mark_same_simple(labels,inds):
    first = inds[:,0]
    labels[...] = first

def mark_same_topk(labels,inds,k=3):
    topk = inds[:,:k]
    pwd = topk[:,None] - topk[None,]
    same = torch.any(pwd==0,2)
    args = torch.max(same,1).indices
    labels[...] = args

def mark_remove(rm_labels,g_labels):
    uniques = torch.unique(g_labels)
    for n in range(uniques.shape[0]):
        unique = uniques[n]
        args = torch.where(unique == g_labels)[0][1:]
        rm_labels[args] = 1

def remove_inds(rm_labels,inds):
    args = torch.where(rm_labels==1)[0]
    inds[args,...] = -1

def reorder_invalids(inds,patch_groups):

    # -- get args --
    valid_args = torch.where(torch.all(inds!=-1,1))[0]
    invalid_args = torch.where(torch.all(inds==-1,1))[0]
    args = torch.cat([valid_args,invalid_args])
    nvalid = valid_args.shape[0]

    # -- reorder --
    inds[...] = inds[args]

    # -- apply to groups --
    for pgroup in patch_groups:
        pgroup[...] = pgroup[args]

    return nvalid

def compute_nbatches(inds,nvalid,nstreams,bsize,thresh=.5):

    # -- num full batches --
    full_batches = nvalid // bsize
    if full_batches == 0: full_batches = 1

    # -- percent of last batch --
    perc_batch = (nvalid % bsize) / (1. * bsize)
    # print("perc_batch: ",perc_batch)
    if perc_batch > thresh: last_batch = 1
    else: last_batch = 0

    # -- num of batches --
    nbatches = full_batches + last_batch
    nbatches = min(nbatches,nstreams)

    return nbatches

def set_trim_breaks(trim_breaks,nstreams,nbatches):

    # -- skip not full batches --
    nskip = nstreams - nbatches
    for idx in range(nskip):
        bidx = nbatches + idx
        trim_breaks[bidx] = True

def reset_skipped_mask(inds,mask,trim_breaks,chnls=3):
    """

    we skip but the "access.shape" has become zero so we "tripped"
    the final

    """

    # -- reset skipped mask --
    for idx,skip in enumerate(trim_breaks):
        if skip is False: continue
        inds_s = inds[idx]
        args = torch.where(torch.all(inds_s != -1,1))[0]
        inds_s = inds_s[args]
        if inds_s.shape[0] == 0: continue

        prev_nmask = mask.sum().item()
        update_mask_inds(mask,inds_s,chnls,0,boost=False,val=1)
        post_nmask = mask.sum().item()
        delta_nmask = post_nmask - prev_nmask
        print("delta nmask: ",delta_nmask)
