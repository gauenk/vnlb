
# -- python deps --
import copy
import torch as th
from easydict import EasyDict as edict

# -- subroutines --
# from .sim_search import runSimSearch
from .bayes_est import runBayesEstimate
from .comp_agg import computeAggregation

# -- project imports --
from vnlb.utils import groups2patches,patches2groups,optional

def estimateSimPatches(noisy,basic,sigma,pidx,flows,params,step):

    # -- unpack --
    t,c,h,w = noisy.shape
    psX = params.sizePatch[step]
    psT = params.sizePatchTime[step]

    # -- cpp exec --
    # sim_results = vnlb.simPatchSearch(noisy,sigma,pidx,flows,params,step)
    # sim_results = edict(sim_results)
    # groupsNoisy = sim_results.groupNoisy
    # groupsBasic = sim_results.groupBasic
    # indices = sim_results.indices

    # sim_groupsNoisy = sim_results.groupNoisy
    # sim_groupsBasic = sim_results.groupBasic
    # sim_indices = sim_results.indices

    # -- sim search --
    params.use_imread = [False,False]
    tensors = edict({k:v for k,v in flows.items()})
    tensors.basic = basic
    sim_results = runSimSearch(noisy,sigma,pidx,tensors,params,step)

    patchesNoisy = sim_results.patchesNoisy
    patchesBasic = sim_results.patchesBasic
    indices = sim_results.indices
    nSimP = sim_results.nSimP
    ngroups = sim_results.ngroups
    groupsNoisy = patches2groups(patchesNoisy,c,psX,psT,ngroups,1)
    groupsBasic = patches2groups(patchesBasic,c,psX,psT,ngroups,1)

    # -- check -- # 563
    # delta = np.sum(np.sort(indices) - np.sort(sim_indices))
    # if delta >1e-3:
    #     print(pidx,step)
    #     print(np.stack([np.sort(indices),np.sort(sim_indices)],-1))
    # assert delta < 1e-3

    # py_order = np.argsort(indices)
    # sim_order = np.argsort(sim_indices)


    # py_patches = patchesNoisy[py_order]
    # sim_patches = groups2patches(sim_groupsNoisy,3,7,2,nSimP)[sim_order]
    # delta = np.abs(py_patches-sim_patches)
    # if np.any(delta>1e-3):
    #     print(np.unique(np.where(delta>1e-3)[0]))
    #     print(np.stack([py_patches[0],sim_patches[0]],-1))
    #     assert False


    # from vnlb.pylib.tests import save_images
    # print("patches.shape: ",patches.shape)
    # patches_rgb = yuv2rgb_cpp(patches)
    # save_images(patches_rgb,"output/patches.png",imax=255.)

    return groupsNoisy,groupsBasic,indices

def computeBayesEstimate(groupNoisy,groupBasic,nSimP,shape,params,step,flatPatch):

    # -- prepare --
    rank_var = 0.

    # -- exec --
    # bayes_results = vnlb.computeBayesEstimate(groupNoisy.copy(),
    #                                             groupBasic.copy(),0.,
    #                                             nSimP,shape,params,step)
    bayes_results = runBayesEstimate(groupNoisy.copy(),groupBasic.copy(),
                                     rank_var,nSimP,shape,params,step,flatPatch)

    # -- format --
    groups = bayes_results['groupNoisy']
    rank_var = bayes_results['rank_var']


    return groups,rank_var

def computeAgg(deno,groupNoisy,indices,weights,mask,nSimP,params,step):

    # -- cpp version --
    # params.isFirstStep[step] = step == 0
    # results = vnlb.computeAggregation(deno,groupNoisy,
    #                                     indices,weights,
    #                                     mask,nSimP,params)
    # deno = results['deno']
    # mask = results['mask']
    # weights = results['weights']
    # nmasked = results['nmasked']

    # -- python version --
    agg_results = computeAggregation(deno,groupNoisy,indices,weights,mask,
                                     nSimP,params,step)
    deno = agg_results['deno']
    weights = agg_results['weights']
    mask = agg_results['mask']
    nmasked = agg_results['nmasked']

    return deno,weights,mask,nmasked

def weightedAggregation(deno,noisy,weights):
    gtz = np.where(weights > 0)
    eqz = np.where(weights == 0)
    for c in range(deno.shape[1]):
        deno[gtz[0],c,gtz[1],gtz[2]] /= weights[gtz]
        # deno[gtz[0],c,gtz[1],gtz[2]] = 0#weights[gtz]
        deno[eqz[0],c,eqz[1],eqz[2]] = noisy[eqz[0],c,eqz[1],eqz[2]]


