

# -- python deps --
import torch
import torch as th
from easydict import EasyDict as edict

# -- local imports --
from .sim_search import runSimSearch
from .bayes_est import runBayesEstimate
from .comp_agg import computeAggregation
# from .proc_nlb import processNLBayes
from .proc_nlm import processNLMeans

# -- project imports --
from vnlb.utils import groups2patches,patches2groups
from vnlb.utils import idx2coords,compute_psnrs

def runNLMeans(noisy,clean,sigma,flows,params,gpuid=0):
    """

    A GPU-Python implementation of the Non-Local Means Method

    """

    # -- place on cuda --
    device = gpuid
    noisy = torch.FloatTensor(noisy).to(device)
    flows = edict({k:torch.FloatTensor(v).to(device) for k,v in flows.items()})
    basic = torch.zeros_like(noisy)

    # -- transfer to cuda --
    clean = torch.FloatTensor(clean).to(device)

    # -- step 1 --
    step_results = processNLMeans(noisy,basic,sigma,0,flows,params,
                                  gpuid=gpuid)#,clean=clean)
    # step_results = processNL(noisy,basic,sigma,0,flows,params,
    #                               gpuid=gpuid,clean=clean)
    step0_basic = step_results.basic.clone()


    # -- step 2 --
    alpha = 0.1
    basic = step0_basic

    psnrs = compute_psnrs(basic.cpu().numpy(),clean.cpu().numpy())
    print("psnrs: ",psnrs)

    niters = 30
    for i in range(niters):
        step_results = processNLMeans(noisy,basic,sigma,1,flows,params,
                                      gpuid=gpuid)#,clean=clean)
        basic = step_results.denoised.clone()
        basic = alpha * basic + ( 1 - alpha ) * noisy
        alpha = 1.2 * alpha
        if i == (niters-2): alpha = 1.
        psnrs = compute_psnrs(basic.cpu().numpy(),clean.cpu().numpy())
        print("psnrs: ",psnrs)
        print(alpha)
        if alpha > 1: break
    print(f"Exec for [{i}] iters.")
    basic = step_results.denoised


    # -- format --
    results = edict()
    results.basic = step0_basic
    results.denoised = basic
    results.ngroups = 0

    # -- oracle --
    step_results = processNLMeans(noisy,basic,sigma,1,flows,params,
                                  gpuid=gpuid,clean=clean)
    basic = step_results.denoised.clone()
    psnrs = compute_psnrs(basic.cpu().numpy(),clean.cpu().numpy())
    print("[oracle] psnrs: ",psnrs)

    return results



