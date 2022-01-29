
# -- python deps --
import torch
import torch as th
from easydict import EasyDict as edict

# -- local imports --
from .sim_search import runSimSearch
from .bayes_est import runBayesEstimate
from .comp_agg import computeAggregation
from .proc_nlb import processNLBayes


# -- project imports --
from vnlb.utils import groups2patches,patches2groups

def runPyAlign(noisy,sigma,flows,params,gpuid=0,clean=None):

    # -- place on cuda --
    device = gpuid
    noisy = torch.FloatTensor(noisy).to(device)
    flows = edict({k:torch.FloatTensor(v).to(device) for k,v in flows.items()})
    basic = torch.zeros_like(noisy)

    # -- align loop --
    niters = 10
    quality = [-1.,]*niters
    for i in range(niters):
        noisy_i,sims_i = exec_alignment(noisy,basic,sigma,0,flows,params)
        if not(clean is None):
            quality[i] = compute_alignment_quality(clean,sims_i)

    # -- print message --
    if not(clean is None):
        print("quality alignment.")
        print(quality)

    return sims
