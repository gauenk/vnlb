"""
Compare the Python API with the C++ Results

"""

# -- python imports --
import os,sys
import numpy as np
import torch as th
import pandas as pd
from collections import defaultdict
from easydict import EasyDict as edict

# -- this package --
import vnlb

# -- local imports --
from vnlb.testing.data_loader import load_dataset


#
#  ---------------- Compare Results ----------------
#

def run_comparison(vnlb_dataset):

    print(f"Running Comparison on {vnlb_dataset}")

    # -- run method using different image IOs when computing Optical Flow  --
    res_vnlb,paths,fmts = load_dataset(vnlb_dataset)
    clean,noisy,sigma = res_vnlb.clean,res_vnlb.noisy,res_vnlb.std

    # -- python denoiser --
    deno,basic,dtime = vnlb.denoise(noisy,sigma)
    res_pyvnlb = edict({'denoised':deno,'basic':basic,'time':dtime})

    # -- compare results --
    results = edict({'basic':edict(),'denoised':edict()})
    for field in results.keys():
        cppField = res_vnlb[field]
        pyField = res_pyvnlb[field].cpu().numpy()
        relError = np.mean(np.abs(cppField - pyField)/(np.abs(cppField)+1e-10))
        rkey = f"Ave Rel. Error ({field})"
        results[field][rkey] = relError

        # -- psnrs --
        cpp = vnlb.utils.compute_psnrs(cppField,clean).mean().item()
        py = vnlb.utils.compute_psnrs(pyField,clean).mean().item()
        rkey = "cpp_psnr"
        results[field][rkey] = cpp
        rkey = "py_psnr"
        results[field][rkey] = py
        rkey = "Abs. Error (PSNR)"
        results[field][rkey] = np.abs(cpp - py)
        rkey = "Rel. Error (PSNR)"
        results[field][rkey] = np.abs(cpp - py)/np.abs(cpp)
    results = pd.DataFrame(results)
    print(results.to_markdown())

#
# -- Comparison Code --
#

def run_method(noisy,sigma):

    #
    #  ---------------- Video Non-Local Bayes ----------------
    #


    #
    #  ---------------- Add Noisy Images to Show IO Changes ----------------
    #

    results = edict()
    results.denoised = deno
    results.basic = basic
    results.time = dtime

    return results

if __name__ == "__main__":

    # -- set seed --
    seed = 123
    np.random.seed(seed)
    th.manual_seed(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False

    # -- dataset example 1 --
    vnlb_dataset = "davis_64x64"
    run_comparison(vnlb_dataset)

    # -- dataset example 2 --
    # vnlb_dataset = "davis"
    # run_comparison(vnlb_dataset)

    # -- dataset example 3 --
    # vnlb_dataset = "gmobil"
    # run_comparison(vnlb_dataset)

