import torch as th
import numpy as np

def compute_psnrs_cuda(deno,clean,imax=255.):
    # -- check imax for warning --
    if np.isclose(imax,255.):
        mm = min(deno.max(),clean.max()).item()
        if mm < 10.:
            print("WARNING: [compute_psnr] imax = 255 but images.max ~= 1.")
    elif np.isclose(imax,1.):
        mm = min(deno.max(),clean.max()).item()
        if mm > 10.:
            print("WARNING: [compute_psnr] imax = 1. but images.max ~= 255.")

    # -- normalize --
    deno = deno/imax
    clean = clean/imax

    psnr = -10 * th.log10(((deno - clean) ** 2).mean(axis=(-3, -2, -1), keepdims=False))
    return psnr.cpu().numpy()


def compute_psnrs(deno,clean,imax=255.):
    if th.is_tensor(deno):
        deno = deno.cpu().numpy()
    if th.is_tensor(clean):
        clean = clean.cpu().numpy()

    # -- check imax for warning --
    if np.isclose(imax,255.):
        mm = min(deno.max(),clean.max())
        if mm < 10.:
            print("WARNING: [compute_psnr] imax = 255 but images.max ~= 1.")
    elif np.isclose(imax,1.):
        mm = min(deno.max(),clean.max())
        if mm > 10.:
            print("WARNING: [compute_psnr] imax = 1. but images.max ~= 255.")

    # -- normalize --
    deno = deno/imax
    clean = clean/imax

    psnr = -10 * np.log10(((deno - clean) ** 2).mean(axis=(-3, -2, -1), keepdims=False))
    return psnr

