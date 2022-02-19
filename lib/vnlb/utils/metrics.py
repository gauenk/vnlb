import torch as th
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

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

def skimage_psnr(clean, noisy, normalized=True, raw=False):
    """Use skimage.meamsure.compare_ssim to calculate SSIM
    Args:
        clean (Tensor): (B, 1, H, W)
        noisy (Tensor): (B, 1, H, W)
        normalized (bool): If True, the range of tensors are [0., 1.]
            else [0, 255]
    Returns:
        SSIM per image: (B, )
    """
    if normalized:
        clean = clean.mul(255).clamp(0, 255)
        noisy = noisy.mul(255).clamp(0, 255)

    clean = clean.cpu().detach().numpy().astype(np.float32).transpose(0,2,3,1)
    noisy = noisy.cpu().detach().numpy().astype(np.float32).transpose(0,2,3,1)

    if raw:
        noisy = (np.uint16(noisy*(2**12-1-240)+240).astype(np.float32)-240)/(2**12-1-240)

    if normalized:
        return np.array([peak_signal_noise_ratio(c, n, data_range=255) for c, n in zip(clean, noisy)]).mean()
    else:
        return np.array([peak_signal_noise_ratio(c, n, data_range=1.0) for c, n in zip(clean, noisy)]).mean()



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

