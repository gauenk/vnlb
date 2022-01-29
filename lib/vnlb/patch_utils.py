# -- python --
import torch,math
import numpy as np
from einops import rearrange,repeat

# -- vision --
import torchvision.utils as tv_utils

# -- project --
import vnlb
from vnlb.utils.gpu_utils import apply_yuv2rgb


def save_patches(patches,postfix,c=3,pt=2,tindex=0,bindex=0):

    # -- reshape --
    if patches.dim() == 3:
        ps = int(math.sqrt(patches.shape[-1]//(pt*c)))
        shape_str = 'b n (pt c ph pw) -> b n pt c ph pw'
        patches = rearrange(patches,shape_str,c=c,pt=pt,ph=ps)

    # -- save as pt file --
    fn = f"./data/patches_{postfix}.pt"
    torch.save(patches.cpu(),fn)

    # -- get patches to save --
    patches = patches[bindex,:,tindex]

    # -- convert --
    apply_yuv2rgb(patches)

    # -- filename --
    fn = f"patches_{postfix}.png"

    # -- num row --
    N = patches.shape[0]
    nrow = int(math.sqrt(N))

    # -- save images from batch index [bindex] --
    tv_utils.save_image(patches/255.,fn,nrow=nrow)

def yuv2rgb_patches(patches,c=3,pt=2):

    # -- reshape --
    if patches.dim() > 3:
        shape_str = 'b n pt c ph pw -> (b c) n (pt ph pw)'
        patches = rearrange(patches,shape_str)

    # -- shapes --
    bc,n,pdim = patches.shape
    ps = int(np.sqrt(pdim//pt))
    b = bc//c

    # -- reshape --
    shape_str = '(b c) n (pt ph pw) -> (b n pt) c ph pw'
    patches = rearrange(patches,shape_str,b=b,pt=pt,ph=ps)

    # -- convert --
    apply_yuv2rgb(patches)

    # -- reshape --
    shape_str = '(b n pt) c ph pw -> b n (pt c ph pw)'
    patches = rearrange(patches,shape_str,b=b,pt=pt,ph=ps)

    return patches

def patches_psnrs(pDeno,pClean,imax=1.):

    # -- reshape if needed --
    if pDeno.dim() > 3:
        pDeno = rearrange(pDeno,'b n pt c ph pw -> b n (pt c ph pw)')
        pClean = rearrange(pClean,'b n pt c ph pw -> b n (pt c ph pw)')

    # -- shape & init --
    eps = 1e-8
    bsize,num,pdim = pDeno.shape

    # -- only 0th index --
    delta = (pDeno/imax - pClean/imax)**2
    # delta = (pDeno[:,:snum,:]/imax - pClean[:,:snum,:]/imax)**2
    # delta = rearrange(delta,'b n p -> (b n) p')
    delta = delta.cpu().numpy()
    delta = np.mean(delta,axis=2) + eps
    log_mse = np.ma.log10(1./delta).filled(-np.infty)
    psnrs = 10 * log_mse
    # psnrs = rearrange(psnrs,'(b n) -> b n',b=bsize)

    # -- ave psnr over all --
    # delta = (pDeno/imax - pClean/imax)**2
    # delta = delta.cpu().numpy()
    # delta = np.mean(delta,axis=2) + eps
    # log_mse = np.ma.log10(1./delta).filled(-np.infty)
    # psnrs = 10 * log_mse
    # psnrs = np.mean(psnrs,axis=1)

    return psnrs
