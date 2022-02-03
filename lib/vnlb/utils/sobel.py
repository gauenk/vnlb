
import torch as th
import torch.nn.functional as F
from einops import rearrange,repeat


def create_sobel_filter():
    # -- get sobel filter to detect rough spots --
    sobel = th.FloatTensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    sobel_t = sobel.t()
    sobel = sobel.reshape(1,3,3)
    sobel_t = sobel_t.reshape(1,3,3)
    weights = th.stack([sobel,sobel_t],dim=0)
    return weights

def apply_sobel_filter(image,thresh=True):
    if len(image.shape) == 3:
        image = image.unsqueeze(0)
    C = image.shape[-3]
    weights = create_sobel_filter()
    weights = weights.to(image.device)
    weights = repeat(weights,'b 1 h w -> b c h w',c=C)
    edges = F.conv2d(image,weights,padding=1,stride=1)
    edges = ( edges[:,0]**2 + edges[:,1]**2 ) ** (0.5)
    # if thresh is True:
    #     edges = rearrange(edges,'b ph pw -> b (ph pw)')
    #     edges = th.quantile(edges,0.9,dim=1,keepdim=True)
    #     edges = rearrange(edges,'b 1 -> b 1 1')
    #     # thresh = edges
    #     # zargs = th.nonzero(edges < thresh)
    #     # edges[zargs] = 0.
    #     # args = th.nonzero(edges > thresh)
    #     # edges[args] = 1.
    return edges

def apply_sobel_to_patches(patches,pshape):

    # -- reshape --
    pt,c,ph,pw = pshape
    bsize,num,dim = patches.shape
    shape_str = 'b n (pt c ph pw) -> (b n pt) c ph pw'
    patches = rearrange(patches,shape_str,pt=pt,c=c,ph=ph)

    # -- apply --
    edges = apply_sobel_filter(patches)

    # -- reshape --
    shape_str = '(b n pt) ph pw -> b n (pt ph pw)'
    edges = rearrange(edges,shape_str,b=bsize,n=num)
    edges = th.mean(edges,2)

    return edges

