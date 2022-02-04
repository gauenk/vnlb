import numpy as np

from vnlb.utils import patches2groups,groups2patches

def index2indices(index,shape):
    t,c,h,w = shape

    tidx = index // (c*h*w)
    t_mod = index % (c*h*w)

    cidx = t_mod // (h*w)
    c_mod = t_mod % (h*w)

    hidx = c_mod // (h)
    h_mod = c_mod % (h)

    widx = h_mod# // w
    # c * wh + index + ht * whc + hy * w + hx
    indices = [tidx,cidx,hidx,widx]
    return indices

def patch_at_index(noisy,index,psX,psT):
    indices = index2indices(index,noisy.shape)
    tslice = slice(indices[0],indices[0]+psT)
    cslice = slice(indices[1],indices[1]+psX)
    hslice = slice(indices[2],indices[2]+psX)
    wslice = slice(indices[3],indices[3]+psX)
    return noisy[tslice,cslice,hslice,wslice]

def patches_at_indices(noisy,indices,psX,psT):
    patches = []
    for index in indices:
        patches.append(patch_at_index(noisy,index,psX,psT))
    patches = np.stack(patches)
    return patches

