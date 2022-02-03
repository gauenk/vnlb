
# -- imports --
import torch
import numpy as np
from einops import rearrange
import matplotlib.pyplot as plt

def explore_gp(deno,noisy,clean):


    # -- messages --
    print("deno.shape:" ,deno.shape)
    print("noisy.shape:" ,noisy.shape)
    print("clean.shape:" ,clean.shape)

    # -- reshape --
    shape_str = 'b n pt c ph pw -> b n c (pt ph pw)'
    deno = rearrange(deno,shape_str)
    noisy = rearrange(noisy,shape_str)
    clean = rearrange(clean,shape_str)

    # -- cpu and numpy --
    deno = deno.cpu().numpy()
    noisy = noisy.cpu().numpy()
    clean = clean.cpu().numpy()

    # -- plot --
    bidx = 0
    cidx = 0
    nelems = 1
    grid = np.arange(deno.shape[3])

    fix,ax = plt.subplots()
    nelems = 3
    ax.plot(grid,noisy[bidx,:nelems,cidx].T,marker='x')
    nelems = 1
    ax.plot(grid,deno[bidx,:nelems,cidx].T,marker='+',label='deno')
    nelems = 1
    ax.plot(grid,clean[bidx,:nelems,cidx].T,marker='o',label='clean')
    ax.legend()
    plt.savefig("explore_gp.png",transparent=True)
    plt.clf()
    plt.cla()
