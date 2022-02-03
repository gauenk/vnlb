
import os
from .utils import read_pacnet_denoised_sequence,read_udvd_denoised_sequence

def proc_nn(noisy,sigma,vid_name,vid_set,deno_model):

    # -- io exec images --
    if deno_model == "pacnet":
        return read_pacnet_denoised_sequence(vid_name,vid_set,sigma)
    elif deno_model == "udvd":
        return read_udvd_denoised_sequence(vid_name,vid_set,sigma)
    else:
        # -- exec models --
        raise NotImplemented("")


