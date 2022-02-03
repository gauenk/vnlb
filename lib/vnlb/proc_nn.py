
import os
from .utils import read_pacnet_denoised_sequence,read_udvd_denoised_sequence

def proc_nn(deno_model,vid_set,vid_name,sigma):

    # -- io exec images --
    if deno_model == "pacnet":
        return read_pacnet_denoised_sequence(vid_set,vid_name,sigma)
    elif deno_model == "udvd":
        return read_udvd_denoised_sequence(vid_set,vid_name,sigma)
    else:
        # -- exec models --
        raise NotImplemented(f"deno_model [{deno_model}]")


