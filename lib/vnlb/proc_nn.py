
import os
from .utils import read_pacnet_denoised_sequence

def proc_nn(noisy,sigma,model):

    # -- io exec images --
    print(model)
    if isinstance(model,str):
        vid_name = model
        return read_pacnet_denoised_sequence(vid_name,sigma)
    else:
        # -- exec models --
        raise NotImplemented("")


