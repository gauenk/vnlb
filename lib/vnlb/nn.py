
import torch as th
from pdnn.nn_arch import get_nn_model

def select_model(sigma):
    if int(sigma) == 30 or int(sigma*255.) == 30:
        path = ""
        return path
    else:
        msg = f"Uknown model for sigma = [{sigma}]"
        raise ValueError(msg)

def load_model(path):
    pass

def denoise_burst(noisy,sigma):

    # -- select model --
    path = select_model(sigma)

    # -- load model --
    model = load_model(path)

    # -- deno each frame --
    deno = []
    for frame in noisy:
        deno.append(model(frame))
    deno = th.stack(deno)

    return deno

