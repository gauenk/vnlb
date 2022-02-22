from .bayes_est import denoise as bayes_denoise
from vnlb.utils import Timer


@Timer("denoise")
def denoise(patches,params,method="bayes"):
    if method == "bayes":
        return bayes_denoise(patches,params)
    if method == "ave":
        return ave_denoise(patches,params)
    if method == "ave_k":
        return ave_k_denoise(patches,params)
    else:
        raise ValueError(f"Uknown denoising method [{method}]")


def ave_denoise(patches,params):
    patches.noisy[...] = patches.noisy.mean(dim=1,keepdim=True)

def ave_k_denoise(patches,params):
    print(params.cpatches)
    cpatches = select_cpatches(patches,params.cpatches)/255.
    print("cpatches.shape: ",cpatches.shape)
    vals = ((cpatches[:,[0],:,:1] - cpatches[...,:1,:,:])**2).mean((-4,-3,-2,-1))
    print("vals.shape: ",vals.shape)
    print(vals[:3,:])
    print(vals[-3:,:])
    # patches.noisy[...] = patches.noisy.mean(dim=1,keepdim=True)
    exit(0)


def select_cpatches(patches,cpatches_str):
    if cpatches_str == "noisy":
        cpatches = patches.noisy
    elif cpatches_str == "basic":
        cpatches = patches.basic
    elif cpatches_str == "clean":
        cpatches = patches.clean
    else:
        raise ValueError(f"Uknown cpatches type [{cpatches_str}]")
    return cpatches
