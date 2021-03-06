from .bayes_est import denoise as bayes_denoise


def denoise(patches,params,method="bayes"):
    if method == "bayes":
        return bayes_denoise(patches,params)
    if method == "ave":
        return ave_denoise(patches,params)
    else:
        raise ValueError(f"Uknown denoising method [{method}]")


def ave_denoise(patches,params):
    return patches.noisy.mean(dim=1)
