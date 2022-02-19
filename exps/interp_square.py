"""

Interpolation square between different methods

"""

# -- python --
from pathlib import Path
from easydict import EasyDict as edict

# -- plotting --
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

# -- dataframe --
import pandas as pd

# -- experiment caching --
import cache_io

# -- linalg --
import numpy as np
import torch as th

# -- package --
import vnlb
from vnlb.utils.metrics import compute_psnrs_cuda
from vnlb.utils import read_video_sequence,read_pacnet_noisy_sequence,save_burst

def main():

    # -- create experiment cache --
    verbose = False
    cache_dir = ".cache_io"
    cache_name = "interp_square_s"
    cache = cache_io.ExpCache(cache_dir,cache_name)
    cache.clear()

    # -- eval across grid --
    set8_names = ["hypersmooth","motorbike","park_joy","rafting",
                  "snowboard","sunflower"]
    exps = {"vid_set":["set8"],
            "vid_name":["motorbike"],
            "sigma":[40],}
    exps = cache_io.mesh_pydicts(exps)
    for exp in exps:
        result = cache.load_exp(exp)
        uuid = cache.get_uuid(exp)
        if result is None:
            result = interp_grid(exp.vid_set,exp.vid_name,exp.sigma)
            cache.save_exp(uuid,exp,result) # save to cache

    # -- (4) print results! --
    records = cache.load_flat_records(exps)
    print(records[['ave_psnr','vid_name','sigma','alpha','pair']])
    records = records[records['sigma'] == 40]
    plot_by_name(records)

def plot_by_name(records):

    # -- make save dir --
    save_dir = Path("./output/interp_square/")
    if not save_dir.exists(): save_dir.mkdir()

    # -- write records --
    for vid_name,vid_df in records.groupby("vid_name"):

        # -- init plot --
        fig,ax = plt.subplots()

        for sigma,sigma_df in vid_df.groupby("pair"):

            # -- data --
            alpha = sigma_df['alpha']
            ave_psnr = sigma_df['ave_psnr']

            ax.plot(alpha,ave_psnr,'-x',label=str(sigma))

        # -- filename --
        save_path = str(save_dir / ("%s.png" % vid_name))

        # -- save and close --
        plt.legend()
        plt.savefig(save_path,transparent=True)
        ax.cla()
        plt.close("all")

def interp_grid(vid_set,vid_name,sigma):

    # -- get misc --
    vid_folder = get_vid_folder(vid_set,vid_name)

    # -- load images --
    imgs = edict()
    imgs.clean = read_video_sequence(vid_folder, 85, "png")
    imgs.pac = vnlb.proc_nn("pacnet",vid_set,vid_name,sigma)
    imgs.udvd = vnlb.proc_nn("udvd",vid_set,vid_name,sigma)
    imgs.vnlb = vnlb.proc_nl_cache(vid_set,vid_name,sigma)
    ifields = list(imgs.keys())

    # -- to cuda --
    device = "cuda:0"
    for key in imgs: imgs[key] = imgs[key].to(device)
    path = Path("output/videos/jpg_sequences/set/set8/")
    path = path / f"interp_grid/{sigma}/{vid_name}"
    if not path.exists(): path.mkdir(parents=True)

    # -- create interpolation grid --
    results = {'psnr':[],'alpha':[],'ave_psnr':[],'pair':[]}
    for i,key1 in enumerate(ifields):
        if key1 == "clean": continue
        for j,key2 in enumerate(ifields):
            if key2 == "clean": continue
            if i >= j: continue
            step = 1./10
            alphas = np.arange(0,1+step,step)
            for alpha in alphas:

                # -- deno --
                deno = alpha * imgs[key2] + (1. - alpha) * imgs[key1]
                psnr = compute_psnrs_cuda(deno,imgs.clean,1.)

                # -- save burst --
                save_path = path / ("%s_%s" % (key1,key2))
                if not save_path.exists(): save_path.mkdir()
                save_path = str(save_path) + "/"
                prefix = "alpha_%d" % ( alpha*100 )
                vnlb.utils.save_burst(deno, save_path, prefix)

                # -- show output --
                results['psnr'].append(psnr)
                results['ave_psnr'].append(psnr.mean().item())
                results['alpha'].append(alpha)
                results['pair'].append("%s-%s"%(key1,key2))

    return results

def order_keys(key1,key2):
    if key1 == "vnlb":
        return key1,key2
    elif key2 == "vnlb":
        return key2,key1
    elif key1 == "pac":
        return key1,key2
    elif key2 == "pac":
        return key2,key1
    else:
        raise ValueError(f"Uknown pair [{key1},{key2}]")

def get_vid_folder(vid_set,vid_name):
    if vid_set == "set8":
        return f"./data/set8/images/{vid_name}/"
    else:
        raise ValueError(f"Uknown video set [{vid_set}]")

if __name__ == "__main__":
    main()
