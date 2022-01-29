


# -- python imports --
import numpy as np
from easydict import EasyDict as edict

# -- local imports --
from .file_io import get_dataset_info,read_result,format_vnlb_results

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#       Menu to Selet Images
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def load_dataset(name,fstart=0,nframes=5,vnlb=True):
    path,fmax,fmt = get_dataset_info(name)
    return load_data(path,fmax,fmt,fstart,nframes,vnlb)

def load_data(path,fmax,fmt,fstart,nframes,vnlb):

    # -- check if path exists --
    if not path.exists():
        print("Please download the davis baseball file from the git repo.")

    # -- read images using opencv --
    clean,cpaths,cfmt = read_result(path,fmt,fstart,nframes)

    # -- read vnlb output files --
    if vnlb:
        data,paths,fmts = load_vnlb_results(path,fstart,nframes)
    else:
        data,paths,fmts = {},{},{}

    # -- combine results --
    data['clean'] = clean
    paths['clean'] = cpaths
    fmts['clean'] = cfmt

    return data,paths,fmts


def load_vnlb_results(path,fstart,nframes):

    # -- path to saved results --
    vnlb_path = path/"vnlb"
    if not vnlb_path.exists():
        return {},{},{}

    # -- load c++ results --
    results = edict()
    results.noisy = read_result(vnlb_path,"%03d.tif",fstart,nframes)
    results.fflow = read_result(vnlb_path,"tvl1_%03d_f.flo",fstart,nframes)
    results.bflow = read_result(vnlb_path,"tvl1_%03d_b.flo",fstart,nframes)
    results.basic = read_result(vnlb_path,"bsic_%03d.tif",fstart,nframes)
    results.denoised = read_result(vnlb_path,"deno_%03d.tif",fstart,nframes)
    results.std = np.loadtxt(str(vnlb_path/"sigma.txt")).item(),"sigma.txt","sigma.txt"
    data,paths,fmts = format_vnlb_results(results)
    return data,paths,fmts

