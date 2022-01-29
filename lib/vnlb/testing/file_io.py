
# -- python imports --
import cv2,sys
import subprocess
import pathlib
import numpy as np
from pathlib import Path
from einops import rearrange
from easydict import EasyDict as edict

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#      Read Files in a Loop
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def read_result(path,fmt,fstart,nframes):
    tensors,paths = [],[]
    for t in range(fstart,fstart+nframes):
        path_t = path / (fmt % t)
        if not path_t.exists():
            print(f"Error: the file {str(path_t)} does not exist.")
            sys.exit(1)
        data = read_file(path_t)
        tensors.append(data)
        paths.append(str(path_t))
    tensors = np.stack(tensors)
    tensors = np.ascontiguousarray(tensors.copy())
    fmt = str(path / fmt)
    return tensors,paths,fmt

def read_file(filename):
    if filename.suffix == ".flo":
        img = read_flo_file(filename)
    else:
        img = cv2.imread(str(filename),-1)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = rearrange(img,'h w c -> c h w')
    return img

def read_flo_file(filename):
    """
    Read from Middlebury .flo file
    :param flow_file: name of the flow file
    :return: optical flow data in matrix
    """
    f = open(filename, 'rb')
    magic = np.fromfile(f, np.float32, count=1)
    data2d = None

    if 202021.25 != magic:
        print('Magic number incorrect. Invalid .flo file')
    else:
        w = np.fromfile(f, np.int32, count=1).item()
        h = np.fromfile(f, np.int32, count=1).item()
        # print("Reading %d x %d flow file in .flo format" % (h, w))
        data2d = np.fromfile(f, np.float32, count=2 * w * h)
        # reshape data into 3D array (columns, rows, channels)
        data2d = np.resize(data2d, (h, w, 2))
    f.close()
    return data2d

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#       Wiring: Data <-> Paths
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def get_dataset_info(name):
    if name == "davis_64x64":
        path = Path("data/davis_baseball_64x64/")
        print_davis_64x64_message(path)
        return path,5,"%05d.jpg"
    elif name == "davis":
        path = Path("data/davis_baseball/")
        print_davis_message(path)
        return path,10,"%05d.jpg"
    elif name == "gmobile":
        path = Path("data/gmobile/")
        print_gmobile_message(path)
        return Path("data/gmobile/"),300,"%03d.png"
    else:
        print("Options include:")
        print(menu)
        raise ValueError(f"Uknown dataset name {name}")

def print_davis_64x64_message(path):
    if not path.exists():
        success = True
        print("Downloading davis_64x64")
        try:
            command = "./scripts/download_davis_64x64.sh"
            process = subprocess.Popen(command,stdout=subprocess.PIPE)
            output, error = process.communicate()
        except:
            print("Failed to download.")
            success = False
        if not(success):
            print("Please run the following commands")
            print("./scripts/download_davis_64x64.sh")
            sys.exit(1)
        return

def print_davis_message(path):
    if not path.exists():
        success = True
        print("Downloading davis")
        try:
            command = "./scripts/download_davis.sh"
            process = subprocess.Popen(command,stdout=subprocess.PIPE)
            output, error = process.communicate()
        except:
            print("Failed to download.")
            success = False
        if not(success):
            print("Please run the following commands")
            print("./scripts/download_davis_64x64.sh")
            sys.exit(1)
        return

def print_gmobile_message():
    if not path.exists():
        print("Please run the following commands")
        print("./scripts/download_gmobile.sh")
        print("Or run the following commands")
        print("mkdir data/gmobile/")
        print("cd data/gmobile/")
        print("wget http://dev.ipol.im/~pariasm/video_nlbayes/videos/gmobile.avi")
        print("ffmpeg -i gmobile.avi -f image2 %03d.png")
        sys.exit(1)

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#             Misc
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def format_vnlb_results(results):
    # -- reshape --
    data,paths,fmts = edict(),edict(),edict()
    for key in results.keys():
        # -- unpack --
        data[key] = results[key][0]
        paths[key] = results[key][1]
        fmts[key] = results[key][2]
    return data,paths,fmts

def merge_images(image_batch, size):
    h,w = image_batch.shape[1], image_batch.shape[2]
    c = image_batch.shape[3]
    img = np.zeros((int(h*size[0]), w*size[1], c))
    for idx, im in enumerate(image_batch):
        i,j = idx % size[1],idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w,:] = im
    img = img.astype(image_batch.dtype)
    return img

def swap_ndarray_fn(tensor,fn):
    # -- swap string and tensor --
    is_str = isinstance(tensor,str)
    is_path = isinstance(tensor,pathlib.Path)
    if is_str or is_path:
        tmp = tensor
        tensor = fn
        fn = tmp
    return tensor,fn

def save_hist(tensor,fn):
    # -- swap string and tensor --
    tensor,fn = swap_ndarray_fn(tensor,fn)

    # -- create hist --
    fig,ax = plt.subplots()
    ax.hist(tensor,bins=30)
    plt.savefig(fn,bbox_inches='tight')
    plt.close("all")

def save_image(tensor,fn,imax=255.):
    # -- swap string and tensor --
    tensor,fn = swap_ndarray_fn(tensor,fn)

    # -- squash image values --
    tensor = tensor.astype(np.float32) / imax
    tensor = np.clip(255.*tensor,0,255)
    tensor = np.uint8(tensor)

    # -- arange --
    save_img = rearrange(tensor,'c h w -> h w c')

    # -- format for cv2 --
    if save_img.shape[-1] == 3:
        save_img = cv2.cvtColor(save_img,cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(fn),save_img)

def save_images(tensor,fn,imax=255.):
    # -- swap string and tensor --
    tensor,fn = swap_ndarray_fn(tensor,fn)

    # -- shaping --
    nframes = tensor.shape[-4]
    if tensor.ndim > 4:
        s = tensor.shape[-3:]
        tensor = tensor.reshape(-1,s[0],s[1],s[2])
    ntotal = len(tensor)
    nrows = ntotal // nframes

    # -- squash image values --
    tensor = tensor.astype(np.float32) / imax
    tensor = np.clip(255.*tensor,0,255)
    tensor = np.uint8(tensor)

    # -- arange --
    tensor = rearrange(tensor,'t c h w -> t h w c')
    save_img = merge_images(tensor, (nrows,nframes))

    # -- format for cv2 --
    if save_img.shape[-1] == 3:
        save_img = cv2.cvtColor(save_img,cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(fn),save_img)


