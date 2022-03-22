import time
import cv2
import os
import gc
from skimage.restoration import estimate_sigma

import shutil
from pathlib import Path

import numpy as np
import torch as th
from einops import rearrange

#
# Save a Frame from "Proc. Video. Set."
#

def save_frame_jpg(opt,vid_name,save_images,method_name=None):
    # -- setup --
    clip_str = 'clip_' if opt.clipped_noise else ''

    #
    # -- setup dirs --
    #

    # -- save root --
    jpg_folder = Path(opt.jpg_out_folder)
    if not(method_name is None):
        jpg_folder = jpg_folder / method_name
    if not jpg_folder.exists():
        jpg_folder.mkdir(parents=True)

    # -- each save image --
    for iname in save_images:

        # -- get image --
        image = save_images[iname]
        nframes = image.shape[0]

        # -- path --
        folder_jpg = jpg_folder / ('%s' % opt.vid_set)
        folder_jpg /= '%s_%s%d' % (iname,clip_str,opt.sigma)
        if not(folder_jpg.exists()):
            folder_jpg.mkdir(parents=True)

        # -- video name --
        folder_jpg /= vid_name
        if iname == "deno" and "alpha" in opt:
            folder_jpg /= "alpha_%d" % (int(opt.alpha*100))
        if folder_jpg.exists():
            shutil.rmtree(str(folder_jpg))
        folder_jpg.mkdir(parents=True)

        for t in range(nframes):
            fid = '/{:05}.jpg'.format(t)
            save_image(image[t, ...],str(folder_jpg),fid)
            fid = '/{:05}.npy'.format(t)
            save_numpy(image[t, ...],str(folder_jpg),fid)


def read_video_sequence(folder_name, max_frame_num, file_ext):
    frame_name = folder_name + '{:05}.{}'.format(0, file_ext)
    print(frame_name)
    frame = cv2.imread(frame_name, cv2.IMREAD_COLOR)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = (np.transpose(np.atleast_3d(frame), (2, 0, 1)) / 255).astype(np.float32)

    frame_h, frame_v = frame.shape[1:3]
    frame_num = min(len(os.listdir(folder_name)), max_frame_num)
    vid = th.full((3, frame_num, frame_h, frame_v), float('nan'))
    vid[:, 0, :, :] = th.from_numpy(frame)

    for i in range(1, frame_num):
        frame_name = folder_name + '{:05}.{}'.format(i, file_ext)
        frame = cv2.imread(frame_name, cv2.IMREAD_COLOR)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = (np.transpose(np.atleast_3d(frame), (2, 0, 1)) / 255).astype(np.float32)
        vid[:, i, :, :] = th.from_numpy(frame)

    vid = vid.unsqueeze(0)
    vid = rearrange(vid,'1 c t h w -> t c h w')

    return vid

"""

Saving computed images

"""

def save_burst(burst, folder_name, prefix):
    # -- make dir if needed --
    path = Path(folder_name)
    if not(path.exists()): path.mkdir(parents=True)
    t,c,h,w = burst.shape
    for ti in range(t):
        name = "%s_%05d.png" % (prefix,ti)
        save_image(burst[ti],folder_name,name)

def save_image(image_to_save, folder_name, image_name):
    # -- make dir if needed --
    path = Path(folder_name)
    if not(path.exists()): path.mkdir(parents=True)

    # -- save --
    image_to_save = float_tensor_to_np_uint8(image_to_save)
    image_to_save = cv2.cvtColor(image_to_save, cv2.COLOR_RGB2BGR)
    fn = folder_name + image_name
    cv2.imwrite(fn, image_to_save)

def save_numpy(image_to_save, folder_name, image_name):
    img = image_to_save.cpu().numpy()
    fn = folder_name + image_name
    np.save(fn,img)

def float_tensor_to_np_uint8(im_in):
    if th.is_tensor(im_in):
        im_in = im_in.clone().cpu().numpy()
    im_out = im_in.clip(0, 1)
    im_out = im_out.transpose(1, 2, 0)
    im_out = np.round(im_out * 255)
    im_out = im_out.astype(np.uint8)
    return im_out
"""

Video-IO for Denoised Sequences

"""

# def read_nl_denoised_sequence(vid_set,vid_name,itype,sigma):
#     return read_nl_sequence(vid_set,vid_name,"deno",sigma)

def read_nl_sequence(vid_set,vid_name,sigma):

    # -- path --
    path = Path("/home/gauenk/Documents/packages/vnlb/")
    path = path / f"output/videos/jpg_sequences/set/set8/nl_{sigma}"
    path = path / vid_name
    assert path.exists(),f"path must exist {path}"
    nframes = 85

    # -- load video --
    vid = []
    for t in range(nframes):

        fn = path / ("%05d.npy" % (t))
        if not(fn.exists()): break
        fn = str(fn)

        frame = np.load(fn)
        vid.append(th.from_numpy(frame))

    vid = th.stack(vid).squeeze()

    return vid

"""

Video-IO for UDVD Sequences


"""

def read_udvd_denoised_sequence(vid_set,vid_name,sigma,nframes=85):
    return read_udvd_sequence(vid_set,vid_name,sigma,"deno",nframes)

def read_udvd_sequence(vid_set,vid_name,sigma,itype,nframes):
    path = Path("/home/gauenk/Documents/packages/")
    path = path / f"udvd/output/{vid_set}/"
    path = path / f"{itype}_{sigma}" / vid_name
    assert path.exists(),f"path must exist {path}"

    # -- load video --
    vid = []
    for t in range(nframes):

        fn = path / ("%s_%05d.npy" % (itype,t))
        if not(fn.exists()): break
        fn = str(fn)

        frame = np.load(fn)
        vid.append(th.from_numpy(frame).type(th.float32))

    vid = th.stack(vid).squeeze()

    return vid


"""

Video-IO for PaCNet Sequences


"""

def read_pacnet_noisy_sequence(vid_set,vid_name,sigma,nframes=85):
    return read_pacnet_sequence(vid_set,vid_name,sigma,"noisy",nframes)

def read_pacnet_denoised_sequence(vid_set,vid_name,sigma,nframes=85):
    return read_pacnet_sequence(vid_set,vid_name,sigma,"denoised",nframes)

def read_pacnet_sequence(vid_set,vid_name,sigma,itype,nframes=85):

    # [vid_set] currently unused.
    path = Path("/home/gauenk/Documents/packages/")
    path = path / "PaCNet-denoiser/output/videos/jpg_sequences/set/"
    path = path / f"{itype}_{sigma}" / vid_name
    assert path.exists(),f"path must exist {path}"

    # -- load video --
    vid = []
    for t in range(nframes):

        fn = path / ("%05d.npy" % (t))
        if not(fn.exists()): break
        fn = str(fn)

        frame = np.load(fn)
        vid.append(th.from_numpy(frame))

    vid = th.stack(vid)

    return vid

