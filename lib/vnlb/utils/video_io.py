import time
import cv2
import os
import gc
from skimage.restoration import estimate_sigma

from pathlib import Path

import numpy as np
import torch as th
from einops import rearrange


def read_video_sequence(folder_name, max_frame_num, file_ext):
    frame_name = folder_name + '{:05}.{}'.format(0, file_ext)
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

def save_image(image_to_save, folder_name, image_name):
    image_to_save = float_tensor_to_np_uint8(image_to_save)
    image_to_save = cv2.cvtColor(image_to_save, cv2.COLOR_RGB2BGR)
    fn = folder_name + image_name
    print("image: ",fn)
    cv2.imwrite(fn, image_to_save)

def save_numpy(image_to_save, folder_name, image_name):
    img = image_to_save.cpu().numpy()
    fn = folder_name + image_name
    print("numpy: ",fn)
    np.save(fn,img)

def float_tensor_to_np_uint8(im_in):
    im_out = im_in.clone().clamp(0, 1).cpu()
    im_out = im_out.permute(1, 2, 0)
    im_out = np.round(im_out.numpy() * 255)
    im_out = im_out.astype(np.uint8)
    return im_out

"""

Video-IO for UDVD Sequences


"""

def read_udvd_denoised_sequence(vid_name,vid_set,sigma,nframes=85):
    return read_udvd_sequence(vid_name,vid_set,sigma,"denoised",nframes)

def read_udvd_sequence(vid_name,vid_set,sigma,itype,nframes):
    path = Path("/home/gauenk/Documents/packages/")
    path = path / f"udvd/output/{vid_set}/"
    path = path / f"{itype}_{sigma}" / vid_name
    assert path.exists()

    # -- load video --
    vid = []
    for t in range(nframes):

        fn = path / ("%s_%05d.npy" % (itype,t))
        if not(fn.exists()): break
        fn = str(fn)

        frame = np.load(fn)
        vid.append(th.from_numpy(frame))

    vid = th.stack(vid)

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
    print(path)
    assert path.exists()

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

