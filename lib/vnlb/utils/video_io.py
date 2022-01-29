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


def read_pacnet_noisy_sequence(name,sigma,nframes=85):
    return read_pacnet_sequence(name,sigma,"noisy",nframes)

def read_pacnet_denoised_sequence(name,sigma,nframes=85):
    return read_pacnet_sequence(name,sigma,"denoised",nframes)

def read_pacnet_sequence(name,sigma,itype,nframes=85):
    path = Path("/home/gauenk/Documents/packages/")
    path = path / "PaCNet-denoiser/output/videos/jpg_sequences/set/"
    path = path / f"{itype}_{sigma}" / name
    assert path.exists()

    # -- load video --
    vid = []
    for t in range(nframes):
        fn = path / ("%05d.npy" % (t))
        print(fn)
        if not(fn.exists()): break
        fn = str(fn)
        frame = np.load(fn)
        vid.append(th.from_numpy(frame))
    vid = th.stack(vid)
    print("vid.shape: ",vid.shape)

    return vid

