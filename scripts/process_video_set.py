# -- python imports --
import sys,os
import cv2
import argparse
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import shutil
from pathlib import Path

# from modules import *
# from functions import *
import torch
import torch as th
import numpy as np
from easydict import EasyDict as edict

from vnlb import deno_n3l,deno_n4
from vnlb.utils import Logger
from vnlb.utils import read_video_sequence,read_pacnet_noisy_sequence
# from vnlb.nn_arch import load_nn_model
from vnlb.utils.video_io import save_image,save_numpy
from vnlb.utils.metrics import compute_psnrs,skimage_psnr

def save_jpg(opt,vid_name,save_images):
    # -- setup --
    clip_str = 'clip_' if opt.clipped_noise else ''

    #
    # -- setup dirs --
    #

    # -- save root --
    jpg_folder = Path(opt.jpg_out_folder)
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
        if iname == "deno": folder_jpg /= "alpha_%d" % (int(opt.alpha*100))
        if folder_jpg.exists():
            shutil.rmtree(str(folder_jpg))
        folder_jpg.mkdir(parents=True)

        for t in range(nframes):
            fid = '/{:05}.jpg'.format(t)
            save_image(image[t, ...],str(folder_jpg),fid)
            fid = '/{:05}.npy'.format(t)
            save_numpy(image[t, ...],str(folder_jpg),fid)

def process_video_set_func():
    """
    Denoise all frames from a video set!

    """
    opt = parse_options()

    torch.manual_seed(opt.seed)
    if opt.gpuid >= 0 and torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        opt.device = 'cuda:%d' % opt.gpuid
    elif opt.gpuid >= 0 and not(torch.cuda.is_available()):
        opt.gpuid = -1
        opt.device = 'cpu'

    clip_str = 'clip_' if opt.clipped_noise else ''
    log_fn = './logs/process_video_set_{}{}_log.txt'.format(clip_str,opt.sigma)
    sys.stdout = Logger(log_fn)

    video_names = sorted(os.listdir(opt.in_folder))
    # video_names = ["salsa"]#sorted(os.listdir(opt.in_folder))
    deno_psnr_list = list()
    nn_psnr_list = list()
    nl_psnr_list = list()

    for i in range(len(video_names)):

        # -- name model --
        vid_name = video_names[i]
        # if vid_name != "park_joy": continue
        # if vid_name != "sunflower": continue
        # if vid_name != "tractor": continue
        vid_folder = opt.in_folder + '{}/'.format(vid_name)

        # -- load clean seq --
        clean = read_video_sequence(vid_folder, opt.max_frame_num, opt.file_ext)

        # -- load noisy from pacnet --
        nframes = clean.shape[0]
        noisy = read_pacnet_noisy_sequence(opt.vid_set, vid_name, opt.sigma, nframes)

        # -- set islice --
        islice = edict()
        islice.t = slice(0,10)
        islice.h = slice(0,-1)
        islice.w = slice(0,-1)
        islice.h = slice(256,256+64)
        islice.w = slice(256,256+64)
        islice = None

        if not(islice is None):
            clean = clean[islice.t,:,islice.h,islice.w]
            noisy = noisy[islice.t,:,islice.h,islice.w]

        if opt.clipped_noise:
            noisy = torch.clamp(noisy, min=0, max=1)
        if opt.gpuid >= 0:
            clean = clean.to(opt.gpuid)
            noisy = noisy.to(opt.gpuid)

        # -- denoise burst --
        # output = deno_n3l(noisy, opt.sigma, opt.alpha, vid_name, opt.clipped_noise,
        #                   opt.gpuid, opt.silent, opt.vid_set, opt.deno_model, islice,
        #                   verbose=opt.verbose)
        output = deno_n4(noisy, opt.sigma, opt.alpha, vid_name, opt.clipped_noise,
                         opt.gpuid, opt.silent, opt.vid_set, opt.deno_model, islice,
                         verbose=opt.verbose)
        deno,deno_nl,deno_nn,tdelta = output

        #
        # -- psnrs --
        #

        # -- psnr [v1] --
        deno_psnr = compute_psnrs(deno,clean,True)
        nl_psnr = compute_psnrs(deno_nl,clean,True)
        nn_psnr = compute_psnrs(deno_nn,clean,True)
        for t_p in range(len(deno_psnr)):
            deno_psnr_list.append(deno_psnr[t_p].item())
            nl_psnr_list.append(nl_psnr[t_p].item())
            nn_psnr_list.append(nn_psnr[t_p].item())

        # deno_psnr_list.append(deno_psnr.mean().item())
        # nl_psnr_list.append(nl_psnr.mean().item())
        # nn_psnr_list.append(nn_psnr.mean().item())

        # -- psnr [v2] --
        for t_p in range(len(noisy)):
            deno_psnr = skimage_psnr(deno[[t_p]],clean[[t_p]],True)
            nl_psnr = skimage_psnr(deno_nl[[t_p]],clean[[t_p]],True)
            nn_psnr = skimage_psnr(deno_nn[[t_p]],clean[[t_p]],True)
            deno_psnr_list.append(deno_psnr)
            nl_psnr_list.append(nl_psnr)
            nn_psnr_list.append(nn_psnr)

        # -- logging --
        print('')
        print('-' * 90)
        nn_mp = nn_psnr.mean()
        nl_mp = nl_psnr.mean()
        deno_mp = deno_psnr.mean()
        msg = '[{}/{}: {} done] '.format(i + 1,len(video_names),vid_name.upper())
        msg += 'nn: {:.2f}, nl: {:.2f}, deno: {:.2f}, '.format(nn_mp,nl_mp,deno_mp)
        msg += "time: {:.2f} ({:.2f} per frame)".format(tdelta, tdelta / clean.shape[1])
        print(msg)
        print('-' * 90)
        print('')
        sys.stdout.flush()

        if opt.save_jpg:
            save_dict = {"noisy":noisy,"nn":deno_nn,"nl":deno_nl,"deno":deno}
            save_vid_name = vid_name + "_test"
            save_jpg(opt,save_vid_name,save_dict)

    print('')
    print('-' * 90)
    print('[sigma {}, alpha {:.2f}] deno: {:.3f}, nl: {:.3f}, nn : {:.3f}'.\
        format(opt.sigma,opt.alpha,np.array(deno_psnr_list).mean(),
               np.array(nl_psnr_list).mean(),np.array(nn_psnr_list).mean()
        )
    )
    print('-' * 90)
    print('')
    sys.stdout.flush()

    return


# Description:
# Parsing command line
#
# Outputs:
# opt - options
def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vid_set', type=str, default='set8', help='name of video')
    parser.add_argument('--alpha', type=float, default=0.20, help='interolation value')
    parser.add_argument('--verbose', type=bool, default=True, help='print VNLB updates')
    parser.add_argument('--deno_model', type=str, default='pacnet', help='name of cached denoised video as input')
    parser.add_argument('--file_ext', type=str, default='jpg', help='file extension: {jpg, png}')
    parser.add_argument('--jpg_out_folder', type=str, default='./output/videos/jpg_sequences/set/', \
        help='path to the output folder for JPG frames')
    parser.add_argument('--avi_out_folder', type=str, default='./output/videos/avi_files/set/', \
        help='path to the output folder for AVI files')
    parser.add_argument('--sigma', type=int, default=30, help='noise sigma')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--clipped_noise', type=int, default=0, help='0: AWGN, 1: clipped Gaussian noise')
    parser.add_argument('--gpuid', type=int, default=0,
                        help='-1 - use CPU, 0 - use GPU 0')
    parser.add_argument('--save_jpg', action='store_true', help='save the denoised video as JPG frames')
    parser.add_argument('--save_avi', action='store_true', help='save the denoised video as AVI file')
    parser.add_argument('--silent', action='store_true', help="don't print 'done' every frame")
    parser.add_argument('--max_frame_num', type=int, default=85, help='maximum number of frames')

    opt = parser.parse_args()

    # -- set infolder --
    if opt.vid_set == "set8":
        opt.in_folder = "./data/set8/images/"
    elif opt.vid_set == "davis":
        opt.in_folder = "./data/davis/"
    else:
        raise ValueError(f"Uknown video set [{opt.vid_set}]")

    return opt


if __name__ == '__main__':

    process_video_set_func()
