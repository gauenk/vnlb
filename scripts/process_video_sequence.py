
import sys,os
import cv2
import argparse
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import shutil
from pathlib import Path
from easydict import EasyDict as edict

# from modules import *
# from functions import *
import torch
import torch as th
import numpy as np

from pyvnlb import denoise
from pyvnlb.utils import Logger
from pyvnlb.utils import read_video_sequence,read_pacnet_noisy_sequence
from pyvnlb.nn_arch import load_nn_model


def save_numpy(image_to_save, folder_name, image_name):
    img = image_to_save.cpu().numpy()
    fn = folder_name + image_name
    np.save(fn,img)

def save_image(image_to_save, folder_name, image_name):
    image_to_save = float_tensor_to_np_uint8(image_to_save)
    image_to_save = cv2.cvtColor(image_to_save, cv2.COLOR_RGB2BGR)
    cv2.imwrite(folder_name + image_name, image_to_save)

def float_tensor_to_np_uint8(im_in):
    im_out = im_in.clamp(0, 1).clone().cpu()
    im_out = im_out.permute(1, 2, 0)
    im_out = np.round(im_out.numpy() * 255)
    im_out = im_out.astype(np.uint8)
    return im_out

def compute_psnrs(deno,clean):
    return -10 * torch.log10(((deno - clean) ** 2).mean(dim=(-3, -2, -1), keepdim=False))

def save_jpg(opt,vid_name,noisy,basic,deno):

    # -- setup --
    clip_str = 'clip_' if opt.clipped_noise else ''

    #
    # -- setup dirs --
    #

    # -- save root --
    jpg_folder = Path(opt.jpg_out_folder)
    if not jpg_folder.exists():
        jpg_folder.mkdir(parents=True)

    # -- noisy --
    noisy_folder_jpg = opt.jpg_out_folder + '/noisy_{}{}/'.format(clip_str, opt.sigma)
    if not os.path.exists(noisy_folder_jpg):
        os.mkdir(noisy_folder_jpg)

    # -- basic --
    basic_folder_jpg = opt.jpg_out_folder
    basic_folder_jpg += '/basic_{}{}/'.format(clip_str, opt.sigma)
    if not os.path.exists(basic_folder_jpg):
        os.mkdir(basic_folder_jpg)

    # -- denoised --
    deno_folder_jpg = opt.jpg_out_folder
    deno_folder_jpg += '/deno_{}{}/'.format(clip_str, opt.sigma)
    if not os.path.exists(deno_folder_jpg):
        os.mkdir(deno_folder_jpg)

    #
    # -- clean old folders --
    #

    # -- noisy --
    noisy_sequence_folder = noisy_folder_jpg + vid_name + '/'
    if os.path.exists(noisy_sequence_folder):
        shutil.rmtree(noisy_sequence_folder)
    os.mkdir(noisy_sequence_folder)

    # -- basic --
    basic_sequence_folder = basic_folder_jpg + vid_name + '/'
    if os.path.exists(basic_sequence_folder):
        shutil.rmtree(basic_sequence_folder)
    os.mkdir(basic_sequence_folder)

    # -- deno --
    deno_sequence_folder = deno_folder_jpg + vid_name + '/'
    if os.path.exists(deno_sequence_folder):
        shutil.rmtree(deno_sequence_folder)
    os.mkdir(deno_sequence_folder)

    # -- save --
    for i in range(deno.shape[0]):
        save_image(noisy[i, ...],
                   noisy_sequence_folder, '{:05}.jpg'.format(i))
        save_numpy(noisy[i, ...],
                   noisy_sequence_folder, '{:05}.npy'.format(i))
        save_image(basic[i,...],
                   basic_sequence_folder,'{:05}.jpg'.format(i))
        save_numpy(basic[i,...],
                   basic_sequence_folder,'{:05}.npy'.format(i))
        save_image(deno[i,...],
                   deno_sequence_folder, '{:05}.jpg'.format(i))
        save_numpy(deno[i,...],
                   deno_sequence_folder, '{:05}.npy'.format(i))

# Description:
# Denoising a benchmark set of video sequences
# from PaCNet github
def process_video_sequence_func():

    # -- parse --
    opt = parse_options()

    # -- setup --
    torch.manual_seed(opt.seed)
    if opt.gpuid >= 0 and torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        opt.device = 'cuda:%d' % opt.gpuid
    elif opt.gpuid >= 0 and not(torch.cuda.is_available()):
        opt.gpuid = -1
        opt.device = 'cpu'
    clip_str = 'clip_' if opt.clipped_noise else ''

    # -- name model --
    vid_folder = opt.in_folder
    vid_name = os.path.basename(os.path.normpath(opt.in_folder))
    model = vid_name

    # -- log file --
    log_fn = "./logs/process_video_seq_"
    log_fn += "{}_{}{}_log.txt".format(vid_name, clip_str, opt.sigma)
    print(f"Log fn: [{log_fn}]")
    sys.stdout = Logger(log_fn)

    # -- slices for img --
    slices = edict()
    slices.t = slice(0,10)
    slices.h = slice(0,-1)
    slices.w = slice(0,-1)

    # -- load clean seq --
    clean = read_video_sequence(vid_folder, opt.max_frame_num, opt.file_ext)
    clean = clean[slices.t,:,slices.h,slices.w]
    print("clean.shape: ",clean.shape)

    # -- load noisy from pacnet --
    nframes = clean.shape[0]
    noisy = read_pacnet_noisy_sequence(vid_name, opt.sigma, nframes)
    noisy = noisy[slices.t,:,slices.h,slices.w]
    # noisy = clean + (opt.sigma / 255) * torch.randn_like(clean)

    # -- clip and xfer to gpu --
    if opt.clipped_noise:
        noisy = torch.clamp(noisy, min=0, max=1)
    if opt.gpuid >= 0:
        clean = clean.to(opt.gpuid)
        noisy = noisy.to(opt.gpuid)

    # -- denoise burst --
    deno,basic,time = denoise(noisy, opt.sigma, opt.clipped_noise,
                              opt.gpuid, opt.silent, model, slices)

    # -- psnrs --
    deno_psnr = compute_psnrs(deno,clean)
    basic_psnr = compute_psnrs(basic,clean)

    # -- logging --
    print('')
    print('-' * 80)
    basic_mpsnr = basic_psnr.mean()
    deno_mpsnr = deno_psnr.mean()
    msg = '[{} done] '.format(vid_name.upper())
    msg += 'basic: {:.2f}, deno: {:.2f}, '.format(basic_mpsnr,deno_mpsnr)
    msg += "time: {:.2f} ({:.2f} per frame)".format(time, time / clean.shape[1])
    print(msg)
    print('-' * 80)
    print('')
    sys.stdout.flush()

    if opt.save_jpg:
        save_jpg(opt,vid_name,noisy,basic,deno)
        print("Result images written.")
    sys.stdout.flush()

    return


# Description:
# Parsing command line
#
# Outputs:
# opt - options
def parse_options():
    parser = argparse.ArgumentParser()

    parser.add_argument('--in_folder', type=str, default='./data/set8/images/park_joy/', help='path to the input folder')
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

    return opt


if __name__ == '__main__':

    process_video_sequence_func()
