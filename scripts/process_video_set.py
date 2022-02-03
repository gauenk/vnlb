
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

from vnlb import denoise
from vnlb.utils import Logger
from vnlb.utils import read_video_sequence,read_pacnet_noisy_sequence
# from vnlb.nn_arch import load_nn_model

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
def process_video_set_func():
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
    sys.stdout = Logger('./logs/process_video_set_{}{}_log.txt'.format(clip_str, opt.sigma))

    video_names = sorted(os.listdir(opt.in_folder))
    # video_names = ["salsa"]#sorted(os.listdir(opt.in_folder))
    deno_psnr_list = list()
    basic_psnr_list = list()
    # model = load_nn_model(opt.sigma,opt.device)
    # model = "salsa"

    for i in range(len(video_names)):
        vid_name = video_names[i]
        if vid_name != "park_joy": continue
        # if vid_name == "salsa":
        #     basic_psnr_list.append(30.40)
        #     deno_psnr_list.append(30.80)
        #     continue

        # -- name model --
        vid_folder = opt.in_folder + '{}/'.format(vid_name)

        # -- load clean seq --
        clean = read_video_sequence(vid_folder, opt.max_frame_num, opt.file_ext)
        print("clean.shape: ",clean.shape)

        # -- load noisy from pacnet --
        nframes = clean.shape[0]
        noisy = read_pacnet_noisy_sequence(vid_name, opt.vid_set, opt.sigma, nframes)
        # noisy = clean + (opt.sigma / 255) * torch.randn_like(clean)

        # -- set islice --
        islice = edict()
        islice.t = slice(0,5)
        islice.h = slice(0,-1)
        islice.w = slice(0,-1)
        islice.h = slice(256,256+128)
        islice.w = slice(256,256+128)
        # islice = None

        if not(islice is None):
            clean = clean[islice.t,:,islice.h,islice.w]
            noisy = noisy[islice.t,:,islice.h,islice.w]
        print("[sliced] clean.shape: ",clean.shape)

        if opt.clipped_noise:
            noisy = torch.clamp(noisy, min=0, max=1)
        if opt.gpuid >= 0:
            clean = clean.to(opt.gpuid)
            noisy = noisy.to(opt.gpuid)

        # -- denoise burst --
        deno,basic,time = denoise(noisy, opt.sigma, vid_name, opt.clipped_noise,
                                  opt.gpuid, opt.silent, opt.vid_set,
                                  opt.deno_model, islice)

        # -- psnrs --
        deno_psnr = compute_psnrs(deno,clean)
        basic_psnr = compute_psnrs(basic,clean)
        deno_psnr_list.append(deno_psnr.mean().item())
        basic_psnr_list.append(basic_psnr.mean().item())

        # -- logging --
        print('')
        print('-' * 80)
        basic_mpsnr = basic_psnr.mean()
        deno_mpsnr = deno_psnr.mean()
        msg = '[{}/{}: {} done] '.format(i + 1,len(video_names),vid_name.upper())
        msg += 'basic: {:.2f}, deno: {:.2f}, '.format(basic_mpsnr,deno_mpsnr)
        msg += "time: {:.2f} ({:.2f} per frame)".format(time, time / clean.shape[1])
        print(msg)
        print('-' * 80)
        print('')
        sys.stdout.flush()

        save_jpg(opt,"tmp",noisy,basic,deno)

        if opt.save_jpg:
            save_jpg(opt,vid_name,noisy,basic,deno)

        # if opt.save_avi:
        #     noisy_folder_avi = opt.avi_out_folder + '/noisy_{}/'.format(opt.sigma)
        #     if not os.path.exists(noisy_folder_avi):
        #         os.mkdir(noisy_folder_avi)

        #     denoised_folder_avi = opt.avi_out_folder + '/denoised_{}/'.format(opt.sigma)
        #     if not os.path.exists(denoised_folder_avi):
        #         os.mkdir(denoised_folder_avi)

        #     save_video_avi(noisy, noisy_folder_avi, vid_name)
        #     save_video_avi(denoised_vid_t, denoised_folder_avi, vid_name)

    print('')
    print('-' * 60)
    print('Denoising a set with sigma {} done, deno psnr: {:.2f}, basic psnr: {:.2f}'.\
        format(opt.sigma,
               np.array(deno_psnr_list).mean(),
               np.array(basic_psnr_list).mean()
        )
    )
    print('-' * 60)
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
