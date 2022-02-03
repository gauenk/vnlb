import cv2
import numpy as np
import torch as th
from PIL import Image
from einops import rearrange,repeat

from vnlb.utils.sobel import apply_sobel_filter

def deno_v_basic(fid,vid_name,sigma):
    # -- root --
    root = "/home/gauenk/Documents/packages/vnlb/"
    out_root = root + "output/videos/jpg_sequences/set/"

    # -- load basic and deno --
    noisy_fn = out_root + f"noisy_{sigma}/{vid_name}/{fid}.npy"
    basic_fn = out_root + f"basic_{sigma}/{vid_name}/{fid}.npy"
    deno_fn = out_root + f"deno_{sigma}/{vid_name}/{fid}.npy"
    noisy = np.load(noisy_fn)
    basic = np.load(basic_fn)
    deno = np.load(deno_fn)
    print("std: ",sigma/255.)

    # -- interpolate --
    alpha_grid = np.arange(0.,1.+0.01,0.01)
    # alpha_grid = [.05,0.10,0.15,0.25,0.50,0.75,1.]
    alpha_grid = []
    for alpha in alpha_grid:
        deno_alpha = alpha * deno + (1 - alpha) * basic
        deno_alpha = th.FloatTensor(deno_alpha)

        # -- edges --
        edges = apply_sobel_filter(deno_alpha)[0]
        me,se = edges.mean().item(),edges.std().item()
        # print("[edges] %2.2f %2.3f %2.3f" % (alpha,me,se))

        # -- res --
        res = noisy - deno_alpha.numpy()
        mr,sr = (res).mean().item()**2,res.std().item()
        error = mr + (sr-sigma/255.)**2

        # -- message --
        msg = "[alpha = %2.2f] %2.3e " % (alpha,error)
        msg += "[(Res) %2.3e %2.3e] " % (mr,sr)
        msg += "[(Edge) %2.3f %2.3f] "% (me,se)
        print(msg)


    # -- interpolate --
    alpha = 1.0
    deno = alpha * deno + (1 - alpha) * basic

    # -- only top half --
    alpha = 0.50
    print("deno.shape: ",deno.shape)
    c,H,W = deno.shape
    # deno[:,:H//2,:] = alpha * deno[:,:H//2,:] + (1 - alpha) * basic[:,:H//2,:]
    # deno[:,H//2:,:] = alpha * deno[:,H//2:,:] + (1 - alpha) * basic[:,H//2:,:]

    # -- load clean image --
    clean_fn = root
    clean_fn += f"data/set8/images/{vid_name}/{fid}.png"
    clean = cv2.cvtColor(cv2.imread(clean_fn,cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)
    clean = (np.transpose(np.atleast_3d(clean),(2,0,1))/255.).astype(np.float32)

    # -- deltas --
    deno_delta = np.abs(deno - clean).sum(0)
    basic_delta = np.abs(basic - clean).sum(0)
    deno_b = deno_delta < basic_delta
    basic_b = deno_delta > basic_delta

    # -- save mask --
    basic_b_img = ((deno_delta > basic_delta)*255.).astype(np.uint8)
    basic_b_img = repeat(basic_b_img,'h w -> h w r',r=3)
    Image.fromarray(basic_b_img).save("mask_basic_b.png")

    deno_b_img = ((deno_delta < basic_delta)*255.).astype(np.uint8)
    deno_b_img = repeat(deno_b_img,'h w -> h w r',r=3)
    Image.fromarray(deno_b_img).save("mask_deno_b.png")

    # -- save pix --
    img = clean.copy()
    inds = np.nonzero(basic_b) # zero out when "basic" is better
    for c in range(3): img[c][inds] = 0
    img = (np.transpose(img,(1,2,0))*255.).astype(np.uint8)
    Image.fromarray(img).save(f"deno_b_{fid}.png")

    img = clean.copy()
    inds = np.nonzero(deno_b) # zero out when "deno" is better
    for c in range(3): img[c][inds] = 0
    img = (np.transpose(img,(1,2,0))*255.).astype(np.uint8)
    Image.fromarray(img).save(f"basic_b_{fid}.png")

    # -- overall --
    deno_delta = np.abs(deno - clean).mean()
    basic_delta = np.abs(basic - clean).mean()
    print("deno: ",deno_delta)
    print("basic: ",basic_delta)

    # -- make psnr --
    i = 0
    # grid = np.arange(0,500,32)
    grid = []#100,200,300]
    for hs in grid:
        for ws in grid:
            sh = slice(hs,hs+32)
            sw = slice(ws,ws+32)
            deno_delta = np.abs(deno[:,sh,sw] - clean[:,sh,sw]).mean()
            basic_delta = np.abs(basic[:,sh,sw] - clean[:,sh,sw]).mean()
            if deno_delta < basic_delta:
                print("WIN!")
                print("hs,ws: %d,%d" % (hs,ws))
                print("deno: ",deno_delta)
                print("basic: ",basic_delta)
                img = rearrange(clean[:,sh,sw],'c h w -> h w c')
                img = (img*255.).astype(np.uint8)
                Image.fromarray(img).save(f"win_{i}.png")
                i+= 1
            else:
                img = rearrange(clean[:,sh,sw],'c h w -> h w c')
                img = (img*255.).astype(np.uint8)
                Image.fromarray(img).save(f"lose_{i}.png")
                i+= 1


if __name__ == "__main__":

    vid_name = "hypersmooth"
    sigma = 20
    fid = "%05d" % 1
    deno_v_basic(fid,vid_name,sigma)

