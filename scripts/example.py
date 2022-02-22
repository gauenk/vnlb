"""

Example use of VNLB for Denoising

"""

import vnlb
import raft
# import svnlb
import torch as th
import numpy as np
import torch.nn.functional as tnnf
from easydict import EasyDict as edict

# -- set seed [randomly order denoised pixels] --
seed = 123
np.random.seed(seed)
th.manual_seed(seed)
th.backends.cudnn.deterministic = True
th.backends.cudnn.benchmark = False

# -- get data --
# clean = vnlb.testing.load_dataset("davis_64x64",vnlb=False)[0]['clean'].copy()[:3]
# clean = vnlb.testing.load_dataset("cup_crop",vnlb=False,nframes=30)[0]['clean'].copy()
# clean = vnlb.testing.load_dataset("candles",vnlb=False,nframes=30)[0]['clean'].copy()
# clean = vnlb.testing.load_dataset("buildings",vnlb=False,nframes=30)[0]['clean'].copy()
# clean = vnlb.testing.load_dataset("brickwall",vnlb=False,nframes=30)[0]['clean'].copy()
clean = vnlb.testing.load_dataset("pinecone_brick",vnlb=False,nframes=30)[0]['clean']
# clean = clean[:,:,512+128-32+32:512+256-32,:128-64]
# clean = clean[:,:,200:328,-128-64:-64]
# clean = clean[:,:,264:328,-128-64:-64]
# clean = clean[:,:,256:256+128,256+64:256+128]
# (nframes,channels,height,width)
print("clean.shape: ",clean.shape)
clean = th.from_numpy(clean)/255.
clean = tnnf.interpolate(clean,scale_factor=0.2,mode="bicubic",align_corners=False)
clean *= 255.
clean = clean.numpy()
# clean = clean[:,:,32:128+32,128+16:128+128+16] # default
clean = clean[:,:,32:64+32,128+16:128+64+16]
print("clean.shape: ",clean.shape)


# -- Compute Flows --
# ftype = "comp"
# ftype = "load"
ftype = "none"
if ftype == "comp":
    fflow,bflow = raft.burst2flow(clean)
    th.save(fflow,"fflow.pth")
    th.save(bflow,"bflow.pth")
    flows = edict({"fflow":fflow,"bflow":bflow})
elif ftype == "load":
    fflow = th.load("fflow.pth")
    bflow = th.load("bflow.pth")
    flows = edict({"fflow":fflow,"bflow":bflow})
else:
    flows = None


# -- Save Examples --
path = "output/example/"
nframes = clean.shape[0]
for t in range(nframes):
    vnlb.utils.save_image(clean[t]/255.,path,"clean_%05d.png" % t)
# exit(0)


# -- add noise --
std = 50.
std_r = 10.
alpha = 20.
noisy = np.random.normal(clean,scale=std)
# noisy = np.random.normal(clean,scale=std) + clean*np.random.normal(clean,scale=1.)
# noisy = np.random.poisson(alpha*clean/255.)*255./alpha
std = (noisy-clean).std()
print("std: ",std)
# print(np.c_[clean.ravel(),noisy.ravel()])
# print(np.mean((noisy - clean)**2))
# exit(0)
th.save(clean,"clean.pth")
th.save(noisy,"noisy.pth")


# 31.415 = standard deno
# .75 -> 2
# .75 -> 10
# .5 -> 2

# -- Video Non-Local Bayes --

# deno,basic,dtime = vnlb.denoise(noisy,std,flows=flows,clean=None,verbose=True)
deno,basic,dtime = vnlb.denoise_mod(noisy,std,flows=flows,clean=clean,verbose=True)

# -- Denoising Quality --
noisy_psnrs = vnlb.utils.compute_psnrs(clean,noisy)
basic_psnrs = vnlb.utils.compute_psnrs(clean,basic)
deno_psnrs = vnlb.utils.compute_psnrs(clean,deno)

print("Denoised PSNRs:")
print(deno_psnrs,deno_psnrs.mean())
print("Basic PSNRs:")
print(basic_psnrs,basic_psnrs.mean())
print("Noisy PSNRs:")
print(noisy_psnrs)
print("Exec Time (sec): %2.2e" % dtime)


# -- Save Examples --
path = "output/example/"
nframes = deno.shape[0]
for t in range(nframes):
    vnlb.utils.save_image(deno[t]/255.,path,"deno_%05d.png" % t)
    vnlb.utils.save_image(noisy[t]/255.,path,"noisy_%05d.png" % t)
    vnlb.utils.save_image(basic[t]/255.,path,"basic_%05d.png" % t)
