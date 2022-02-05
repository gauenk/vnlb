"""

Example use of VNLB for Denoising

"""

import vnlb
import numpy as np

# -- get data --
clean = vnlb.testing.load_dataset("davis_64x64",vnlb=False)[0]['clean'].copy()[:3]
# (nframes,channels,height,width)

# -- add noise --
std = 20.
noisy = np.random.normal(clean,scale=std)

# -- Video Non-Local Bayes --
deno,basic,dtime = vnlb.denoise(noisy,std)


# -- Denoising Quality --
noisy_psnrs = vnlb.utils.compute_psnrs(clean,noisy)
basic_psnrs = vnlb.utils.compute_psnrs(clean,basic)
deno_psnrs = vnlb.utils.compute_psnrs(clean,deno)

print("Denoised PSNRs:")
print(deno_psnrs)
print("Basic PSNRs:")
print(basic_psnrs)
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
