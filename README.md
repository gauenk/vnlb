Video Non-Local Bayes (VNLB)
=========================================
A Python Implementation for Video Non-Local Bayesian Denoiser. 


Install
-------

The package is available through Python pip,

```
$ python -m pip install vnlb --user
```

Or the package can be downloaded through github,

```
$ git clone https://github.com/gauenk/vnlb/
$ cd vnlb
$ python -m pip install -r requirements.txt --user
$ python -m pip install -e ./lib --user
```

Usage
-----

We expect the images to be shaped `(nframes,channels,height,width)` with
pixel values in range `[0,...,255.]`. The color channels are ordered RGB. Common examples of noise levels are 10, 20 and 50. See [scripts/example.py](https://github.com/gauenk/vnlb/blob/master/scripts/example.py) for more details.

```python
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

# -- compute denoising quality --
deno_psnr = vnlb.utils.compute_psnrs(clean,deno)
basic_psnr = vnlb.utils.compute_psnrs(clean,basic)
print("Denoised PSNRs:")
print(deno_psnrs)
print("Basic PSNRs:")
print(basic_psnrs)
print("Execution Time (s): %2.2e" % dtime)

```

Comparing with C++ Code
---

The outputs from this VNLB code and the C++ Code are almost equal. The primary difference between to two method is the way in which we achieve parallelism. This difference impacts the final PSNR, especially on smaller images. More details are [included in docs/COMPARE.md](https://github.com/gauenk/vnlb/blob/master/docs/COMPARE.md). Note too there is no optical flow computed within this method to account for large motion changes. This must be computed separately and passed to the function call.


Credits
--------

This code provides is a Python+GPU implementation of the video denoising method (VNLB) described in:

[P. Arias, J.-M. Morel. "Video denoising via empirical Bayesian estimation of
space-time patches", Journal of Mathematical Imaging and Vision, 60(1),
January 2018.](https://link.springer.com/article/10.1007%2Fs10851-017-0742-4)

Additionally, [the original C++ code](https://github.com/pariasm/vnlb) is from Pablo Arias. For easier interfacing, a Swig-Python Wrapper of the original C++ Code is [available here](https://github.com/gauenk/svnlb).


LICENSE
-------

Licensed under the GNU Affero General Public License v3.0, see `LICENSE`.
