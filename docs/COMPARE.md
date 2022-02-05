Comparing with C++ Code
---

This code is a Python API for Video Non-Local Bayesian Denoising (VNLB). [The original C++ code](https://github.com/pariasm/vnlb) is from Pablo Arias. For easier interfacing, a Swig-Python Wrapper of the original C++ Code is [available here](https://github.com/gauenk/svnlb).

CPU v.s. GPU Parallelism
---

The outputs from this VNLB code and the C++ Code are almost equal. The primary difference between to two method is the way in which we achieve parallelism. This difference impacts the final PSNR, especially on smaller images. The C++ code uses OPENMP to execute parallel CPU threads and chunks the image into sections. In this code base, a batch of random pixels are selected to searched and this searching is executed in parallel on a GPU. A secondary (and seemingly less important) difference between the two methods is how the precision of the inherent different of matrix multiplication between the C++ LAPLACK package the Python Pytorch package. However, since the image denoising community is interested in differences of approximately 1e-3 (and smaller), these small errors can potentially meaningfully change the final denoised PSNR.


Numerical Comparison
---

To demonstrate the difference between the two methods, we provide the `compare_cpp.py` script. We have two examples from the [C++ Code](https://github.com/pariasm/vnlb) provided in the `data/` folder. For reproducibility, details to re-create the C++ Code results are included in the [docs/VNLB.md](https://github.com/gauenk/pyvnlb/blob/master/docs/VNLB.md) file. To run the comparison, run the following:

```
$ python scripts/compare_cpp.py
```

The script prints the below table. Each element of the table is the sum of the absolute relative error between the outputs from the Python API and C++ Code.

|                           |         basic |      denoised |
|:--------------------------|--------------:|--------------:|
| Ave Rel. Error (basic)    |   0.023508    | nan           |
| cpp_psnr                  |  31.4275      |  31.6715      |
| py_psnr                   |  31.4137      |  31.6513      |
| Abs. Error (PSNR)         |   0.0138222   |   0.0201801   |
| Rel. Error (PSNR)         |   0.000439814 |   0.000637169 |
| Ave Rel. Error (denoised) | nan           |   0.0213285   |


Compute Time Comparison
---

On small images, the CPU is x10 faster than the GPU code. On larger images, the two methods are approximately equal... The Python code takes about 3 seconds longer than the C++ Code to execute.

First, export a short-cut to the directory

```
$ export PYVNLB_HOME="/pick/your/path/"
```

such that `ls $PYTHON_HOME/` shows "data/" with "data/davis_baseball" as a subdirectory.

To time the C++ algorithms, please [install the original C++ code](https://github.com/pariasm/vnlb). Then run the following command in the "scripts" directory,

```
$ time `./vnlb-gt.sh $PYVNLB_HOME/data/davis_baseball/%05d.jpg 0 4 20 $PYVNLB_HOME/data/davis_baseball_64x64/vnlb/ "-px1 7 -pt1 2 -px2 7 -pt2 2 -verbose"`

#davis
real	2m25.276s
user	12m25.894s
sys	3m5.471s
```


```
$ cd vnlb/
$ rm -r ./__pycache__
$ time `python ./scripts/example.py`

#davis
real	2m28.202s
user	12m18.676s
sys	3m10.941s
```

On the `davis` example, the original execution time of the C++ Code and Python API is X and Y, respectively. ...
