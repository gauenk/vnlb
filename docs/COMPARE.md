Comparing with C++ Code
---

This code is a Python API for Video Non-Local Bayesian Denoising ([C++ code originally from Pablo Arias](https://github.com/pariasm/vnlb)). The numerical outputs from the Python API and the C++ Code are exactly equal. The Python code takes about 3 seconds longer than the C++ Code to execute.

Numerical Comparison
---

To demonstrate this claim, we provide the `compare_cpp.py` script. We have two examples from the [C++ Code](https://github.com/pariasm/vnlb) provided in the `data/` folder. For reproducibility, details to re-create the C++ Code results are included in the [docs/VNLB.md](https://github.com/gauenk/pyvnlb/blob/master/docs/VNLB.md) file. To run the comparison with the provided C++ outputs, type:

```
$ export OMP_NUM_THREADS=4
$ python compare_cpp.py
```

The script prints the below table. Each element of the table is the sum of the relative error between the outputs from the Python API and C++ Code.

|                   |   noisyForFlow |   noisyForVnlb |   fflow |   bflow |   basic |   denoised |
|:------------------|---------------:|---------------:|--------:|--------:|--------:|-----------:|
| Total Error (cv2) |    0.000505755 |              0 | 504.308 |  21.643 |       0 |          0 |
| Total Error (cpp) |    0           |              0 |   0     |   0     |       0 |          0 |


The following describes each column:

* __noisyForFlow__: the images used to compute the optical flow
* __noisyForVnlb__: the images to be denoised
* __fflow__: the forward optical flow (t -> t+1)
* __bflow__: the backward optical flow (t -> t-1)
* __basic__: the "basic" esimate, as described in the VNLM method
* __denoised__: the denoised images

The graphic below depicts the input-output relationship of the columns:

```           
noisyForFlow -> (fflow, bflow)
(noisyForVnlb, fflow, bflow) -> (basic, denoised)  
```

In the above table, the two rows identify two methods used to read image data. The first row uses opencv (`cv2`) and the second row uses the original, wrapped C++ functions (`cpp`). Images for optical flow (`noisyForFlow`) are read with the [iio library](https://github.com/pariasm/vnlb/tree/master/lib/iio). Images for the VNLB method (`noisyForVnlb`) are read with the [VidUtils library](https://github.com/pariasm/vnlb/tree/master/src/VidUtils). 

For optical flow, images read with opencv are slightly different from the images read using the C++ function, as indicated by the 0.0005 under the `noisyForFlow` column. This yields a change in the the optical flow outputs (`fflow` and `bflow`). However, this small change in optical flow yields no difference in the final denoising results (`basic` and `denoising`). 

For the VNLB method, images read with opencv are exactly equal the images read using the C++ function, as indicated by the 0 in the `noisyForVnlb` column.


Compute Time Comparison
---

The Python code takes about 3 seconds longer than the C++ Code to execute. To time the algorithms, one can execute both methods within a `time` bracket. 

```
$ cd vnlb/build/bin/
$ time `./vnlb-gt.sh $PYVNLB_HOME/data/davis_baseball_64x64/%05d.jpg 0 4 20 $PYVNLB_HOME/data/davis_baseball_64x64/vnlb/ "-px1 7 -pt1 2 -px2 7 -pt2 2 -verbose"`

#davis_64x64
real	0m5.917s
user	0m30.099s
sys	0m7.698s

#davis
real	2m25.276s
user	12m25.894s
sys	3m5.471s
```

```
$ cd pyvnlb/
$ rm -r ./__pycache__
$ time `python ./scripts/example.py`

#davis_64x64 
real	0m8.867s
user	0m41.212s
sys	0m10.648s

#davis
real	2m28.202s
user	12m18.676s
sys	3m10.941s
```

On the `davis` example, the original execution time of the C++ Code and Python API is 2:25 and 2:28, respectively. This increase in time is from an increase in execution time within the C++ routines themselves, rather than the Python wrapper. See the `scripts/example.py` and the `runVnlbTimed` function for more information.
