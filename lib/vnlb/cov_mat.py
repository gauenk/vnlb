

from scipy import linalg as scipy_linalg
import numpy as np
from einops import rearrange
from easydict import EasyDict as edict

def computeCovMat(groups,rank,covMat=None):

    # -- shapes --
    ndim = groups.ndim
    if groups.ndim > 2:
        if groups.ndim == 5:
            in_shape = 'p pst ps1 ps2 n'
        elif groups.ndim == 6:
            in_shape = 'p 1 pst ps1 ps2 n'
        # elif groups.ndim == 7:
        #     in_shape = 'p n pst ps1 ps2 1 1'
        else:
            raise ValueError(f"unknown ndim: {ndim}")
        groups = rearrange(groups,in_shape + ' -> (p pst ps1 ps2) n')
        groups = groups.copy()
    pdim,nSimP = groups.shape

    # -- cov mat --
    if covMat is None:
        groups = groups.astype(np.float32)
        covMat = np.matmul(groups,groups.transpose(1,0))/nSimP
        covMat = covMat.astype(np.float32)

    # -- eigen stuff --
    kwargs = {"compute_v":1,"range":'I',"lower":0,"vl":-1,"vu":0,
              "il":pdim-rank+1,"iu":pdim,"abstol":0,"overwrite_a":1}
              # "lwork":pdim*8}
    results = scipy_linalg.lapack.ssyevx(covMat,**kwargs)

    # -- format eigs --
    eigVals,eigVecs = results[0],results[1]
    eigVals = eigVals.astype(np.float32)
    eigVecs = eigVecs.astype(np.float32)

    # -- format output --
    results = edict()
    results.covMat = covMat
    results.covEigVals = eigVals
    results.covEigVecs = eigVecs

    return results
