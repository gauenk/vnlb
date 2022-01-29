import os,sys
import cv2
import torch
import numpy as np
from einops import rearrange
import svnlb


def divUp(a,b): return (a-1)//b+1

def get_patch_shapes_from_params(params,channels):
    step1 = params.isFirstStep
    sWx = params.sizeSearchWindow
    sWt_f = params.sizeSearchTimeFwd
    sWt_b = params.sizeSearchTimeBwd
    sWt = sWt_f + sWt_b + 1
    sPx = params.sizePatch
    sPt = params.sizePatchTime
    patch_num = sWx * sWx * sWt
    patch_dim_c = channels if params.coupleChannels else 1
    patch_dim = sPx * sPx * sPt * patch_dim_c
    patch_chnls = 1 if params.coupleChannels else channels
    return patch_num,patch_dim,patch_chnls

def assign_swig_args(args,sargs):
    for key,val in args.items():
        sval = optional_swig_ptr(val)
        setattr(sargs,key,sval)
    return sargs

def check_and_expand_flows(pyargs,t):
    fflow,bflow = pyargs['fflow'],pyargs['bflow']
    nfflow = fflow.shape[0]
    nbflow = bflow.shape[0]
    assert nfflow == nbflow,"num flows must be equal."
    if nfflow == t-1:
        expand_flows(pyargs)
    elif nfflow < t-1:
        msg = "The input flows are the wrong shape.\n"
        msg += "(nframes,two,height,width)"
        raise ValueError(msg)

def check_omp_num_threads(nthreads=4):
    omp_nthreads = omp_num_threads()
    check_eq = omp_nthreads == nthreads
    # check_geq = omp_nthreads >= nthreads
    # check_bool = check_geq if geq else check_eq
    if not(check_eq):
        msg = f"Please run `export OMP_NUM_THREADS={nthreads}` "
        msg += "before running this file.\n"
        msg += f"Currently set to [{omp_nthreads}] threads"
        print(msg)
        sys.exit(1)

def omp_num_threads():
    omp_nthreads = os.getenv('OMP_NUM_THREADS')
    omp_nthreads = 0 if omp_nthreads is None else int(omp_nthreads)
    return omp_nthreads

def ndarray_ctg_dtype(ndarray,dtype,verbose):
    in_dtype = ndarray.dtype
    if in_dtype != dtype:
        if verbose:
            print(f"Warning: converting burst image from {in_dtype} to {dtype}.")
        ndarray = ndarray.astype(np.float32)
    # ndarray = np.ascontiguousarray(ndarray.copy())
    return ndarray

def rgb2bw(burst):
    burst = burst.astype(np.float32)
    burst_bw = []
    for t in range(burst.shape[0]):
        frame = burst[t]
        frame = rearrange(frame,'c h w -> h w c')
        # frame = .299 * frame[...,0] + .587 * frame[...,1] + .114 * frame[...,2]
        frame = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
        frame = rearrange(frame,'h w -> 1 h w')
        burst_bw.append(frame)
    burst_bw = np.stack(burst_bw)
    return burst_bw

def compute_psnrs(img1,img2,imax=255.):

    # -- same num of dims --
    assert img1.ndim == img2.ndim,"both must have same dims."

    # -- give batch dim if not exist --
    if img1.ndim == 3:
        img1 = img1[None,:]
        img2 = img2[None,:]

    # -- compute --
    eps=1e-16
    b = img1.shape[0]
    img1 = img1/imax
    img2 = img2/imax
    delta = (img1 - img2)**2
    mse = delta.reshape(b,-1).mean(axis=1) + eps
    log_mse = np.ma.log10(1./mse).filled(-np.infty)
    psnr = 10 * log_mse
    return psnr

def optional_pair(pydict,key,default,dtype):
    value = optional(pydict,key,default,dtype)
    if not(hasattr(value,"__getitem__")):
        value = np.array([value,value],dtype=dtype)
    return value

def optional(pydict,key,default,dtype=None):

    # -- get elem --
    rtn = default
    if not(pydict is None):
        # -- "key" can be a list of options; we take first one. --
        if isinstance(key,list):
            for _key in key:
                if _key in pydict:
                    rtn = pydict[_key]
                    break
        elif key in pydict:
            rtn = pydict[key]

    # -- convert to correct numpy type --
    if isinstance(rtn,list):
        if dtype is None: dtype = np.float32
        rtn = np.array(rtn,dtype=dtype)

    return rtn

def optional_swig_ptr(elem):
    swig_xfer = isinstance(elem,np.ndarray)
    swig_xfer = swig_xfer or isinstance(elem,str)
    swig_xfer = swig_xfer or isinstance(elem,bytes)
    if not swig_xfer:
        return elem
    # elem = np.ascontiguousarray(elem)
    return svnlb.swig_ptr(elem)

def check_flows(pyargs):
    fflow = optional(pyargs,'fflow',None)
    bflow = optional(pyargs,'bflow',None)
    return check_none(fflow,'neq') and check_none(bflow,'neq')

def check_none(pyobj,mode):
    if mode == "eq":
        return type(pyobj) == type(None)
    elif mode == "neq":
        return type(pyobj) != type(None)
    else:
        raise ValueError(f"unknown mode [{mode}]")

def expand_flows(pydict,axis=0):
    fflow = pydict['fflow']
    if torch.is_tensor(fflow):
        return expand_flows_th(pydict,axis)
    else:
        return expand_flows_np(pydict,axis)

def expand_flows_th(pydict,axis=0):
    # -- unpack --
    fflow,bflow = pydict['fflow'],pydict['bflow']

    # -- expand according to original c++ repo --
    if axis == 0:
        fflow = torch.cat([fflow,fflow[[-1]]],dim=axis)
        bflow = torch.cat([bflow[[0]],bflow],dim=axis)
    elif axis == 1:
        fflow = torch.cat([fflow,fflow[:,[-1]]],dim=1)
        bflow = torch.cat([bflow[:,[0]],bflow],dim=1)
    else:
        raise ValueError(f"Invalid axis {axis}")

    # -- update --
    pydict['fflow'],pydict['bflow'] = fflow,bflow


def expand_flows_np(pydict,axis=0):
    """
    CPP requires the flows be repeated so
    the number of temporal flows matches
    the number of frames in a burst.
    """

    # -- unpack --
    fflow,bflow = pydict['fflow'],pydict['bflow']
    np.cat = np.concatenate

    # -- expand according to original c++ repo --
    if axis == 0:
        fflow = np.cat([fflow,fflow[[-1]]],axis=0)
        bflow = np.cat([bflow[[0]],bflow],axis=0)
    elif axis == 1:
        fflow = np.cat([fflow,fflow[:,[-1]]],axis=1)
        bflow = np.cat([bflow[:,[0]],bflow],axis=1)
    else:
        raise ValueError(f"Invalid axis {axis}")

    # -- update --
    pydict['fflow'],pydict['bflow'] = fflow,bflow


def groups2patches(group,c=None,psX=None,psT=None,npatches=None):

    # -- shapes --
    if (c is None) or (psX is None) or (psT is None) or (npatches is None):
        _,c,psT,psX,_,npatches = group.shape

    # -- setup --
    ncat = np.concatenate
    size = psX * psX * psT * c
    numNz = npatches * psX * psX * psT * c
    group_f = group.ravel()[:numNz]

    # -- [og -> img] --
    group = group_f.reshape(c,psT,-1)
    group = ncat(group,axis=1)
    group = group.reshape(c*psT,psX**2,npatches).transpose(2,0,1)
    group = ncat(group,axis=0)

    # -- final reshape --
    group = group.reshape(npatches,psT,c,psX,psX)

    return group


def patches2groups(patches,c=None,psX=None,psT=None,nsearch=None,nParts=None):

    # -- shapes --
    shape = patches.shape
    if c is None:
        c = shape[2]
    if psX is None:
        psX = shape[3]
    if psT is None:
        psT = shape[1]
    if nsearch is None:
        nsearch = shape[0]
    if nParts is None:
        nParts = 1

    # -- setup --
    npatches = patches.shape[0]
    ncat = np.concatenate
    size = psX * psX * psT * c
    numNz = npatches * psX * psX * psT * c
    group = patches.ravel()[:numNz]

    # -- [img -> og] --
    group = group.reshape(npatches,psX*psX,c*psT).transpose(1,2,0)
    group = ncat(group,axis=0)
    group = group.reshape(psT,c,npatches*psX*psX)
    group = ncat(group,axis=1)

    # -- fill with zeros --
    group_f = group.ravel()[:numNz]
    group = np.zeros(size*nsearch,dtype=np.float32)
    group[:size*npatches] = group_f[...]
    group = group.reshape(nParts,c,psT,psX,psX,nsearch)

    return group
