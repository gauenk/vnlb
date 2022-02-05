
# -- python --
import torch,time
torch.set_printoptions(precision=6)
import cv2,tqdm,copy
import numpy as np
import unittest
import vnlb
import tempfile
import sys
from einops import rearrange
import shutil
from pathlib import Path
from easydict import EasyDict as edict

# -- package helper imports --
from vnlb.agg import agg_patches
from vnlb.testing.data_loader import load_dataset
from vnlb.testing.file_io import save_images
from vnlb.testing import exec_search_testing
from vnlb.utils import groups2patches,patches2groups,patches_at_indices

# -- python impl --
import svnlb
# from svnlb.gpu import runSimSearch
from svnlb.utils import idx2coords

# -- check if reordered --
from scipy import optimize
SAVE_DIR = Path("./output/tests/")

def check_if_reordered(data_a,data_b):
    delta = np.zeros((len(data_a),len(data_b)))
    for a_i in range(len(data_a)):
        for b_i in range(len(data_b)):
            delta[a_i,b_i] = np.sum(np.abs(data_a[a_i]-data_b[b_i]))
    row_ind,col_ind = optimize.linear_sum_assignment(delta)
    perc_nz = (delta[row_ind, col_ind] > 0.).astype(np.float32).mean()*100
    return perc_nz

def print_value_order(group_og,gt_patches_og,c,psX,psT,nSimP):

    # -- create patches --
    order = []
    size = psX * psX * psT * c
    start,pidx = 5000,0
    gidx = 30
    group_og_f = group_og.ravel()[:size*nSimP]
    patch_cmp = gt_patches_og.ravel()[:size*nSimP]

    # -- message --
    print("Num Eq: ",len(np.where(np.abs(patch_cmp - group_og_f)<1e-10)[0]))
    print(np.where(np.abs(patch_cmp - group_og_f)<1e-10)[0])

    print("Num Neq: ",len(np.where(np.abs(patch_cmp - group_og_f)>1e-10)[0]))
    print(np.where(np.abs(patch_cmp - group_og_f)>1e-10)[0])

    # -- the two are zero at _exactly_ the same indices --
    # pzeros = np.where(np.abs(patch_cmp)<1e-10)[0]
    # print(pzeros,len(pzeros))
    # gzeros = np.where(np.abs(group_og_f)<1e-10)[0]
    # print(gzeros,len(gzeros))
    # print(np.sum(np.abs(gzeros-pzeros)))

    return

def print_neq_values(group_og,gt_patches_og):
    order = []
    skip,pidx = 0,0
    gidx = 20
    for gidx in range(0,103):
        group_og_f = group_og[0,...,gidx].ravel()
        patch_cmp = gt_patches_og[0,...,gidx].ravel()
        for i in range(patch_cmp.shape[0]):
            idx = np.where(np.abs(patch_cmp[i] - group_og_f)<1e-10)[0]
            if (i+skip*pidx) in idx: idx = i
            elif len(idx) > 0: idx = idx[0]
            else: idx = -1
            if idx != i:
                print(gidx,i,np.abs(patch_cmp[i] - group_og_f[i]))

def print_neq_values_fix_pix(group_og,gt_patches_og):
    order = []
    skip,pidx,gidx = 0,0,20
    shape = group_og.shape
    nParts,nSimP = shape[0],shape[1]
    for gidx in range(1,2):
        group_og_f = group_og.reshape(nParts,-1,nSimP).ravel()#[:,1,:].ravel()
        patch_cmp = gt_patches_og.reshape(nParts,-1,nSimP)[:,gidx,:].ravel()
        for i in range(patch_cmp.shape[0]):
            idx = np.where(np.abs(patch_cmp[i] - group_og_f)<1e-10)[0]
            print(gidx,i,idx)

def check_pairwise_diff(vals,tol=1e-4):
    # all the "same" value;
    # the order change is
    # due to small (< tol) differences
    nvals = vals.shape[0]
    # delta = np.zeros(nvals,nvals)
    for i in range(vals.shape[0]):
        for j in range(vals.shape[0]):
            # delta[i,j] = np.abs(vals[i] - vals[j])
            assert np.abs(vals[i] - vals[j]) < tol

#
#
# -- Primary Testing Class --
#
#

class TestSimSearch(unittest.TestCase):

    #
    # -- Load Data --
    #

    def do_load_data(self,vnlb_dataset):

        #  -- Read Data (Image & VNLB-C++ Results) --
        res_vnlb,paths,fmts = load_dataset(vnlb_dataset)
        clean,noisy,std = res_vnlb.clean,res_vnlb.noisy,res_vnlb.std
        fflow,bflow = res_vnlb.fflow,res_vnlb.bflow

        #  -- TV-L1 Optical Flow --
        flow_params = {"nproc":0,"tau":0.25,"lambda":0.2,"theta":0.3,
                       "nscales":100,"fscale":1,"zfactor":0.5,"nwarps":5,
                       "epsilon":0.01,"verbose":False,"testing":False,'bw':True}
        fflow,bflow = svnlb.swig.runPyFlow(noisy,std,flow_params)

        # -- pack data --
        data = edict()
        data.noisy = noisy
        data.fflow = fflow
        data.bflow = bflow

        return data,std

    def do_load_rand_data(self,t,c,h,w):

        # -- create data --
        data = edict()
        data.noisy = np.random.rand(t,c,h,w)*255.
        data.fflow = (np.random.rand(t,2,h,w)-0.5)*5.
        data.bflow = (np.random.rand(t,2,h,w)-0.5)*5.
        sigma = 20.
        data.fflow = np.zeros_like(data.fflow)
        data.bflow = np.zeros_like(data.bflow)

        for key,val in data.items():
            data[key] = data[key].astype(np.float32)

        return data,sigma


    #
    # -- [Exec] Sim Search --
    #

    def do_run_sim_search_full(self,tensors,sigma,in_params,save=True):

        # -- unpack shapes --
        # noisy = tensors.noisy[:3,:,:16,:16]
        # noisy = tensors.noisy[:3,:,:8,:8]
        noisy = tensors.noisy[:3,:,:32,:32]
        # noisy = tensors.noisy
        t,c,h,w = noisy.shape
        step = 0
        device = 0

        # -- parse parameters --
        params = svnlb.swig.setVnlbParams(noisy.shape,sigma,params=in_params)
        ps,ps_t = params.sizePatch[step],params.sizePatchTime[step]
        params.use_imread = [True,True]
        flows = {'fflow':tensors['fflow'],'bflow':tensors['bflow']}
        tensors = {'fflow':tensors['fflow'],'bflow':tensors['bflow']}

        # -- setup agg function --
        npatches = 100
        t,c,h,w = noisy.shape
        tf32 = torch.float32
        noisy_th = torch.FloatTensor(noisy).to(device)
        deno = torch.zeros_like(noisy_th)
        # patches = torch.zeros(npatches,t,ps_t,c,ps,ps,h,w).to(device)
        cs = torch.cuda.default_stream()
        cs_ptr = cs.cuda_stream
        pidx = 100

        # -- python exec --
        gpu_params = copy.deepcopy(params)
        gpu_params.nstreams = 4
        gpu_tensors = {k:torch.FloatTensor(v) for k,v in tensors.items()}
        start = time.perf_counter()
        py_inds = exec_search_testing(noisy,sigma,pidx)
        # py_data = runSimSearch(noisy,sigma,gpu_tensors,gpu_params,step)
        # py_inds = py_data.indices.clone()
        end = time.perf_counter() - start
        print("exec time: ",end)
        print("py_inds.shape: ",py_inds.shape)

        # -- python stats --
        n_invalid = torch.sum(py_inds == -1).item()
        n_zero = torch.sum(py_inds == 0).item()
        n_elems = py_inds.numel()*1.
        perc_invalid = n_invalid / n_elems * 100
        perc_zero = n_zero / n_elems * 100
        print("py_inds.shape: ",py_inds.shape)
        print("[Py] Percent Invalid: %2.1f" % perc_invalid)
        print("[Py] Percent Zero: %2.1f" % perc_zero)

        # -- python weights --
        pshape = (py_inds.shape[0],py_inds.shape[1],ps_t,c,ps,ps)
        patches = torch.zeros(pshape).to(device)
        python_weights = torch.zeros(t,h,w).type(tf32).to(device)

        # vnlb.gpu.compute_agg_batch(deno,patches,py_inds,
        #                            python_weights,ps,ps_t,cs_ptr)

        weights = python_weights.cpu().numpy()[:,None]
        wmax = weights.max().item()
        save_images(SAVE_DIR / "python_weights.png",weights,imax=wmax)

        # -- cpp exec --
        cpp_params = copy.deepcopy(params)
        cpp_data = svnlb.swig.simPatchSearch(noisy.copy(),sigma,pidx,{},
                                             copy.deepcopy(params),step)
        cpp_inds = cpp_data['indices']
        print("cpp_inds.shape: ",cpp_inds.shape)
        # cpp_inds = svnlb.swig.simSearchImage(noisy,noisy,sigma,flows,cpp_params,step)
        # cpp_inds = torch.IntTensor(cpp_inds).to(device)
        # cpp_inds = rearrange(cpp_inds,'n bT bH bW -> (bT bH bW) n')
        # print("cpp_inds.shape: ",cpp_inds.shape)

        # -- cpp stats --
        n_invalid = torch.sum(cpp_inds == -1).item()
        n_zero = torch.sum(cpp_inds == 0).item()
        n_elems = cpp_inds.numel()*1.
        perc_invalid = n_invalid / n_elems * 100
        perc_zero = n_zero / n_elems * 100
        print("cpp_inds.shape: ",cpp_inds.shape)
        print("[Cpp] Percent Invalid: %2.1f" % perc_invalid)
        print("[Cpp] Percent Zero: %2.1f" % perc_zero)

        # -- cpp weights --
        cpp_weights = torch.zeros(t,h,w).type(tf32).to(device)
        # vnlb.gpu.compute_agg_batch(deno,patches,cpp_inds,
        #                            cpp_weights,ps,ps_t,cs_ptr)
        weights = cpp_weights.cpu().numpy()[:,None]
        wmax = weights.max().item()
        save_images(SAVE_DIR / "cpp_weights.png",weights,imax=wmax)

        # -- delta --
        delta = torch.sum(torch.abs(python_weights - cpp_weights))
        delta = delta.item()
        assert delta < 1e-8

    def do_run_sim_search(self,tensors,sigma,in_params,save=True):

        # -- unpack shapes --
        # noisy = tensors.noisy[:3,:,:16,:16]
        # noisy = tensors.noisy[:3,:,:32,:32]
        noisy = tensors.noisy[:3,:,:64,:64]
        # noisy = tensors.noisy
        t,c,h,w = noisy.shape
        nframes,height,width = t,h,w
        chw = c*h*w
        hw = h*w
        step = 1
        device = 0

        print("do_run_sim_search")

        # -- parse parameters --
        params = svnlb.swig.setVnlbParams(noisy.shape,sigma,params=in_params)
        ps,ps_t = params.sizePatch[step],params.sizePatchTime[step]
        params.use_imread = [True,True]
        flows = {'fflow':tensors['fflow'],'bflow':tensors['bflow']}
        tensors = {'fflow':tensors['fflow'],'bflow':tensors['bflow']}
        tensors['basic'] = noisy.copy()
        tchecks,nchecks = 200,0
        checks = np.random.permutation(h*w*(t-1))[:10]
        # checks[0] = 2518
        for pidx in checks:

            # -- check boarder --
            pidx = pidx.item()
            pidx3 = pidx
            print("pidx3: ",pidx3)
            ti = pidx // (height*width)
            hi = (pidx % hw) // width
            wi = pidx % width
            # ti,_,wi,hi = idx2coords(pidx,1,h,w)
            valid_w = (wi + params.sizePatch[step]-1) < w
            valid_h = (hi + params.sizePatch[step]-1) < h
            # print(pidx,ti,ci,wi,hi,w,h,c,valid_w,valid_h)
            if not(valid_w and valid_h): continue

            # -- cpp exec --
            # cpp_data = svnlb.swig.simPatchSearch(noisy.copy(),sigma,pidx,
            #                                      tensors,copy.deepcopy(params),step)
            pidx4 = ti*chw + hi*w + wi
            print("pidx4: ",pidx4)
            cpp_data = svnlb.cpu.runSimSearch(noisy.copy(),sigma,pidx,
                                              tensors,copy.deepcopy(params),step)
            # -- unpack --
            cpp_patches = cpp_data["patchesNoisy"]
            cpp_group = cpp_data["groupNoisy"]
            cpp_indices = cpp_data['indices']
            cpp_psX,cpp_psT = cpp_data['psX'],cpp_data['psT']
            cpp_nSimP = cpp_data['npatches']
            cpp_ngroups = cpp_data['ngroups']
            # cpp_access = cpp_data['access']
            # cpp_nParts = cpp_data['nparts_omp']
            # cpp_vals = cpp_data['values']
            # print("-- cpp_vals --")
            # print(cpp_vals)

            # -- python exec --
            gpu_params = copy.deepcopy(params)
            gpu_params.nstreams = 4
            gpu_params.rand_mask = False
            gpu_tensors = {k:torch.FloatTensor(v) for k,v in tensors.items()}
            start = time.perf_counter()
            py_data = exec_search_testing(noisy,sigma,pidx,gpu_params,step)
            end = time.perf_counter() - start
            print("exec time: ",end)

            # -- unpack --
            py_patches = py_data.patches
            py_vals = py_data.values
            py_indices_full = py_data.indices.clone()
            py_indices = py_data.indices.cpu().numpy()
            py_ngroups = 0#py_data.ngroups
            nSimP = 0#len(py_indices)
            nflat = 0#py_data.nflat
            psX,psT = cpp_psX,cpp_psT#py_data['psX'],py_data['psT']

            # -- stats about indices --
            nzero = np.sum(py_indices == 0).item()
            nelems = py_indices.size * 1.
            print("[py_inds (all)]: perc zero %2.3f" % (nzero/nelems*100))

            nzero = np.sum(py_indices == -1).item()
            nelems = py_indices.size * 1.
            print("[py_inds (all)]: perc invalid %2.3f" % (nzero/nelems*100))

            # -- allow for swapping of "close" values --
            # print(py_access.shape,cpp_access.shape)
            # pidx = ti*height*width + hi*width + width
            # print(pidx,py_indices.shape,py_vals.shape)

            print(py_indices.shape,pidx3)#,pidx4)
            py_indices = py_indices[0,:]#.cpu().numpy()
            py_vals = py_vals[0,:].cpu().numpy()
            # py_indices = py_indices[pidx3,:]#.cpu().numpy()
            # py_vals = py_vals[pidx3,:].cpu().numpy()
            # py_vals = py_vals[:,ti,wi,hi].cpu().numpy()
            # py_access = py_access[...,ti,wi,hi].cpu().numpy()

            # -- stats about indices --
            nzero = np.sum(py_indices == 0).item()
            nelems = py_indices.size * 1.
            print("[py_inds]: perc zero %2.3f" % (nzero/nelems*100))

            nzero = np.sum(py_indices == -1).item()
            nelems = py_indices.size * 1.
            print("[py_inds]: perc invalid %2.3f" % (nzero/nelems*100))

            nzero = np.sum(cpp_indices == 0).item()
            nelems = cpp_indices.size * 1.
            print("[cpp_inds]: perc zero %2.3f" % (nzero/nelems*100))


            # -- create sim search image --
            # cpp_params = copy.deepcopy(params)
            # cpp_inds = vnlb.cpu.simSearchImage(noisy,noisy,sigma,
            #                                    flows,cpp_params,step)
            # cpp_inds = torch.IntTensor(cpp_inds).to(device)

            # -- save weights --
            # print(py_indices_full.shape,cpp_inds.shape)
            inds = py_indices_full
            # inds = cpp_inds
            t,c,h,w = noisy.shape
            tf32 = torch.float32
            noisy_th = torch.FloatTensor(noisy).to(device)
            deno = torch.zeros_like(noisy_th)
            # patches[ni,ti,pt,ci,pi,pj,hi,wi]
            pshape = (inds.shape[0],inds.shape[1],ps_t,c,ps,ps)
            patches = torch.zeros(pshape).to(device)
            # inds = torch.zeros(100,t,h,w).type(torch.int32).to(device)
            weights = torch.zeros(t,h,w).type(tf32).to(device)
            cs = torch.cuda.default_stream()
            cs_ptr = cs.cuda_stream
            # vnlb.gpu.compute_agg_batch(deno,patches,inds,
            #                            weights,ps,ps_t,cs_ptr)
            weights = weights.cpu().numpy()[:,None]
            wmax = weights.max().item()
            print("saving weights image.")
            save_images(SAVE_DIR/"weights.png",weights,imax=wmax)

            # print(py_access.shape,cpp_access.shape)
            # print(pidx)
            # for i in range(3):
            #     access = np.stack([py_access[i],cpp_access[i]],-1)
            #     delta = np.abs(py_access[i]-cpp_access[i]).sum().item()
            #     print("[%d] delta: %2.3f" % (i,delta))
            #     # print(access)
            # indices = np.stack([py_indices.ravel(),cpp_indices.ravel()],-1)
            # vals = np.stack([py_vals.ravel(),cpp_vals.ravel()],-1)
            # print(ti,ci,wi,hi)
            # print(np.c_[indices,vals])

            # args = np.where(py_indices_full[:,0].cpu().numpy() == 13647)[0]
            # print(args)
            # print(13647./args)


            pwd = np.any((py_indices[:,None] - cpp_indices[None,:])**2 < 1e-8,1)
            pmean = pwd.astype(np.float).mean()*100
            print("%% equal: %2.2f" % (pmean))

            print(py_indices/cpp_indices)
            np.testing.assert_array_equal(py_indices,cpp_indices)
            try:
                np.testing.assert_array_equal(py_indices,cpp_indices)
            except:
                # np.testing.assert_array_equal(np.sort(py_indices),np.sort(cpp_indices))
                neq_idx = np.where(cpp_indices != py_indices)
                check_pairwise_diff(py_vals[neq_idx])
            # py_order = np.array([np.where(cpp_indices==pidx)[0] for pidx in py_indices]).ravel()

            # # -- compare patches --
            # assert np.abs(cpp_ngroups - py_ngroups) == 0
            # np.testing.assert_allclose(py_patches[py_order],cpp_patches,rtol=1e-7)

            # # -- compare groups --
            # cpp_group = cpp_group.ravel()[:cpp_nSimP*psT*psX*psX*c].reshape((1,c,psT,psX,psX,nSimP))
            # py_group = py_group.ravel()[:cpp_nSimP*psT*psX*psX*c].reshape((1,c,psT,psX,psX,nSimP))
            # np.testing.assert_allclose(py_group[...,py_order],cpp_group,rtol=1e-7)

            # -- check to break --
            nchecks += 1
            if nchecks >= tchecks: break


    #
    # -- Call the Tests --
    #

    def test_sim_search(self):

        # -- init save path --
        np.random.seed(123)
        save_dir = SAVE_DIR
        if not save_dir.exists():
            save_dir.mkdir(parents=True)

        # -- no args --
        pyargs = {}
        vnlb_dataset = "davis_64x64"
        tensors,sigma = self.do_load_data(vnlb_dataset)
        # self.do_run_sim_search_full(tensors,sigma,pyargs)
        self.do_run_sim_search(tensors,sigma,pyargs)

        # -- modify patch size --
        pyargs = {'ps_x':3,'ps_t':2}
        self.do_run_sim_search(tensors,sigma,pyargs)

        # -- random data & modified patch size --
        pyargs = {}
        tensors,sigma = self.do_load_rand_data(5,3,854,480)
        tensors,sigma = self.do_load_rand_data(5,3,32,32)
        self.do_run_sim_search(tensors,sigma,pyargs)

        # -- random data & modified patch size --
        pyargs = {'ps_x':3,'ps_t':2}
        tensors,sigma = self.do_load_rand_data(5,3,32,32)
        self.do_run_sim_search(tensors,sigma,pyargs)

