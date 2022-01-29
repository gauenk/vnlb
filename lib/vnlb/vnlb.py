
# -- python deps --
import copy
import numpy as np
import torch
import torch as th
from PIL import Image
from einops import rearrange
from easydict import EasyDict as edict

# -- local imports --
# from .sim_search import runSimSearch
# from .bayes_est import runBayesEstimate
# from .comp_agg import computeAggregation
# from .proc_nlb import processNLBayes

# -- project imports --
import svnlb # for params only
# from svnlb.gpu import processNLBayes as proc_nlb
from .proc_nn import proc_nn
from .proc_nlb import processNLBayes as proc_nlb
# from vnlb.utils import groups2patches,patches2groups,compute_psnrs
# from vnlb.testing.file_io import save_images,save_image

def get_params(shape,sigma):
    params = svnlb.swig.setVnlbParams(shape,sigma)
    # params['nSimilarPatches'][0] = 100
    # params['useWeights'] = [False,False]
    # params['simPatchRefineKeep'] = [100,100]
    # params['cleanSearch'] = [True,True]
    # params['cleanSearch'] = [False,False]
    # params['variThres'] = [0.,0.]
    # params['useWeights'] = [False,False]
    # params['nfilter'] = [-1,-1]
    return params

def get_flows(shape,device):
    t,c,h,w = shape
    fflow = th.zeros(t,2,h,w).to(device)
    bflow = th.zeros(t,2,h,w).to(device)
    flows = edict({"fflow":fflow,"bflow":bflow})
    return flows

def denoise(noisy, sigma, clipped_noise, gpuid, silent, model=None):

    # -- load nn --
    # model = load_nn_model(model,sigma,gpuid)

    # -- denoise with model --
    print("noisy.device: ",noisy.device)
    basic = proc_nn(noisy,sigma,model)
    basic = basic.to(noisy.device)

    # -- denoise with vnlb --
    params = get_params(noisy.shape,sigma)
    flows = get_flows(noisy.shape,noisy.device)
    deno = proc_nlb(noisy*255.,basic*255.,sigma,1,flows,params,gpuid=gpuid)
    # processNLBayes(noisy,basic,sigma,step,flows,params,gpuid=0,clean=None)

    return deno,basic

def load_denoise_burst():

    # -- load paths --
    fns = []
    for i in range(5):
        fns.append(f"./data/dcrop_fdvd/n30_FastDVDnet_{i}.png")

    # -- load images --
    burst = []
    for fn in fns:
        img = Image.open(fn).convert('RGB')
        img = th.FloatTensor(np.array(img))
        img = rearrange(img,'h w c -> c h w')
        burst.append(img)
    burst = th.stack(burst)
    print("burst.shape: ",burst.shape)
    burst = burst.to("cuda:0")

    return burst

def runPythonVnlb_2step(noisy,sigma,flows,params,gpuid=0,clean=None):
    """

    A GPU-Python implementation of the C++ code.

    """

    # -- place on cuda --
    device = gpuid
    noisy = torch.FloatTensor(noisy).to(device)
    flows = edict({k:torch.FloatTensor(v).to(device) for k,v in flows.items()})
    basic = torch.zeros_like(noisy)

    # -- step 1 --
    # params['nSimilarPatches'][0] = 100
    # params['simPatchRefineKeep'] = [-1,-1]
    # params['nstreams'] = [1,1]
    # params['offset'] = [0.,0.]
    # step_results = processNLBayes(noisy,basic,sigma,0,flows,params,clean=clean)
    # basic_out = step_results.basic.clone()
    # basic = step_results.basic.clone()
    # py_psnr = compute_psnrs(basic.cpu().numpy(),clean)
    # print(f"[basic (clean)] psnrs: ", py_psnr)

    # params['nSimilarPatches'][0] = 100
    # params['simPatchRefineKeep'] = [-1,-1]
    # params['nstreams'] = [1,1]
    # params['offset'] = [2*(sigma/255.)**2.,0.]
    # basic = torch.zeros_like(noisy)
    # step_results = processNLBayes(noisy,basic,sigma,0,flows,params)
    # basic_out = step_results.basic.clone()
    # basic = step_results.basic.clone()
    # py_psnr = compute_psnrs(basic.cpu().numpy(),clean)
    # print(f"[basic (clean)] psnrs: ", py_psnr)

    # -- step 1 --
    # nfilter = 250 or 200 gives 29.00!
    print("sigma: ",sigma)
    print("sizeSearchWindow: ",params['sizeSearchWindow'])
    # params['sizeSearchWindow'][0] = 15
    params['nSimilarPatches'][0] = 100
    params['useWeights'] = [False,False]
    # params['simPatchRefineKeep'] = [100,100]
    # params['cleanSearch'] = [True,True]
    params['cleanSearch'] = [False,False]
    # params['variThres'] = [0.,0.]
    params['useWeights'] = [False,False]
    params['nfilter'] = [-1,-1]
    # params['nfilter'] = [300,-1]
    # params['nfilter'] = [400,-1]
    # params['simPatchRefineKeep'] = [40,60]
    params['simPatchRefineKeep'] = [100,60]
    params['offset'] = [2*(sigma/255.)**2.,0.]
    params['variThres'] = [2.7,.2]
    step_results = processNLBayes(noisy,basic,sigma,0,flows,params,clean=clean)
    # params['nSimilarPatches'][0] = 500
    # params['simPatchRefineKeep'] = [-1,-1]
    # params['offset'] = [0.,0.]
    # step_results = processNLBayes(noisy,basic,sigma,0,flows,params,clean=clean)
    basic_out = step_results.basic.clone()
    basic = step_results.basic.clone()
    py_psnr = compute_psnrs(basic.cpu().numpy(),clean)
    print(f"[basic] psnrs: ", py_psnr,np.mean(py_psnr))
    tmp = basic

    # -- proccess using NN --
    basic = load_denoise_burst()#noisy,sigma)
    print("basic.shape: ",basic.shape)
    py_psnr = compute_psnrs(basic.cpu().numpy(),clean)
    print(f"[basic:2] psnrs: ", py_psnr,np.mean(py_psnr))

    # params['nfilter'] = [-1,-1]
    # params['cleanSearch'] = [True,True]
    # step_results = processNLBayes(noisy,basic,sigma,0,flows,params,clean=clean)
    # basic_out = step_results.basic.clone()
    # basic = step_results.basic.clone()
    # py_psnr = compute_psnrs(basic.cpu().numpy(),clean)
    # print(f"[basic-clean] psnrs: ", py_psnr)
    # exit(0)

    # -- misc --
    # params['nSimilarPatches'][1] = params['nSimilarPatches'][0]
    # params['sigmaBasic'] = [sigma,sigma]
    # params['variThres'] = [2.7,2.7]

    # -- standard step 1 --
    # params['simPatchRefineKeep'] = [-1,-1]
    # params['nSimilarPatches'][1] = 250
    # params['simPatchRefineKeep'] = [25,25]

    # params['nfilter'][1] = 200
    params['cleanSearch'][1] = False
    params['nSimilarPatches'][1] = 60
    params['variThres'][1] = .2
    params['simPatchRefineKeep'][1] = 60
    alpha = 1.0
    in_basic = alpha * basic + (1 - alpha) * noisy
    step_results = processNLBayes(noisy,in_basic,sigma,1,flows,params)#,clean=clean)
    tmp = step_results.denoised.clone()
    py_psnr = compute_psnrs(tmp.cpu().numpy(),clean)
    print("[step 1]: ",py_psnr,np.mean(py_psnr))
    exit()
    # fn = "./output/deno_wclean.png"
    # stmp = tmp.cpu().numpy()
    # save_images(stmp,fn,imax=255.)


    # -- modified step 1 --
    # alpha = 0.5
    # params['variThres'][1] = 1.5
    # params['nSimilarPatches'][1] = 100
    # in_basic = alpha * basic + (1 - alpha) * noisy
    # step_results = processNLBayes(noisy,in_basic,sigma,1,flows,params)#,clean=clean)
    # tmp = step_results.denoised.clone()
    # py_psnr = compute_psnrs(tmp.cpu().numpy(),clean)
    # print("[(mod) step 1]: ",py_psnr,np.mean(py_psnr))


    # exit()

    # -- other --
    # params['nSimilarPatches'][1] = 50
    # params['variThres'][1] = 0.2
    # alpha = 0.8
    # in_basic = alpha * tmp + (1 - alpha) * noisy
    # step_results = processNLBayes(noisy,in_basic,sigma,1,flows,
    #                               params)#,clean=clean)
    # tmp = step_results.denoised.clone()

    # py_psnr = compute_psnrs(tmp.cpu().numpy(),clean)
    # print("[step 2]: ",py_psnr)

    # -- explore --
    # explore_depth(noisy,tmp,sigma,flows,params,clean)
    # tmp = basic
    explore_alpha_patch_sched(noisy,tmp,sigma,flows,params,clean)
    exit()

    # -- step 2 --
    # alpha_sched = [0.1,0.15,0.25,0.3,0.4,0.45]

    # alpha_sched = [0.5,0.6,0.7,0.8]#,0.9]
    # patch_sched = [150,60,60,40]#,40]

    alpha_sched = [.5,.8,.9,1.]#,0.9]
    patch_sched = [100,50,50,50]#,40]
    thresh_sched = [0.8,0.1,0.8,.2]

    # alpha_sched = [0.5,0.6,0.7,0.8,0.9]
    # patch_sched = [150,120,100,80,60,50,40,20,10]
    # for idx in range(len(alpha_sched)):

    #     alpha = alpha_sched[idx]
    #     patches = patch_sched[idx]
    #     thresh = thresh_sched[idx]
    #     params['nSimilarPatches'][1] = patches
    #     params['variThres'][1] = thresh

    #     in_basic = alpha * basic + (1 - alpha) * noisy
    #     step_results = processNLBayes(noisy,in_basic,sigma,1,flows,
    #                                       params,clean=clean)
    #     denoised = step_results.denoised.clone()
    #     basic = step_results.denoised.clone()

    #     py_psnr = compute_psnrs(denoised.cpu().numpy(),clean)
    #     m_psnr = np.mean(py_psnr)
    #     print(f"[alpha = {alpha:.2}, patches = {patches}] psnrs: ", py_psnr,m_psnr)
    denoised = tmp

    # -- format --
    results = edict()
    results.basic = basic_out
    results.denoised = denoised

    return results

def explore_depth(noisy,basic,sigma,flows,params,clean):
    banner = "Exploring depth!"
    print(banner)
    alpha_sched = [1.,1.,1.]
    thresh_sched = [2.7,2.7,2.7]
    patch_sched = [50,50,50]
    for idx in range(len(thresh_sched)):
        alpha = alpha_sched[idx]
        patches = patch_sched[idx]
        thresh = thresh_sched[idx]

        params['nSimilarPatches'][1] = patches
        params['variThres'][1] = thresh
        in_basic = alpha * basic + (1 - alpha) * noisy

        try:
            step_results = processNLBayes(noisy,in_basic,sigma,1,
                                          flows,params)#,clean=clean)
            denoised = step_results.denoised.clone()
            py_psnr = compute_psnrs(denoised.cpu().numpy(),clean)
            m_psnr = np.mean(py_psnr)
            print(f"[alpha = {alpha:.2}, patches = {patches}, thresh = {thresh:.2}] psnrs: ", py_psnr,m_psnr)
        except:
            print(f"[alpha = {alpha:.2}, patches = {patches}, thresh = {thresh:.2}]: error... skipping.")
        basic = denoised

def explore_alpha_patch_sched(noisy,basic,sigma,flows,params,clean):
    # alpha_sched = [0.,0.1,0.5,0.8,0.9,1.]
    # alpha_sched = [0.1,0.25,0.5,0.8,0.9,1.]
    # alpha_sched = [0.5,0.8,0.9,1.]
    alpha_sched = [1.]
    # alpha_sched = [0.8,0.9,1.]
    # alpha_sched = [0.9,1.]
    # alpha_sched = [1.]
    # alpha_sched = [0.8,0.9,1.]
    # alpha_sched = [0.1,0.5,0.6,0.7,0.8,0.9,0.99]
    # patch_sched = [300,200,150,100,50,10]
    # patch_sched = [300,200,150,50]#,100,50,10]
    # patch_sched = [300,200,150,120,100,80,60,50,40,20,10]
    # patch_sched = [500,300,100,50]#,75,50,25]#,100,50,10]
    patch_sched = [80,60,30]#,30,20,10]
    # patch_sched = [200,176,150,125,100,50]#,100,50,10]
    # thresh_sched = [2.7,1.5,0.8,.2]
    # thresh_sched = [2.7,1.5,0.8,0.4,.2,.1,.05]
    thresh_sched = [.2]#,1.]
    # thresh_sched = [.2,2.7,1.5]
    nfilter_sched = [-1,200,100]

    for kdx in range(len(thresh_sched)):
        for idx in range(len(nfilter_sched)):
        # for idx in range(len(alpha_sched)):
            for jdx in range(len(patch_sched)):

                # -- vars --
                alpha = alpha_sched[0]
                nfilter = nfilter_sched[idx]
                # alpha = alpha_sched[idx]
                patches = patch_sched[jdx]
                thresh = thresh_sched[kdx]
                # params['nfilter'] = [200,-1]

                # -- skip conds --
                if nfilter <= patches: continue

                # -- params --
                # print(params['nSimilarPatches'][1])
                params['nSimilarPatches'][1] = patches
                params['variThres'][1] = thresh
                params['nfilter'][1] = nfilter
                params['simPatchRefineKeep'][1] = 60


                in_basic = alpha * basic.clone() + (1 - alpha) * noisy.clone()
                try:
                    _params = copy.deepcopy(params)
                    step_results = processNLBayes(noisy.clone(),in_basic,sigma,1,
                                                  flows,_params)#,clean=clean)
                    denoised = step_results.denoised.clone()
                    py_psnr = compute_psnrs(denoised.cpu().numpy(),clean)
                    m_psnr = np.mean(py_psnr)
                    print(f"[nfilter = {nfilter:3}, patches = {patches}, thresh = {thresh:.2}, psnrs = {m_psnr:2.4}]")
                    #print(f"[alpha = {alpha:.2}, patches = {patches}, thresh = {thresh:.2}, psnrs = {m_psnr:2.4}]")
                    # print(f"[alpha = {alpha:.2}, patches = {patches}, thresh = {thresh:.2}] psnrs: ", py_psnr,m_psnr)
                except Exception as e:
                    print(e)
                    print(f"[alpha = {alpha:.2}, patches = {patches}, thresh = {thresh:.2}]: error... skipping.")

    """

    alpha = 0.9
    patches = 60
    thresh = 0.2
    28.94

    alpha = 0.8
    patches = 60
    thresh = 1.5

    [step 1]

    alpha = 0.
    pathcces = 125
    thresh = 2.7

    alpha = 0.1
    patches = 100
    thresh = 2.7

    alpha = 0.5
    patches = 100
    thresh = 0.8

    alpha = 0.8
    patches = 50
    thresh = 0.1

    alpha = 0.9
    patches = 50
    thresh = 0.8

    alpha = 1.0
    patches = 50
    thresh = 0.2

    [step 2]

    

    """

def runPythonVnlb_clean(noisy,clean,sigma,flows,params,gpuid=0):
    """

    A GPU-Python implementation of the C++ code.
    using the a clean reference to estimate the eigenvals

    """

    # -- place on cuda --
    device = gpuid
    noisy = torch.FloatTensor(noisy).to(device)
    clean = torch.FloatTensor(clean).to(device)
    flows = edict({k:torch.FloatTensor(v).to(device) for k,v in flows.items()})
    basic = torch.zeros_like(noisy)
    params['bsize_s'] = [128,128]
    params['nSimilarPatches'] = [100,100]

    # -- step using clean values --
    step_results = processNLBayes(noisy,noisy,sigma,1,flows,params)
    deno = step_results.denoised.clone()

    # -- format --
    results = edict()
    results.basic = noisy
    results.denoised = deno

    return results
