def wait_streams(waiting,waitOn):

    # -- create events for each stream --
    events = []
    for stream in waitOn:
        event = torch.cuda.Event(blocking=False)
        event.record(stream)
        # stream.record_event(event)
        events.append(event)

    # -- issue wait for all streams in waitOn and all events --
    for stream in waiting:
        for event in events:
            # stream.wait_event(event)
            event.wait(stream)

def get_hw_batches(h,w,bsize):
    hbatches = torch.arange(0,h,bsize)
    wbatches = torch.arange(0,w,bsize)
    return hbatches,wbatches


def exec_step(noisy,basic,weights,sigma,flows,params,step):
    """

    Primary sub-routine of VNLB

    """


    # -- init denoised image --
    shape = noisy.shape
    t,c,h,w = noisy.shape
    chnls = c
    # deno = basic if step == 0 else np.zeros_like(noisy)
    deno = np.zeros_like(noisy)

    # -- init mask --
    minfo = initMask(noisy.shape,params,step)
    # minfo = vnlb.init_mask(noisy.shape,params,step)
    mask,n_groups = minfo['mask'],minfo['ngroups']

    # -- color xform --
    noisy_yuv = apply_color_xform_cpp(noisy)
    basic_yuv = apply_color_xform_cpp(basic)

    # -- init looping vars --
    npixels = t*h*w
    g_remain = n_groups
    g_counter = 0

    # -- run npixels --
    for pidx in range(npixels):

        # -- pix index to coords --
        # pidx = t*wh + y*width + x;
        ti = pidx // (w*h)
        hi = (pidx - ti*w*h) // w
        wi = pidx - ti*w*h - hi*w

        # pidx3 = t*whc + c*wh + y*width + x

        # ti,ci,hi,wi = idx2coords(pidx,c,h,w)
        # pidx3 = coords2idx(ti,hi,wi,1,h,w)
        # t1,c1,h1,w1 = idx2coords(pidx3,1,h,w)
        pidx3 = ti*w*h*c + hi*w + wi

        # -- skip masked --
        if not(mask[ti,hi,wi] == 1): continue
        # print("mask: ",mask[0,0,0],mask[0,0,1],mask[0,0,2],mask[0,0,3])

        # -- inc counter --
        # if g_counter > 2: break
        # print("group_counter: %d" % g_counter)
        g_counter += 1
        # print("(t,h,w,-): %d,%d,%d,%d" %(ti,hi,wi,mask[1,0,24]))
        # print("ij,ij3: %d,%d\n" % (pidx,pidx3))

        # -- sim search --
        sim_results = estimateSimPatches(noisy,basic,sigma,pidx3,flows,params,step)
        groupNoisy,groupBasic,indices = sim_results
        nSimP = len(indices)

        # -- optional flat patch --
        flatPatch = False
        if params.flatAreas[step]:
            # flatPatch = runFlatAreas(groupNoisy,groupBasic,nSimP,chnls)
            psX,psT = params.sizePatch[step],params.sizePatchTime[step]
            gamma = params.gamma[step]
            flatPatch = runFlatAreas(groupNoisy,psX,psT,nSimP,chnls,gamma,sigma)

        # -- bayes estimate --
        rank_var = 0.
        groupNoisy,rank_var = computeBayesEstimate(groupNoisy,groupBasic,
                                                   nSimP,shape,params,
                                                   step,flatPatch)
        # print(groupNoisy.ravel()[0])

        # -- debug zone. --
        # from vnlb.pylib.tests import save_images
        # print(groups.shape)
        # patches_yuv = groups2patches(groups,c,7,2,groups.shape[-1])[:100]
        # patches_yuv = groups2patches(groups,c,7,2,groups.shape[-1],1)
        # patches_rgb = yuv2rgb_cpp(patches_yuv)
        # print(patches_rgb)
        # save_images(patches_rgb,f"output/patches_{pidx}.png",imax=255.)

        # -- aggregate results --
        deno,weights,mask,nmasked = computeAgg(deno,groupNoisy,indices,weights,
                                               mask,nSimP,params,step)
        # print("deno.ravel()[0]: ",deno.ravel()[0])
        # print("deno.ravel()[1]: ",deno.ravel()[1])
        g_remain -= nmasked

    # -- reduce using weighted ave --
    weightedAggregation(deno,noisy_yuv,weights)
    # deno = numpy_div0(deno,weights[:,None],0.)
    # print(weights[0,0,0])
    # print(deno[0,0,0])

    # -- re-colorize --
    deno = yuv2rgb_cpp(deno)
    # basic = yuv2rgb_cpp(basic)

    # -- pack results --
    results = edict()
    results.denoised = deno if step == 1 else np.zeros_like(deno)
    results.basic = deno
    results.ngroups = g_counter

    return results

