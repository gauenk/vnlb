
# -- torch --
import torch as th
import torch.optim as optim

# -- local imports --
from .models import SingleSTN
from vnlb.utils import save_image,compute_psnrs

def stn_basic_est(images,args):

    images.basic = None
    sigma = args.sigma
    basic = stn_basic_est_loop(images,args,sigma)
    for i in range(0):
        images.basic = basic
        sigma_b = sigma
        basic = stn_basic_est_loop(images,args,sigma_b)
    return basic

def stn_basic_est_loop(images,args,sigma):

    # -- run each sample --
    nframes = images.shape[0]
    basic = []
    for t in range(nframes):
        basic_t = stn_basic_est_ref(images,args,t,sigma)
        basic.append(basic_t)
    basic = th.stack(basic)
    # print("basic.shape: ",basic.shape)
    return basic

def stn_basic_est_ref(images,args,ref,sigma):

    # -- unpack --
    if images.basic is None: burst = images.noisy/255.
    else: burst = images.basic/255.
    burst = images.clean/255.
    sigma = 0.
    burst = burst.double()
    device,shape = burst.device,burst.shape

    # -- create model --
    model = SingleSTN(sigma, ref, shape, device)
    optimizer = optim.Adam(model.parameters(),lr=1e-4)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    # -- run steps --
    loss_prev = 10000
    niters = 1000
    for i in range(niters):

        # -- scheduler --
        if (i % 10) == 0 and i > 100:
            scheduler.step()

        # -- zero grad --
        optimizer.zero_grad()
        model.zero_grad()

        # -- state loss --
        loss = model.align_loss(burst,images.clean/255.)
        if i % 100 == 0:
            print("loss.item(): ",loss.item())

        # -- info --
        if i in [0]:
            # -- Save Examples --
            warped = model(burst).detach()
            path = "output/example/"
            nframes = warped.shape[0]
            for t in range(nframes):
                fn = "iter_%d_warped_%05d.png" % (i,t)
                save_image(warped[t],path,fn)

        # -- update --
        loss.backward()
        optimizer.step()

        # -- dloss --
        dloss = abs(loss.item() - loss_prev)
        loss_prev = loss.item()
        if dloss < 1e-10: break

    # -- compute basic estimate --
    warped = model(burst).detach()

    # -- Save Examples --
    path = "output/example/"
    nframes = warped.shape[0]
    for t in range(nframes):
        save_image(warped[t],path,"warped_%05d.png" % t)
        save_image(burst[t],path,"pwarped_%05d.png" % t)

    # -- mean --
    burst = images.noisy.double()/255.
    basic = model.wmean(burst).detach()*255.

    # -- compare --
    print("[Warped] PSNRS:")
    warped = model(images.clean.double()).detach()
    psnrs = compute_psnrs(images.clean[[ref]],warped)
    print(psnrs)
    print(psnrs.mean())

    # print("[Clean-Ref] PSNRS:")
    # psnrs = compute_psnrs(images.clean[[ref]],images.clean)
    # print(psnrs)
    # print(psnrs.mean())

    psnrs = compute_psnrs(basic,images.clean[ref])
    print("[Basic] PSNRS:")
    print(psnrs)

    psnrs = compute_psnrs(images.noisy,images.clean)
    print("[Noisy] PSNRS:")
    print(psnrs)
    print(psnrs.mean())


    return basic


