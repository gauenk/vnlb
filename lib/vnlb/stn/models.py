import torch
import torch as th
import torch.nn as nn
from torch.distributions.normal import Normal
import torch.nn.functional as F
from einops import rearrange,repeat

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):
        # transform the input
        x = self.stn(x)

        # Perform the usual forward pass
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class SingleSTN(nn.Module):

    def __init__(self, sigma, ref, shape, device, igrid=None, mode='bilinear'):
        super(SingleSTN, self).__init__()
        self.sigma = sigma
        self.ref = ref
        self.mode = mode
        self.shape = shape
        self.device = device
        t,c,h,w = shape
        self.scale = 2
        self.nonref = [ti for ti in range(t) if ti != self.ref]
        self.nonref = th.LongTensor(self.nonref).to(device)

        # -- theta --
        # self.theta = self.init_theta(t)
        # self.theta = nn.parameter.Parameter(self.theta)

        # -- grid --
        # self.grid = th.zeros((t,2,h,w)).to(device).double()
        if igrid is None: self.grid = self.init_grid(t,h,w,device,th.double)
        else: self.grid = igrid.clone()
        # self.grid = th.zeros((t,2,h//self.scale,w//self.scale)).to(device).double()
        # self.upsample = nn.Upsample(scale_factor=self.scale)
        self.grid = nn.parameter.Parameter(self.grid)

        # -- pooling --
        self.apool = nn.AvgPool2d(3,stride=1)
        self.mpool = nn.MaxPool2d(7,stride=1)
        self.upool = nn.Upsample(size=(h,w))

    def apply_pool(self,images,ksize,stride):
        nn_pool = nn.AvgPool2d(ksize,stride=1)
        return nn_pool(images)

    def init_theta(self,t):
        theta = th.zeros((t,2,3)).to(self.device).double()
        theta[:,0,0] = 1
        theta[:,1,1] = 1
        return theta

    def init_grid(self,t,h,w,device,dtype):
        vectors = [th.arange(0, s, device=device, dtype=dtype)/s for s in [w,h]]
        vectors = [2*(vectors[i]-0.5) for i in range(len(vectors))]
        grids = th.meshgrid(vectors)
        grid  = th.stack(grids) # y, x, z
        grid  = th.unsqueeze(grid, 0)  #add batch
        grid = repeat(grid,'1 c w h -> t c h w',t=t)
        return grid

    def forward(self, x):
        return self.forward_v1(x)
        # return self.forward_v2(x)

    def forward_v2(self,x):
        theta = self.theta
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid, mode="bicubic")
        return x

    def forward_v1(self,x):
        # ugrid = self.upsample(self.grid)
        ugrid = self.grid
        ugrid = rearrange(ugrid,'t c h w -> t h w c')
        x = F.grid_sample(x, ugrid, mode="bicubic")
        return x

    def align_loss(self, burst, clean=None):
        t,c,h,w = burst.shape

        # -- warp loss --
        warp = self.forward(burst)
        offset = 0#2*(30./255.)**2
        tloss = self.warp_loss(warp,burst,offset)

        warp_s = self.apply_pool(warp,3,3)
        burst_s = self.apply_pool(burst,3,3)
        offset = 0#2*(30./255.)**2/9.
        tloss += self.warp_loss(warp_s,burst_s,offset)

        # warp_s = self.apool(warp_s)
        # burst_s = self.apool(burst_s)
        # if not(clean is None):
        #     print("clean.shape: ",clean.shape)
        #     print("burst.shape: ",burst.shape)
        #     est_std = (burst - clean).std()
        #     print(est_std*255.)
        #     clean_s = self.apply_pool(clean,3,3)
        #     print("burst_s.shape: ",burst_s.shape)
        #     est_std = (burst_s - clean_s).std()
        #     print(est_std*255.)
        #     clean_s = self.apool(clean)
        #     print("clean_s.shape: ",clean_s.shape)
        #     print(est_std,2*(30./255.)**2)
        #     clean_s = self.apool(clean_s)
        #     est_std = ((burst_s - clean_s)**2).mean()
        #     print(est_std,(2*(30./255.)**2)/est_std)

        # warp_s = self.apply_pool(warp,5,3)
        # burst_s = self.apply_pool(burst,5,3)
        # offset = 2*(30./255.)**2/(25.)
        # tloss += self.warp_loss(warp_s,burst_s,offset)

        # -- smooth grid --

        return tloss

    def warp_loss(self,warp,burst,offset):
        mwarp = warp[0:].mean(0,keepdim=True)
        dwarp = (burst[[self.ref]] - warp)**2
        # dwarp += (burst[[self.ref]] - mwarp)**2
        # dwarp = dwarp.mean((0,1))

        ref = self.ref
        nonref = self.nonref
        rwarp = dwarp[ref].mean()
        owarp = dwarp[nonref].mean((-2,-1))
        # print("owarp.shape: ",owarp.shape)
        owarp = th.abs(dwarp[nonref] - offset).mean()
        dwarp = (rwarp + owarp)/2.
        # dwarp = dwarp.mean()
        return dwarp

    def wmean(self, burst):

        # -- warp loss --
        warp = self.forward(burst)
        # print("warp.shape: ",warp.shape)
        vals = ((burst[[self.ref]] - warp)**2).mean(1,keepdim=True)

        # -- weights --
        weights = th.exp(-vals/2)
        weights /= weights.sum(0,keepdim=True)
        # print(weights.shape)
        # print(weights)

        # -- pool weights --
        # pweights = self.mpool(weights)
        # weights = self.upool(pweights)

        # -- weighted mean --
        wmean = (weights * warp).sum(0)
        print("wmean.shape: ",wmean.shape)
        return wmean

class SpatialTransformer(nn.Module):
    """
    [SpatialTransformer] represesents a spatial transformation block
    that uses the output from the UNet to preform an grid_sample
    https://pytorch.org/docs/stable/nn.functional.html#grid-sample
    """
    def __init__(self, mode='bilinear'):
        super(SpatialTransformer, self).__init__()
        self.mode = mode

    def gen_grid(self, flow):
        vectors = [torch.arange(0, s, device=flow.device, dtype=flow.dtype) for s in flow.shape[2:]]
        grids = torch.meshgrid(vectors)
        grid  = torch.stack(grids) # y, x, z
        grid  = torch.unsqueeze(grid, 0)  #add batch
        #grid = grid.type(torch.FloatTensor)
        return grid

    def forward(self, flow, *args):
        """
        Push the src and flow through the spatial transform block
            :param src: the original moving image
            :param flow: the output from the U-Net
        """
        new_locs = self.gen_grid(flow) + flow

        shape = flow.shape[2:]

        # Need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:,i,...] = 2*(new_locs[:,i,...]/(shape[i]-1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1,0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2,1,0]]

        if len(args) == 1:
            return F.grid_sample(args[0], new_locs, mode=self.mode, align_corners=True)
        else:
            return tuple(F.grid_sample(arg, new_locs, mode=self.mode, align_corners=True) for arg in args)
