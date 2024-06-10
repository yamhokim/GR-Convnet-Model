# V1Block as defined in https://proceedings.neurips.cc/paper/2020/hash/98b17f068d5d9b7668e19fb8ae470841-Abstract.html
# Code adapted from https://github.com/dicarlolab/vonenet

import torch
from torch import nn 
import numpy as np
import torch.nn.functional as F 
import scipy.stats as stats

np.seterr(divide='ignore', invalid='ignore')

def sample_dist(hist, bins, ns, scale='linear'):
    rand_sample = np.random.rand(ns)
    if scale == 'linear':
        rand_sample = np.interp(rand_sample, np.hstack(([0], hist.cumsum())), bins)
    elif scale == 'log2':
        rand_sample = np.interp(rand_sample, np.hstack(([0], hist.cumsum())), np.log2(bins))
        rand_sample = 2**rand_sample
    elif scale == 'log10':
        rand_sample = np.interp(rand_sample, np.hstack(([0], hist.cumsum())), np.log10(bins))
        rand_sample = 10**rand_sample
    return rand_sample

class GaborKernel(nn.Module):

    def __init__(self, kernel_size):
        super(GaborKernel, self).__init__()
        self.kernel_size = kernel_size

    def forward(self, freq, sigma_x, sigma_y, theta, offset):
        w = self.kernel_size // 2
        grid_val = torch.arange(-w, w+1, dtype=torch.float)
        x, y = torch.meshgrid(grid_val, grid_val, indexing="ij")
        rotx = x * torch.cos(theta) + y * torch.sin(theta)
        roty = -x * torch.sin(theta) + y * torch.cos(theta)
        g = torch.zeros(y.shape)
        g[:] = torch.exp(-0.5 * (rotx ** 2 / sigma_x ** 2 + roty ** 2 / sigma_y ** 2))
        g /= 2 * torch.pi * sigma_x * sigma_y
        g *= torch.cos(2 * torch.pi * freq * rotx + offset)
        return g

class GaborFilterBank(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        image_size=224,
        kernel_size=25,
        stride=4,
        visual_degrees=8
    ):
        super(GaborFilterBank, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (stride, stride)
        self.padding = (kernel_size // 2, kernel_size // 2)
        self.image_size = image_size
        self.visual_degrees = visual_degrees

        # Param instatiations
        self.weight = torch.zeros((out_channels, in_channels, kernel_size, kernel_size))
        self.gabor_kernel = GaborKernel(kernel_size=self.kernel_size[0])
        self.initialize()

    def forward(self, x):
        return F.conv2d(x, self.weight, None, self.stride, self.padding)
    
    def generate_gabor_parameters(self, sf_corr=0.75, sf_max=9, sf_min=0):
        # Generates random sample
        features = self.out_channels

        phase_bins = np.array([0, 360])
        phase_dist = np.array([1])

        # DeValois 1982a
        ori_bins = np.array([-22.5, 22.5, 67.5, 112.5, 157.5])
        ori_dist = np.array([66, 49, 77, 54])
        ori_dist = ori_dist / ori_dist.sum()

        # Schiller 1976
        cov_mat = np.array([[1, sf_corr], [sf_corr, 1]])

        # Ringach 2002b
        nx_bins = np.logspace(-1, 0.2, 6, base=10)
        ny_bins = np.logspace(-1, 0.2, 6, base=10)
        n_joint_dist = np.array([[2.,  0.,  1.,  0.,  0.],
                                [8.,  9.,  4.,  1.,  0.],
                                [1.,  2., 19., 17.,  3.],
                                [0.,  0.,  1.,  7.,  4.],
                                [0.,  0.,  0.,  0.,  0.]])
        n_joint_dist = n_joint_dist / n_joint_dist.sum()
        nx_dist = n_joint_dist.sum(axis=1)
        nx_dist = nx_dist / nx_dist.sum()
        ny_dist_marg = n_joint_dist / n_joint_dist.sum(axis=1, keepdims=True)

        # DeValois 1982b
        sf_bins = np.array([0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0, 5.6, 8])
        sf_dist = np.array([4,  4,  8, 25, 32, 26, 28, 12])

        sfmax_ind = np.where(sf_bins <= sf_max)[0][-1]
        sfmin_ind = np.where(sf_bins >= sf_min)[0][0]

        sf_bins = sf_bins[sfmin_ind:sfmax_ind+1]
        sf_dist = sf_dist[sfmin_ind:sfmax_ind]

        sf_dist = sf_dist / sf_dist.sum()

        phase = sample_dist(phase_dist, phase_bins, features)
        ori = sample_dist(ori_dist, ori_bins, features)
        ori[ori < 0] = ori[ori < 0] + 180

        samps = np.random.multivariate_normal([0, 0], cov_mat, features)
        samps_cdf = stats.norm.cdf(samps)

        nx = np.interp(samps_cdf[:,0], np.hstack(([0], nx_dist.cumsum())), np.log10(nx_bins))
        nx = 10**nx

        ny_samp = np.random.rand(features)
        ny = np.zeros(features)
        for samp_ind, nx_samp in enumerate(nx):
            bin_id = np.argwhere(nx_bins < nx_samp)[-1]
            ny[samp_ind] = np.interp(ny_samp[samp_ind], np.hstack(([0], ny_dist_marg[bin_id, :].cumsum())),
                                            np.log10(ny_bins))
        ny = 10**ny
        sf = np.interp(samps_cdf[:,1], np.hstack(([0], sf_dist.cumsum())), np.log2(sf_bins))
        sf = 2**sf
        return_val = (sf, ori, phase, nx, ny)
        return (torch.from_numpy(i) for i in return_val)
        
    def initialize(self):
        ppd = self.image_size / self.visual_degrees
        sf, theta, phase, sigx, sigy = self.generate_gabor_parameters()

        sf /= ppd
        sigx /= sf
        sigy /= sf
        theta *= torch.pi/180
        phase *= torch.pi/180

        random_channel = torch.randint(0, self.in_channels, (self.out_channels,))
        for i in range(self.out_channels):
            self.weight[i, random_channel[i]] = self.gabor_kernel(
                freq=sf[i],
                sigma_x=sigx[i],
                sigma_y=sigy[i],
                theta=theta[i],
                offset=phase[i]
            )
        self.weight = nn.Parameter(self.weight, requires_grad=False)

class V1Block(nn.Module):

    def __init__(
        self,
        in_channels, out_channels,
        image_size=224, kernel_size=25, stride=4,
        simple_channels=128, complex_channels=128, visual_degrees=8, k_exc=25
    ):
        super(V1Block, self).__init__()
        self.intermediate_channels = simple_channels + complex_channels
        self.simple_channels = simple_channels
        self.k_exc = k_exc
        self.complex_channels = complex_channels
        self.noise_scale, self.noise_level = 1, 1
        self.noise_mode, self.fixed_noise = None, None

        self.simple_conv_q0 = GaborFilterBank(
            image_size=image_size, kernel_size=kernel_size, stride=stride,
            in_channels=in_channels, out_channels=self.intermediate_channels,
            visual_degrees=visual_degrees
        )
        self.simple_conv_q1 = GaborFilterBank(
            image_size=image_size, kernel_size=kernel_size, stride=stride,
            in_channels=in_channels, out_channels=self.intermediate_channels,
            visual_degrees=visual_degrees
        )
        self.output_bottleneck_layer = nn.Conv2d(
            in_channels=self.intermediate_channels, out_channels=out_channels,
            kernel_size=(1, 1), stride=(1, 1), bias=False
        )
        
        self.simple_activation = nn.ReLU()
        self.complex_activation = nn.Identity()

    def gabor_function(self, x):
        q0_activation = self.simple_conv_q0(x)
        q1_activation = self.simple_conv_q1(x)
        c = torch.sqrt(
            (q0_activation[:, self.simple_channels:, :, :] ** 2
            + q1_activation[:, self.simple_channels:, :, :] ** 2) / 2
        )
        s = F.relu(q0_activation[:, 0:self.simple_channels, :, :])
        return self.k_exc * torch.cat((s, c), 1)

    def noise_function(self, x):
        eps = 10e-5
        x *= self.noise_scale
        x += self.noise_level
        if self.fixed_noise is not None:
            x += self.fixed_noise * torch.sqrt(F.relu(x.clone()) + eps)
        else:
            x += torch.distributions.normal.Normal(torch.zeros_like(x), scale=1).rsample() * \
                    torch.sqrt(F.relu(x.clone()) + eps)
        x -= self.noise_level
        x /= self.noise_scale
        return F.relu(x)

    def forward(self, x):
        x = self.gabor_function(x)
        x = self.noise_function(x)
        return self.output_bottleneck_layer(x)


if __name__ == "__main__":
    block = V1Block(4, 64)
    x = torch.randn(32, 4, 224, 224)
    output = block(x)
    print("extra line")