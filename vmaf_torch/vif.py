"""VIF metric

Based on:
    http://live.ece.utexas.edu/research/Quality/vifvec_release.zip
    https://github.com/Netflix/vmaf/blob/master/libvmaf/src/feature/vif.c

References:
    https://ieeexplore.ieee.org/abstract/document/1576816/

Todo:
    check that padding modes are the same as C code for nonstandard resolutions

"""

import torch
import torch.nn.functional as F
from .utils import gaussian_kernel


class VIF(torch.nn.Module):

    def __init__(self, NEG=False):
        super().__init__()

        if NEG:
            self.vif_enhn_gain_limit = 1.
        else:
            self.vif_enhn_gain_limit = 100.

        self.sigma_nsq = 2
        self.sigma_max_inv = 4. / (255.*255.)

        self.scales = (0, 1, 2, 3)
        for scale in self.scales:
            N = pow(2, 4 - scale) + 1
            win = gaussian_kernel(kernel_size=N, sigma=N/5)
            self.register_buffer(f'gaussian_kernel_{scale}', win)

    def forward(self, ref, dist):
        return self.vif_features(ref, dist)  # all VIF features are used for VMAF score regresion

    def vif_features(self, ref, dist):
        num, den = self.vif_num_den(ref, dist)
        features = num/den
        return features               # [batch_size, num_scales]

    def vif_score(self, ref, dist):
        num, den = self.vif_num_den(ref, dist)
        score = torch.sum(num, dim=-1)/torch.sum(den, dim=-1)
        return score                  # [batch_size]

    def vif_features_and_score(self, ref, dist):
        num, den = self.vif_num_den(ref, dist)
        features = num/den
        score = torch.sum(num, dim=-1)/torch.sum(den, dim=-1)
        return features, score

    def vif_num_den(self, ref, dist):

        assert len(ref.shape) == 4 and len(ref.shape) == 4, f'Expected tensors in [b,c,h,w] format, got {ref.shape} and {dist.shape}'

        num = []
        den = []

        ref = ref - 128
        dist = dist - 128

        w = ref.shape[-1]
        h = ref.shape[-2]

        for scale in self.scales:

            win = self.get_buffer(f'gaussian_kernel_{scale}')
            kernel_size = win.shape[-1]
            pad_size = [(kernel_size-1)//2]*4

            if (scale > 0):
                ref = F.conv2d(F.pad(ref, pad_size, mode='reflect'), weight=win, stride=2)
                dist = F.conv2d(F.pad(dist, pad_size, mode='reflect'), weight=win, stride=2)
                w = w//2
                h = h//2
                ref = ref[:, :, :h, :w]
                dist = dist[:, :, :h, :w]

            # this is the slowest part of VMAF
            # mu1 = F.conv2d(F.pad(ref, pad_size, mode='reflect'), weight=win)
            # mu2 = F.conv2d(F.pad(dist, pad_size, mode='reflect'), weight=win)
            # mu1_sq = mu1 * mu1
            # mu2_sq = mu2 * mu2
            # mu1_mu2 = mu1 * mu2
            # sigma1_sq = F.conv2d(F.pad(ref**2, pad_size, mode='reflect'), weight=win) - mu1_sq
            # sigma2_sq = F.conv2d(F.pad(dist**2, pad_size, mode='reflect'), weight=win) - mu2_sq
            # sigma12 = F.conv2d(F.pad(ref*dist, pad_size, mode='reflect'), weight=win) - mu1_mu2

            # same as commented code above but ~2x faster on cpu, slightly faster on gpu
            win_ = torch.cat([win]*5, dim=0)
            input_ = torch.cat([ref, dist, ref**2, dist**2, ref*dist,], dim=1)
            input_ = F.pad(input_, pad_size, mode='reflect')
            output_ = F.conv2d(input_, win_, groups=5)
            mu1, mu2, sigma1_sq, sigma2_sq, sigma12 = [t.unsqueeze(1) for t in torch.unbind(output_, dim=1)]
            sigma1_sq = sigma1_sq - mu1**2
            sigma2_sq = sigma2_sq - mu2**2
            sigma12 = sigma12 - mu1*mu2

            sigma1_sq = F.relu(sigma1_sq)
            sigma2_sq = F.relu(sigma2_sq)

            g = torch.divide(sigma12, (sigma1_sq+1e-10))

            sv_sq = sigma2_sq - torch.multiply(g, sigma12)

            g = g.masked_fill(sigma1_sq < 1e-10, 0)
            sv_sq[sigma1_sq < 1e-10] = sigma2_sq[sigma1_sq < 1e-10]
            sigma1_sq = sigma1_sq.masked_fill(sigma1_sq < 1e-10, 0)

            g = g.masked_fill(sigma2_sq < 1e-10, 0)
            sv_sq = sv_sq.masked_fill(sigma2_sq < 1e-10, 0)

            sv_sq[g < 0] = sigma2_sq[g < 0]
            g = F.relu(g)
            sv_sq = sv_sq.masked_fill(sv_sq <= 1e-10, 1e-10)

            g = torch.clamp(g, max=self.vif_enhn_gain_limit)

            num_ar = torch.log2(1 + ((g**2) * sigma1_sq)/(sv_sq+self.sigma_nsq))
            den_ar = torch.log2(1 + sigma1_sq/self.sigma_nsq)

            num_ar[sigma12 < 0] = 0
            num_ar[sigma1_sq < self.sigma_nsq] = 1 - sigma2_sq[sigma1_sq < self.sigma_nsq] * self.sigma_max_inv
            den_ar = den_ar.masked_fill(sigma1_sq < self.sigma_nsq, 1)

            num.append(torch.sum(num_ar, dim=(-1, -2, -3)))
            den.append(torch.sum(den_ar, dim=(-1, -2, -3)))

        num = torch.stack(num, dim=-1)  # [batch_size, num_scales]
        den = torch.stack(den, dim=-1)

        return num, den
