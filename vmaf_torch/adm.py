"""ADM metric

Based on:
    https://github.com/Netflix/vmaf/blob/master/libvmaf/src/feature/adm.c

References:
    https://www.researchgate.net/publication/220516648_Image_Quality_Assessment_by_Separately_Evaluating_Detail_Losses_and_Additive_Impairments

"""

import numpy as np
import torch
import torch.nn.functional as F
from .utils import vmaf_pad


class ADM(torch.nn.Module):

    def __init__(self, NEG=False):
        super().__init__()
        self.scales = (0, 1, 2, 3)
        self.border_factor = 0.1
        self.rfactors = [[0.017382, 0.017382, 0.005891],
                         [0.031985, 0.031985, 0.014299],
                         [0.043373, 0.043373, 0.024397],
                         [0.045673, 0.045673, 0.031313]]
        self.eps = 1e-30

        cm_kernel = torch.tensor([[1/30, 1/30, 1/30],
                                  [1/30, 1/15, 1/30],
                                  [1/30, 1/30, 1/30]]).unsqueeze(0).unsqueeze(0)
        self.register_buffer('cm_kernel', cm_kernel)

        dwt2_db2_coeffs_lo = [0.482962913144690, 0.836516303737469, 0.224143868041857, -0.129409522550921]
        dwt2_db2_coeffs_hi = [-0.129409522550921, -0.224143868041857, 0.836516303737469, -0.482962913144690]
        dwt2_db2_coeffs_lo_hor = torch.tensor(dwt2_db2_coeffs_lo).reshape([1, 1, 1, 4])
        self.register_buffer('dwt2_db2_coeffs_lo_hor', dwt2_db2_coeffs_lo_hor)
        dwt2_db2_coeffs_lo_vert = torch.tensor(dwt2_db2_coeffs_lo).reshape([1, 1, 4, 1])
        self.register_buffer('dwt2_db2_coeffs_lo_vert', dwt2_db2_coeffs_lo_vert)
        dwt2_db2_coeffs_hi_hor = torch.tensor(dwt2_db2_coeffs_hi).reshape([1, 1, 1, 4])
        self.register_buffer('dwt2_db2_coeffs_hi_hor', dwt2_db2_coeffs_hi_hor)
        dwt2_db2_coeffs_hi_vert = torch.tensor(dwt2_db2_coeffs_hi).reshape([1, 1, 4, 1])
        self.register_buffer('dwt2_db2_coeffs_hi_vert', dwt2_db2_coeffs_hi_vert)

        if NEG:
            self.adm_enhn_gain_limit = 1.
        else:
            self.adm_enhn_gain_limit = 100.

    def forward(self, X_ref, X_dst):
        return self.adm_score(X_ref, X_dst)  # only adm score is used as a feature for VMAF score regresion

    def adm_features(self, X_ref, X_dst):
        num, den = self.adm_num_den(X_ref, X_dst)
        features = num/den
        return features               # [batch_size, num_scales]

    def adm_score(self, X_ref, X_dst):
        image_width = X_ref.shape[-1]
        image_height = X_ref.shape[-2]
        num, den = self.adm_num_den(X_ref, X_dst)
        num_sum = num.sum(dim=-1, keepdim=True)     # [batch_size, 1]
        den_sum = den.sum(dim=-1, keepdim=True)     # [batch_size, 1]
        numden_limit = 1e-10 * (image_width * image_height) / (1920 * 1080)
        num_sum = num_sum.masked_fill(num_sum < numden_limit,  0.)
        den_sum = den_sum.masked_fill(den_sum < numden_limit,  0.)
        score = (num_sum/den_sum).masked_fill(den_sum == 0., 1.)
        return score                                # [batch_size, 1]

    def adm_features_and_score(self, X_ref, X_dst):
        image_width = X_ref.shape[-1]
        image_height = X_ref.shape[-2]
        num, den = self.adm_num_den(X_ref, X_dst)
        num_sum = num.sum(dim=-1, keepdim=True)     # [batch_size, 1]
        den_sum = den.sum(dim=-1, keepdim=True)     # [batch_size, 1]
        numden_limit = 1e-10 * (image_width * image_height) / (1920 * 1080)
        num_sum = num_sum.masked_fill(num_sum < numden_limit,  0.)
        den_sum = den_sum.masked_fill(den_sum < numden_limit,  0.)
        score = (num_sum/den_sum).masked_fill(den_sum == 0., 1.)

        features = num/den
        return features, score

    def adm_num_den(self, X_ref, X_dst):
        '''Compute ADM numerator and denominator for several scales'''

        assert len(X_ref.shape) == 4 and len(X_ref.shape) == 4, f'Expected tensors in [b,c,h,w] format, got {X_ref.shape} and {X_dst.shape}'

        image_width = X_ref.shape[-1]
        image_height = X_ref.shape[-2]

        num = []
        den = []

        O_a = X_ref
        T_a = X_dst

        for scale in self.scales:

            width = torch.tensor(image_width/(2**(scale+1))).ceil().long()
            height = torch.tensor(image_height/(2**(scale+1))).ceil().long()

            O_a, O_h, O_v, O_d = self.adm_dwt2(O_a)
            T_a, T_h, T_v, T_d = self.adm_dwt2(T_a)

            R_h, R_v, R_d, A_h, A_v, A_d = self.adm_decouple(O_h, O_v, O_d, T_h, T_v, T_d, width, height, scale)

            den_scale = self.adm_csf_den(O_h, O_v, O_d, width, height, scale)

            csf_A_h, csf_A_v, csf_A_d = self.adm_csf(A_h, A_v, A_d, width, height, scale)

            num_scale = self.adm_cm(R_h, R_v, R_d, csf_A_h, csf_A_v, csf_A_d, width, height, scale)

            den.append(den_scale)
            num.append(num_scale)

        num = torch.cat(num, dim=-1)      # [batch_size, num_scales]
        den = torch.cat(den, dim=-1)      # [batch_size, num_scales]

        return num, den

    def adm_dwt2(self, X):
        '''Compute discrete wavelet transform'''

        Y_lo_vert = self.dwt2_vertical_pass(X, self.dwt2_db2_coeffs_lo_vert)
        Y_hi_vert = self.dwt2_vertical_pass(X, self.dwt2_db2_coeffs_hi_vert)

        Y_a = self.dwt2_horizontal_pass(Y_lo_vert, self.dwt2_db2_coeffs_lo_hor)
        Y_v = self.dwt2_horizontal_pass(Y_lo_vert, self.dwt2_db2_coeffs_hi_hor)
        Y_h = self.dwt2_horizontal_pass(Y_hi_vert, self.dwt2_db2_coeffs_lo_hor)
        Y_d = self.dwt2_horizontal_pass(Y_hi_vert, self.dwt2_db2_coeffs_hi_hor)

        return Y_a, Y_h, Y_v, Y_d

    def dwt2_vertical_pass(self, X, filter_vert):
        # [b, c, h, w] -> [b, c, ceil(h/2), w]
        # assert filter_vert.shape==[1, 1, 4, 1]
        padding_size_top = 1
        padding_size_bottom = 1 if X.shape[-2] % 2 == 0 else 2
        X_pad = vmaf_pad(X, (0, 0, padding_size_top, padding_size_bottom))
        Y = F.conv2d(X_pad, filter_vert, padding="valid", stride=(2, 1))
        return Y

    def dwt2_horizontal_pass(self, X, filter_hor):
        # [b, c, h, w] -> [b, c, h, ceil(w/2)]
        # assert filter_hor.shape==[1, 1, 1, 4]
        padding_size_left = 1
        padding_size_right = 1 if X.shape[-1] % 2 == 0 else 2
        X_pad = vmaf_pad(X, (padding_size_left, padding_size_right, 0, 0))
        Y = F.conv2d(X_pad, filter_hor, padding="valid", stride=(1, 2))
        return Y

    def adm_decouple(self, O_h, O_v, O_d, T_h, T_v, T_d, width, height, scale):
        '''Compute Restored image and Additive impairment image, eq. (12), (15)'''

        # eq. (10)
        K_h = torch.clamp(torch.divide(T_h, O_h + self.eps), 0., 1.)
        K_v = torch.clamp(torch.divide(T_v, O_v + self.eps), 0., 1.)
        K_d = torch.clamp(torch.divide(T_d, O_d + self.eps), 0., 1.)

        # eq. (11) & (15)
        R_h = K_h * O_h
        R_v = K_v * O_v
        R_d = K_d * O_d

        # same as eq. (14), see https://github.com/Netflix/vmaf/blob/master/libvmaf/src/feature/adm_tools.c#L162
        ot_dp = O_h * T_h + O_v * T_v
        o_mag_sq = O_h * O_h + O_v * O_v
        t_mag_sq = T_h * T_h + T_v * T_v
        cos_1deg_sq = np.cos(1.0 * np.pi / 180.0)**2
        angle_flag = (ot_dp >= 0.) & (ot_dp * ot_dp >= cos_1deg_sq * o_mag_sq * t_mag_sq)

        R_h = torch.where(angle_flag & (R_h > 0.0), torch.minimum(R_h * self.adm_enhn_gain_limit, T_h), R_h)
        R_h = torch.where(angle_flag & (R_h < 0.0), torch.maximum(R_h * self.adm_enhn_gain_limit, T_h), R_h)

        R_v = torch.where(angle_flag & (R_v > 0.0), torch.minimum(R_v * self.adm_enhn_gain_limit, T_v), R_v)
        R_v = torch.where(angle_flag & (R_v < 0.0), torch.maximum(R_v * self.adm_enhn_gain_limit, T_v), R_v)

        R_d = torch.where(angle_flag & (R_d > 0.0), torch.minimum(R_d * self.adm_enhn_gain_limit, T_d), R_d)
        R_d = torch.where(angle_flag & (R_d < 0.0), torch.maximum(R_d * self.adm_enhn_gain_limit, T_d), R_d)

        # eq. (12)
        A_h = T_h - R_h
        A_v = T_v - R_v
        A_d = T_d - R_d

        return R_h, R_v, R_d, A_h, A_v, A_d

    def adm_csf_den(self, O_h, O_v, O_d, width, height, scale):
        '''Compute denominator for eq. (19). Perform CSF weighting (without CM) on original image, and spatial and subband Minkowski pooling'''

        left = (width * self.border_factor - 0.5).floor().long()
        top = (height * self.border_factor - 0.5).floor().long()
        right = width - left
        bottom = height - top

        rfactor = self.rfactors[scale]

        # perform CSF on O
        # do not use coefs on the border of the image
        # using a mask instead of indexing because it leads to graph breaks during torch.compile which slow down the computation
        ind_hor, ind_vert = torch.arange(width, device=O_h.device), torch.arange(height, device=O_h.device)
        border_mask = ((ind_hor < left) | (ind_hor >= right)).view(1, -1) | ((ind_vert < top) | (ind_vert >= bottom)).view(-1, 1)  # [height, width]
        abs_csf_o_val_h = torch.abs(rfactor[0] * O_h.masked_fill(border_mask, 0))
        abs_csf_o_val_v = torch.abs(rfactor[1] * O_v.masked_fill(border_mask, 0))
        abs_csf_o_val_d = torch.abs(rfactor[2] * O_d.masked_fill(border_mask, 0))

        # perform Minkowski pooling with p=3
        den_scale_h = torch.linalg.vector_norm(abs_csf_o_val_h, 3, dim=(-1, -2))
        den_scale_v = torch.linalg.vector_norm(abs_csf_o_val_v, 3, dim=(-1, -2))
        den_scale_d = torch.linalg.vector_norm(abs_csf_o_val_d, 3, dim=(-1, -2))

        den = den_scale_h + den_scale_v + den_scale_d + 3*torch.pow((bottom-top)*(right-left)/32., 1./3.)

        return den  # [batch_size, 1]

    def adm_csf(self, A_h, A_v, A_d, width, height, scale):
        '''Apply Contrast Sensitivty Function (CSF) on additive impairments image A'''

        rfactor = self.rfactors[scale]

        csf_A_h = rfactor[0] * A_h
        csf_A_v = rfactor[1] * A_v
        csf_A_d = rfactor[2] * A_d

        return csf_A_h, csf_A_v, csf_A_d

    def adm_cm(self, R_h, R_v, R_d, csf_A_h, csf_A_v, csf_A_d, width, height, scale):
        '''Compute numerator for eq. (19) Perform Contrast Masking on restored image R, and spatial and subband Minkowski pooling'''

        rfactor = self.rfactors[scale]

        left = (width * self.border_factor - 0.5).floor().long()
        top = (height * self.border_factor - 0.5).floor().long()
        right = width - left
        bottom = height - top

        # eq. (18), compute thresholds
        csf_A_h_padded = vmaf_pad(csf_A_h, (1, 1, 1, 1))
        thr_h = F.conv2d(torch.abs(csf_A_h_padded), self.cm_kernel, padding="valid")

        csf_A_v_padded = vmaf_pad(csf_A_v, (1, 1, 1, 1))
        thr_v = F.conv2d(torch.abs(csf_A_v_padded), self.cm_kernel, padding="valid")

        csf_A_d_padded = vmaf_pad(csf_A_d, (1, 1, 1, 1))
        thr_d = F.conv2d(torch.abs(csf_A_d_padded), self.cm_kernel, padding="valid")

        # do not use coefs on the border of the image
        # using a mask instead of indexing because it leads to graph breaks during torch.compile which slow down the computation
        ind_hor, ind_vert = torch.arange(width, device=R_h.device), torch.arange(height, device=R_h.device)
        border_mask = ((ind_hor < left) | (ind_hor >= right)).view(1, -1) | ((ind_vert < top) | (ind_vert >= bottom)).view(-1, 1)  # [height, width]

        thr_h = thr_h.masked_fill(border_mask, 0)
        thr_v = thr_v.masked_fill(border_mask, 0)
        thr_d = thr_d.masked_fill(border_mask, 0)

        R_h = R_h.masked_fill(border_mask, 0)
        R_v = R_v.masked_fill(border_mask, 0)
        R_d = R_d.masked_fill(border_mask, 0)

        thr = thr_h + thr_v + thr_d

        # perform CSF and CM on R
        xh = F.relu(torch.abs(R_h * rfactor[0]) - thr)
        xv = F.relu(torch.abs(R_v * rfactor[1]) - thr)
        xd = F.relu(torch.abs(R_d * rfactor[2]) - thr)

        # perform Minkowski pooling with p=3
        num_scale_h = torch.linalg.vector_norm(xh, 3, dim=(-1, -2))
        num_scale_v = torch.linalg.vector_norm(xv, 3, dim=(-1, -2))
        num_scale_d = torch.linalg.vector_norm(xd, 3, dim=(-1, -2))

        num = num_scale_h + num_scale_v + num_scale_d + 3*torch.pow((bottom-top)*(right-left)/32., 1./3.)

        return num  # [batch_size, 1]
