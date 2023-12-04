"""Motion feature

Based on:
    https://github.com/Netflix/vmaf/blob/master/libvmaf/src/feature/motion.c

Todo:
    check that padding modes are the same as C code for nonstandard resolutions

"""

import torch
from .utils import gaussian_kernel, gaussian_filter


class Motion(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.frame_dim = 0
        blur_win = gaussian_kernel(kernel_size=5)
        self.register_buffer('blur_win', blur_win)

    def forward(self, x):
        return self.motion2(x)

    def motion(self, x):
        # SAD(frame(i),frame(i-1))
        ref_blur = gaussian_filter(x, self.blur_win, padding='vmaf')
        motion = torch.mean(torch.abs(ref_blur-torch.roll(ref_blur, 1, self.frame_dim)), dim=(-1, -2))
        motion[0] = 0
        return motion             # [num_frames, 1]

    def motion2(self, x):
        # min( SAD(frame(i),frame(i-1)) , SAD(frame(i),frame(i+1)) )
        ref_blur = gaussian_filter(x, self.blur_win, padding='vmaf')
        motion_i_minus_1 = torch.mean(torch.abs(ref_blur-torch.roll(ref_blur, 1, self.frame_dim)), dim=(-1, -2))

        # motion_minus1 = torch.mean(torch.abs(ref_blur-torch.roll(ref_blur, -1, self.frame_dim)), dim=(-1, -2))
        motion_i_plus_1 = torch.roll(motion_i_minus_1, -1, self.frame_dim)   # same as previous line but faster

        motion2 = torch.minimum(motion_i_minus_1, motion_i_plus_1)
        motion2[0] = 0
        motion2[-1] = motion_i_minus_1[-1]
        return motion2            # [num_frames, 1]
