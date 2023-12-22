import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yuvio
import subprocess


def gaussian_kernel_1d(kernel_size, sigma=1):
    # equal to matlab fspecial('gaussian',hsize,sigma)
    x = torch.arange(-(kernel_size//2), kernel_size//2+1)
    gauss = torch.exp(-(x**2 / (2.0 * sigma**2)))
    gauss = gauss/gauss.sum()
    return gauss


def gaussian_kernel(kernel_size, sigma=1):
    x, y = torch.meshgrid(torch.arange(-(kernel_size//2), kernel_size//2+1),
                          torch.arange(-(kernel_size//2), kernel_size//2+1),
                          indexing='ij')
    dst = x**2+y**2
    gauss = torch.exp(-(dst / (2.0 * sigma**2)))
    gauss = gauss/gauss.sum()
    gauss = gauss.reshape((1, 1, kernel_size, kernel_size))
    return gauss


def vmaf_pad(input, pad):
    '''Pad image using padding mode: dcb|abcdef|fed - same as in VMAF C code'''
    padding_left, padding_right, padding_top, padding_bottom = pad
    w = input.shape[-1]
    h = input.shape[-2]
    if padding_left <= 1 and padding_right <= 1 and padding_top <= 1 and padding_bottom <= 1:
        padded = F.pad(input, (padding_left, 0, padding_top, 0), mode='reflect')
        padded = F.pad(padded, (0, padding_right, 0, padding_bottom), mode='replicate')
    else:
        # pad right
        if padding_right > 0:
            padded = F.pad(input, (0, padding_right-1, 0, 0), mode='reflect')                               # make abcdef|ed
            padded = torch.cat((padded[:, :, :, :w], padded[:, :, :, w-1:w], padded[:, :, :, w:]), dim=-1)  # insert f manually so we have abcdef|fed
        else:
            padded = input
        # pad left
        padded = F.pad(padded, (padding_left, 0, 0, 0), mode='reflect')
        # pad bottom
        if padding_bottom > 0:
            padded = F.pad(padded, (0, 0, 0, padding_bottom-1), mode='reflect')
            padded = torch.cat((padded[:, :, :h, :], padded[:, :, h-1:h, :], padded[:, :, h:, :]), dim=-2)
        # pad top
        padded = F.pad(padded, (0, 0, padding_top, 0), mode='reflect')
    return padded


def yuv_to_tensor(yuv_path, width, height, num_frames, channel='y'):
    '''Read yuv from disk and return as a float tensor or tuple of tensors
    if channel=='y' return [b,1,h,v] float tensor
    if channel=='yuv' return tuple of [b,1,h,v], [b,1,h//2,v//2], [b,1,h//2,v//2] tensors
    '''
    if channel == 'y':
        t_y = [torch.tensor(fr.y) for fr in yuvio.mimread(yuv_path, width, height, "yuv420p", index=0, count=num_frames)]
        t_y = torch.stack(t_y, dim=0).to(torch.float32).unsqueeze(1)
        return t_y
    elif channel == 'yuv':
        # TODO rewrite more efficiently
        t_y = [torch.tensor(fr.y) for fr in yuvio.mimread(yuv_path, width, height, "yuv420p", index=0, count=num_frames)]
        t_y = torch.stack(t_y, dim=0).to(torch.float32).unsqueeze(1)

        t_u = [torch.tensor(fr.u) for fr in yuvio.mimread(yuv_path, width, height, "yuv420p", index=0, count=num_frames)]
        t_u = torch.stack(t_u, dim=0).to(torch.float32).unsqueeze(1)

        t_v = [torch.tensor(fr.v) for fr in yuvio.mimread(yuv_path, width, height, "yuv420p", index=0, count=num_frames)]
        t_v = torch.stack(t_v, dim=0).to(torch.float32).unsqueeze(1)
        return t_y, t_u, t_v
    else:
        raise ValueError("Only 'y' and 'yuv' channel options are supported")


def tensor_to_yuv(t_y, t_u, t_v, yuv_path):
    '''Write tuple of tensors to disk as yuv
    expected shapes t_y [b,1,h,v] t_u [b,1,h//2,v//2] t_v [b,1,h//2,v//2]
    '''
    frames_y = [x.squeeze().detach().cpu().numpy().astype(np.uint8) for x in torch.unbind(t_y, dim=0)]
    frames_u = [x.squeeze().detach().cpu().numpy().astype(np.uint8) for x in torch.unbind(t_u, dim=0)]
    frames_v = [x.squeeze().detach().cpu().numpy().astype(np.uint8) for x in torch.unbind(t_v, dim=0)]
    frames = [yuvio.frame(x, "yuv420p") for x in list(zip(frames_y, frames_u, frames_v))]
    yuvio.mimwrite(yuv_path, frames)
    # return frames


class VMAF_C():
    '''Utility class for calling reference vmaf executable'''

    def __init__(self, vmaf_executable="vmaf", vmaf_model_version="default", verbose=True):
        self.vmaf_executable = vmaf_executable
        self.verbose = verbose

        if vmaf_model_version == "default":
            self.vmaf_model_param = ""
        elif vmaf_model_version == "NEG":
            self.vmaf_model_param = "--model version=vmaf_v0.6.1neg"

    def table_from_path(self, ref_path, dist_path, width, height, num_frames):
        vmaf_out_csv_path = 'vmaf_out.csv'
        vmaf_param = f"{self.vmaf_executable} -r {ref_path} -d {dist_path} -w {width} -h {height} --frame_cnt {num_frames} -p 420 -b 8 --threads 16 -q --csv -o {vmaf_out_csv_path} {self.vmaf_model_param}"
        if self.verbose:
            print('Executing:', vmaf_param)
        p = subprocess.run(vmaf_param.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if self.verbose:
            print('Reading:', vmaf_out_csv_path)
        df = pd.read_csv(vmaf_out_csv_path)
        return df

    def score_from_path(self, ref_path, dist_path, width, height, num_frames):
        df = self.table_from_path(ref_path, dist_path, width, height, num_frames)
        score = df['vmaf'].iloc[:num_frames].mean()
        return score

    def table_from_tensors(self, ref_tup, dist_tup):
        # TODO check shapes and rounding here
        ref_save_path = './ref.yuv'
        dist_save_path = './dist.yuv'
        width = ref_tup[0].shape[-1]
        height = ref_tup[0].shape[-2]
        num_frames = ref_tup[0].shape[0]
        if self.verbose:
            print('Saving tensors to:', ref_save_path, 'and', dist_save_path)
        tensor_to_yuv(*ref_tup, yuv_path=ref_save_path)
        tensor_to_yuv(*dist_tup, yuv_path=dist_save_path)
        return self.table_from_path(ref_save_path, dist_save_path, width, height, num_frames)

    def score_from_tensors(self, ref_tup, dist_tup):
        df = self.table_from_tensors(ref_tup, dist_tup)
        score = df['vmaf'].mean()
        return score
