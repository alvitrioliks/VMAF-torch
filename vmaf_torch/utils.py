import numpy as np
import torch
import torch.nn.functional as F
import yuvio


def gaussian_kernel_1d(kernel_size, sigma=1):

    # corresponds to matlab fspecial('gaussian',hsize,sigma)
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


def fast_gaussian_blur(x, weight, stride=1):
    '''Fast gaussian blur using separable filter

    Args:
        x: input image of shape (b,1,h,w)
        weight: 1d gaussian kernel of shape (1,1,1,kernel_size)
    '''
    return F.conv2d(F.conv2d(x, weight.view(1, 1, 1, -1), stride=(1, stride)), weight.view(1, 1, -1, 1), stride=(stride, 1))


def yuv_to_tensor(yuv_path, width, height, num_frames, channel='y'):
    '''Read yuv file from disk and return it as a float tensor or a tuple of tensors

    Args:
        yuv_path: path to yuv file
        width: width of the video
        height: height of the video
        num_frames: number of frames in the video
        channel: 'y' or 'yuv' whether read only Y channel or all YUV channels. Defaults to 'y'.

    Returns:
        if channel=='y' return [b,1,h,v] float tensor, if channel=='yuv' return tuple of [b,1,h,v], [b,1,h//2,v//2], [b,1,h//2,v//2] tensors
    '''
    frames = yuvio.mimread(yuv_path, width, height, "yuv420p", index=0, count=num_frames)
    if channel == 'y':
        t_y = torch.stack([torch.tensor(fr.y) for fr in frames], dim=0).to(torch.float32).unsqueeze(1)
        return t_y
    elif channel == 'yuv':
        t_y = torch.stack([torch.tensor(fr.y) for fr in frames], dim=0).to(torch.float32).unsqueeze(1)
        t_u = torch.stack([torch.tensor(fr.u) for fr in frames], dim=0).to(torch.float32).unsqueeze(1)
        t_v = torch.stack([torch.tensor(fr.v) for fr in frames], dim=0).to(torch.float32).unsqueeze(1)
        return t_y, t_u, t_v
    else:
        raise ValueError("Only 'y' and 'yuv' channel options are supported")


def tensor_to_yuv(t_y, t_u, t_v, yuv_path):
    '''Write tensors with YUV channels to disk as yuv file

    Args:
        t_y: [b,1,h,v] tensor with Y channel
        t_u: [b,1,h//2,v//2] tensor with U channel
        t_v: [b,1,h//2,v//2] tensor with V channel
        yuv_path: path to save yuv file to
    '''
    frames_y = [x.clamp(0, 255).round().squeeze().detach().cpu().numpy().astype(np.uint8) for x in torch.unbind(t_y, dim=0)]
    frames_u = [x.clamp(0, 255).round().squeeze().detach().cpu().numpy().astype(np.uint8) for x in torch.unbind(t_u, dim=0)]
    frames_v = [x.clamp(0, 255).round().squeeze().detach().cpu().numpy().astype(np.uint8) for x in torch.unbind(t_v, dim=0)]
    frames = [yuvio.frame(x, "yuv420p") for x in list(zip(frames_y, frames_u, frames_v))]
    yuvio.mimwrite(yuv_path, frames)
