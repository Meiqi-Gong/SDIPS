import sys

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def mmd_rbf(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    '''
    计算源域数据和目标域数据的MMD距离
	Return:
		loss: MMD loss
    '''

    source = (source-torch.min(source))/(torch.max(source) - torch.min(source))
    target = (target - torch.min(target)) / (torch.max(target) - torch.min(target))
    batch_size, c, h, w = source.shape
    unfold = nn.Unfold(kernel_size=(3, 3), stride=12)
    patch_source = unfold(source)
    patch_source=patch_source.reshape(batch_size, c, -1, patch_source.shape[-1])
    patch_source=torch.mean(patch_source, 2).permute(0, 2, 1).reshape(-1, c)
    unfold = nn.Unfold(kernel_size=(3, 3), stride=48)
    patch_target = unfold(target)
    patch_target=patch_target.reshape(batch_size, c, -1, patch_target.shape[-1])
    patch_target=torch.mean(patch_target, 2).permute(0, 2, 1).reshape(-1, c)

    b = patch_target.shape[0]
    n = int(patch_source.size()[0])
    m = int(patch_target.size()[0])
    kernels = guassian_kernel(patch_source, patch_target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:n, :n]
    YY = kernels[n:, n:]
    XY = kernels[:n, n:]
    YX = kernels[n:, :n]
    XX = torch.div(XX, n * n).sum(dim=1).view(1, -1)
    XY = torch.div(XY, -n * m).sum(dim=1).view(1, -1)

    YX = torch.div(YX, -m * n).sum(dim=1).view(1, -1)
    YY = torch.div(YY, m * m).sum(dim=1).view(1, -1)
    loss = (XX + XY).sum() + (YX + YY).sum()
    return loss

def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    '''
	Return:
		sum(kernel_val): 多个核矩阵之和
    '''
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)#/len(kernel_val)

def _tf_fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x_data, y_data = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]

    x_data = np.expand_dims(x_data, axis=-1)
    x_data = np.expand_dims(x_data, axis=-1)

    y_data = np.expand_dims(y_data, axis=-1)
    y_data = np.expand_dims(y_data, axis=-1)

    x = torch.tensor(x_data, dtype=torch.float32)
    y = torch.tensor(y_data, dtype=torch.float32)
    g = torch.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    # print(g.shape)
    # sys.exit(0)

    # g = tf.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    return g / torch.sum(g)


def SSIM_single(img1, img2, size=11, sigma=1.5):
    window = _tf_fspecial_gauss(size, sigma).cuda()
    k1 = 0.01
    k2 = 0.03
    L = 1
    c1 = (k1 * L) ** 2
    c2 = (k2 * L) ** 2
    window = window.permute(2, 3, 0, 1)
    mu1 = F.conv2d(img1, window, stride=1, padding=0)
    mu2 = F.conv2d(img2, window, stride=1, padding=0)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, stride=1, padding=0) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, stride=1, padding=0) - mu2_sq
    sigma1_2 = F.conv2d(img1 * img2, window, stride=1, padding=0) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma1_2 + c2)) / (
            (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    value = torch.mean(ssim_map, dim=[1, 2, 3])
    return value

def SSIM_loss(img1, img2):
    channel = img1.shape[1]
    s=0
    for i in range(channel):
        s+=(1-torch.mean(SSIM_single(img1[:,i:i+1,:,:], img2[:,i:i+1,:,:])))
    return s
