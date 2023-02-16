from __future__ import print_function, division
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


""""basic registration loss"""
"""NCC."""


def ncc_global(sources, targets):
    size = sources.size(2) * sources.size(3)
    sources_mean = torch.mean(sources, dim=(1, 2, 3)).view(sources.size(0), 1, 1, 1)
    targets_mean = torch.mean(targets, dim=(1, 2, 3)).view(sources.size(0), 1, 1, 1)
    sources_std = torch.std(sources, dim=(1, 2, 3)).view(sources.size(0), 1, 1, 1)
    targets_std = torch.std(targets, dim=(1, 2, 3)).view(sources.size(0), 1, 1, 1)
    ncc = (1 / size) * torch.sum((sources - sources_mean) * (targets - targets_mean) / (sources_std * targets_std),
                                 dim=(1, 2, 3))
    return ncc



def ncc_losses_global(sources, targets, device='cpu', **params):
    ncc = ncc_global(sources, targets, device=device, **params)
    ncc = torch.mean(ncc)
    if ncc != ncc:
        return torch.autograd.Variable(torch.Tensor([1]), requires_grad=True).to(device)
    return -ncc


def ncc_loss_global(source, target, device='cpu', **params):
    # return ncc_losses_global(source.view(1, 1, source.size(0), source.size(1)), target.view(1, 1, target.size(0), target.size(1)), device=device, **params)
    return ncc_losses_global(source, target, device=device, **params)


class NCCLoss(nn.Module):
    """
    The normalized cross correlation loss is a measure for image pairs with a linear
    intensity relation.
    .. math::
        \mathcal{S}_{\text{NCC}} := \frac{\sum I_F\cdot (I_M\circ f)
                - \sum\text{E}(I_F)\text{E}(I_M\circ f)}
                {\vert\mathcal{X}\vert\cdot\sum\text{Var}(I_F)\text{Var}(I_M\circ f)}
    """

    def __init__(self, device='cpu'):
        super(NCCLoss, self).__init__()
        self.device = device
        self.NCC = ncc_loss_global

    def forward(self, sources, targets):
        NCC = self.NCC(sources, targets, self.device)
        if sources.shape[1] == 3:
            return NCC / 3
        else:
            return NCC


class NCCLoss_Nomean(nn.Module):
    """
    The normalized cross correlation loss is a measure for image pairs with a linear
    intensity relation.
    .. math::
        \mathcal{S}_{\text{NCC}} := \frac{\sum I_F\cdot (I_M\circ f)
                - \sum\text{E}(I_F)\text{E}(I_M\circ f)}
                {\vert\mathcal{X}\vert\cdot\sum\text{Var}(I_F)\text{Var}(I_M\circ f)}
    """

    def __init__(self):
        super(NCCLoss_Nomean, self).__init__()
        self.NCC = ncc_global

    def forward(self, sources, targets):
        ncc = self.NCC(sources, targets)
        if sources.shape[1] == 3:
            return -ncc / 3
        else:
            return -ncc
        # return ncc



"""NGF."""
class NGFLoss(nn.Module):
    def __init__(self, device='cpu', epsilon=1e-5):
        super(NGFLoss, self).__init__()
        self.device = device
        self.epsilon = epsilon

    def forward(self, sources, targets):
        # normal of sources  需要考虑输入图片通道数
        # print(sources.shape, targets.shape)
        if sources.shape[0] == 1:
            sources_dx = (sources[..., 1:, 1:] - sources[..., :-1, 1:])
            sources_dy = (sources[..., 1:, 1:] - sources[..., 1:, :-1])
        else:  # 3 channel
            sources_dx = (sources[:, 0, 1:, 1:] - sources[:, 0, :-1, 1:])
            sources_dy = (sources[:, 0, 1:, 1:] - sources[:, 0, 1:, :-1])
            sources_dx = torch.unsqueeze(sources_dx, dim=1)
            sources_dy = torch.unsqueeze(sources_dy, dim=1)

        sources_norm = torch.sqrt(
            sources_dx.type(torch.float64).pow(2) + sources_dy.type(torch.float64).pow(2) + self.epsilon ** 2)
        ng_sourses = F.pad(torch.cat((sources_dx, sources_dy), dim=1) / sources_norm, (0, 1, 0, 1))
        # normal of targets
        if targets.shape[1] == 1:
            targets_dx = (targets[..., 1:, 1:] - targets[..., :-1, 1:])
            targets_dy = (targets[..., 1:, 1:] - targets[..., 1:, :-1])
        else:  # 3 channel
            targets_dx = (targets[:, 0, 1:, 1:] - targets[:, 0, :-1, 1:])
            targets_dy = (targets[:, 0, 1:, 1:] - targets[:, 0, 1:, :-1])
            targets_dx = torch.unsqueeze(targets_dx, dim=1)
            targets_dy = torch.unsqueeze(targets_dy, dim=1)
        targets_norm = torch.sqrt(
            targets_dx.type(torch.float64).pow(2) + targets_dy.type(torch.float64).pow(2) + self.epsilon ** 2)
        ng_targets = F.pad(torch.cat((targets_dx, targets_dy), dim=1) / targets_norm, (0, 1, 0, 1))
        value = 0
        # for dim in range(targets.shape[1]): # ode version
        for dim in range(ng_targets.shape[1]):
            value = value + ng_sourses[:, dim, ...] * ng_targets[:, dim, ...]
        # NGF = -torch.mean(value.type(torch.float64).pow(2))
        NGF = -(torch.sum(value.type(torch.float64).pow(2)) / (torch.sum(value.type(torch.float64) != 0) + 1e-5))
        # print(torch.sum(sources), torch.sum(targets))
        print(NGF, torch.sum(value.type(torch.float64) != 0))
        return NGF


def compute_entropy(sources, targets, device='cpu'):
    entropy = 0
    for dim in range(sources.size(1)):
        with torch.no_grad():
            img_stack = torch.stack([sources[:, dim, ...], targets[:, dim, ...]], dim=0).view(2, sources.size(0), -1)
            values, counts = torch.unique(img_stack, return_counts=True, dim=2)  # 按照像素值统计个数(256*256)
            counts = counts / (torch.tensor(img_stack.size(2)).type(torch.float))  # 计算Pij
            counts = counts * torch.log2(counts + 1e-10)  # # 计算单通道图像二维熵
            entropy += -torch.sum(counts)
    entropy = entropy / (sources.size(0) * sources.size(1))
    return entropy




"""MSE."""
class MSELoss(nn.Module):
    """
    The mean square error loss is a simple and fast to compute point-wise measure
    which is well suited for monomodal image registration.
    .. math::
        \mathcal{S}_{\text{MSE}} := \frac{1}{\vert \mathcal{X} \vert}\sum_{x\in\mathcal{X}}
        \Big(I_M\big(x+f(x)\big) - I_F\big(x\big)\Big)^2
    """

    def __init__(self):
        super(MSELoss, self).__init__()


    def forward(self, sources, targets):
        """
        inputs:[B,C,H,W],
        outputs:mse of single batch.
        """
        MSE = torch.mean((sources - targets).pow(2))
        return MSE



class LNCC(torch.nn.Module):
    """
    local (over window) normalized cross correlation
    """
    def __init__(self,  win=7, eps=1e-8,):
        super(LNCC, self).__init__()
        self.win = win
        self.eps = eps
        self.w_temp = win

    def forward(self, I, J):
        ndims = 3
        win_size = self.w_temp

        # set window size
        if self.win is None:
            self.win = [5] * ndims
        else:
            self.win = [self.w_temp] * ndims

        weight_win_size = self.w_temp
        weight = torch.ones((1, 1, weight_win_size, weight_win_size, weight_win_size), device=I.device, requires_grad=False)
        conv_fn = F.conv3d

        # compute CC squares
        I2 = I*I
        J2 = J*J
        IJ = I*J

        # compute filters
        # compute local sums via convolution
        I_sum = conv_fn(I, weight, padding=int(win_size/2))
        J_sum = conv_fn(J, weight, padding=int(win_size/2))
        I2_sum = conv_fn(I2, weight, padding=int(win_size/2))
        J2_sum = conv_fn(J2, weight, padding=int(win_size/2))
        IJ_sum = conv_fn(IJ, weight, padding=int(win_size/2))

        # compute cross correlation
        win_size = np.prod(self.win)
        u_I = I_sum/win_size
        u_J = J_sum/win_size

        cross = IJ_sum - u_J*I_sum - u_I*J_sum + u_I*u_J*win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I*u_I*win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J*u_J*win_size

        cc = cross * cross / (I_var * J_var + self.eps)

        # return negative cc.
        return -1.0 * torch.mean(cc)



"""load criterion."""
def load_criterion(criterion):
    if criterion == 'NCC':
        return NCCLoss_Nomean()
    if criterion == 'LNCC':
        return LNCC()
    elif criterion == 'NGF':
        return NGFLoss()
    elif criterion == 'MSE':
        return MSELoss()



