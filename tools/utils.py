import os
import math
import random
import warnings
from numbers import Number
import logging
import numpy as np
import torch
from torch.nn import functional as F
# use to approx logdet
from torch.backends import cudnn
from torch.optim import Optimizer
from torch import nn


def normalize_point_clouds(pc: torch.Tensor):
    B, N, _ = pc.shape
    centroid = torch.mean(pc, dim=1, keepdim=True)
    pc = pc - centroid
    furthest_distance = torch.amax(torch.sqrt(torch.sum(pc ** 2, dim=-1, keepdim=True)), dim=1, keepdim=True)
    pc = pc / furthest_distance
    return pc


class EMA(Optimizer):
    def __init__(self, opt, ema_decay):
        self.ema_decay = ema_decay
        self.apply_ema = self.ema_decay > 0.
        self.optimizer = opt
        self.state = opt.state
        self.param_groups = opt.param_groups
        self.defaults = opt.defaults

    def step(self, *args, **kwargs):
        retval = self.optimizer.step(*args, **kwargs)

        # stop here if we are not applying EMA
        if not self.apply_ema:
            return retval

        ema, params = {}, {}
        for group in self.optimizer.param_groups:
            for i, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                state = self.optimizer.state[p]

                # State initialization
                if 'ema' not in state:
                    state['ema'] = p.data.clone()

                if p.shape not in params:
                    params[p.shape] = {'idx': 0, 'data': []}
                    ema[p.shape] = []

                params[p.shape]['data'].append(p.data)
                ema[p.shape].append(state['ema'])

            for i in params:
                params[i]['data'] = torch.stack(params[i]['data'], dim=0)
                ema[i] = torch.stack(ema[i], dim=0)
                ema[i].mul_(self.ema_decay).add_(params[i]['data'], alpha=1. - self.ema_decay)

            for p in group['params']:
                if p.grad is None:
                    continue
                idx = params[p.shape]['idx']
                self.optimizer.state[p]['ema'] = ema[p.shape][idx, :]
                params[p.shape]['idx'] += 1

        return retval

    def load_state_dict(self, state_dict):
        super(EMA, self).load_state_dict(state_dict)
        # load_state_dict loads the data to self.state and self.param_groups. We need to pass this data to
        # the underlying optimizer too.
        self.optimizer.state = self.state
        self.optimizer.param_groups = self.param_groups

    def swap_parameters_with_ema(self, store_params_in_ema):
        """ This function swaps parameters with their ema values. It records original parameters in the ema
        parameters, if store_params_in_ema is true."""

        # stop here if we are not applying EMA
        if not self.apply_ema:
            # warnings.warn('swap_parameters_with_ema was called when there is no EMA weights.')
            return

        for group in self.optimizer.param_groups:
            for i, p in enumerate(group['params']):
                if not p.requires_grad:
                    continue
                if 'ema' not in self.optimizer.state[p]:
                    continue
                ema = self.optimizer.state[p]['ema']
                if store_params_in_ema:
                    tmp = p.data.detach()
                    p.data = ema.detach()
                    self.optimizer.state[p]['ema'] = tmp
                else:
                    p.data = ema.detach()


def get_activation(activation):
    if activation is None:
        return nn.Identity()
    if activation.lower() == 'gelu':
        return nn.GELU()
    elif activation.lower() == 'rrelu':
        return nn.RReLU(inplace=True)
    elif activation.lower() == 'selu':
        return nn.SELU(inplace=True)
    elif activation.lower() == 'silu':
        return nn.SiLU(inplace=True)
    elif activation.lower() == 'hardswish':
        return nn.Hardswish(inplace=True)
    elif activation.lower() == 'leakyrelu':
        return nn.LeakyReLU(inplace=True)
    elif activation.lower() == 'leakyrelu0.2':
        return nn.LeakyReLU(negative_slope=0.2, inplace=True)
    elif activation.lower() == 'swish':
        return nn.SiLU()
    else:
        return nn.ReLU(inplace=True)


class LayerNorm(nn.Module):
    def __init__(self, channels, elementwise_affine):
        super().__init__()
        self.norm = nn.LayerNorm(channels, elementwise_affine=elementwise_affine, eps=1e-6)

    def forward(self, x):
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class BatchNorm1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.BatchNorm1d(channels)

    def forward(self, x):
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


# class get_norm(nn.Module):
#     def __init__(self, channels, type="group_norm", groups=16, elementwise_affine=False, axis=1):
#         super(get_norm, self).__init__()
#         if type is None:
#             self.norm = nn.Identity()
#         type = type.lower()
#         if type == "batch_norm":
#             self.norm = BatchNorm1d(channels)
#         elif type == "layer_norm":
#             self.norm = LayerNorm(channels, elementwise_affine)
#         elif type == "group_norm":
#             assert groups > 0
#             self.norm = nn.GroupNorm(min(channels // 4, groups), channels, eps=1e-6)
#         else:
#             raise TypeError("norm not support")
#         self.axis = axis
#
#     def forward(self, x):
#         if self.axis == 1:
#             return self.norm(x.transpose(1, 2)).transpose(1, 2)
#         else:
#             return self.norm(x)

def get_norm(channels, type="group_norm", groups=16, elementwise_affine=False, axis=1):
        if type is None:
            return nn.Identity()
        type = type.lower()
        if type == "batch_norm":
            norm = BatchNorm1d(channels)
        elif type == "layer_norm":
            norm = LayerNorm(channels, elementwise_affine)
        elif type == "group_norm":
            assert groups > 0
            norm = nn.GroupNorm(min(channels // 4, groups), channels, eps=1e-6)
        else:
            raise TypeError("norm not support")
        return norm


def trace_df_dx_hutchinson(f, x, noise, no_autograd):
    """
    Hutchinson's trace estimator for Jacobian df/dx, O(1) call to autograd
    """
    if no_autograd:
        # the following is compatible with checkpointing
        torch.sum(f * noise).backward()
        # torch.autograd.backward(tensors=[f], grad_tensors=[noise])
        jvp = x.grad
        trJ = torch.sum(jvp * noise, dim=[1, 2, 3])
        x.grad = None
    else:
        jvp = torch.autograd.grad(f, x, noise, create_graph=False)[0]
        trJ = torch.sum(jvp * noise, dim=[1, 2, 3])
        # trJ = torch.einsum('bijk,bijk->b', jvp, noise)  # we could test if there's a speed difference in einsum vs sum

    return trJ


def standard_normal_logprob(z):
    logZ = -0.5 * math.log(2 * math.pi)
    return logZ - z.pow(2) / 2


def logsumexp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation
    value.exp().sum(dim, keepdim).log()
    """
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0), dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        if isinstance(sum_exp, Number):
            return m + math.log(sum_exp)
        else:
            return m + torch.log(sum_exp)


def logabsdet(x):
    """Returns the log absolute determinant of square matrix x."""
    # Note: torch.logdet() only works for positive determinant.
    _, res = torch.slogdet(x)
    return res


def sum_except_batch(x, num_batch_dims=1):
    """Sums all elements of `x` except for the first `num_batch_dims` dimensions."""
    reduce_dims = list(range(num_batch_dims, x.ndimension()))
    return torch.sum(x, dim=reduce_dims)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def orthogonalize_tensor(tensor):
    assert len(tensor.shape) == 2
    # flattened = tensor.new(rows, cols).normal_(0, 1)

    # Compute the qr factorization
    q, r = torch.qr(tensor)
    # Make Q uniform according to https://arxiv.org/pdf/math-ph/0609050.pdf
    d = torch.diag(r, 0)
    ph = d.sign()
    q *= ph
    tensor.view_as(q).copy_(q)
    return tensor


def random_orthogonal(size):
    """
    Returns a random orthogonal matrix as a 2-dim tensor of shape [size, size].
    """

    # Use the QR decomposition of a random Gaussian matrix.
    x = torch.randn(size, size)
    q, _ = torch.qr(x)
    return q


def common_init(seed):
    """set random seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = True


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum = self.sum + self.val * n
        self.count += n
        self.avg = self.sum / self.count


def variance_scaling(scale, mode, distribution,
                     in_axis=1, out_axis=0,
                     dtype=torch.float32,
                     device='cpu'):
    """Ported from JAX. """

    def _compute_fans(shape, in_axis=1, out_axis=0):
        receptive_field_size = np.prod(shape) / shape[in_axis] / shape[out_axis]
        fan_in = shape[in_axis] * receptive_field_size
        fan_out = shape[out_axis] * receptive_field_size
        return fan_in, fan_out

    def init(shape, dtype=dtype, device=device):
        fan_in, fan_out = _compute_fans(shape, in_axis, out_axis)
        if mode == "fan_in":
            denominator = fan_in
        elif mode == "fan_out":
            denominator = fan_out
        elif mode == "fan_avg":
            denominator = (fan_in + fan_out) / 2
        else:
            raise ValueError(
                "invalid mode for variance scaling initializer: {}".format(mode))
        variance = scale / denominator
        if distribution == "normal":
            return torch.randn(*shape, dtype=dtype, device=device) * np.sqrt(variance)
        elif distribution == "uniform":
            return (torch.rand(*shape, dtype=dtype, device=device) * 2. - 1.) * np.sqrt(3 * variance)
        else:
            raise ValueError("invalid distribution for variance scaling initializer")

    return init


def default_init(scale=1.):
    """The same initialization used in DDPM."""
    scale = 1e-10 if scale == 0 else scale
    return variance_scaling(scale, 'fan_avg', 'uniform')
