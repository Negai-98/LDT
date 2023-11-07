import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_ops import pointnet2_utils
from .ops import sample_mask
from tools.utils import get_norm, get_activation
from ..layers import ResidualBlock


class InitialSet(nn.Module):
    def __init__(self, dim_seed, max_outputs, n_mixtures=4):
        super().__init__()
        self.dim_seed = dim_seed
        self.max_outputs = max_outputs
        if max_outputs is None:
            self.n_mixtures = n_mixtures
            self.logits = nn.Parameter(torch.ones(n_mixtures, ))
            self.mu = nn.Parameter(torch.randn(n_mixtures, dim_seed))
            self.sig = nn.Parameter(torch.randn(n_mixtures, dim_seed).abs() / math.sqrt(n_mixtures))
            self.output = nn.Sequential(nn.Linear(dim_seed, dim_seed), nn.SiLU(), nn.Linear(dim_seed, dim_seed))
        else:
            self.prior = nn.Parameter(torch.rand((self.max_outputs, self.dim_seed), requires_grad=True))

    def forward(self, output_sizes):
        """
        Sample from prior
        :param output_sizes: Tensor([B, N])
        :return: Tensor([B, N, 2D])
        """
        bsize = output_sizes[0]
        if self.max_outputs is not None:
            x_mask = sample_mask(output_sizes, self.max_outputs).to("cuda")
            x = self.prior[None, :, :].expand(bsize, -1, -1)
            B, N, D = x.shape
            x = x[~x_mask, :].view((B, output_sizes[1], D)).transpose(1, 2)
        else:
            eps = torch.randn([bsize, output_sizes[1], self.n_mixtures, self.dim_seed]).to("cuda")
            x = (eps * self.sig[None, None, :, :] + self.mu[None, None, :, :]) \
                * F.softmax(self.logits, dim=0)[None, None, :, None]
            x = self.output(x.sum(2)).transpose(1, 2)
        return x


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx


def cluster(xyz, N, k, center=None):
    S = N
    xyz = xyz.contiguous()  # xyz [btach, points, xyz]
    with torch.no_grad():
        if center is None:
            center_idx = pointnet2_utils.furthest_point_sample(xyz, S).long()  # [B, npoint]
            new_xyz = index_points(xyz, center_idx)  # [B, npoint, 3]
        else:
            new_xyz = center
            center_idx = None
        group_idx = knn_point(k, xyz, new_xyz)
    return new_xyz, center_idx, group_idx


class ConvBNReLU1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, bias=True, activation='relu'):
        super(ConvBNReLU1D, self).__init__()
        self.act = get_activation(activation)
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias),
            nn.BatchNorm1d(out_channels),
            self.act
        )

    def forward(self, x):
        return self.net(x)


class ConvBNReLURes1D(nn.Module):
    def __init__(self, channel, kernel_size=1, groups=1, res_expansion=1.0, bias=True, activation='relu'):
        super(ConvBNReLURes1D, self).__init__()
        self.act = get_activation(activation)
        self.net1 = nn.Sequential(
            nn.Conv1d(in_channels=channel, out_channels=int(channel * res_expansion),
                      kernel_size=kernel_size, groups=groups, bias=bias),
            nn.BatchNorm1d(int(channel * res_expansion)),
            self.act
        )
        if groups > 1:
            self.net2 = nn.Sequential(
                nn.Conv1d(in_channels=int(channel * res_expansion), out_channels=channel,
                          kernel_size=kernel_size, groups=groups, bias=bias),
                nn.BatchNorm1d(channel),
                self.act,
                nn.Conv1d(in_channels=channel, out_channels=channel,
                          kernel_size=kernel_size, bias=bias),
                nn.BatchNorm1d(channel),
            )
        else:
            self.net2 = nn.Sequential(
                nn.Conv1d(in_channels=int(channel * res_expansion), out_channels=channel,
                          kernel_size=kernel_size, bias=bias)
            )

    def forward(self, x):
        return self.act(self.net2(self.net1(x)) + x)


class PreExtraction(nn.Module):
    def __init__(self, channels, out_channels,  blocks=1, groups=1, res_expansion=1, bias=True,
                 activation='relu', use_xyz=True):
        """
        input: [b,g,k,d]: output:[b,d,g]
        :param channels:
        :param blocks:
        """
        super(PreExtraction, self).__init__()
        in_channels = 3+2*channels if use_xyz else 2*channels
        self.transfer = ConvBNReLU1D(in_channels, out_channels, bias=bias, activation=activation)
        operation = []
        for _ in range(blocks):
            operation.append(
                ConvBNReLURes1D(out_channels, groups=groups, res_expansion=res_expansion,
                                bias=bias, activation=activation)
            )
        self.operation = nn.Sequential(*operation)

    def forward(self, x):
        b, n, s, d = x.size()  # torch.Size([32, 512, 32, 6])
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(-1, d, s)
        x = self.transfer(x)
        batch_size, _, _ = x.size()
        x = self.operation(x)  # [b, d, k]
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = x.reshape(b, n, -1).permute(0, 2, 1)
        return x


# class DGCNN_Grouper(nn.Module):
#     def __init__(self, hidden_dim, z_scale):
#         super().__init__()
#         '''
#         K has to be 16
#         '''
#         self.z_scale = z_scale
#         self.input_trans = nn.Conv1d(3, 8, 1)
#
#         self.layer1 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=1, bias=False),
#                                     nn.GroupNorm(4, 32),
#                                     nn.LeakyReLU(negative_slope=0.2)
#                                     )
#
#         self.layer2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
#                                     nn.GroupNorm(4, 64),
#                                     nn.LeakyReLU(negative_slope=0.2)
#                                     )
#
#         self.layer3 = nn.Sequential(nn.Conv2d(128, hidden_dim, kernel_size=1, bias=False),
#                                     nn.GroupNorm(4, hidden_dim),
#                                     nn.LeakyReLU(negative_slope=0.2)
#                                     )
#
#     @staticmethod
#     def fps_downsample(coor, x, num_group):
#         xyz = coor.transpose(1, 2).contiguous()  # b, n, 3
#         fps_idx = pointnet2_utils.furthest_point_sample(xyz, num_group)
#
#         combined_x = torch.cat([coor, x], dim=1)
#
#         new_combined_x = (
#             pointnet2_utils.gather_operation(
#                 combined_x, fps_idx
#             )
#         )
#
#         new_coor = new_combined_x[:, :3]
#         new_x = new_combined_x[:, 3:]
#
#         return new_coor, new_x
#
#     @staticmethod
#     def get_graph_feature(coor, x, k, N):
#         # coor: bs, 3, np, x: bs, c, np
#         batch_size = x.size(0)
#         center = coor if N == coor.shape[1] else None
#         coor = coor.transpose(2, 1)
#         coor_center, center_idx, group_idx = cluster(coor, N, k, center)
#         x = x.transpose(2, 1).contiguous()
#         f_center = index_points(x, center_idx) if center_idx is not None else x
#         f_group = index_points(x, group_idx)
#         f_group = f_group.view(batch_size, k, N, -1).permute(0, 3, 2, 1).contiguous()
#         f_center = f_center.view(batch_size, -1, N, 1).expand(-1, -1, -1, k)
#         feature = torch.cat((f_group - f_center, f_center), dim=1)
#         return feature, coor_center.transpose(2, 1)
#
#     def forward(self, x):
#         # x: bs, 3, np
#
#         # bs 3 N(128)   bs C(224)128 N(128)
#         coor = x
#         f = self.input_trans(x)
#         # downsample and graph conv1
#         f, coor_center = self.get_graph_feature(coor, f, k=16, N=256)
#         f = self.layer1(f)
#         f = f.max(dim=-1, keepdim=False)[0]
#         coor = coor_center
#
#         # graph conv2
#         f, _ = self.get_graph_feature(coor, f, k=16, N=coor.shape[-1])
#         f = self.layer2(f)
#         f = f.max(dim=-1, keepdim=False)[0]
#
#         # downsample and graph conv3
#         f, coor_center = self.get_graph_feature(coor, f, k=16, N=self.z_scale)
#         f = self.layer3(f)
#         f = f.max(dim=-1, keepdim=False)[0]
#         coor = coor_center
#         return coor, f

class LocalGrouper(nn.Module):
    def __init__(self, in_channels, use_xyz=True, normalize="anchor"):
        super(LocalGrouper, self).__init__()
        self.use_xyz = use_xyz
        add_channel = 3 if self.use_xyz else 0
        if normalize is not None:
            self.normalize = normalize.lower()
        else:
            self.normalize = None
        if self.normalize not in ["center", "anchor"]:
            print("Unrecognized normalize parameter (self.normalize), set to None. Should be one of [center, anchor].")
            self.normalize = None
        if self.normalize is not None:
            self.affine_alpha = nn.Parameter(torch.ones([1, 1, 1, in_channels + add_channel]))
            self.affine_beta = nn.Parameter(torch.zeros([1, 1, 1, in_channels + add_channel]))
        self.extraction = PreExtraction(in_channels, in_channels)

    def forward(self, xyz, feature, groups, k):
        """
        Give xyz[B,N,3] and fea[B,N,D], return new_xyz[b,g,3] and new_fea[b,g,k,d]
        :param xyz: [B,N,3]
        :param feature: [B,N,D]
        :param groups: groups number
        :param k: k-nerighbors
        """
        xyz = xyz.transpose(1, 2)
        feature = feature.transpose(1, 2)
        B, N, _ = xyz.shape
        S = groups
        new_xyz, fps_idx, idx = cluster(xyz, groups, k)
        new_feature = index_points(feature, fps_idx)
        # idx = query_ball_point(radius, nsample, xyz, new_xyz)
        grouped_xyz = index_points(xyz, idx)  # [B, npoint, k, 3]
        grouped_feature = index_points(feature, idx)  # [B, npoint, k, d]
        if self.use_xyz:
            grouped_feature = torch.cat([grouped_feature, grouped_xyz], dim=-1)  # [B, npoint, k, d+z]
        if self.normalize is not None:
            if self.normalize == "center":
                mean = torch.mean(grouped_feature, dim=2, keepdim=True)
            if self.normalize == "anchor":
                mean = torch.cat([new_feature, new_xyz], dim=-1) if self.use_xyz else feature
                mean = mean.unsqueeze(dim=-2)  # [B, npoint, 1, d+3]
            std = torch.std((grouped_feature - mean).reshape(B, -1), dim=-1, keepdim=True).unsqueeze(dim=-1).unsqueeze(
                dim=-1)
            grouped_feature = (grouped_feature - mean) / (std + 1e-5)
            grouped_feature = self.affine_alpha * grouped_feature + self.affine_beta
        x = torch.cat([grouped_feature, new_feature.view(B, S, 1, -1).repeat(1, 1, k, 1)], dim=-1)
        x = self.extraction(x)
        return new_xyz.transpose(1, 2), x

