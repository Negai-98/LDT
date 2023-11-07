import math

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from tools.utils import get_activation, get_norm


def swish(x):
    return x * torch.sigmoid(x)


class TimeEmbedding(nn.Module):
    def __init__(self, dim_embed, dim_out):
        super(TimeEmbedding, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(dim_embed, dim_out), nn.SiLU(), nn.Linear(dim_out, dim_out))
        self.t_emb_dim = dim_embed

    def calc_t_emb(self, ts, max_period=10000):
        """
        Embed time steps into a higher dimension space
        """
        assert self.t_emb_dim % 2 == 0
        # input is of shape (B) of integer time steps
        # output is of shape (B, t_emb_dim)
        ts = ts.unsqueeze(1)
        half_dim = self.t_emb_dim // 2
        t_emb = np.log(max_period) / (half_dim - 1)
        t_emb = torch.exp(torch.arange(half_dim) * -t_emb)
        t_emb = t_emb.to(ts.device)  # shape (half_dim)
        # ts is of shape (B,1)
        t_emb = ts * t_emb
        t_emb = torch.cat((torch.sin(t_emb), torch.cos(t_emb)), 1)

        return t_emb

    def forward(self, t):
        t_emb = self.calc_t_emb(t)
        t_emb = self.mlp(t_emb)
        return t_emb


class LabelEmbedding(nn.Module):
    def __init__(self, num_categorys, dim_embed, dim_out):
        super(LabelEmbedding, self).__init__()
        self.label_emb = nn.Embedding(num_categorys, dim_embed)
        self.mlp = nn.Sequential(nn.Linear(dim_embed, dim_out), nn.SiLU(), nn.Linear(dim_out, dim_out))
        self.t_emb_dim = dim_embed

    def forward(self, label):
        return self.mlp(self.label_emb(label))


class ActNorm(nn.Module):
    '''
    Base class for activation normalization [1].

    References:
        [1] Glow: Generative Flow with Invertible 1×1 Convolutions,
            Kingma & Dhariwal, 2018, https://arxiv.org/abs/1807.03039
    '''

    def __init__(self, num_features, z_scale, data_dep_init=True, eps=1e-6, feature_type='set'):
        super().__init__()
        self.num_features = num_features
        self.z_scale = z_scale
        self.data_dep_init = data_dep_init
        self.eps = eps
        self.feature_type = feature_type
        self.register_buffer('initialized', torch.zeros(1) if data_dep_init else torch.ones(1))
        self.register_params()

    def data_init(self, x):
        self.initialized += 1.
        with torch.no_grad():
            x_mean, x_std = self.compute_stats(x)
            self.shift.data = x_mean
            self.log_scale.data = torch.log(x_std + self.eps)

    def init(self):
        self.initialized += 1.

    def compute_stats(self, x):
        '''Compute x_mean and x_std'''
        if self.feature_type == 'set':
            x_mean = torch.mean(x.view(-1, 1, self.num_features), dim=0, keepdim=True)
            x_std = torch.std(x.view(-1, 1, self.num_features), dim=0, keepdim=True)
        else:
            x_mean = torch.mean(x, dim=0, keepdim=True)
            x_std = torch.std(x, dim=0, keepdim=True)
        return x_mean, x_std

    def register_params(self):
        '''Register parameters shift and log_scale'''
        if self.feature_type == 'set':
            self.register_parameter('shift', nn.Parameter(torch.zeros(1, 1, self.num_features)))
            self.register_parameter('log_scale', nn.Parameter(torch.zeros(1, 1, self.num_features)))
        else:
            self.register_parameter('shift', nn.Parameter(torch.zeros(1, self.z_scale, self.num_features)))
            self.register_parameter('log_scale', nn.Parameter(torch.zeros(1, self.z_scale, self.num_features)))

    def forward(self, x):
        x = x.transpose(1, 2)
        if self.training and not self.initialized: self.data_init(x)
        z = (x - self.shift) * torch.exp(-self.log_scale)
        return z.transpose(1, 2)


class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, n_hidden, activation='gelu', residual=False, dropout_p=0.):
        super().__init__()
        self.act = get_activation(activation)
        self.residual = residual
        self.dropout_p = dropout_p
        if dropout_p > 0:
            self.dropout = nn.Dropout(p=dropout_p)
        self.fc = nn.ModuleList()
        for i in range(n_hidden):
            self.fc.append(nn.Sequential(nn.Conv1d(dim_in if i == 0 else dim_hidden, dim_hidden, 1)))
        if self.residual:
            self.shortcut = nn.Conv1d(dim_in, dim_out, 1) if dim_in != dim_out else nn.Identity()
        self.out = nn.Conv1d(dim_hidden if n_hidden > 0 else dim_in, dim_out, 1)

    def forward(self, x):  # [B, C]
        x_input = x
        for fc in self.fc:
            x = self.act(fc(x))
            x = self.dropout(x) if hasattr(self, 'dropout') else x
        x = self.out(x)
        if self.residual:
            x = x + self.shortcut(x_input)
        return x


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class ResidualBlock(nn.Module):
    """Set Transformer"""
    def __init__(self, dim_in, dim_kv, dim_c, num_heads, norm=None,
                 mlp_ratio=4.0, dropout_att=0., dropout_mlp=0., rescale=False,
                 dim_out=None, AdaLN=True, act=None):
        super().__init__()
        self.dim_in = dim_in
        self.dim_kv = dim_kv
        self.dim_c = dim_c
        self.num_heads = num_heads
        self.dim_split = dim_in // num_heads if dim_in >= num_heads else 1
        self.dropout_att = dropout_att
        self.dropout_mlp = dropout_mlp
        if dim_out is not None:
            if dim_out != dim_in:
                self.shortcut = nn.Conv1d(dim_in, dim_out, 1)
        else:
            dim_out = dim_in
            self.shortcut = nn.Identity()
        self.fc_q = nn.Conv1d(dim_in, dim_out, 1)
        self.fc_kv = nn.Conv1d(dim_kv, 2 * dim_out, 1)
        self.fc_o = nn.Conv1d(dim_out, dim_out, 1)
        self.norm = norm
        self.norm1 = get_norm(dim_in, norm, 16, elementwise_affine=True if dim_c is None else False, axis=2)
        self.norm2 = get_norm(dim_out, norm, 16, elementwise_affine=True if dim_c is None else False, axis=2)
        self.rescale = rescale
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.AdaLN = AdaLN
        if dim_c is not None:
            if AdaLN:
                if dim_in == dim_out:
                    self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim_c, 6 * dim_out))
                else:
                    self.adaLN1 = nn.Sequential(nn.SiLU(), nn.Linear(dim_c, 2 * dim_in))
                    self.adaLN2 = nn.Sequential(nn.SiLU(), nn.Linear(dim_c, 4 * dim_out))
            else:
                self.pos_embedding = nn.Sequential(nn.SiLU(), nn.Linear(dim_c, dim_in))

        self.dropout = nn.Dropout(p=dropout_att)
        self.mlp = MLP(dim_out, int(mlp_ratio * dim_out), dim_out, 1, dropout_p=dropout_mlp)
        self.act = get_activation(act)

    def compute_attention(self, x, y):
        if y is None:
            y = x
        query = self.fc_q(x)  # [B, N, Dv]
        kv = self.fc_kv(y)  # [B, M, Dv]
        B, D, N = query.shape
        key, value = kv[:, :D, :], kv[:, D:, :]
        B, C, N = query.shape
        _, _, M = key.shape
        q = query.reshape(B, self.num_heads, C // self.num_heads, N).permute(0, 1, 3, 2)  # B H N D
        k = key.reshape(B, self.num_heads, C // self.num_heads, M).permute(0, 1, 3, 2)
        v = value.reshape(B, self.num_heads, C // self.num_heads, M).permute(0, 1, 3, 2)
        w = ((q @ k.transpose(-2, -1)) * ((C // self.num_heads) ** -0.5))
        w = w.softmax(dim=-1)
        att = (w @ v).reshape(B, N, C).transpose(1, 2)
        att = self.fc_o(att)
        att = self.dropout(att) if hasattr(self, 'dropout') else att
        return att

    def forward(self, x, y=None, c=None):
        """
        :param x: Tensor([B, N, C]) Query
        :param y: Tensor([B, M, D]) Key and Value
        :param c: Tensor([B, C]) Condition on time and label
        :return: Tensor([B, N, C])
        """

        if c is not None:
            c = c[:, None, :] if c.dim() == 2 else c.transpose(1, 2)
            if self.AdaLN:
                if self.dim_in == self.dim_out:
                    shift_msa, scale_sma, gate_msa, shift_mlp, scale_mlp, gate_mlp, = self.adaLN(c).transpose(1, 2).chunk(6, dim=1)
                else:
                    shift_msa, scale_sma = self.adaLN1(c).transpose(1, 2).chunk(2, dim=1)
                    gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN2(c).transpose(1, 2).chunk(4, dim=1)
                x = self.shortcut(x) + gate_msa * self.compute_attention(modulate(self.norm1(x), shift_msa, scale_sma), y)
                x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
            else:
                x = self.act(self.norm1(x)) + self.pos_embedding(c)
                x = self.shortcut(x) + self.compute_attention(x, y)
                x = x + self.mlp(self.act(self.norm2(x)))
        else:
            x = self.shortcut(x) + self.compute_attention(self.act(self.norm1(x)), y)
            x = x + self.mlp(self.act(self.norm2(x)))
        if self.rescale:
            x = x / math.sqrt(2)
        return x


class FinalLayer(nn.Module):
    def __init__(self, dim_in, dim_out, dim_c, norm):
        super(FinalLayer, self).__init__()
        self.norm = get_norm(dim_in, norm, 16, elementwise_affine=True if dim_c is None else False, axis=2)
        if dim_c is not None:
            self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim_c, 2*dim_in))
        self.ln = nn.Conv1d(dim_in, dim_out, 1)

    def forward(self, x, c=None):
        if c is not None:
            c = c[:, None, :] if c.dim() == 2 else c.transpose(1, 2)
            shift, scale = self.adaLN(c).transpose(1, 2).chunk(2, dim=1)
            x = modulate(self.norm(x), shift, scale)
            x = self.ln(x)
        else:
            x = self.ln(self.norm(x))
        return x

