import torch
import torch.nn as nn
import yaml
from model.Compressor.layers import InitialSet, LocalGrouper
from model.layers import ResidualBlock, FinalLayer, ActNorm, LabelEmbedding, MLP
# from model.Compressor.layers import LocalGrouper
from tools.io import dict2namespace
from tools.utils import get_norm
import torch.nn.functional as F


def log_p_var_normal(samples, mu, logvar):
    log_p = (- 0.5 * torch.square(samples - mu) / torch.exp(logvar) - 0.5 * logvar - 0.9189385332)
    return log_p


def log_p_normal(samples):
    log_p = (- 0.5 * torch.square(samples) - 0.9189385332)
    return log_p


def compute_kl(mu, logvar):
    return -0.5 * (logvar - torch.exp(logvar) - mu.pow(2) + 1).view(mu.shape[0], -1).sum(dim=-1)


def sample(mu, logvar):
    eps = torch.randn(mu.shape).to(mu)
    z = mu + torch.exp(logvar / 2.) * eps
    return z


class Encoder(nn.Module):
    def __init__(self, dim_in, p_dim, num_heads, norm, mlp_ratio=4.0, dropout_p=0., num_layers=1, AdaLN=True):
        super().__init__()
        self.atts = nn.ModuleList()
        for i in range(num_layers):
            self.atts.append(ResidualBlock(dim_in, dim_in, p_dim, num_heads, norm, mlp_ratio,
                                           rescale=False, dropout_att=dropout_p, dropout_mlp=dropout_p, AdaLN=AdaLN))
        self.conv_out = FinalLayer(dim_in, dim_in, p_dim, norm)

    def forward(self, x, pos_embedding):
        for layer in self.atts:
            x = layer(x, x, pos_embedding)
        o = self.conv_out(x, pos_embedding)
        return x, o


class DecoderBlock(nn.Module):
    """ABL (Attentive Bottleneck Layer)"""

    def __init__(self, dim_in, dim_z, dim_o, num_heads, norm, mlp_ratio=4.0, dropout_p=0., min_sigma=-30., act=None, c_dim=None):
        super().__init__()
        self.min_sigma = min_sigma
        self.att = ResidualBlock(dim_in, dim_in, c_dim, num_heads, norm, mlp_ratio, dropout_p, act=act)
        # self.prior = FinalLayer(dim_in, 2 * dim_z, None, norm)
        # self.global_embedding = MiniPointnet(dim_in, dim_in)
        self.prior = nn.Sequential(nn.SiLU(), nn.Conv1d(dim_in, 2 * dim_z, 1))
        self.att1 = ResidualBlock(dim_in, dim_in, c_dim, num_heads, norm, mlp_ratio, dropout_p, act=act)
        self.ln = nn.Conv1d(dim_z, dim_in, 1)

    def compute_posterior(self, x, o=None, c=None):
        """
        Estimate residual posterior parameters from prior parameters and top-down features
        :param x: Tensor([B, N', D])
        :param o: Tensor([B, N, D])
        :return: Tensor([B, M, Dz]), Tensor([B, M, Dz])
        """
        if o is not None:
            # global_feature = self.global_embedding(o).unsqueeze(-1)
            # x = x + global_feature
            x = self.att(x, o, c)
            posterior = self.prior(x)
        else:
            x = self.att(x, x, c)
            posterior = self.prior(x)
        mu = posterior[:, :posterior.shape[1] // 2, :]  # [B, M, Dz]
        logvar = posterior[:, posterior.shape[1] // 2:, :].clamp(self.min_sigma, 10.)
        return mu, logvar

    def forward(self, o, x, c=None):
        x = self.ln(x)
        o = self.att1(o, x, c)
        return o


class MiniPointnet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MiniPointnet, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, 128, 1)
        self.conv2 = nn.Conv1d(128, 256, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc = nn.Linear(256, output_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 256)
        ms = self.fc(x)
        return ms



class Compressor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.input_dim = cfg.input_dim
        self.max_outputs = cfg.max_outputs
        self.n_layers = cfg.n_layers
        self.z_dim = cfg.z_dim
        self.hidden_dim = cfg.hidden_dim
        self.num_heads = cfg.num_heads
        self.norm = cfg.norm
        self.encoder_dropout_p = cfg.encoder_dropout_p
        self.decoder_dropout_p = cfg.decoder_dropout_p
        self.activation = cfg.activation
        self.z_scales = cfg.z_scales
        self.p_dim = cfg.p_dim
        self.input = nn.Conv1d(self.input_dim, self.hidden_dim, 1)
        self.ActNorm = cfg.ActNorm
        if self.ActNorm is not None:
            self.conv_in = ActNorm(self.hidden_dim, self.z_scales, feature_type=cfg.ActNorm)
        self.outsize = cfg.outsize
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.upsample = nn.ModuleList()
        self.neighbors = cfg.neighbors
        self.mlp_ratio = cfg.mlp_ratio
        self.min_sigma = cfg.min_sigma
        self.encoder_layers = cfg.encoder_layers
        self.decoder_act = cfg.decoder_act
        self.AdaLN = cfg.AdaLN
        self.group = LocalGrouper(self.hidden_dim, True, normalize=cfg.cluster_norm)
        if cfg.pos_embedding == 'mlp':
            self.pos_embedding = MLP(dim_in=3, dim_hidden=self.p_dim, dim_out=self.p_dim, n_hidden=1)
        else:
            self.pos_embedding = MiniPointnet(3, self.p_dim)
        self.class_condition = cfg.class_condition
        if cfg.class_condition:
            self.LabelEmbedding = LabelEmbedding(cfg.num_categorys, self.p_dim, self.p_dim)
            self.label_dim = self.p_dim
        else:
            self.label_dim = None
        for i in range(self.n_layers):
            self.encoder.append(
                Encoder(self.hidden_dim, self.p_dim, self.num_heads, norm=self.norm,
                        dropout_p=self.encoder_dropout_p, num_layers=self.encoder_layers, mlp_ratio=self.mlp_ratio))
            self.decoder.append(
                DecoderBlock(self.hidden_dim, cfg.z_dim, self.hidden_dim if i != self.n_layers - 1 else None,
                             self.num_heads, norm=self.norm, dropout_p=self.decoder_dropout_p, mlp_ratio=self.mlp_ratio,
                             min_sigma=self.min_sigma, act=cfg.decoder_act, c_dim=self.label_dim))
        # self.output = FinalLayer(self.hidden_dim, 3, None, self.norm)
        self.output = nn.Conv1d(self.hidden_dim, 3, 1)
        self.init_set = InitialSet(self.hidden_dim, self.max_outputs)
        self.norm_input = cfg.norm_input
        self.pre_group = cfg.pre_group
        if cfg.pre_group:
            self.pre_grouper = LocalGrouper(self.hidden_dim, True, normalize=cfg.cluster_norm)

        # self.initialize_weights()

    def init(self):
        if self.ActNorm is not None:
            self.conv_in.init()

    def norm_pts(self, pts):
        mean = torch.mean(pts, dim=1, keepdim=True)
        std = torch.std(pts, dim=1, keepdim=True)
        pts = (pts - mean)/std
        return pts

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        # for encoder in self.encoder:
        #     for block in encoder.atts:
        #         nn.init.constant_(block.adaLN[-1].weight, 0)
        #         nn.init.constant_(block.adaLN[-1].bias, 0)
        # Zero-out output layers:
        nn.init.constant_(self.output.weight, 0)
        nn.init.constant_(self.output.bias, 0)

    def bottom_up(self, pts, label=None):
        if self.norm_input:
            pts = self.norm_pts(pts)
        pts = pts.transpose(1, 2)
        x = self.input(pts)
        if self.pre_group:
            pts, x = self.pre_grouper(pts, x, 256, 32)
        center, x = self.group(pts, x, self.z_scales, pts.shape[2]//self.z_scales*2)
        pos = self.pos_embedding(center)
        if label is not None:
            pos = pos + label
        # center, x = self.group(pts)
        if self.ActNorm is not None:
            x = self.conv_in(x)
        outputs = list()
        for i, layer in enumerate(self.encoder):
            x, o = layer(x, pos)
            outputs.append(o)
        return {"outputs": outputs, "max": x.max()}

    def top_down(self, encoder_out, num_points=None, label=None):
        """ Stochastic top-down decoding
        :param cardinality: Tensor([B,])
        :param bottom_up_h: List([Tensor([B, M, D])]) in top-down order
        :return:
        """
        B, _, N = encoder_out[0].shape
        o = self.init_set((B, num_points) if num_points is not None else (B, self.outsize))
        alphas, posteriors, all_eps, kls, all_logqz, all_logpz = [], [(o, None, None)], [], [], [], []
        for idx, layer in enumerate(reversed(self.decoder)):
            x = encoder_out[-idx - 1]
            mu, logvar = layer.compute_posterior(x, o if idx != 0 else None, c=label)
            eps = sample(mu, logvar)
            # Flow?
            logqz = log_p_var_normal(eps, mu, logvar)
            logpz = log_p_normal(eps)
            kl = logqz - logpz
            o = layer(o, eps, label)
            all_eps.append(eps)
            posteriors.append((eps, mu, logvar))
            kls.append(kl)
            all_logqz.append(logqz)
            all_logpz.append(logpz)
        o = self.output(o).transpose(1, 2)  # [B, D, N]
        return {'set': o, 'posteriors': posteriors,
                'kls': kls, 'all_logqz': all_logqz, 'all_eps': all_eps}

    def forward(self, x, num_points=None, label=None):
        """ Bidirectional inference
        :param x: Tensor([B, N, Di])
        :return: Tensor([B, N, Do]), Tensor([B, N]), List([Tensor([H, B, N, M]), Tensor([H, B, N, M])]) * 2
        """
        if label is not None and self.class_condition:
            l_emb = self.LabelEmbedding(label)
        else:
            l_emb = None
        bup = self.bottom_up(x, label=l_emb)
        tdn = self.top_down(encoder_out=bup["outputs"], num_points=num_points, label=l_emb)
        all_eps = torch.cat(tdn["all_eps"], dim=1).transpose(1, 2)
        o = self.postprocess(tdn['set'])
        return {'set': o, 'posteriors': tdn['posteriors'], 'kls': tdn['kls'], 'all_eps': all_eps,
                'all_logqz': tdn['all_logqz'], "max": bup["max"]}

    def sample(self, shape, given_eps=None):
        """ Top-down generation
        :param given_eps: List([Tensor([B, ?, D])])
        :return: Tensor([B, N, Do]), Tensor([B, N]), List([Tensor([B, M, D])]),
                 List([Tensor([H, B, N, M]), Tensor([H, B, N, M])])
        """
        B, num_points = shape[0], shape[1]
        o = self.init_set((B, num_points) if num_points is not None else (B, self.outsize))
        if given_eps is None:
            given_eps = torch.randn((o.shape[0], self.z_scales, self.n_layers * self.z_dim)).to(o)
        given_eps = given_eps.transpose(1, 2)
        given_eps = torch.split(given_eps, [self.z_dim, ] * self.n_layers, dim=1)
        for idx, layer in enumerate(reversed(self.decoder)):
            eps = given_eps[idx]
            o = layer(o, eps)
        o = self.output(o).transpose(1, 2)  # [B, N, Do]
        o = self.postprocess(o)
        return o

    @staticmethod
    def postprocess(x):
        if x.shape[-1] == 2:  # MNIST, xy
            return (torch.tanh(x) + 1) / 2.  # [B, N, Do], [0, 1] range
        elif x.shape[-1] == 3:  # ShapeNet, xyz
            return x  # [B, N, Do]
        elif x.shape[-1] == 4:  # KITTI, xyzc
            x = x.clone()
            x[..., -1] = (torch.tanh(x[..., -1]) + 1) / 2.
            return x  # [B, N, Do]

    def multi_gpu_wrapper(self, f):
        self.input = f(self.input)
        self.encoder = f(self.encoder)
        self.decoder = f(self.decoder)
        self.output = f(self.output)


if __name__ == '__main__':
    def get_config(path):
        with open(path, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        config = dict2namespace(config)
        return config


    with torch.no_grad():
        path = "config.yaml"
        cfg = get_config(path)
        label = torch.randint(10, (16,)).cuda()
        data = torch.randn(16, 128, 3).cuda()
        print("===> testing vae ...")
        model = Compressor(cfg.vae).cuda()
        out = model(data, label=label)
        eps = out["all_eps"]
        rec = model.sample(data.shape, eps)
    print(out["set"].shape)
    print(rec.shape)
