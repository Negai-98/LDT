import torch
import torch.nn as nn
import yaml
from torch.nn import functional as F
import numpy as np
from torch.optim import Adam
from torchvision import models
from model.Compressor.layers import LocalGrouper
from model.layers import TimeEmbedding, LabelEmbedding, ResidualBlock, FinalLayer
from tools.io import dict2namespace


class ConditionNet(nn.Module):
    def __init__(self, hidden_size, p_dim, patch_size=16, img_condition=True, pt_condition=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.img_condition = img_condition
        self.pt_condition = pt_condition
        self.patch_size = patch_size
        if pt_condition:
            self.pc_conv_in = nn.Conv1d(3, 128, 1)
            self.group = LocalGrouper(128, True, normalize='center')
            self.pc_conv_out = nn.Conv1d(128, self.hidden_size, 1)
        if img_condition:
            base = models.resnet18(pretrained=False)
            self.resnet = nn.Sequential(*list(base.children())[:-4])
            self.ln = nn.Linear(128, p_dim)

        self.conv_out = nn.Conv1d(self.hidden_size, self.hidden_size, 1)

    def forward(self, condition):
        x, img = None, None
        if 'img' in condition and self.img_condition:
            img = condition['img'].to('cuda')
            img = F.adaptive_max_pool2d(self.resnet(img), 1).squeeze()
            img = self.ln(img)
        if 'pts' in condition and self.pt_condition:
            pts = condition['pts'].transpose(1, 2).to('cuda')
            x = self.pc_conv_in(pts)
            _, x = self.group(pts, x, self.patch_size, x.shape[1] // self.patch_size * 2)
            x = self.pc_conv_out(x)
        pts_condition = x if x is not None else 0.
        img_conditon = img if img is not None else 0.
        return pts_condition, img_conditon


class Score(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.z_dim = cfg.z_dim
        self.out_dim = self.z_dim
        self.z_scale = cfg.z_scale
        self.hidden_size = cfg.hidden_size
        self.num_heads = cfg.num_heads
        self.condition = cfg.condition
        self.num_steps = cfg.num_steps
        self.norm = cfg.norm
        self.t_dim = cfg.t_dim
        self.num_blocks = cfg.num_blocks
        self.dropout = cfg.dropout
        self.learn_sigma = cfg.learn_sigma
        self.unet = cfg.unet
        self.AdaLN = cfg.AdaLN
        if self.condition:
            self.c_net = ConditionNet(self.hidden_size, self.t_dim, patch_size=self.z_scale)
        if self.unet:
            self.Transformer_Up = nn.ModuleList([ResidualBlock(self.hidden_size, self.hidden_size,
                                                               self.t_dim, self.num_heads,
                                                               norm=self.norm, dropout_att=self.dropout,
                                                               dropout_mlp=self.dropout, act=cfg.act, AdaLN=self.AdaLN)
                                                 for _ in range(self.num_blocks // 2)])
            self.Transformer_Mid = ResidualBlock(self.hidden_size, self.hidden_size,
                                                 self.t_dim, self.num_heads,
                                                 norm=self.norm, dropout_att=self.dropout,
                                                 dropout_mlp=self.dropout, act=cfg.act, AdaLN=self.AdaLN)
            self.Transformer_Down = nn.ModuleList([ResidualBlock(self.hidden_size * 2, self.hidden_size * 2,
                                                                 self.t_dim, self.num_heads,
                                                                 norm=self.norm, dropout_att=self.dropout,
                                                                 dropout_mlp=self.dropout,
                                                                 dim_out=self.hidden_size, act=cfg.act,
                                                                 AdaLN=self.AdaLN)
                                                   for _ in range(self.num_blocks // 2)])
        else:
            self.Transformer = nn.ModuleList([ResidualBlock(self.hidden_size, self.hidden_size,
                                                            self.t_dim, self.num_heads,
                                                            norm=self.norm, dropout_att=self.dropout,
                                                            dropout_mlp=self.dropout, act=cfg.act, AdaLN=self.AdaLN)
                                              for _ in range(self.num_blocks)])
        if cfg.num_categorys > 1:
            self.LabelEmbedding = LabelEmbedding(cfg.num_categorys, self.t_dim, self.t_dim)
        else:
            self.label_dim = None

        self.ln_in = nn.Conv1d(self.z_dim, self.hidden_size, 1)
        self.TimeEmbedding = TimeEmbedding(self.t_dim // 4, self.t_dim)
        self.ln_out = FinalLayer(self.hidden_size, self.z_dim, self.t_dim, self.norm)
        # self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Conv1d):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)
        nn.init.normal_(self.TimeEmbedding.mlp[-1].weight, std=0.02)
        for block in self.Transformer:
            nn.init.constant_(block.adaLN[-1].weight, 0)
            nn.init.constant_(block.adaLN[-1].bias, 0)
            # Zero-out output layers:
        nn.init.constant_(self.ln_out.weight, 0)
        nn.init.constant_(self.ln_out.bias, 0)

    def forward(self, x, t, label=None, condition=None):
        """
        :param x: (bs, chs, dim) Hidden features
        :param t: (bs,) Input times
        :param kwargs: dict (label(bs,), condtition(bs, chs, dim))
        :return: (bs, chs, dim) predict noise
        """
        # t = t / self.num_steps
        if label is not None:
            l_emb = self.LabelEmbedding(label)
        else:
            l_emb = None
        if condition is not None:
            if isinstance(condition, dict):
                condition = self.c_net(condition)
        else:
            condition = (None, 0.)
        t_emb = self.TimeEmbedding(t)
        c = t_emb + l_emb if l_emb is not None else t_emb + condition[1]
        x = x.transpose(1, 2)
        x = self.ln_in(x)
        if self.unet:
            x_list = [x]
            for idx, layer in enumerate(self.Transformer_Up):
                x = layer(x, condition[0], c)
                x_list.append(x)
            x = self.Transformer_Mid(x, condition[0], c)
            for idx, layer in enumerate(self.Transformer_Down):
                x = torch.cat((x, x_list.pop()), dim=1)
                x = layer(x, condition[0], c)
        else:
            for idx, layer in enumerate(self.Transformer):
                x = layer(x, condition[0] if idx % 2 == 0 else None, c)
        out = self.ln_out(x, c).transpose(1, 2)
        return out


if __name__ == '__main__':
    def get_config(path):
        with open(path, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        config = dict2namespace(config)
        return config


    path = "config.yaml"
    cfg = get_config(path)
    img = torch.rand(16, 3, 224, 224).cuda() * 255
    pts = torch.randn(16, 256, 3).cuda()
    condition = {'pts': pts, 'img': img}
    label = None
    # condition = torch.randn(16, 64, 256).cuda()
    time = torch.rand(16, ).cuda()
    # label = torch.randint(10, (16,)).cuda()
    print("===> testing scorenet ...")
    model = Score(cfg.score).cuda()
    data = torch.randn(16, 32, 120).cuda()
    dict = {"label": label, "condition": condition}
    out = model(data, time, **dict)
    print(out.shape)
