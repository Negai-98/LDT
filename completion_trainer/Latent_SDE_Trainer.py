import math
import os
import time
import numpy as np
import pandas as pd
import torch
from pointnet2_ops import pointnet2_utils
from torch import optim
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from tqdm import tqdm
from diffusion.diffusion_continuous import DiffusionVPSDE, DiffusionSubVPSDE, DiffusionVESDE
from model.Compressor.layers import index_points
from tools.utils import EMA
from tools.vis_utils import render_3D
from trainer.base import BaseTrainer

try:
    from StructuralLosses.nn_distance import nn_distance


    def distChamfer(x, y):
        return nn_distance(x, y)
except:
    print("distChamferCUDA not available; fall back to slower version.")


    def distChamfer(a, b):
        x, y = a, b
        bs, num_points, points_dim = x.size()
        xx = torch.bmm(x, x.transpose(2, 1))
        yy = torch.bmm(y, y.transpose(2, 1))
        zz = torch.bmm(x, y.transpose(2, 1))
        diag_ind = torch.arange(0, num_points).to(a).long()
        rx = xx[:, diag_ind, diag_ind].unsqueeze(1).expand_as(xx)
        ry = yy[:, diag_ind, diag_ind].unsqueeze(1).expand_as(yy)
        P = (rx.transpose(2, 1) + ry - 2 * zz)
        return P.min(1)[0], P.min(2)[0]


def L2_ChamferEval_1000(array1, array2):
    dist1, dist2 = distChamfer(array1, array2)
    dist = torch.mean(dist1) + torch.mean(dist2)
    return dist * 1000


def F1Score(array1, array2, threshold=0.001):
    dist1, dist2 = distChamfer(array1, array2)
    precision_1 = torch.mean((dist1 < threshold).float(), dim=1)
    precision_2 = torch.mean((dist2 < threshold).float(), dim=1)
    fscore = 2 * precision_1 * precision_2 / (precision_1 + precision_2)
    fscore[torch.isnan(fscore)] = 0
    return fscore, precision_1, precision_2


class Trainer(BaseTrainer):

    def __init__(self, cfg, model, compressor, device):
        super(Trainer, self).__init__(cfg, device)
        if cfg.sde.sde_type == 'vpsde':
            self.SDE = DiffusionVPSDE(cfg.sde)
        elif cfg.sde.sde_type == 'sub_vpsde':
            self.SDE = DiffusionSubVPSDE(cfg.sde)
        elif cfg.sde.sde_type == 'vesde':
            self.SDE = DiffusionVESDE(cfg.sde)
        else:
            raise TypeError
        self.sde_type = cfg.sde.sde_type
        self.num_points = cfg.data.tr_max_sample_points
        self.device = device
        # score network
        self.model = model.to(device)
        self.optimizer = Adam(model.parameters(), lr=cfg.opt.lr, weight_decay=cfg.opt.weight_decay,
                              betas=(cfg.opt.beta1, cfg.opt.beta2))
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.cfg.common.epochs, 0)
        # compressor
        self.compressor = compressor.to(device)
        self.optimizer = EMA(Adam(model.parameters(), lr=cfg.opt.lr, betas=(cfg.opt.beta1, cfg.opt.beta2),
                                  weight_decay=cfg.opt.weight_decay), ema_decay=cfg.opt.ema_decay)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.cfg.common.epochs, 0)
        # sampling
        self.sample_time_eps = cfg.sde.sample_time_eps
        self.sample_N = cfg.sde.sample_N
        self.sample_mode = cfg.sde.sample_mode
        self.ode_tol = cfg.sde.ode_tol

        # training
        self.N = cfg.sde.train_N
        self.discrete = cfg.opt.discrete
        self.time_eps = cfg.sde.time_eps
        self.timesteps = torch.linspace(1.0, self.sample_time_eps, self.N)

    def score_fn(self, t, x, label=None, condition=None):
        t = t.to(x)
        params = self.model(x, t, label=label, condition=condition)
        var = self.SDE.var(t)[:, None, None]
        return -params / torch.sqrt(var), params

    def update(self, data, condition=None):
        cates = None
        if self.itr > 0:
            self.optimizer.swap_parameters_with_ema(store_params_in_ema=True)
        point = data.to("cuda")
        with torch.no_grad():
            output = self.compressor(point)
        eps = output["all_eps"]
        loss = self.update_score(eps, cates=cates, discrete=self.discrete, condition=condition)
        if self.itr > 0:
            self.optimizer.swap_parameters_with_ema(store_params_in_ema=True)
        self.itr += 1
        return loss

    def update_score(self, eps, condition=None, cates=None, discrete=True):
        self.warm_up(self.optimizer, self.itr)
        eps = eps.detach()
        self.model.train()
        self.optimizer.zero_grad()
        size = eps.shape[0]
        if discrete:
            idx = torch.from_numpy(np.random.choice(np.arange(self.N), size, replace=True))
            t = self.timesteps.index_select(0, idx).to("cuda")
            e2int_f = self.SDE.e2int_f(t)[:, None, None]
            var = self.SDE.var(t)[:, None, None]
            weight_p = torch.ones(1, device='cuda')
        else:
            t, var, e2int_f, weight_p, _, g2 = self.SDE.iw_quantities(size, time_eps=self.time_eps,
                                                                      iw_sample_mode=self.cfg.sde.iw_sample_p_mode,
                                                                      iw_subvp_like_vp_sde=True if self.sde_type == 'sub_vpsde' else False)
            e2int_f = e2int_f[:, :, None]
            var = var[:, :, None]
        eta = torch.randn_like(eps)
        std = torch.sqrt(var)
        xt = eps * e2int_f + std * eta
        params = self.model(xt, t, condition=condition, label=cates)
        if self.cfg.opt.loss_type == "l1":
            distance = torch.abs(eta - params)
        else:
            distance = torch.square(eta - params)
        loss_score = (distance * weight_p).mean()
        loss = loss_score
        loss.backward()
        if self.cfg.opt.grad_norm_clip_value is not None:
            clip_grad_norm_(self.model.parameters(), self.cfg.opt.grad_norm_clip_value)
        self.optimizer.step()
        return loss_score

    def sample(self, num_samples, num_points=None, label=None, condition=None):
        self.optimizer.swap_parameters_with_ema(store_params_in_ema=True)
        shape = (num_samples, self.num_points if num_points is None else num_points)
        if condition is not None:
            condition = self.model.c_net(condition)
        if self.sample_mode == "continuous":
            eps, nfe_count, _ = self.SDE.sample_model_ode(score_fn=self.score_fn, num_samples=num_samples,
                                                          shape=(self.cfg.score.z_scale, self.cfg.score.z_dim),
                                                          label=label,
                                                          ode_eps=self.sample_time_eps, enable_autocast=False,
                                                          ode_solver_tol=self.ode_tol, condition=condition)
        else:
            eps = self.SDE.sample_discrete(score_fn=self.score_fn, N=self.cfg.sde.sample_N,
                                           corrector=self.cfg.sde.corrector,
                                           predictor=self.cfg.sde.predictor,
                                           corrector_steps=self.cfg.sde.corrector_steps,
                                           shape=(self.cfg.score.z_scale, self.cfg.score.z_dim),
                                           time_eps=self.sample_time_eps, label=label,
                                           denoise=self.cfg.sde.denoise, device=self.device,
                                           num_samples=num_samples, probability_flow=self.cfg.sde.probability_flow,
                                           snr=self.cfg.sde.snr, condition=condition)
        sample = self.compressor.sample(shape, given_eps=eps)
        self.optimizer.swap_parameters_with_ema(store_params_in_ema=True)
        return sample

    def valsample(self, test_loader, vis=False, full=False):
        with torch.no_grad():
            self.model.eval()
            self.compressor.eval()
            all_ref, all_part, all_smp = [], [], [],
            use_time = 0.
            tbar = tqdm(test_loader)
            for i, data in enumerate(tbar):
                views, pc, pc_part = data
                pc, pc_part = pc.to('cuda'), pc_part.to('cuda')
                pc_center, pc_part_center = pointnet2_utils.furthest_point_sample(pc, 2048).long(), \
                                            pointnet2_utils.furthest_point_sample(pc_part, 2048).long()
                ref_pts, pc_part = index_points(pc, pc_center), index_points(pc_part, pc_part_center)
                views = views.float()
                condition = {'img': views, 'pts': pc_part}
                T = time.time()
                smp_pts = self.sample(num_samples=ref_pts.size(0), condition=condition)
                use_time += time.time() - T
                # denormalize
                all_smp.append(smp_pts)
                all_ref.append(ref_pts)
                all_part.append(pc_part)
                smp = torch.cat(all_smp, dim=0)
                ref = torch.cat(all_ref, dim=0)
                cd = L2_ChamferEval_1000(smp, ref)
                f1score, p1, p2 = F1Score(smp, ref)
                tbar.set_postfix(
                    {'cd': '{:.8f}'.format(float(cd.cpu())), 'f1_score': '{:.8f}'.format(float(f1score.mean().cpu()))}
                )
                # fixme
                if not full:
                    if smp.size(0) > 1000:
                        break
            smp = torch.cat(all_smp, dim=0)
            ref = torch.cat(all_ref, dim=0)
            part = torch.cat(all_part, dim=0)
            if vis:
                vis_smp = smp.detach().cpu().numpy()
                path = os.path.join(self.cfg.log.save_path, "vis")
                if not os.path.exists(path):
                    os.mkdir(path)
                render_3D(path=os.path.join(self.cfg.log.save_path, "vis"), sample=vis_smp)
            cd = L2_ChamferEval_1000(smp, ref)
            f1score, p1, p2 = F1Score(smp, ref)
            np.save(
                os.path.join(self.cfg.log.save_path, 'part_ep%d' % self.epoch + ".npy"),
                part.detach().cpu().numpy()
            )
            np.save(
                os.path.join(self.cfg.log.save_path, 'smp_ep%d' % self.epoch + ".npy"),
                smp.detach().cpu().numpy()
            )
            np.save(
                os.path.join(self.cfg.log.save_path, 'ref_ep%d' % self.epoch + ".npy"),
                ref.detach().cpu().numpy()
            )
        all_res = {'cd': cd, 'f1score': f1score.mean()}
        print("Validation Sample (unit) Epoch:%d " % self.epoch, all_res)
        return all_res


    def reconstrustion(self, test_loader):
        with torch.no_grad():
            self.compressor.eval()
            all_ref, all_rec, all_inp, all_smp, all_ref_denorm = [], [], [], [], []
            count = 0
            for i, data in enumerate(test_loader):
                views, pc, pc_part = data
                pc = pc.to('cuda')
                pc_center = pointnet2_utils.furthest_point_sample(pc, 2048).long()
                ref_pts = index_points(pc, pc_center)
                output = self.compressor(ref_pts)
                eps, rec_pts = output["all_eps"], output["set"]
                all_rec.append(rec_pts)
                all_ref.append(ref_pts)
                count += rec_pts.shape[0]
            rec = torch.cat(all_rec, dim=0)
            ref = torch.cat(all_ref, dim=0)
            # mask = sample_mask(500, ref.shape[0]).to("cuda")
            # rec, ref = rec[~mask, :], ref[~mask, :]
            np.save(
                os.path.join(self.cfg.log.save_path,
                             'rec_ep%d.npy' % self.epoch),
                rec.detach().cpu().numpy()
            )
            cd = L2_ChamferEval_1000(rec, ref)
            f1score, p1, p2 = F1Score(rec, ref)
        all_res = {'cd': cd, 'f1score': f1score.mean()}
        # print("Validation Sample (unit) Epoch:%d " % self.epoch, all_res)
        return all_res


    def save(self, **kwargs):
        path = os.path.join(self.cfg.log.save_path, 'checkpt_{:}.pth'.format(self.epoch))
        torch.save({
            'cfg': self.cfg,
            'score_state_dict': self.model.state_dict(),
            'score_optim_state_dict': self.optimizer.state_dict(),
            "score_scheduler": self.scheduler.state_dict(),
            'compressor_state_dict': self.compressor.state_dict(),
            'epoch': self.epoch,
            'itr': self.itr,
            "time": self.time
        }, path)

    def resume(self, epoch=None, strict=False, load_optim=True, finetune=False,
               **kwargs):
        if finetune:
            load_optim = False
            strict = False
        if epoch is None:
            path = os.path.join(self.cfg.log.save_path, 'training.csv')
            tsdf = pd.read_csv(os.path.join(path))
            epoch = tsdf["epoch"].values[-1]
        path = os.path.join(self.cfg.log.save_path, 'checkpt_{:}.pth'.format(epoch))
        checkpt = torch.load(path, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(checkpt["score_state_dict"], strict=strict)
        self.compressor.load_state_dict(checkpt["compressor_state_dict"], strict=strict)
        self.compressor.init()
        if load_optim:
            self.optimizer.load_state_dict(checkpt["score_optim_state_dict"])
        if finetune:
            self.epoch = 1
            self.itr = 0
        else:
            self.epoch = checkpt["epoch"] + 1
            self.itr = checkpt["itr"]
        self.time = checkpt["time"]
        self.scheduler.base_lrs = [self.cfg.opt.lr]

    def load_pretrain(self):
        path = os.path.join(self.cfg.compressor.pretrain_path)
        checkpt = torch.load(path)
        self.compressor.load_state_dict(checkpt["state_dict"], strict=True)
        self.compressor.init()
