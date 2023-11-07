import math
import os
import time
import numpy as np
import pandas as pd
import torch
from torch import optim
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from tqdm import tqdm
from diffusion.diffusion_continuous import DiffusionVPSDE, DiffusionSubVPSDE, DiffusionVESDE
from evaluation import compute_all_metrics
try:
    from evaluation.loss import CD_loss, EMD_loss
except:
    pass
from tools.utils import EMA, normalize_point_clouds
from trainer.base import BaseTrainer


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
        self.num_categorys = cfg.data.num_categorys
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
        self.compressor_warmup = cfg.opt.compressor_warmup
        # training
        self.N = cfg.sde.train_N
        self.discrete = cfg.opt.discrete
        self.time_eps = cfg.sde.time_eps
        self.timesteps = torch.linspace(1.0, self.sample_time_eps, self.N)
        self.compressor_optimizer = Adam(compressor.parameters(), lr=cfg.opt.lr, betas=(cfg.opt.compressor_beta1, cfg.opt.compressor_beta2),
                                  weight_decay=cfg.opt.weight_decay)
        self.compressor_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.compressor_optimizer, self.cfg.common.epochs, 0)
        self.alpha = cfg.opt.alpha

    def score_fn(self, t, x, label=None, condition=None):
        t = t.to(x)
        params = self.model(x, t, label=label, condition=condition)
        var = self.SDE.var(t)[:, None, None]
        return -params / torch.sqrt(var), params

    def update(self, data, condition=None, train_individual=True):
        if self.cfg.data.num_categorys > 1:
            cates = data["cate_idx"].to("cuda")
        else:
            cates = None
        if self.itr > 0:
            self.optimizer.swap_parameters_with_ema(store_params_in_ema=True)
        point = data['tr_points'].to("cuda")
        eps, recon, kl, rec = self.clc_compressor(point, cates=cates, discrete=self.discrete,
                                                  condition=condition, train_score=not train_individual)
        if train_individual:
            loss_score = self.update_score(eps, cates=cates, condition=condition)
        else:
            loss_score = torch.zeros_like(kl)
        if self.itr > 0:
            self.optimizer.swap_parameters_with_ema(store_params_in_ema=True)
        self.itr += 1
        return loss_score, kl, rec

    def update_score(self, eps, condition=None, cates=None):
        self.warm_up(self.optimizer, self.itr)
        eps = eps.detach()
        self.model.train()
        self.optimizer.zero_grad()
        size = eps.shape[0]
        idx = torch.from_numpy(np.random.choice(np.arange(self.N), size, replace=True))
        t = self.timesteps.index_select(0, idx).to("cuda")
        e2int_f = self.SDE.e2int_f(t)[:, None, None]
        var = self.SDE.var(t)[:, None, None]
        weight_p = torch.ones(1, device='cuda')
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

    def clc_compressor(self, point, cates=None, condition=None, discrete=False, train_score=True):
        self.compressor.train()
        output = self.compressor(point)
        recon, logqz = output['set'], output['all_logqz']
        logqz = torch.cat(logqz, dim=1).transpose(1, 2)
        eps = output["all_eps"]
        size = point.shape[0]
        if discrete:
            idx = torch.from_numpy(np.random.choice(np.arange(self.N), size, replace=True))
            t = self.timesteps.index_select(0, idx).to("cuda")
            e2int_f = self.SDE.e2int_f(t)[:, None, None]
            var = self.SDE.var(t)[:, None, None]
            weight_q = self.SDE.g2(t)[:, None,  None] / (2 * var)
        else:
            t, var, e2int_f, weight_q, _, g2 = self.SDE.iw_quantities(size, time_eps=self.time_eps,
                                                                             iw_sample_mode=self.cfg.sde.iw_sample_q_mode,
                                                                             iw_subvp_like_vp_sde=True if self.sde_type == 'sub_vpsde' else False)
            e2int_f = e2int_f[:, :, None]
            var = var[:, :, None]
            weight_q = weight_q[:, :, None]
        eta = torch.randn_like(eps)
        std = torch.sqrt(var)
        xt = eps * e2int_f + std * eta
        params = self.model(xt, t, condition=condition, label=cates)
        distance = torch.square(eta - params)
        cross_entropy_const = 0.5 * (
                1.0 + torch.log(2.0 * np.pi * self.SDE.var(t=torch.tensor(self.time_eps, device='cuda'))))
        logpz = -(distance * weight_q + cross_entropy_const)
        kl_loss = (logqz - logpz).mean()
        target_set = point
        # cd_loss = CD_loss(recon, None, target_set, output_mask)
        emd_loss = EMD_loss(recon, target_set)
        cd_loss = CD_loss(recon, target_set)
        rec_loss = (cd_loss + emd_loss).mean()
        if self.epoch < self.cfg.opt.compressor_warmup:
            alpha = self.alpha / 10
        else:
            alpha = self.alpha
        compressor_loss = rec_loss + kl_loss * alpha
        self.compressor_optimizer.zero_grad()
        compressor_loss.backward()
        self.compressor_optimizer.step()
        if train_score:
            self.optimizer.zero_grad()
            self.optimizer.step()
        return eps, recon, kl_loss, rec_loss

    def sample(self, num_samples, label=None, condition=None):
        self.optimizer.swap_parameters_with_ema(store_params_in_ema=True)
        shape = (num_samples, self.num_points)
        t = time.time()
        if self.sample_mode == "continuous":
            eps, nfe_count, _ = self.SDE.sample_model_ode(score_fn=self.score_fn, num_samples=num_samples,
                                                          shape=(self.cfg.score.z_scale, self.cfg.score.z_dim), label=label,
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
            nfe_count = self.cfg.sde.sample_N
        print('NFE:{:}, NFEs{:.2f}/s'.format(nfe_count, nfe_count/(time.time()-t)))
        sample = self.compressor.sample(shape, given_eps=eps)
        self.optimizer.swap_parameters_with_ema(store_params_in_ema=True)
        return sample

    def valsample(self, test_loader, val_cate=0, vis=False):
        with torch.no_grad():
            self.model.eval()
            self.compressor.eval()
            all_ref, all_inp, all_smp = [], [], [],
            use_time = 0.
            if self.cfg.data.num_categorys == 1:
                tbar = tqdm(test_loader)
                for i, data in enumerate(tbar):
                    ref_pts = data['te_points'].cuda()
                    inp_pts = data['tr_points'].cuda()
                    T = time.time()
                    condition = None
                    smp_pts = self.sample(num_samples=inp_pts.size(0), condition=condition)
                    use_time += time.time() - T
                    # denormalize
                    all_inp.append(inp_pts)
                    all_smp.append(smp_pts)
                    all_ref.append(ref_pts)
                smp = torch.cat(all_smp, dim=0)
                ref = torch.cat(all_ref, dim=0)
            else:
                for data in test_loader:
                    idx = data["cate_idx"] == val_cate
                    # m, s = data['mean'][idx].float(), data['std'][idx].float()
                    all_ref.append(data['te_points'][idx])
                    all_inp.append(data['tr_points'][idx])
                ref = torch.cat(all_ref, dim=0).to("cuda")
                # inp = torch.cat(all_inp, dim=0).to("cuda")
                bsize = self.cfg.data.test_batch_size
                T = time.time()
                for _ in tqdm(range(0, math.ceil(ref.shape[0] / bsize))):
                    cates = (torch.ones(bsize) * val_cate).int().to("cuda")
                    smp_pts = self.sample(num_samples=bsize, label=cates)
                    all_smp.append(smp_pts)
                use_time += time.time() - T
                # smp = torch.cat(all_smp, dim=0)[:ref.shape[0]]
                smp = torch.cat(all_smp, dim=0)
                ref = torch.cat(all_ref, dim=0)
            print("Sample rate: %.8f " % (smp.shape[0] / use_time))
            np.save(
                os.path.join(self.cfg.log.save_path, 'smp{:}_ep%d' % self.epoch + ".npy"),
                smp.detach().cpu().numpy()
            )
            if vis:
                from tools.vis_utils import render_3D
                vis_smp = smp.detach().cpu().numpy()
                path = os.path.join(self.cfg.log.save_path, "vis")
                if not os.path.exists(path):
                    os.mkdir(path)
                render_3D(path=os.path.join(self.cfg.log.save_path, "vis"), sample=vis_smp)
            gen_res = compute_all_metrics(
                smp, ref,
                batch_size=64
            )
        all_res = {
            ("val/gen/%s" % k):
                (v if isinstance(v, float) else v.item())
            for k, v in gen_res.items()}
        print("Validation Sample (unit) Epoch:%d " % self.epoch, gen_res)
        return all_res

    def valrecon(self, test_loader, val_cate=0, *args, **kwargs):
        with torch.no_grad():
            self.model.eval()
            self.compressor.eval()
            all_ref, all_inp, all_rec = [], [], []
            tbar = tqdm(test_loader)
            if self.cfg.data.num_categorys == 1:
                tbar = tqdm(test_loader)
                all_rec, all_smp, all_inp = [], [], []
                for i, data in enumerate(tbar):
                    ref_pts = data['te_points'].cuda()
                    inp_pts = data['tr_points'].cuda()
                    output = self.model(ref_pts)
                    eps, rec_pts = output["all_eps"], output["set"]
                    # rec_pts = self.model.sample(ref_pts.shape, eps)
                    shift, scale = data["shift"].float().cuda(), data["scale"].float().cuda()
                    ref_pts = ref_pts * scale + shift
                    rec_pts = rec_pts * scale + shift
                    all_rec.append(rec_pts)
                    all_ref.append(ref_pts)
                rec = torch.cat(all_rec, dim=0)
                ref = torch.cat(all_ref, dim=0)

            else:
                all_rec, all_smp, all_inp = [], [], []
                all_shift, all_scale = [], []
                for data in test_loader:
                    idx = data["cate_idx"] == val_cate
                    m, s = data['mean'][idx].float(), data['std'][idx].float()
                    all_ref.append(data['te_points'][idx])
                    all_inp.append(data['tr_points'][idx])
                    all_shift.append(m)
                    all_scale.append(s)
                pts = torch.cat(all_ref, dim=0).to("cuda")
                shift = torch.cat(all_shift, dim=0).to("cuda")
                scale = torch.cat(all_scale, dim=0).to("cuda")
                ref = pts * scale + shift
                bsize = self.cfg.data.test_batch_size
                for _ in tqdm(range(0, math.ceil(pts.shape[0] / bsize))):
                    points = pts[:bsize]
                    output = self.compressor(points)
                    rec_pts = output['set']
                    all_rec.append(rec_pts)
                    pts = pts[bsize:]
                rec = torch.cat(all_rec, dim=0)[:ref.shape[0]]
                rec = rec * scale + shift
            np.save(
                os.path.join(self.cfg.log.save_path, 'rec_ep%d' % self.epoch + ".npy"),
                rec.detach().cpu().numpy()
            )
            gen_res = compute_all_metrics(
                rec, ref,
                batch_size=256
            )
        all_res = {
            ("val/gen/%s" % k):
                (v if isinstance(v, float) else v.item())
            for k, v in gen_res.items()}
        print("Validation Sample (unit) Epoch:%d " % self.epoch, gen_res)
        return all_res

    def save(self, **kwargs):
        path = os.path.join(self.cfg.log.save_path, 'checkpt_{:}.pth'.format(self.epoch))
        torch.save({
            'cfg': self.cfg,
            'score_state_dict': self.model.state_dict(),
            'score_optim_state_dict': self.optimizer.state_dict(),
            "score_scheduler": self.scheduler.state_dict(),
            'compressor_state_dict': self.compressor.state_dict(),
            'compressor_optim_state_dict': self.compressor_optimizer.state_dict(),
            "compressor_scheduler": self.compressor_scheduler.state_dict(),
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
            self.scheduler.load_state_dict(checkpt["score_scheduler"])
            self.compressor_optimizer.load_state_dict(checkpt["compressor_optim_state_dict"])
            self.compressor_scheduler.load_state_dict(checkpt["compressor_scheduler"])
        if finetune:
            self.epoch = 1
            self.itr = 0
        else:
            self.epoch = checkpt["epoch"] + 1
            self.itr = checkpt["itr"]
        self.time = checkpt["time"]

    def load_pretrain(self):
        path = os.path.join(self.cfg.opt.pretrain_path)
        checkpt = torch.load(path)
        self.model.load_state_dict(checkpt["score_state_dict"], strict=True)
        self.compressor.load_state_dict(checkpt["compressor_state_dict"], strict=True)
        self.compressor.init()
