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
from tools.vis_utils import render_3D
from evaluation.loss import EMD_loss, CD_loss
from evaluation import compute_all_metrics
from trainer.base import BaseTrainer


class Trainer(BaseTrainer):

    def __init__(self, cfg, model, device):
        super(Trainer, self).__init__(cfg, device)
        self.num_points = cfg.data.tr_max_sample_points
        self.device = device
        self.kl_weight = cfg.opt.kl_weight
        self.model = model.to(device)
        self.optimizer = Adam(model.parameters(), lr=cfg.opt.lr, betas=(cfg.opt.beta1, cfg.opt.beta2),
                                    weight_decay=cfg.opt.weight_decay)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.cfg.common.epochs, 0)

    def update(self, data):
        cates = data["cate_idx"].to("cuda")
        data = data['tr_points'].to(self.device)
        self.model.train()
        self.warm_up(self.optimizer, self.itr)
        loss, kl_loss, rec_loss = self.compute_loss(data, cates)
        self.optimizer.zero_grad()
        loss.backward()
        if self.cfg.opt.grad_norm_clip_value is not None:
            clip_grad_norm_(self.model.parameters(), self.cfg.opt.grad_norm_clip_value)
        self.optimizer.step()
        self.itr += 1
        return loss, kl_loss, rec_loss

    def compute_loss(self, target_set, label):
        output = self.model(target_set, label=label)
        output_set, kls = output['set'], output['kls']
        emd_loss = EMD_loss(output_set, target_set)
        cd_loss = CD_loss(output_set, target_set)
        rec_loss = cd_loss + emd_loss
        kl_loss = torch.cat(kls, dim=1)
        kl_loss = kl_loss.mean()
        loss = self.kl_weight * kl_loss + rec_loss
        return loss, kl_loss, rec_loss

    def sample(self, num_samples, num_points, given_eps=None):
        shape = (num_samples, num_points)
        self.model.eval()
        with torch.no_grad():
            sample = self.model.sample(shape, given_eps=given_eps)
        return sample

    def valsample(self, test_loader, sample_points, vis=False):
        with torch.no_grad():
            self.model.eval()
            all_ref, all_rec, all_smp, all_ref_denorm = [], [], [], []
            all_rec_gt, all_inp_denorm, all_inp = [], [], []
            tbar = tqdm(test_loader)
            use_time = 0.
            for i, data in enumerate(tbar):
                ref_pts = data['te_points'].cuda()
                inp_pts = data['tr_points'].cuda()
                T = time.time()
                rec_pts = self.sample(num_samples=inp_pts.size(0), num_points=sample_points)
                use_time += time.time() - T
                all_inp.append(inp_pts)
                all_rec.append(rec_pts)
                all_ref.append(ref_pts)
                smp = torch.cat(all_rec, dim=0)
                ref = torch.cat(all_ref, dim=0)
            print("Sample rate: %.8f " % (smp.shape[0] / use_time))
            np.save(
                os.path.join(self.cfg.log.save_path,
                             'smp_ep%d.npy' % self.epoch),
                smp.detach().cpu().numpy()
            )
            if vis:
                vis_smp = smp.detach().cpu().numpy()
                path = os.path.join(self.cfg.log.save_path, "vis")
                if not os.path.exists(path):
                    os.mkdir(path)
                render_3D(path=os.path.join(self.cfg.log.save_path, "vis"), sample=vis_smp)
            gen_res = compute_all_metrics(
                smp, ref,
                batch_size=128
            )
        all_res = {
            ("val/gen/%s" % k):
                (v if isinstance(v, float) else v.item())
            for k, v in gen_res.items()}
        print("Validation Sample (unit) Epoch:%d " % self.epoch, gen_res)
        return all_res

    def reconstrustion(self, test_loader, val_cate=0):
        with torch.no_grad():
            self.model.eval()
            all_ref, all_rec, all_inp, all_smp, all_ref_denorm = [], [], [], [], []
            tbar = tqdm(test_loader)
            if self.cfg.data.num_categorys == 1:
                for i, data in enumerate(tbar):
                    ref_pts = data['te_points'].cuda()
                    inp_pts = data['tr_points'].cuda()
                    output = self.model(ref_pts)
                    eps, rec_pts = output["all_eps"], output["set"]
                    # rec_pts = self.model.sample(ref_pts.shape, eps)
                    shift, scale = data["shift"].float().cuda(), data["scale"].float().cuda()
                    ref_pts = ref_pts * scale + shift
                    rec_pts = rec_pts * scale + shift
                    inp_pts = inp_pts * scale + shift
                    all_rec.append(rec_pts)
                    all_ref.append(ref_pts)
                    all_inp.append(inp_pts)
                rec = torch.cat(all_rec, dim=0)
                ref = torch.cat(all_ref, dim=0)
                inp = torch.cat(all_inp, dim=0)
            else:
                shift = []
                scale = []
                for data in test_loader:
                    idx = data["cate_idx"] == val_cate
                    shift.append(data["shift"][idx].float().cuda())
                    scale.append(data["scale"][idx].float().cuda())
                    all_ref.append(data['te_points'][idx])
                    all_inp.append(data['tr_points'][idx])
                ref = torch.cat(all_ref, dim=0).to("cuda")
                bsize = self.cfg.data.test_batch_size
                for idx in tqdm(range(0, math.ceil(ref.shape[0] / bsize))):
                    pts = ref[idx * bsize:(idx + 1) * bsize]
                    cates = (torch.ones(pts.shape[0]) * val_cate).int().to("cuda")
                    rec_pts = self.model(ref[idx * bsize:(idx+1)*bsize], label=cates)['set']
                    all_rec.append(rec_pts)
                rec = torch.cat(all_rec, dim=0).to("cuda")
                ref = torch.cat(all_ref, dim=0).to("cuda")
                shift = torch.cat(shift, dim=0)
                scale = torch.cat(scale, dim=0)
                ref = ref * scale + shift
                rec = rec * scale + shift

            np.save(
                os.path.join(self.cfg.log.save_path,
                             'rec_ep%d.npy' % self.epoch),
                rec.detach().cpu().numpy()
            )
            gen_res = compute_all_metrics(
                rec, ref,
                batch_size=128
            )
        all_res = {
            ("val/gen/%s" % k):
                (v if isinstance(v, float) else v.item())
            for k, v in gen_res.items()}
        print("Validation Sample (unit) Epoch:%d " % self.epoch, gen_res)
        return all_res

    def resume(self, epoch=None, finetune=False, strict=False, load_optim=True):
        if epoch is None:
            path = os.path.join(self.cfg.log.save_path, 'training.csv')
            tsdf = pd.read_csv(os.path.join(path))
            epoch = tsdf["epoch"].values[-1]
        path = os.path.join(self.cfg.log.save_path, 'checkpt_{:}.pth'.format(epoch))
        checkpt = torch.load(path, map_location=lambda storage, loc: storage)
        if not finetune:
            self.model.load_state_dict(checkpt["state_dict"], strict=strict)
            if load_optim:
                if "optim_state_dict" in checkpt.keys():
                    self.optimizer.load_state_dict(checkpt["optim_state_dict"])
                    # Manually move optimizer state to device.
                    for state in self.optimizer.state.values():
                        for k, v in state.items():
                            if torch.is_tensor(v):
                                state[k] = v.to(self.device, non_blocking=True)
            self.epoch = checkpt["epoch"] + 1
            self.scheduler.base_lrs = [self.cfg.opt.lr]
            self.scheduler.step(self.epoch)
            self.itr = checkpt["itr"]
            self.time = checkpt["time"]


        else:
            self.model.load_state_dict(checkpt["state_dict"], strict=False)
        self.model.init()