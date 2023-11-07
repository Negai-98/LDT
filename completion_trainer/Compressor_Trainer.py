import importlib
import math
import os
import random
import time
import numpy as np
import pandas as pd
import torch
from pointnet2_ops import pointnet2_utils
from torch import optim
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from tqdm import tqdm

from completion_trainer.Latent_SDE_Trainer import L2_ChamferEval_1000, F1Score
from model.Compressor.layers import index_points
try:
    from evaluation.loss import EMD_loss, CD_loss
except:
    pass
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
        self.model.train()
        self.warm_up(self.optimizer, self.itr)
        loss, kl_loss, rec_loss, max_feature = self.compute_loss(data)
        self.optimizer.zero_grad()
        loss.backward()
        if self.cfg.opt.grad_norm_clip_value is not None:
            clip_grad_norm_(self.model.parameters(), self.cfg.opt.grad_norm_clip_value)
        self.optimizer.step()
        self.itr += 1
        return loss, kl_loss, rec_loss, max_feature

    def compute_loss(self, target_set):
        output = self.model(target_set)
        output_set, kls, max_feature = output['set'], output['kls'], output['max']
        emd_loss = EMD_loss(output_set, target_set)
        cd_loss = CD_loss(output_set, target_set)
        rec_loss = cd_loss + emd_loss
        kl_loss = torch.cat(kls, dim=1)
        kl_loss = kl_loss.mean()
        loss = self.kl_weight * kl_loss + rec_loss
        return loss, kl_loss, rec_loss, max_feature

    def sample(self, num_samples, num_points, given_eps=None):
        shape = (num_samples, num_points)
        self.model.eval()
        with torch.no_grad():
            sample = self.model.sample(shape, given_eps=given_eps)
        return sample

    def reconstrustion(self, test_loader):
        with torch.no_grad():
            self.model.eval()
            all_ref, all_rec, all_inp, all_smp, all_ref_denorm = [], [], [], [], []
            count = 0
            tbar = tqdm(test_loader)
            for i, data in enumerate(tbar):
                views, pc, pc_part = data
                pc = pc.to('cuda')
                pc_center = pointnet2_utils.furthest_point_sample(pc, 2048).long()
                ref_pts = index_points(pc, pc_center)
                output = self.model(ref_pts)
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
        print("Validation Sample (unit) Epoch:%d " % self.epoch, all_res)
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

    def load_pretrain(self):
        path = os.path.join(self.cfg.model.pretrain_path)
        checkpt = torch.load(path)
        self.model.load_state_dict(checkpt["state_dict"], strict=True)
        self.model.init()
