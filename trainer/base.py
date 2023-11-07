import os
import random
import time

import numpy as np
import pandas as pd
import torch
from torch import optim
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from tqdm import tqdm

from evaluation import compute_all_metrics
from tools.log import logger


class BaseTrainer(object):

    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device
        self.logger = logger(cfg)
        self.itr = 0
        self.epoch = 1
        self.time = 0
        self.tmp = time.time()

    def updata_time(self):
        self.time = self.time + time.time() - self.tmp
        self.tmp = time.time()

    def warm_up(self, optimizer, itr):
        if itr < self.cfg.opt.warmup_iters:
            iter_frac = min(float(itr + 1) / max(self.cfg.opt.warmup_iters, 1), 1.0)
            lr = self.cfg.opt.lr * iter_frac
            for i, param_group in enumerate(optimizer.param_groups):
                param_group["lr"] = lr

    def epoch_end(self):
        if self.epoch % self.cfg.log.save_epoch_freq == 0:
            self.save()
        self.epoch += 1
        pass

    def write_log(self, message, mode="train"):
        self.logger.write(message, mode)

    def info(self, message):
        self.logger.info(message)

    def save(self):
        path = os.path.join(self.cfg.log.save_path, 'checkpt_{:}.pth'.format(self.epoch))
        torch.save({
            'cfg': self.cfg,
            'state_dict': self.model.state_dict(),
            'optim_state_dict': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'epoch': self.epoch,
            'itr': self.itr,
            "time": self.time
        }, path)