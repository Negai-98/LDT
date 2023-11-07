import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils import data
import random
import tqdm
import numbers
import math

# taken from https://github.com/optas/latent_3d_points/blob/
# 8e8f29f8124ed5fc59439e8551ba7ef7567c9a37/src/in_out.py
synsetid_to_cate = {
    '02691156': 'airplane', '02773838': 'bag', '02801938': 'basket',
    '02808440': 'bathtub', '02818832': 'bed', '02828884': 'bench',
    '02876657': 'bottle', '02880940': 'bowl', '02924116': 'bus',
    '02933112': 'cabinet', '02747177': 'can', '02942699': 'camera',
    '02954340': 'cap', '02958343': 'car', '03001627': 'chair',
    '03046257': 'clock', '03207941': 'dishwasher', '03211117': 'monitor',
    '04379243': 'table', '04401088': 'telephone', '02946921': 'tin_can',
    '04460130': 'tower', '04468005': 'train', '03085013': 'keyboard',
    '03261776': 'earphone', '03325088': 'faucet', '03337140': 'file',
    '03467517': 'guitar', '03513137': 'helmet', '03593526': 'jar',
    '03624134': 'knife', '03636649': 'lamp', '03642806': 'laptop',
    '03691459': 'speaker', '03710193': 'mailbox', '03759954': 'microphone',
    '03761084': 'microwave', '03790512': 'motorcycle', '03797390': 'mug',
    '03928116': 'piano', '03938244': 'pillow', '03948459': 'pistol',
    '03991062': 'pot', '04004475': 'printer', '04074963': 'remote_control',
    '04090263': 'rifle', '04099429': 'rocket', '04225987': 'skateboard',
    '04256520': 'sofa', '04330267': 'stove', '04530566': 'vessel',
    '04554684': 'washer', '02992529': 'cellphone',
    '02843684': 'birdhouse', '02871439': 'bookshelf',
    # '02858304': 'boat', no boat in our dataset, merged into vessels
    # '02834778': 'bicycle', not in our taxonomy
}
cate_to_synsetid = {v: k for k, v in synsetid_to_cate.items()}


def normalize_point_cloud(inputs, verbose=False):
    """
    input: pc [N, P, 3]
    output: pc, centroid, furthest_distance
    """
    # print("shape",input.shape)
    C = inputs.shape[-1]
    pc = inputs[:, :, :3]
    if C > 3:
        nor = inputs[:, :, 3:]

    centroid = np.mean(pc, axis=1, keepdims=True)
    pc = inputs[:, :, :3] - centroid
    furthest_distance = np.amax(
        np.sqrt(np.sum(pc ** 2, axis=-1, keepdims=True)), axis=1, keepdims=True)
    pc = pc / furthest_distance
    if C > 3:
        return np.concatenate([pc, nor], axis=-1)
    else:
        if verbose:
            return pc, [centroid, furthest_distance]
        else:
            return pc


class Uniform15KPC(Dataset):
    def __init__(self, root_dir, subdirs, tr_sample_size=10000,
                 te_sample_size=10000, split='train',
                 random_subsample=False, boundary=True,):
        self.root_dir = root_dir
        self.split = split
        self.in_tr_sample_size = tr_sample_size
        self.in_te_sample_size = te_sample_size
        self.subdirs = subdirs
        self.random_subsample = random_subsample
        self.input_dim = 3
        self.all_cate_mids = []
        self.cate_idx_lst = []
        self.all_points = []
        for cate_idx, subd in tqdm.tqdm(enumerate(self.subdirs), total=len(self.subdirs)):
            # NOTE: [subd] here is synset id
            sub_path = os.path.join(root_dir, subd, self.split)
            if not os.path.isdir(sub_path):
                print("Directory missing : %s" % sub_path)
                continue

            all_mids = []
            for x in os.listdir(sub_path):
                if not x.endswith('.npy'):
                    continue
                all_mids.append(os.path.join(self.split, x[:-len('.npy')]))

            # NOTE: [mid] contains the split: i.e. "train/<mid>"
            # or "val/<mid>" or "test/<mid>"
            for mid in all_mids:
                # obj_fname = os.path.join(sub_path, x)
                obj_fname = os.path.join(root_dir, subd, mid + ".npy")
                try:
                    point_cloud = np.load(obj_fname)  # (15k, 3)
                except:  # nofa: E722
                    continue

                assert point_cloud.shape[0] == 15000
                self.all_points.append(point_cloud[np.newaxis, ...])
                self.cate_idx_lst.append(cate_idx)
                self.all_cate_mids.append((subd, mid))

        # Shuffle the index deterministically (based on the number of examples)
        self.shuffle_idx = list(range(len(self.all_points)))
        random.Random(38383).shuffle(self.shuffle_idx)
        self.cate_idx_lst = [self.cate_idx_lst[i] for i in self.shuffle_idx]
        self.all_points = [self.all_points[i] for i in self.shuffle_idx]
        self.all_cate_mids = [self.all_cate_mids[i] for i in self.shuffle_idx]

        # Normalization
        self.all_points = np.concatenate(self.all_points)  # (N, 15000, 3)
        if boundary:
            self.all_points, [self.per_points_shift, self.per_points_scale] = normalize_point_cloud(self.all_points,
                                                                                                    verbose=True)
        else:
            self.per_points_shift = np.zeros((self.all_points.shape[0], 1, 3))
            self.per_points_scale = np.ones((self.all_points.shape[0], 1, 3))

        self.train_points = self.all_points[:, :10000]
        self.test_points = self.all_points[:, 10000:]
        self.tr_sample_size = min(10000, tr_sample_size)
        self.te_sample_size = min(5000, te_sample_size)
        print("Total number of data:%d" % len(self.train_points))
        print("Min number of points: (train)%d (test)%d"
              % (self.tr_sample_size, self.te_sample_size))
        # Default display axis order
        self.display_axis_order = [0, 1, 2]

    def get_standardize_stats(self, idx):
        shift = self.per_points_shift[idx].reshape(1, self.input_dim)
        scale = self.per_points_scale[idx].reshape(1, -1)
        return shift, scale

    def renormalize(self, mean, std):
        self.all_points = self.all_points * self.all_points_std + \
                          self.all_points_mean
        self.all_points_mean = mean
        self.all_points_std = std
        self.all_points = (self.all_points - self.all_points_mean) / \
                          self.all_points_std
        self.train_points = self.all_points[:, :10000]
        self.test_points = self.all_points[:, 10000:]

    def __len__(self):
        return len(self.train_points)

    def __getitem__(self, idx):
        tr_out = self.all_points[idx]
        if self.random_subsample:
            tr_idxs = np.random.choice(tr_out.shape[0], self.tr_sample_size)
        else:
            tr_idxs = np.arange(self.tr_sample_size)
        tr_out = torch.from_numpy(tr_out[tr_idxs, :]).float()
        te_out = self.test_points[idx]

        if self.random_subsample:
            te_idxs = np.random.choice(te_out.shape[0], self.te_sample_size)
        else:
            te_idxs = np.arange(self.te_sample_size)
        te_out = torch.from_numpy(te_out[te_idxs, :]).float()
        cate_idx = self.cate_idx_lst[idx]
        sid, mid = self.all_cate_mids[idx]
        shift, scale = self.get_standardize_stats(idx)
        return {
            'idx': idx,
            'tr_points': tr_out,
            'te_points': te_out,
            'cate_idx': cate_idx,
            'sid': sid, 'mid': mid,
            'shift': shift, 'scale': scale,
            'display_axis_order': self.display_axis_order
        }


class ShapeNet15kPointClouds(Uniform15KPC):
    def __init__(self, root_dir="data/ShapeNetCore.v2.PC15k",
                 categories=['airplane'], tr_sample_size=10000,
                 te_sample_size=2048,
                 split='train', random_subsample=False, boundary=True):
        self.root_dir = root_dir
        self.split = split
        assert self.split in ['train', 'test', 'val']
        self.tr_sample_size = tr_sample_size
        self.te_sample_size = te_sample_size
        self.cates = categories
        if 'all' in categories:
            self.synset_ids = list(cate_to_synsetid.values())
        else:
            self.synset_ids = [cate_to_synsetid[c] for c in self.cates]

        # assert 'v2' in root_dir, "Only supporting v2 right now."
        self.gravity_axis = 1
        self.display_axis_order = [0, 2, 1]
        super(ShapeNet15kPointClouds, self).__init__(
            root_dir, self.synset_ids,
            tr_sample_size=tr_sample_size,
            te_sample_size=te_sample_size,
            split=split,
            random_subsample=random_subsample,
            boundary=boundary)


def get_datasets(cfg, args):
    tr_dataset = ShapeNet15kPointClouds(
        categories=cfg.cates, split='train',
        tr_sample_size=cfg.tr_max_sample_points,
        te_sample_size=cfg.te_max_sample_points,
        root_dir=cfg.data_dir,
        random_subsample=True, boundary=cfg.boundary)

    eval_split = getattr(args, "eval_split", "val")
    te_dataset = ShapeNet15kPointClouds(
        categories=cfg.cates, split=eval_split,
        tr_sample_size=cfg.tr_max_sample_points,
        te_sample_size=cfg.te_max_sample_points,
        root_dir=cfg.data_dir, boundary=cfg.boundary
    )
    return tr_dataset, te_dataset


def get_data_loaders(cfg, args):
    tr_dataset, te_dataset = get_datasets(cfg, args)
    train_loader = data.DataLoader(
        dataset=tr_dataset, batch_size=cfg.batch_size,
        shuffle=True, num_workers=cfg.num_workers, drop_last=True,
        pin_memory=True
    )
    test_loader = data.DataLoader(
        dataset=te_dataset, batch_size=cfg.test_batch_size,
        shuffle=False, num_workers=cfg.num_workers, drop_last=False
    )

    loaders = {
        "test_loader": test_loader,
        'train_loader': train_loader,
    }
    return loaders
