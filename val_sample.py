import os
import argparse
import logging
from datetime import datetime
import torch
import yaml

from evaluation.evaluation_metrics import compute_CD_metrics


def normalize_point_clouds(pc):
    B, N, _ = pc.shape
    centroid = torch.mean(pc, dim=1, keepdim=True)
    pc = pc - centroid
    furthest_distance = torch.amax(torch.sqrt(torch.sum(pc ** 2, dim=-1, keepdim=True)), dim=1, keepdim=True)
    pc = pc / furthest_distance
    return pc


torch.set_printoptions(precision=20)


def main(args, cfg):
    # cfg args
    from datasets.ShapeNet_55 import get_data_loaders
    from evaluation import compute_all_metrics
    import numpy as np
    from tqdm import tqdm
    cfg.data.cates = [args.dataset]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sample = np.load(os.path.join('test', args.sample_path, args.dataset, args.sample))
    smp = torch.from_numpy(sample).to(device)
    loaders = get_data_loaders(cfg.data, args)
    test_loader = loaders['test_loader']
    with torch.no_grad():
        all_ref, all_rec, all_smp, all_ref_denorm = [], [], [], []
        all_rec_gt, all_inp_denorm, all_inp = [], [], []
        tbar = tqdm(test_loader)

        for i, data in enumerate(tbar):
            ref_pts = data['te_points'].cuda()
            inp_pts = data['tr_points'].cuda()
            smp_pts = smp[:ref_pts.shape[0]]  # * s + m
            smp = smp[ref_pts.shape[0]:]
            all_inp.append(inp_pts)
            all_rec.append(smp_pts)
            all_ref.append(ref_pts)

        smp = torch.cat(all_rec, dim=0)
        ref = torch.cat(all_ref, dim=0)
        if args.norm:
            smp = normalize_point_clouds(smp)
            ref = normalize_point_clouds(ref)
        gen_res = compute_CD_metrics(
            smp, ref,
            batch_size=256
        )
    logging.basicConfig(filename="val.txt",
                        level=logging.INFO, filemode="a")
    logging.info(args.dataset + ":" + args.sample)
    all_res = {
        ("val/gen/%s" % k):
            (v if isinstance(v, float) else v.item())
        for k, v in gen_res.items()}
    for k, v in all_res.items():
        logging.info('[%s] %.8f' % (k, v))


def get_parser():
    parser = argparse.ArgumentParser('val samples')
    parser.add_argument('--sample', type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--sample_path", default="smp", type=str)
    parser.add_argument("--norm", default=False, type=eval, choices=[True, False])
    return parser.parse_args()


def get_config():
    from tools.io import dict2namespace
    path = os.path.join('test', "val_config.yaml")
    with open(path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = dict2namespace(config)
    return config


if __name__ == "__main__":
    args = get_parser()
    cfg = get_config()
    main(args, cfg)
