import argparse
import os
import torch
import yaml
from pointnet2_ops import pointnet2_utils
from tqdm import tqdm
from datasets.ViPC import get_data_loaders
from model.Compressor.Network import Compressor
from model.Compressor.layers import index_points
from tools.io import dict2namespace
from tools.utils import AverageMeter, common_init
from completion_trainer.Compressor_Trainer import Trainer


def main(args, cfg):
    # cfg args
    common_init(cfg.common.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # create data
    loaders = get_data_loaders(cfg.data)
    train_loader = loaders['train_loader']
    test_loader = loaders['test_loader']
    # model
    model = Compressor(cfg.model)
    trainer = Trainer(cfg, model, device)

    path = os.path.join(args.save, args.trainer_type, 'completion', args.dataset, "config.yaml")
    with open(path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    trainer.info(config)
    # resume
    if args.resume:
        trainer.resume(epoch=args.resume_epoch, finetune=args.finetune, strict=args.strict,
                       load_optim=args.load_optimizer)
        trainer.optimizer.defaults['lr'] = cfg.opt.lr
    else:
        trainer.load_pretrain()
    # record
    loss_meter = AverageMeter()
    kl_loss_meter = AverageMeter()
    rec_meter = AverageMeter()
    max_meter = AverageMeter()
    # train
    if not args.evaluate:
        while trainer.epoch < cfg.common.epochs:
            for epoch in range(trainer.epoch, cfg.common.epochs + 1):
                if trainer.itr > cfg.opt.warmup_iters:
                    trainer.scheduler.step(trainer.epoch)
                tbar = tqdm(train_loader, ncols=160)
                tbar.set_description("Epoch {}".format(epoch))
                for data in tbar:
                    views, pc, pc_part = data
                    pc, pc_part = pc.to('cuda'), pc_part.to('cuda')
                    pc_center, pc_part_center = pointnet2_utils.furthest_point_sample(pc, 2048).long(), \
                                                pointnet2_utils.furthest_point_sample(pc_part, 2048).long()
                    pc, pc_part = index_points(pc, pc_center), index_points(pc, pc_part_center)
                    views = views.float()
                    condition = (views, pc_part)
                    loss, kl_loss, rec_loss, max_feature = trainer.update(pc)
                    loss_meter.update(loss)
                    kl_loss_meter.update(kl_loss)
                    rec_meter.update(rec_loss)
                    max_meter.update(max_feature)
                    tbar.set_postfix(
                        {'loss': '{0:1.5f}({1:1.5f})'.format(loss_meter.val, loss_meter.avg),
                         'kl': '{0:1.5f}({1:1.5f})'.format(kl_loss_meter.val, kl_loss_meter.avg),
                         'rec': '{0:1.5f}({1:1.5f})'.format(rec_meter.val, rec_meter.avg),
                         'max': '{0:1.5f}({1:1.5f})'.format(max_meter.val, max_meter.avg)
                         }
                    )
                    if torch.isnan(loss_meter.avg) or torch.isinf(loss_meter.avg) or max_meter.avg > 10000:
                        break
                if trainer.epoch % cfg.log.log_epoch_freq == 0:
                    trainer.updata_time()
                    message = [epoch, trainer.itr, loss_meter.avg, kl_loss_meter.avg, rec_meter.avg, max_meter.avg, trainer.time]
                    trainer.write_log(message=message, mode="train")
                trainer.epoch_end()

                if (trainer.epoch - 1) % cfg.log.eval_epoch_freq == 0:
                    all_res = trainer.reconstrustion(test_loader=test_loader)
                    message = [trainer.epoch - 1] + list(all_res.values())
                    trainer.info("epoch{:}:".format(trainer.epoch - 1) + str(all_res))
                    try:
                        trainer.write_log(message=message, mode="eval")
                    except:
                        print("write log failed")
                    # trainer.info("epoch{:}:".format(trainer.epoch - 1) + all_res)
                    # all_res = trainer.valsample(test_loader=test_loader, sample_points=2048)
                    # message = [trainer.epoch-1] + list(all_res.values())
                    # trainer.write_log(message=message, mode="eval")
                if torch.isnan(loss_meter.avg) or torch.isinf(loss_meter.avg) or max_meter.avg > 10000:
                    trainer.resume(epoch=(trainer.epoch - 10) // 10 * 10, finetune=False, strict=True,
                                   load_optim=True)
                    trainer.optimizer.defaults['lr'] = cfg.opt.lr / 2
                    loss_meter.reset()
                    kl_loss_meter.reset()
                    rec_meter.reset()
                    max_meter.reset()
                    break
                loss_meter.reset()
                kl_loss_meter.reset()
                rec_meter.reset()
                max_meter.reset()

    else:
        # all_res = trainer.valsample(test_loader=test_loader, sample_points=2048)
        all_res = trainer.reconstrustion(test_loader=test_loader)
        message = [trainer.epoch - 1] + list(all_res.values())
        trainer.write_log(message=message, mode="eval")
        # trainer.info("epoch{:}:".format(trainer.epoch - 1) + all_res)


def get_parser():
    parser = argparse.ArgumentParser('SDE Point Cloud')
    # model
    parser.add_argument('--trainer_type', type=str, default="Compressor_Trainer")
    # distributed
    parser.add_argument('--gpu', default=0, type=int,
                        help='GPU id to use. None means using all '
                             'available GPUs.')

    # common
    parser.add_argument("--dataset", default='sofa', type=str)
    parser.add_argument('--save', type=str, default=os.path.join('experiments'))
    parser.add_argument('--resume', type=eval, default=False, choices=[True, False])
    parser.add_argument('--resume_epoch', type=int, default=17)
    parser.add_argument('--load_optimizer', type=eval, default=True, choices=[True, False])
    parser.add_argument('--evaluate', type=eval, default=False, choices=[True, False])
    parser.add_argument('--strict', type=eval, default=True, choices=[True, False])
    parser.add_argument('--finetune', type=eval, default=False, choices=[True, False])
    return parser.parse_args()


def get_config(args):
    path = os.path.join(args.save, args.trainer_type, 'completion', args.dataset, "config.yaml")
    with open(path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = dict2namespace(config)
    return config


if __name__ == "__main__":
    args = get_parser()
    cfg = get_config(args)
    main(args, cfg)
