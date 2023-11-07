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
from tools.utils import AverageMeter, common_init, count_parameters
from completion_trainer.Latent_SDE_Trainer import Trainer
from model.scorenet.score import Score

def main(args, cfg):
    common_init(cfg.common.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # create data
    model = Score(cfg.score)
    compressor = Compressor(cfg.compressor)
    # print(count_parameters(model))
    # 457012344
    # print(count_parameters(compressor))
    # 8059001
    loaders = get_data_loaders(cfg.data)
    train_loader = loaders['train_loader']
    test_loader = loaders['test_loader']
    # model
    # create trainer
    trainer = Trainer(cfg, model=model, compressor=compressor, device=device)

    # resume
    if args.resume:
        trainer.resume(epoch=args.resume_epoch, strict=args.strict,
                       load_optim=args.load_optimizer, finetune=args.finetune)
        trainer.optimizer.defaults['lr'] = cfg.opt.lr
        trainer.itr = 0
    else:
        trainer.load_pretrain()
    loss_meter = AverageMeter()
    # train
    if not args.evaluate:
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
                pc, pc_part = index_points(pc, pc_center), index_points(pc_part, pc_part_center)
                views = views.float()
                condition = {'img': views, 'pts': pc_part}
                loss = trainer.update(pc, condition)
                loss_meter.update(loss.detach())
                tbar.set_postfix(
                    {'loss_score': '{0:1.5f}({1:1.5f})'.format(loss_meter.val, loss_meter.avg)}
                )
                # log
            trainer.epoch_end()
            if (trainer.epoch - 1) % cfg.log.log_epoch_freq == 0:
                trainer.updata_time()
                message = [epoch, trainer.itr, loss_meter.avg, trainer.time]
                trainer.write_log(message=message, mode="train")
                loss_meter.reset()

            if (trainer.epoch - 1) % cfg.log.eval_epoch_freq == 0:
                all_res = trainer.valsample(test_loader=test_loader)
                message = [trainer.epoch - 1] + list(all_res.values())
                trainer.info("epoch{:}:".format(trainer.epoch - 1) + str(all_res))
                try:
                    trainer.write_log(message=message, mode="eval")
                except:
                    print("write log failed")
    else:
        # tbar = tqdm(test_loader, ncols=160)
        # for data in tbar:
        #     loss = trainer.val_loss(data)
        #     loss_meter.update(loss.detach())
        #     tbar.set_postfix(
        #         {'loss_score': '{0:1.5f}({1:1.5f})'.format(loss_meter.val, loss_meter.avg)}
        #     )
        # trainer.updata_time()
        # message = [trainer.epoch, trainer.itr, loss_meter.avg, trainer.time]
        # trainer.write_log(message=message, mode="test")

        all_res = trainer.valsample(test_loader=test_loader, full=True)
        # all_res = trainer.reconstrustion(test_loader=test_loader)
        message = [trainer.epoch - 1] + list(all_res.values())
        trainer.write_log(message=message, mode="eval")


def get_parser():
    parser = argparse.ArgumentParser('Latent Shape Diffusion')
    # dataset
    parser.add_argument("--dataset", default='table', type=str)
    # model
    parser.add_argument('--trainer_type', type=str, default="Latent_Diffusion_Trainer")
    # distributed
    parser.add_argument('--gpu', default=0, type=int,
                        help='GPU id to use. None means using all '
                             'available GPUs.')
    # common
    parser.add_argument('--save', type=str, default=os.path.join('experiments'))
    parser.add_argument('--resume', type=eval, default=True, choices=[True, False])
    parser.add_argument('--resume_epoch', type=int, default=428)
    parser.add_argument('--evaluate', type=eval, default=False, choices=[True, False])
    parser.add_argument('--strict', type=eval, default=True, choices=[True, False])
    parser.add_argument('--finetune', type=eval, default=False, choices=[True, False])
    parser.add_argument('--load_optimizer', type=eval, default=False, choices=[True, False])

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
