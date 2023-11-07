import argparse
import os
import torch
import yaml
from tqdm import tqdm
from datasets.ShapeNet_55 import get_data_loaders
from model.Compressor.Network import Compressor
from tools.io import dict2namespace
from tools.utils import AverageMeter, common_init
# from trainer.Latent_Diffusion_Trainer import Trainer
from trainer.Hybrid_Trainer import Trainer
from model.scorenet.score import Score

def main(args, cfg):
    common_init(cfg.common.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # create data
    loaders = get_data_loaders(cfg.data, args)
    train_loader = loaders['train_loader']
    test_loader = loaders['test_loader']
    # model
    model = Score(cfg.score)
    compressor = Compressor(cfg.compressor)
    # print(count_parameters(model))
    # print(count_parameters(compressor))
    # create trainer
    trainer = Trainer(cfg, model=model, compressor=compressor, device=device)
    # resume
    if args.resume:
        trainer.resume(epoch=args.resume_epoch, strict=args.strict,
                       load_optim=args.load_optimizer, finetune=args.finetune)
    else:
        trainer.load_pretrain()
    loss_score_meter = AverageMeter()
    kl_meter = AverageMeter()
    rec_meter = AverageMeter()

    # train
    if not args.evaluate:
        for epoch in range(trainer.epoch, cfg.common.epochs + 1):
            if trainer.itr > cfg.opt.warmup_iters:
                trainer.scheduler.step(trainer.epoch)
            tbar = tqdm(train_loader, ncols=160)
            tbar.set_description("Epoch {}".format(epoch))
            for data in tbar:
                loss_score, kl, rec = trainer.update(data)
                loss_score_meter.update(loss_score.detach())
                kl_meter.update(kl.detach())
                rec_meter.update(rec.detach())
                tbar.set_postfix(
                    {'loss_score': '{0:1.5f}({1:1.5f})'.format(loss_score_meter.val, loss_score_meter.avg),
                     'kl': '{0:1.5f}({1:1.5f})'.format(kl_meter.val, kl_meter.avg),
                     'rec': '{0:1.5f}({1:1.5f})'.format(rec_meter.val, rec_meter.avg)
                     }
                )
                # log
            trainer.epoch_end()
            if (trainer.epoch - 1) % cfg.log.log_epoch_freq == 0:
                trainer.updata_time()
                message = [epoch, trainer.itr, loss_score_meter.avg, kl_meter.avg, rec_meter.avg, trainer.time]
                trainer.write_log(message=message, mode="train")
                loss_score_meter.reset()
                kl_meter.reset()
                rec_meter = AverageMeter()

            if (trainer.epoch - 1) % cfg.log.eval_epoch_freq == 0:
                all_res = trainer.valsample(test_loader=test_loader)
                message = [trainer.epoch - 1] + list(all_res.values())
                trainer.info("epoch{:}:".format(trainer.epoch - 1) + str(all_res))
                try:
                    trainer.write_log(message=message, mode="eval")
                except:
                    print("write log failed")
    else:
        # print(torch.randn(16, ))
        all_res = trainer.valsample(test_loader=test_loader)
        # tbar = tqdm(test_loader, ncols=160)
        # all_res = trainer.reconstrustion(test_loader=test_loader)
        message = [trainer.epoch - 1] + list(all_res.values())
        trainer.write_log(message=message, mode="eval")


def get_parser():
    parser = argparse.ArgumentParser('Latent Shape Diffusion')
    # dataset
    parser.add_argument("--dataset", default='airplane', type=str)

    # model
    parser.add_argument('--trainer_type', type=str, default="Hybrid_Trainer")
    # distributed
    parser.add_argument('--gpu', default=0, type=int,
                        help='GPU id to use. None means using all '
                             'available GPUs.')
    # common
    parser.add_argument('--save', type=str, default=os.path.join('experiments'))
    parser.add_argument('--resume', type=eval, default=False, choices=[True, False])
    parser.add_argument('--resume_epoch', type=int, default=None)
    parser.add_argument('--evaluate', type=eval, default=False, choices=[True, False])
    parser.add_argument('--strict', type=eval, default=True, choices=[True, False])
    parser.add_argument('--finetune', type=eval, default=False, choices=[True, False])
    parser.add_argument('--load_optimizer', type=eval, default=True, choices=[True, False])

    return parser.parse_args()


def get_config(args):
    path = os.path.join(args.save, args.trainer_type, args.dataset, "config.yaml")
    with open(path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = dict2namespace(config)
    return config


if __name__ == "__main__":
    args = get_parser()
    cfg = get_config(args)
    # for NFE in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
    # cfg.sde.sample_N = NFE
    main(args, cfg)
