import argparse
import copy
import functools
import json
import os
import random
from time import time

import numpy as np
import torch

from networks.CBDNet import CBDNet, CBDNet_EfficientNet
from framework import MyFrame
from dice_bce_loss import dice_bce_loss
from data import ImageFolder

SHAPE = (256, 256)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=['resnet34', 'effb0', 'effb3'], default='resnet34')
    p.add_argument('--epochs', type=int, default=150)
    p.add_argument('--data_root', type=str, required=True,
                    help='train images are read from <data_root>/train')
    p.add_argument('--name', type=str, required=True)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out_dir', type=str, required=True)
    p.add_argument('--resume', action='store_true',
                    help='continue from <out_dir>/<name>/last.th if it exists')
    p.add_argument('--early_stop', action='store_true',
                    help='enable the no-improvement-for-6-epochs and lr<5e-7 stop rules')
    p.add_argument('--no_pretrained', action='store_true',
                    help='skip loading pretrained backbone weights (fast local testing only)')
    return p.parse_args()


def build_model_fn(model_name, pretrained):
    if model_name == 'resnet34':
        return CBDNet
    variant = 'b0' if model_name == 'effb0' else 'b3'
    return functools.partial(CBDNet_EfficientNet, variant=variant, pretrained=pretrained)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    set_seed(args.seed)

    train_root = os.path.join(args.data_root, 'train') + os.sep
    run_dir = os.path.join(args.out_dir, args.name)
    os.makedirs(run_dir, exist_ok=True)

    last_ckpt_path = os.path.join(run_dir, 'last.th')
    log_path = os.path.join(run_dir, 'train.log')
    config_path = os.path.join(run_dir, 'config.json')

    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)

    imagelist = filter(lambda x: x.find('sat') != -1, os.listdir(train_root))
    trainlist = list(map(lambda x: x[:x.find('_sat')], imagelist))

    model_fn = build_model_fn(args.model, pretrained=not args.no_pretrained)
    solver = MyFrame(model_fn, dice_bce_loss, args.lr)

    batchsize = max(torch.cuda.device_count(), 1) * args.batch_size

    dataset = ImageFolder(trainlist, train_root)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batchsize,
        shuffle=False,
        num_workers=4)

    start_epoch = 0
    no_optim = 0
    train_epoch_best_loss = 100.
    best_state_dict = None

    if args.resume and os.path.exists(last_ckpt_path):
        start_epoch, no_optim, train_epoch_best_loss = solver.load_checkpoint(last_ckpt_path)
        print(f'Resuming from {last_ckpt_path} at epoch {start_epoch}')
    else:
        print('Starting fresh training run')

    mylog = open(log_path, 'a' if start_epoch > 0 else 'w')
    tic = time()
    last_epoch = start_epoch

    for epoch in range(start_epoch + 1, args.epochs + 1):
        data_loader_iter = iter(data_loader)
        train_epoch_loss = 0
        for img, mask in data_loader_iter:
            solver.set_input(img, mask)
            train_loss = solver.optimize()
            train_epoch_loss += train_loss
        train_epoch_loss /= len(data_loader_iter)

        msg = f'epoch:{epoch}, time:{int(time()-tic)}, train_loss:{train_epoch_loss}'
        print('--------')
        print(msg)
        print(msg, file=mylog)

        if train_epoch_loss >= train_epoch_best_loss:
            no_optim += 1
        else:
            no_optim = 0
            train_epoch_best_loss = train_epoch_loss
            best_state_dict = copy.deepcopy(solver.net.state_dict())

        last_epoch = epoch
        solver.save_checkpoint(last_ckpt_path, epoch=epoch, no_optim=no_optim,
                                best_loss=train_epoch_best_loss)

        if args.early_stop and no_optim > 6:
            stop_msg = 'early stop at %d epoch' % epoch
            print(stop_msg)
            print(stop_msg, file=mylog)
            break

        if no_optim > 3:
            if args.early_stop and solver.old_lr < 5e-7:
                stop_msg = 'stopping: lr below 5e-7 at epoch %d' % epoch
                print(stop_msg)
                print(stop_msg, file=mylog)
                break
            if best_state_dict is not None:
                solver.net.load_state_dict(best_state_dict)
            solver.update_lr(5.0, factor=True, mylog=mylog)

        mylog.flush()

    print('Finish!', file=mylog)
    mylog.close()

    if last_epoch >= args.epochs:
        print(f'Training complete: reached final epoch {last_epoch}/{args.epochs}.')
    else:
        print(f'Stopped after epoch {last_epoch} (target was epoch {args.epochs}).')
        print(f'Run again with --resume to continue training from epoch {last_epoch}.')


if __name__ == '__main__':
    main()
