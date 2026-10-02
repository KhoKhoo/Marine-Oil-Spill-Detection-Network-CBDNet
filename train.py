import argparse
import copy
import csv
import datetime
import json
import os
import random
import shutil
import sys
from time import time

import numpy as np
import psutil
import torch

from framework import MyFrame
from dice_bce_loss import dice_bce_loss
from data import ImageFolder
from models_factory import build_model

SHAPE = (256, 256)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=['resnet34', 'effb0', 'effb3'], default='resnet34')
    p.add_argument('--epochs', type=int, default=150)
    p.add_argument('--data_root', type=str, required=True,
                    help='train/holdout images are read from <data_root>/train and <data_root>/holdout')
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
    p.add_argument('--num_workers', type=int, default=4)
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def list_ids(root):
    imagelist = filter(lambda x: x.find('sat') != -1, os.listdir(root))
    return list(map(lambda x: x[:x.find('_sat')], imagelist))


def safe_div(n, d):
    return n / d if d > 0 else 0.0


def get_memory_stats():
    process = psutil.Process(os.getpid())

    ram_gb = process.memory_info().rss / (1024 ** 3)

    gpu_peak_gb = None
    if torch.cuda.is_available():
        gpu_peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

    shm_used_mb = None
    if os.path.isdir('/dev/shm'):
        shm_used_mb = shutil.disk_usage('/dev/shm').used / (1024 ** 2)

    num_fds = None
    if os.path.isdir('/proc'):
        num_fds = len(os.listdir('/proc/self/fd'))

    num_children = len(process.children(recursive=True))

    return ram_gb, gpu_peak_gb, shm_used_mb, num_fds, num_children


def evaluate_holdout(solver, holdout_loader):
    solver.net.eval()
    tp = fp = fn = tn = 0
    with torch.no_grad():
        for img, mask in holdout_loader:
            img = img.to(solver.device)
            mask = mask.to(solver.device)
            pred = solver.net(img)
            pred_bin = (pred > 0.5)
            mask_bin = (mask > 0.5)
            tp += (pred_bin & mask_bin).sum().item()
            fp += (pred_bin & ~mask_bin).sum().item()
            fn += ((~pred_bin) & mask_bin).sum().item()
            tn += ((~pred_bin) & (~mask_bin)).sum().item()
    solver.net.train()

    iou_fg = safe_div(tp, tp + fp + fn)
    iou_bg = safe_div(tn, tn + fp + fn)
    miou = (iou_fg + iou_bg) / 2
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)

    return {
        'miou': miou,
        'iou_fg': iou_fg,
        'iou_bg': iou_bg,
        'f1': f1,
        'precision': precision,
        'recall': recall,
    }


def main():
    args = parse_args()
    set_seed(args.seed)

    train_root = os.path.join(args.data_root, 'train') + os.sep
    holdout_root = os.path.join(args.data_root, 'holdout') + os.sep
    run_dir = os.path.join(args.out_dir, args.name)
    os.makedirs(run_dir, exist_ok=True)

    last_ckpt_path = os.path.join(run_dir, 'last.th')
    best_ckpt_path = os.path.join(run_dir, 'best.th')
    best_json_path = os.path.join(run_dir, 'best.json')
    log_path = os.path.join(run_dir, 'train.log')
    config_path = os.path.join(run_dir, 'config.json')
    metrics_path = os.path.join(run_dir, 'metrics.csv')
    status_path = os.path.join(run_dir, 'status.json')

    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)

    trainlist = list_ids(train_root)

    if not os.path.isdir(holdout_root) or not list_ids(holdout_root):
        sys.exit(f'Error: holdout set not found or empty at {holdout_root}. '
                 'Use prepare_data.py to create a <data_root>/holdout split before training.')
    holdoutlist = list_ids(holdout_root)

    model_fn = build_model(args.model, pretrained=not args.no_pretrained)
    solver = MyFrame(model_fn, dice_bce_loss, args.lr)

    batchsize = max(torch.cuda.device_count(), 1) * args.batch_size

    dataset = ImageFolder(trainlist, train_root)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batchsize,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0)

    holdout_dataset = ImageFolder(holdoutlist, holdout_root, augment=False)
    holdout_loader = torch.utils.data.DataLoader(
        holdout_dataset,
        batch_size=batchsize,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0)

    start_epoch = 0
    no_optim = 0
    train_epoch_best_loss = 100.
    best_state_dict = None
    best_holdout_miou = -1.0

    if args.resume and os.path.exists(last_ckpt_path):
        start_epoch, no_optim, train_epoch_best_loss, best_state_dict = solver.load_checkpoint(last_ckpt_path)
        print(f'Resuming from {last_ckpt_path} at epoch {start_epoch}', flush=True)
    else:
        print('Starting fresh training run', flush=True)

    if args.resume and os.path.exists(best_json_path):
        with open(best_json_path, 'r') as f:
            best_holdout_miou = json.load(f).get('miou', -1.0)

    mylog = open(log_path, 'a' if start_epoch > 0 else 'w')
    tic = time()
    last_epoch = start_epoch

    write_header = not (start_epoch > 0 and os.path.exists(metrics_path))
    metrics_file = open(metrics_path, 'a' if start_epoch > 0 else 'w', newline='')
    metrics_writer = csv.writer(metrics_file)
    if write_header:
        metrics_writer.writerow(['epoch', 'train_loss', 'holdout_miou', 'holdout_iou_fg',
                                  'holdout_f1', 'holdout_precision', 'holdout_recall',
                                  'lr', 'epoch_seconds'])
        metrics_file.flush()

    for epoch in range(start_epoch + 1, args.epochs + 1):
        epoch_tic = time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        data_loader_iter = iter(data_loader)
        train_epoch_loss = 0
        for img, mask in data_loader_iter:
            solver.set_input(img, mask)
            train_loss = solver.optimize()
            train_epoch_loss += train_loss
        train_epoch_loss /= len(data_loader_iter)

        holdout_metrics = evaluate_holdout(solver, holdout_loader)
        epoch_seconds = time() - epoch_tic

        ram_gb, gpu_peak_gb, shm_used_mb, num_fds, num_children = get_memory_stats()
        mem_str = f'ram:{ram_gb:.2f}GB'
        if gpu_peak_gb is not None:
            mem_str += f', gpu_peak:{gpu_peak_gb:.2f}GB'
        if shm_used_mb is not None:
            mem_str += f', shm:{shm_used_mb:.2f}MB'
        if num_fds is not None:
            mem_str += f', fds:{num_fds}'
        mem_str += f', children:{num_children}'

        msg = (f'epoch:{epoch}, time:{int(time()-tic)}, train_loss:{train_epoch_loss}, '
               f'holdout_miou:{holdout_metrics["miou"]:.4f}, {mem_str}')
        print('--------', flush=True)
        print(msg, flush=True)
        print(msg, file=mylog, flush=True)

        with open(status_path, 'w') as f:
            json.dump({
                'epoch': epoch,
                'timestamp': datetime.datetime.now().isoformat(),
                'ram_gb': ram_gb,
                'gpu_peak_gb': gpu_peak_gb,
                'shm_used_mb': shm_used_mb,
                'num_fds': num_fds,
                'num_children': num_children,
            }, f, indent=2)

        metrics_writer.writerow([
            epoch, train_epoch_loss, holdout_metrics['miou'], holdout_metrics['iou_fg'],
            holdout_metrics['f1'], holdout_metrics['precision'], holdout_metrics['recall'],
            solver.old_lr, epoch_seconds,
        ])
        metrics_file.flush()

        if holdout_metrics['miou'] > best_holdout_miou:
            best_holdout_miou = holdout_metrics['miou']
            torch.save(solver.net.state_dict(), best_ckpt_path)
            with open(best_json_path, 'w') as f:
                json.dump({'epoch': epoch, 'miou': best_holdout_miou}, f, indent=2)
            best_msg = f'New best holdout mIoU {best_holdout_miou:.4f} at epoch {epoch} -> saved {best_ckpt_path}'
            print(best_msg, flush=True)
            print(best_msg, file=mylog, flush=True)

        if train_epoch_loss >= train_epoch_best_loss:
            no_optim += 1
        else:
            no_optim = 0
            train_epoch_best_loss = train_epoch_loss
            best_state_dict = copy.deepcopy(solver.net.state_dict())

        last_epoch = epoch
        solver.save_checkpoint(last_ckpt_path, epoch=epoch, no_optim=no_optim,
                                best_loss=train_epoch_best_loss, best_state_dict=best_state_dict)

        if args.early_stop and no_optim > 6:
            stop_msg = 'early stop at %d epoch' % epoch
            print(stop_msg, flush=True)
            print(stop_msg, file=mylog, flush=True)
            break

        if no_optim > 3:
            if args.early_stop and solver.old_lr < 5e-7:
                stop_msg = 'stopping: lr below 5e-7 at epoch %d' % epoch
                print(stop_msg, flush=True)
                print(stop_msg, file=mylog, flush=True)
                break
            if best_state_dict is not None:
                solver.net.load_state_dict(best_state_dict)
            solver.update_lr(5.0, factor=True, mylog=mylog)

        mylog.flush()

    print('Finish!', file=mylog, flush=True)
    mylog.close()
    metrics_file.close()

    if last_epoch >= args.epochs:
        print(f'Training complete: reached final epoch {last_epoch}/{args.epochs}.', flush=True)
    else:
        print(f'Stopped after epoch {last_epoch} (target was epoch {args.epochs}).', flush=True)
        print(f'Run again with --resume to continue training from epoch {last_epoch}.', flush=True)


if __name__ == '__main__':
    main()
