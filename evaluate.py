import argparse
import json
import os
import random
import time

import cv2
import numpy as np
import torch

from data import ImageFolder, default_loader
from models_factory import build_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=['resnet34', 'effb0', 'effb3'], required=True)
    p.add_argument('--weights', type=str, required=True)
    p.add_argument('--data_root', type=str, required=True,
                    help='images are read from <data_root>/<split>')
    p.add_argument('--out_dir', type=str, required=True)
    p.add_argument('--split', choices=['val', 'holdout'], default='val')
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--no_pretrained', action='store_true')
    return p.parse_args()


def list_ids(root):
    imagelist = filter(lambda x: x.find('sat') != -1, os.listdir(root))
    return list(map(lambda x: x[:x.find('_sat')], imagelist))


def load_weights(net, path, device):
    state_dict = torch.load(path, map_location=device)
    if any(k.startswith('module.') for k in state_dict):
        state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
    net.load_state_dict(state_dict)


def safe_div(n, d):
    return n / d if d > 0 else 0.0


def compute_metrics(tp, fp, fn, tn):
    iou_fg = safe_div(tp, tp + fp + fn)
    iou_bg = safe_div(tn, tn + fp + fn)
    miou = (iou_fg + iou_bg) / 2
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    accuracy = safe_div(tp + tn, tp + fp + fn + tn)
    return {
        'iou_fg': iou_fg, 'iou_bg': iou_bg, 'miou': miou,
        'f1': f1, 'precision': precision, 'recall': recall, 'accuracy': accuracy,
    }


def sensor_of(sample_id):
    lower = sample_id.lower()
    if 'palsar' in lower:
        return 'palsar'
    if 'sentinel' in lower:
        return 'sentinel'
    return None


def save_example_panel(sample_id, iou, split_root, net, device, out_path):
    img_np, mask_np = default_loader(sample_id, split_root, augment=False)
    img_batch = torch.tensor(img_np).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = net(img_batch)
    pred_bin = (pred > 0.5).float().squeeze().cpu().numpy()

    img_vis = ((img_np.transpose(1, 2, 0) + 1.6) / 3.2 * 255.0).clip(0, 255).astype(np.uint8)
    mask_vis = (mask_np.squeeze(0) * 255).astype(np.uint8)
    pred_vis = (pred_bin * 255).astype(np.uint8)

    mask_vis_3c = cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2BGR)
    pred_vis_3c = cv2.cvtColor(pred_vis, cv2.COLOR_GRAY2BGR)

    panel = np.concatenate([img_vis, mask_vis_3c, pred_vis_3c], axis=1)
    cv2.imwrite(out_path, panel)


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    split_root = os.path.join(args.data_root, args.split) + os.sep
    ids = list_ids(split_root)
    if not ids:
        raise SystemExit('Error: no samples found at {}'.format(split_root))

    os.makedirs(args.out_dir, exist_ok=True)
    examples_dir = os.path.join(args.out_dir, 'examples')
    os.makedirs(examples_dir, exist_ok=True)

    model_fn = build_model(args.model, pretrained=not args.no_pretrained)
    net = model_fn().to(device)
    load_weights(net, args.weights, device)
    net.eval()

    num_params_m = sum(p.numel() for p in net.parameters()) / 1e6

    dataset = ImageFolder(ids, split_root, augment=False)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Warm up before timing.
    warm_img, _ = dataset[0]
    warm_img = warm_img.unsqueeze(0).to(device)
    with torch.no_grad():
        for _ in range(3):
            net(warm_img)
    if device.type == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)

    per_sample_iou = {}
    tp_total = fp_total = fn_total = tn_total = 0
    sensor_counts = {}
    inference_times = []

    idx = 0
    with torch.no_grad():
        for img, mask in loader:
            img = img.to(device)
            mask = mask.to(device)

            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.time()
            pred = net(img)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t1 = time.time()

            batch_n = img.shape[0]
            inference_times.append((t1 - t0) / batch_n)

            pred_bin = pred > 0.5
            mask_bin = mask > 0.5

            for b in range(batch_n):
                sample_id = ids[idx]
                idx += 1
                p = pred_bin[b]
                m = mask_bin[b]
                tp = (p & m).sum().item()
                fp = (p & ~m).sum().item()
                fn = ((~p) & m).sum().item()
                tn = ((~p) & (~m)).sum().item()

                tp_total += tp
                fp_total += fp
                fn_total += fn
                tn_total += tn

                sensor = sensor_of(sample_id)
                if sensor:
                    counts = sensor_counts.setdefault(sensor, [0, 0, 0, 0])
                    counts[0] += tp
                    counts[1] += fp
                    counts[2] += fn
                    counts[3] += tn

                per_sample_iou[sample_id] = safe_div(tp, tp + fp + fn)

    overall = compute_metrics(tp_total, fp_total, fn_total, tn_total)
    per_sensor = {
        sensor: compute_metrics(*counts) for sensor, counts in sensor_counts.items()
    }

    ms_per_img = (sum(inference_times) / len(inference_times)) * 1000.0
    peak_gpu_memory_mb = None
    if device.type == 'cuda':
        peak_gpu_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    # Example panels: 6 random samples (fixed seed) + 3 worst by per-image IoU.
    rng = random.Random(42)
    random_ids = rng.sample(ids, min(6, len(ids)))
    worst_ids = [sid for sid, _ in sorted(per_sample_iou.items(), key=lambda kv: kv[1])[:3]]

    selected_ids = []
    for sid in random_ids + worst_ids:
        if sid not in selected_ids:
            selected_ids.append(sid)

    for sid in selected_ids:
        iou = per_sample_iou[sid]
        fname = '{}_iou{:.3f}.png'.format(sid, iou)
        save_example_panel(sid, iou, split_root, net, device, os.path.join(examples_dir, fname))

    result = {
        'model': args.model,
        'split': args.split,
        'weights': args.weights,
        'num_samples': len(ids),
        'overall': overall,
        'per_sensor': per_sensor,
        'params_millions': num_params_m,
        'ms_per_image': ms_per_img,
        'peak_gpu_memory_mb': peak_gpu_memory_mb,
        'device': str(device),
    }

    out_json_path = os.path.join(args.out_dir, 'eval_{}.json'.format(args.split))
    with open(out_json_path, 'w') as f:
        json.dump(result, f, indent=2)

    print('\n{:<10} {:<10} {:>8} {:>8} {:>10} {:>8} {:>10} {:>8}'.format(
        'Model', 'Split', 'mIoU', 'F1', 'Precision', 'Recall', 'Params(M)', 'ms/img'))
    print('{:<10} {:<10} {:>7.2f}% {:>7.2f}% {:>9.2f}% {:>7.2f}% {:>10.2f} {:>8.2f}'.format(
        args.model, args.split, overall['miou'] * 100, overall['f1'] * 100,
        overall['precision'] * 100, overall['recall'] * 100, num_params_m, ms_per_img))

    print('\n| model | split | mIoU | F1 | Precision | Recall | Params | ms/img |')
    print('|---|---|---|---|---|---|---|---|')
    print('| {} | {} | {:.2f}% | {:.2f}% | {:.2f}% | {:.2f}% | {:.2f}M | {:.2f} |'.format(
        args.model, args.split, overall['miou'] * 100, overall['f1'] * 100,
        overall['precision'] * 100, overall['recall'] * 100, num_params_m, ms_per_img))

    print('\nSaved metrics to {}'.format(out_json_path))
    print('Saved example panels to {}'.format(examples_dir))


if __name__ == '__main__':
    main()
