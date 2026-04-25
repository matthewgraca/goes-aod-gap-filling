"""
Train UNet 3+ on a cached dataset and emit a gap-filled AOD frame.
Reports paper-style metrics (RMSE, r, NMB) on val pixels and against AERONET.

Usage:
    python train.py --cache-dir cache_96 --epochs 20
"""
import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from libs.cache import (
    TARGET_SOURCE,
    CachedAODDataset,
    GapFillPipeline,
    compute_norm_stats,
)
from libs.metrics import (
    evaluate_aeronet,
    evaluate_pixel,
    predict_frame,
)
from libs.viz import (
    plot_aeronet_scatter,
    plot_all_timesteps_panel,
    plot_single_prediction,
    plot_training_curve,
)
from model.model import Unet3p

DEFAULT_EXTENT = (-118.615, -117.70, 33.60, 34.35)


def masked_mse(pred, target, valid):
    n = valid.sum().clamp_min(1.0)
    return (((pred - target) ** 2) * valid).sum() / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cache-dir', default='cache_96x96')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--out', default='outputs', help='Root output dir; each run lands in a timestamped subdir')
    p.add_argument('--run-name', default=None, help='Override run subdir name (defaults to YYYY-MM-DD_HHMMSS)')
    p.add_argument('--aeronet-extent', nargs=4, type=float, default=DEFAULT_EXTENT,
                   metavar=('LON_MIN', 'LON_MAX', 'LAT_MIN', 'LAT_MAX'),
                   help='Bounding box used only to colocate AERONET stations to grid cells')
    args = p.parse_args()

    run_name = args.run_name or datetime.now().strftime('%Y-%m-%d_%H%M%S')
    args.out = os.path.join(args.out, run_name)
    os.makedirs(args.out, exist_ok=True)
    print(f'Run output dir: {args.out}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    pipe = GapFillPipeline(cache_dir=args.cache_dir)
    pipe.load()
    pipe.summary()

    indices = pipe.valid_indices()
    print(f'Valid samples: {len(indices)}')

    print('Computing normalization stats...')
    x_min, x_max, y_min, y_max = compute_norm_stats(pipe, indices)
    print(f'  target {TARGET_SOURCE} range: [{y_min:.4g}, {y_max:.4g}]')

    ds = CachedAODDataset(pipe, indices, x_min, x_max, y_min, y_max)
    n_train = max(1, int(len(ds) * 0.8))
    train_set = Subset(ds, list(range(n_train)))
    val_set = Subset(ds, list(range(n_train, len(ds))))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size) if len(val_set) > 0 else None
    if len(val_set):
        print(f'Train: t_idx {indices[0]}..{indices[n_train-1]}  '
              f'Val: t_idx {indices[n_train]}..{indices[-1]}')
    else:
        print(f'Train: t_idx {indices[0]}..{indices[-1]}  (no val)')

    model = Unet3p(in_channels=pipe.num_input_channels).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f'Model: Unet3p({pipe.num_input_channels} channels), '
          f'{sum(p.numel() for p in model.parameters())/1e6:.1f}M params')

    history = {'train': [], 'val': []}
    for epoch in range(args.epochs):
        print(f'\nEpoch {epoch+1}/{args.epochs}')
        total_batches = len(train_loader) + (len(val_loader) if val_loader is not None else 0)
        bar = tqdm(total=total_batches, unit='batch', file=sys.stdout, mininterval=0,
                   bar_format='{n_fmt}/{total_fmt} [{bar}] {elapsed} - {rate_fmt}{postfix}')

        model.train()
        run, nb = 0.0, 0
        for X, y, v in train_loader:
            X, y, v = X.to(device), y.to(device), v.to(device)
            loss = masked_mse(model(X), y, v)
            opt.zero_grad(); loss.backward(); opt.step()
            run += loss.item(); nb += 1
            bar.set_postfix_str(f'loss: {run/nb:.5f}')
            bar.update(1)
        tl = run / max(nb, 1)

        vl = float('nan')
        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                vrun, vb = 0.0, 0
                for X, y, v in val_loader:
                    X, y, v = X.to(device), y.to(device), v.to(device)
                    vrun += masked_mse(model(X), y, v).item(); vb += 1
                    bar.set_postfix_str(f'loss: {tl:.5f} - val_loss: {vrun/vb:.5f}')
                    bar.update(1)
                vl = vrun / max(vb, 1)
        bar.close()

        history['train'].append(tl)
        history['val'].append(vl)

    plot_training_curve(history, os.path.join(args.out, 'training_curve.png'))

    train_m, _, _ = evaluate_pixel(model, ds, list(range(0, n_train)), device)
    val_m, _, _ = evaluate_pixel(model, ds, list(range(n_train, len(ds))), device)
    print('\n=== Pixel-wise metrics (vs GOES AOD) ===')
    print(f"  train: RMSE {train_m['rmse']:.4f}  r {train_m['r']:.3f}  "
          f"NMB {train_m['nmb']*100:+.1f}%  N={train_m['n']:,}")
    print(f"  val  : RMSE {val_m['rmse']:.4f}  r {val_m['r']:.3f}  "
          f"NMB {val_m['nmb']*100:+.1f}%  N={val_m['n']:,}")

    aero = evaluate_aeronet(model, pipe, indices, x_min, x_max, y_min, y_max,
                            tuple(args.aeronet_extent), device)
    if aero is not None:
        am = aero['metrics']
        print('\n=== AERONET-collocated metrics (550 nm) ===')
        print(f"  RMSE {am['rmse']:.4f}  r {am['r']:.3f}  "
              f"NMB {am['nmb']*100:+.1f}%  N={am['n']}")
        plot_aeronet_scatter(aero['obs'], aero['pred'], am,
                             os.path.join(args.out, 'aeronet_scatter.png'))
    else:
        print('\nNo AERONET observations colocated with valid timesteps; skipping AERONET metrics.')

    out_metrics = {
        'pixel_train': train_m,
        'pixel_val': val_m,
        'aeronet': aero['metrics'] if aero is not None else None,
        'history': history,
        'cache_dir': args.cache_dir,
        'epochs': args.epochs,
        'lr': args.lr,
        'batch_size': args.batch_size,
        'in_channels': int(pipe.num_input_channels),
        'aeronet_extent': list(args.aeronet_extent),
        'dim': int(pipe.dim),
    }
    with open(os.path.join(args.out, 'metrics.json'), 'w') as f:
        json.dump(out_metrics, f, indent=2,
                  default=lambda o: float(o) if isinstance(o, (np.floating,)) else None)

    snapshot = {
        'state_dict': model.state_dict(),
        'in_channels': int(pipe.num_input_channels),
        'x_min': x_min,
        'x_max': x_max,
        'y_min': float(y_min),
        'y_max': float(y_max),
        'dim': int(pipe.dim),
    }
    torch.save(snapshot, os.path.join(args.out, 'snapshot.pt'))

    # Single-frame prediction (last valid timestep)
    t_eval = indices[-1]
    X, y_obs = pipe.get_sample(t_eval)
    pred = predict_frame(model, X, x_min, x_max, y_min, y_max, device)
    np.save(os.path.join(args.out, 'prediction.npy'), pred)
    np.save(os.path.join(args.out, 'target_observed.npy'), y_obs[0])
    plot_single_prediction(y_obs[0], pred, pipe.timestamps[t_eval],
                           os.path.join(args.out, 'prediction.png'))

    # Contact sheet across every valid timestep
    is_val_idx = set(range(n_train, len(ds)))
    samples = []
    for pos, t_idx in enumerate(indices):
        X, y = pipe.get_sample(t_idx)
        pr = predict_frame(model, X, x_min, x_max, y_min, y_max, device)
        samples.append({
            'timestamp': pipe.timestamps[t_idx],
            'obs': y[0],
            'pred': pr,
            'split': 'VAL' if pos in is_val_idx else 'train',
        })
    plot_all_timesteps_panel(samples, os.path.join(args.out, 'panel_all_timesteps.png'))

    print(f'\nOutput files in {args.out}/:')
    for f in sorted(os.listdir(args.out)):
        print(f'  {f}')


if __name__ == '__main__':
    main()
