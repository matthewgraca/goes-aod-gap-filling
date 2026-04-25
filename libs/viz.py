"""Plotting helpers used by train.py."""
import os

import matplotlib.pyplot as plt
import numpy as np


def plot_training_curve(history, out_path, title='Training History — UNet 3+ AOD Gap-Fill'):
    """Train + val loss per epoch with best-val/best-epoch reference lines."""
    train_loss = history['train']
    val_loss = history['val']
    n_epochs = len(train_loss)
    epochs = np.arange(1, n_epochs + 1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, train_loss, label='Training Loss', lw=2, color='blue')
    has_val = not np.all(np.isnan(val_loss))
    if has_val:
        ax.plot(epochs, val_loss, label='Validation Loss', lw=2, color='orange')
        best_val = float(np.nanmin(val_loss))
        best_epoch = int(np.nanargmin(val_loss)) + 1
        ax.axhline(y=best_val, color='r', ls='--', alpha=0.5)
        ax.axvline(x=best_epoch, color='g', ls='--', alpha=0.3)
        textstr = (f'Best Val: {best_val:.4f}\nBest Epoch: {best_epoch}\n'
                   f'Final Train: {train_loss[-1]:.4f}\nFinal Val: {val_loss[-1]:.4f}')
    else:
        textstr = f'Final Train: {train_loss[-1]:.4f}'
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(title, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.text(0.98, 0.98, textstr, transform=ax.transAxes, fontsize=9,
            va='top', ha='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_aeronet_scatter(obs, pred, m, out_path):
    """1:1 scatter of model prediction vs AERONET 550 nm with paper-style stats."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(obs, pred, s=18, alpha=0.7)
    lim = max(obs.max(), pred.max(), 0.01) * 1.1
    ax.plot([0, lim], [0, lim], 'k--', lw=1, alpha=0.5)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel('AERONET AOD 550 nm')
    ax.set_ylabel('UNet 3+ AOD 550 nm')
    ax.set_title(
        f"vs AERONET   RMSE={m['rmse']:.3f}  r={m['r']:.3f}  "
        f"NMB={m['nmb']*100:+.1f}%  N={m['n']}"
    )
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_single_prediction(obs, pred, timestamp, out_path):
    """3-panel: observed | predicted | (pred - obs)."""
    finite_obs = obs[np.isfinite(obs)]
    vmin = float(min(finite_obs.min() if finite_obs.size else 0.0, pred.min()))
    vmax = float(max(finite_obs.max() if finite_obs.size else 1.0, pred.max()))
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(obs, vmin=vmin, vmax=vmax)
    axes[0].set_title(f'GOES AOD obs\n{timestamp}')
    axes[1].imshow(pred, vmin=vmin, vmax=vmax)
    axes[1].set_title('UNet3+ prediction')
    axes[2].imshow(pred - np.where(np.isnan(obs), pred, obs), cmap='RdBu_r')
    axes[2].set_title('pred - obs (NaNs zeroed)')
    for a in axes:
        a.axis('off')
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_all_timesteps_panel(samples, out_path):
    """
    Contact sheet: obs / pred / diff for every entry in `samples`.

    samples: list of dicts with keys {'timestamp', 'obs', 'pred', 'split'}
        where 'obs' and 'pred' are (H, W) arrays, 'split' is 'train' or 'VAL'.
    """
    n = len(samples)
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n), squeeze=False)

    all_obs = [s['obs'][np.isfinite(s['obs'])] for s in samples]
    all_pred = [s['pred'].flatten() for s in samples]
    flat_obs = np.concatenate(all_obs) if all_obs else np.array([0.0])
    flat_pred = np.concatenate(all_pred)
    vmn = float(min(flat_obs.min() if flat_obs.size else 0.0, flat_pred.min()))
    vmx = float(max(flat_obs.max() if flat_obs.size else 1.0, flat_pred.max()))

    for row, s in enumerate(samples):
        obs, pred = s['obs'], s['pred']
        axes[row, 0].imshow(obs, vmin=vmn, vmax=vmx)
        axes[row, 0].set_title(f"{s['timestamp']}  [{s['split']}] — obs", fontsize=9)
        axes[row, 1].imshow(pred, vmin=vmn, vmax=vmx)
        axes[row, 1].set_title('pred', fontsize=9)
        diff = pred - np.where(np.isnan(obs), pred, obs)
        dlim = max(abs(diff.min()), abs(diff.max()), 1e-6)
        axes[row, 2].imshow(diff, cmap='RdBu_r', vmin=-dlim, vmax=dlim)
        axes[row, 2].set_title('pred - obs', fontsize=9)
        for c in range(3):
            axes[row, c].axis('off')
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
