"""Metrics + evaluation helpers (RMSE / r / NMB on pixels and against AERONET)."""
import os

import numpy as np
import pandas as pd
import torch

FILL = 0.0


def metrics(pred, obs):
    """RMSE, Pearson r, NMB over a flat pair of arrays."""
    pred = np.asarray(pred, dtype=np.float64)
    obs = np.asarray(obs, dtype=np.float64)
    n = len(pred)
    if n == 0:
        return dict(rmse=float('nan'), r=float('nan'), nmb=float('nan'), n=0)
    rmse = float(np.sqrt(np.mean((pred - obs) ** 2)))
    r = float(np.corrcoef(pred, obs)[0, 1]) if n > 1 and obs.std() > 0 and pred.std() > 0 else float('nan')
    denom = float(np.abs(obs).sum())
    nmb = float((pred - obs).sum() / denom) if denom > 0 else float('nan')
    return dict(rmse=rmse, r=r, nmb=nmb, n=int(n))


def predict_frame(model, X, x_min, x_max, y_min, y_max, device):
    """Run model on a single (C,H,W) numpy frame; return de-normalized (H,W)."""
    rng = (x_max - x_min); rng[rng == 0] = 1.0
    Xn = (X - x_min[:, None, None]) / rng[:, None, None]
    Xn = np.where(np.isnan(Xn), FILL, Xn).astype(np.float32)
    with torch.no_grad():
        pred_n = model(torch.from_numpy(Xn).unsqueeze(0).to(device)).cpu().numpy()[0, 0]
    return pred_n * (y_max - y_min) + y_min


def evaluate_pixel(model, ds, indices_subset, device):
    """De-normalized pixel-wise pred vs obs over a list of dataset indices."""
    preds, obses = [], []
    model.eval()
    for i in indices_subset:
        X_norm, y_norm, valid = ds[i]
        with torch.no_grad():
            pred_n = model(torch.from_numpy(X_norm).unsqueeze(0).to(device)).cpu().numpy()[0, 0]
        pred = pred_n * ds.y_rng + ds.y_min
        obs = y_norm[0] * ds.y_rng + ds.y_min
        m = valid[0] > 0
        preds.append(pred[m])
        obses.append(obs[m])
    if not preds:
        return metrics([], []), np.array([]), np.array([])
    p = np.concatenate(preds); o = np.concatenate(obses)
    return metrics(p, o), p, o


def evaluate_aeronet(model, pipe, valid_indices, x_min, x_max, y_min, y_max, extent, device):
    """Collocate model prediction at AERONET station lat/lon for each valid timestep."""
    aero_path = os.path.join(pipe.cache_dir, 'aeronet.parquet')
    if not os.path.exists(aero_path):
        return None
    aero = pd.read_parquet(aero_path)
    if aero.empty:
        return None

    lon_min, lon_max, lat_min, lat_max = extent
    dim = pipe.dim
    aero_ts = pd.to_datetime(aero['timestamp'])
    obs_pred_pairs = []
    model.eval()

    for t_idx in valid_indices:
        ts = pipe.timestamps[t_idx]
        sub = aero[aero_ts == ts]
        if sub.empty:
            continue
        X, _ = pipe.get_sample(t_idx)
        pred = predict_frame(model, X, x_min, x_max, y_min, y_max, device)
        for _, row in sub.iterrows():
            lat, lon, aod = float(row['lat']), float(row['lon']), float(row['aod_550'])
            if not np.isfinite(aod):
                continue
            if not (lon_min <= lon <= lon_max and lat_min <= lat <= lat_max):
                continue
            col = int(np.clip((lon - lon_min) / (lon_max - lon_min) * dim, 0, dim - 1))
            r_idx = int(np.clip((lat_max - lat) / (lat_max - lat_min) * dim, 0, dim - 1))
            obs_pred_pairs.append((aod, float(pred[r_idx, col]), str(row['sensor_name']), str(ts)))

    if not obs_pred_pairs:
        return None
    obs = np.array([p[0] for p in obs_pred_pairs])
    prd = np.array([p[1] for p in obs_pred_pairs])
    return dict(metrics=metrics(prd, obs), obs=obs, pred=prd, pairs=obs_pred_pairs)
