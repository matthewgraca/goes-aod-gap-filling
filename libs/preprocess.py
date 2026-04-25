"""
Preprocress.py assembles per-timestep input tensors.

Cache contents:
    goes.npz          keys: aod, frp, adp_smoke, adp_dust
    hrrr.npz          keys: temp_2m, dewpoint_2m, rh_2m, ...
    ndvi.npz          keys: ndvi
    aeronet.parquet   point observations (used for evaluation)
    timestamps.npy
"""
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

FILL = 0.0
TARGET_SOURCE = 'goes.aod'


@dataclass
class ChannelSpec:
    name: str
    source: str        # "file.key" e.g. "goes.aod", "hrrr.temp_2m"
    lags: list[int]    # hours back (0 = current, 1 = 1h ago, ...)
    log_transform: bool = False


DEFAULT_CHANNELS = [
    # GOES AOD — lag 0 is the target; only past lags are inputs
    ChannelSpec('goes_aod',         'goes.aod',              lags=[1, 2, 3], log_transform=True),
    ChannelSpec('goes_frp',         'goes.frp',              lags=[1, 2, 3]),
    ChannelSpec('goes_adp_smoke',   'goes.adp_smoke',        lags=[1, 2, 3]),
    ChannelSpec('goes_adp_dust',    'goes.adp_dust',         lags=[1, 2, 3]),
    # HRRR meteorology — current + recent history
    ChannelSpec('temp_2m',          'hrrr.temp_2m',          lags=[0, 1, 2, 3]),
    ChannelSpec('dewpoint_2m',      'hrrr.dewpoint_2m',      lags=[0, 1, 2, 3]),
    ChannelSpec('rh_2m',            'hrrr.rh_2m',            lags=[0, 1, 2, 3]),
    ChannelSpec('pressure_surface', 'hrrr.pressure_surface', lags=[0, 1, 2, 3]),
    ChannelSpec('pressure_msl',     'hrrr.pressure_msl',     lags=[0, 1, 2, 3]),
    ChannelSpec('pbl_height',       'hrrr.pbl_height',       lags=[0, 1, 2, 3]),
    ChannelSpec('u_wind',           'hrrr.u_wind',           lags=[0, 1, 2, 3]),
    ChannelSpec('v_wind',           'hrrr.v_wind',           lags=[0, 1, 2, 3]),
    ChannelSpec('smoke_massden',    'hrrr.smoke_massden',    lags=[0, 1, 2, 3]),
    ChannelSpec('ndvi',             'ndvi.ndvi',             lags=[0]),
]


class GapFillPipeline:
    """Loads cached .npz arrays and assembles per-timestep input tensors."""

    def __init__(self, cache_dir, channels=None):
        self.cache_dir = cache_dir
        self.channels = channels if channels is not None else DEFAULT_CHANNELS
        self.sources = {}
        self.timestamps = None
        self.dim = None
        self.max_lag = max(max(ch.lags) for ch in self.channels)

    def load(self):
        ts_path = os.path.join(self.cache_dir, 'timestamps.npy')
        self.timestamps = pd.DatetimeIndex(np.load(ts_path, allow_pickle=True))
        T = len(self.timestamps)

        all_sources = {ch.source for ch in self.channels} | {TARGET_SOURCE}
        npz_keys = {}
        for src in all_sources:
            f, k = src.split('.', 1)
            npz_keys.setdefault(f, []).append(k)

        for npz_file, keys in npz_keys.items():
            path = os.path.join(self.cache_dir, f'{npz_file}.npz')
            if not os.path.exists(path):
                raise FileNotFoundError(f'{path} missing — required by channels {keys}')
            data = np.load(path)
            for key in keys:
                if key not in data:
                    raise KeyError(f"key '{key}' not in {npz_file}.npz")
                arr = data[key]
                if self.dim is None:
                    self.dim = arr.shape[-1]
                self.sources[f'{npz_file}.{key}'] = self._align(arr, T)

    def _align(self, arr, T):
        if len(arr) > T:
            return arr[:T]
        if len(arr) < T:
            pad = np.full((T - len(arr), self.dim, self.dim), np.nan)
            return np.concatenate([arr, pad], axis=0)
        return arr

    @property
    def num_input_channels(self):
        return sum(len(ch.lags) for ch in self.channels)

    def get_sample(self, t_idx):
        if t_idx < self.max_lag:
            raise IndexError(f't_idx={t_idx} < max_lag={self.max_lag}')
        frames = []
        for ch in self.channels:
            arr = self.sources[ch.source]
            for lag in ch.lags:
                frame = arr[t_idx - lag].copy()
                if ch.log_transform:
                    frame = np.log(np.maximum(frame, 0.0) + 1e-5)
                frames.append(frame)
        X = np.stack(frames, axis=0).astype(np.float32)
        y = self.sources[TARGET_SOURCE][t_idx][None].astype(np.float32)
        return X, y

    def valid_indices(self):
        return list(range(self.max_lag, len(self.timestamps)))

    def summary(self):
        T = len(self.timestamps) if self.timestamps is not None else 0
        print(f'Cache      : {self.cache_dir}')
        print(f'Grid       : {self.dim} x {self.dim}')
        print(f'Timestamps : {T}')
        if T > 0:
            print(f'Range      : {self.timestamps[0]} -> {self.timestamps[-1]}')
        print(f'Input ch   : {self.num_input_channels}')
        print(f'Max lag    : {self.max_lag} hours')
        print()
        print('Sources:')
        for key in sorted(self.sources):
            arr = self.sources[key]
            nan_pct = np.isnan(arr).mean() * 100
            lo, hi = np.nanmin(arr), np.nanmax(arr)
            print(f'  {key:25s}  {str(arr.shape):18s}  '
                  f'NaN {nan_pct:5.1f}%  [{lo:.4g}, {hi:.4g}]')
        print()
        print('Channels:')
        for ch in self.channels:
            tag = ' (log)' if ch.log_transform else ''
            print(f'  {ch.name:25s}  lags={ch.lags}  -> {len(ch.lags)} ch{tag}')


class CachedAODDataset(Dataset):
    """Wraps a GapFillPipeline with normalization for use in a torch DataLoader."""

    def __init__(self, pipe, indices, x_min, x_max, y_min, y_max):
        self.pipe = pipe
        self.indices = indices
        self.x_min = x_min[:, None, None].astype(np.float32)
        rng = (x_max - x_min).astype(np.float32)
        rng[rng == 0] = 1.0
        self.x_rng = rng[:, None, None]
        self.y_min = float(y_min)
        self.y_rng = float(max(y_max - y_min, 1e-6))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        X, y = self.pipe.get_sample(self.indices[i])
        valid = (~np.isnan(y)).astype(np.float32)
        Xn = (X - self.x_min) / self.x_rng
        Xn = np.where(np.isnan(Xn), FILL, Xn).astype(np.float32)
        yn = (y - self.y_min) / self.y_rng
        yn = np.where(np.isnan(yn), FILL, yn).astype(np.float32)
        return Xn, yn, valid


def compute_norm_stats(pipe, indices):
    """Per-channel min/max + target min/max scanned across `indices`."""
    C = pipe.num_input_channels
    x_min = np.full(C, np.inf, dtype=np.float64)
    x_max = np.full(C, -np.inf, dtype=np.float64)
    y_min, y_max = np.inf, -np.inf
    for t in indices:
        X, y = pipe.get_sample(t)
        for c in range(C):
            ch = X[c][np.isfinite(X[c])]
            if ch.size:
                x_min[c] = min(x_min[c], ch.min())
                x_max[c] = max(x_max[c], ch.max())
        yv = y[np.isfinite(y)]
        if yv.size:
            y_min = min(y_min, yv.min())
            y_max = max(y_max, yv.max())
    x_min = np.where(np.isfinite(x_min), x_min, 0.0)
    x_max = np.where(np.isfinite(x_max), x_max, 1.0)
    if not np.isfinite(y_min): y_min = 0.0
    if not np.isfinite(y_max): y_max = 1.0
    return x_min.astype(np.float32), x_max.astype(np.float32), float(y_min), float(y_max)
