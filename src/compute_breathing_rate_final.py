from __future__ import annotations
import argparse, math, os
from pathlib import Path
import numpy as np
import pandas as pd
from . import ibirsa   # project-specific utilities for loading and processing IBI/beat data


# --- Internal helper: refine FFT peak estimate with quadratic interpolation
def _parabolic_interp(f, Pxx, k):
    # Avoid edges: no interpolation possible at boundaries
    if k <= 0 or k >= len(Pxx) - 1:
        return float(f[k])
    # Fit a parabola around the local maximum to estimate sub-bin frequency
    y0, y1, y2 = Pxx[k-1], Pxx[k], Pxx[k+1]
    d = (y0 - 2*y1 + y2)
    delta = 0.5 * (y0 - y2) / d if d != 0 else 0.0
    return float(f[k] + delta * (f[k+1] - f[k]))


# --- Core routine: compute breathing rate windows using FFT
def _bpm_windows_fft(x, valid_mask, fs, win_s, fmin, fmax, min_cov, peak_prom):
    win_len = int(round(win_s * fs))
    # Bail out if data too short for even a single window
    if win_len < 2 or x.size < win_len:
        return np.array([], dtype=int), np.array([], dtype=float), np.array([], dtype=bool)

    # Reshape into non-overlapping windows
    nwin = x.size // win_len
    x = x[: nwin*win_len].reshape(nwin, win_len)
    v = valid_mask[: nwin*win_len].reshape(nwin, win_len)

    t0_idx = np.arange(nwin, dtype=int)
    bpm = np.full(nwin, np.nan, float)   # store results
    ok = np.zeros(nwin, bool)            # window validity flag

    # FFT setup: pad to nearest power of two for efficiency
    nfft = 1 << int(np.ceil(np.log2(max(256, win_len))))
    w = np.hanning(win_len)
    wnorm = np.sum(w**2) * fs
    freqs = np.fft.rfftfreq(nfft, d=1.0/fs)

    # Restrict to plausible breathing band (fmin–fmax)
    band = (freqs >= fmin) & (freqs <= fmax)
    bidx = np.flatnonzero(band)
    if bidx.size < 3:
        return t0_idx, bpm, ok

    for i in range(nwin):
        # Require sufficient valid coverage in this window
        if np.mean(v[i]) < min_cov:
            continue
        xi = x[i] - np.nanmean(x[i])
        if not np.all(np.isfinite(xi)):
            continue

        # Power spectrum
        X = np.fft.rfft(xi * w, n=nfft)
        Pxx = (np.abs(X)**2) / wnorm
        Pb = Pxx[band]

        # Identify peak and refine via parabolic interpolation
        k_rel = int(np.argmax(Pb))
        k = bidx[k_rel]
        f_hat = _parabolic_interp(freqs, Pxx, k) if 0 < k < len(Pxx)-1 else freqs[k]

        # Assess peak prominence relative to background
        peak = Pxx[k]
        bg = np.median(Pb) if np.isfinite(np.median(Pb)) else 0.0
        prom = peak / max(bg, 1e-12)

        # Accept if prominent and finite
        if prom >= peak_prom and np.isfinite(f_hat):
            bpm[i] = 60.0 * f_hat
            ok[i] = True

    return t0_idx, bpm, ok


# --- Public API: compute breathing rate at 15s resolution
def compute_15s(input_path, output_csv,
                fs=4.0, win_s=15.0, fmin=0.1, fmax=0.5,
                min_cov=0.7, peak_prom=2.0, round_decimals=1):
    # Load raw records from JSON/NDJSON logs
    records = ibirsa.load_stream(input_path)
    if not records:
        raise SystemExit("No records loaded.")

    # Extract and normalize beat times
    ts_vals = np.array([rec["ts"] for rec in records if "ts" in rec], float)
    ts_unit = ibirsa.detect_ts_unit(ts_vals)
    beat_t, ibi = ibirsa.reconstruct_beats(records, ts_unit)
    if beat_t.size == 0:
        raise SystemExit("No beats reconstructed.")

    # Clean IBI series (reject outliers outside 0.3–2.0 s)
    ok = ibirsa.clean_ibi(ibi, lo=0.3, hi=2.0)
    beat_t, ibi = beat_t[ok], ibi[ok]
    if beat_t.size < 5:
        raise SystemExit("Too few valid beats.")

    # Resample onto uniform grid for FFT analysis
    t_grid, ibi_grid, valid_grid = ibirsa.to_uniform_series(beat_t, ibi, fs=fs, max_gap_s=2.0)
    if t_grid.size == 0:
        raise SystemExit("Failed to create uniform series.")

    # Align windowing to clean 15s segments
    step = 1.0 / fs
    start_aligned = ibirsa.align_grid_start(t_grid[0], win_s)
    start_idx = int(np.ceil((start_aligned - t_grid[0]) / step))
    if start_idx >= t_grid.size:
        raise SystemExit("Data too short after alignment.")
    t_grid, ibi_grid, valid_grid = t_grid[start_idx:], ibi_grid[start_idx:], valid_grid[start_idx:]

    # Trim to integer number of windows
    nwin = (ibi_grid.size // int(win_s * fs))
    trim = nwin * int(win_s * fs)
    t_grid, ibi_grid, valid_grid = t_grid[:trim], ibi_grid[:trim], valid_grid[:trim]

    # Bandpass filter around respiratory range
    x_bp = ibirsa.ban_
