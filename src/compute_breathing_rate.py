"""
Compute breathing rate from RR interval logs at multiple window sizes.

This script reads a JSON log of ts/hr/rr, reconstructs beats, creates a uniform
time series, bandpasses in 0.1–0.5 Hz, estimates the dominant respiration
frequency per window using FFT with parabolic peak interpolation, converts to
bpm, and writes CSVs for 15s/30s/60s/120s/240s windows.

Each CSV has:
    - time: window start (HH:MM:SS PDT)
    - bpm: breaths per minute, rounded to one decimal
"""

from __future__ import annotations
import argparse
import math
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from . import ibirsa


def _summarize(bpm: np.ndarray, valid: np.ndarray) -> str:
    if bpm.size == 0:
        return "No output windows."
    valid_bpm = bpm[valid & ~np.isnan(bpm)]
    if valid_bpm.size == 0:
        return "No valid windows."
    return (
        f"Windows: {bpm.size}, valid: {valid_bpm.size} "
        f"({100.0*valid_bpm.size/max(1,bpm.size):.1f}%). "
        f"Median bpm: {np.nanmedian(valid_bpm):.1f}"
    )


def _parabolic_interp(f: np.ndarray, Pxx: np.ndarray, k: int) -> float:
    if k <= 0 or k >= len(Pxx) - 1:
        return float(f[k])
    y0, y1, y2 = Pxx[k - 1], Pxx[k], Pxx[k + 1]
    denom = (y0 - 2 * y1 + y2)
    delta = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
    return float(f[k] + delta * (f[k + 1] - f[k]))


def _bpm_windows_fft(
    x: np.ndarray,
    valid_mask: np.ndarray,
    fs: float,
    win_s: float,
    fmin: float,
    fmax: float,
    min_coverage: float,
    peak_prom: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    win_len = int(round(win_s * fs))
    if win_len < 2 or x.size < win_len:
        return np.array([], dtype=int), np.array([], dtype=float), np.array([], dtype=bool)

    nwin = x.size // win_len
    x = x[: nwin * win_len]
    valid_mask = valid_mask[: nwin * win_len]
    x = x.reshape(nwin, win_len)
    v = valid_mask.reshape(nwin, win_len)

    t0_idx = np.arange(nwin, dtype=int)
    bpm = np.full(nwin, np.nan, dtype=float)
    ok = np.zeros(nwin, dtype=bool)

    nfft = 1 << int(np.ceil(np.log2(max(256, win_len))))
    w = np.hanning(win_len)
    wnorm = np.sum(w ** 2) * fs

    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
    band = (freqs >= fmin) & (freqs <= fmax)
    band_idx = np.flatnonzero(band)
    if band_idx.size < 3:
        return t0_idx, bpm, ok

    for i in range(nwin):
        cov = np.mean(v[i])
        if cov < min_coverage:
            continue
        xi = x[i]
        xi = xi - np.nanmean(xi)
        if np.any(~np.isfinite(xi)):
            continue
        X = np.fft.rfft(xi * w, n=nfft)
        Pxx = (np.abs(X) ** 2) / wnorm

        P_band = Pxx[band]
        if not np.all(np.isfinite(P_band)):
            continue

        k_rel = int(np.argmax(P_band))
        k = band_idx[k_rel]

        # reject edges where interpolation isn’t possible
        if k == 0 or k == len(Pxx) - 1:
            f_hat = freqs[k]
        else:
            f_hat = _parabolic_interp(freqs, Pxx, k)

        peak = Pxx[k]
        bg = np.median(P_band) if np.isfinite(np.median(P_band)) else 0.0
        prom = peak / max(bg, 1e-12)

        if prom >= peak_prom and np.isfinite(f_hat):
            bpm[i] = 60.0 * f_hat
            ok[i] = True

    return t0_idx, bpm, ok


def compute_file(
    input_path: str,
    output_path: str,
    fs: float = 4.0,
    win_s: float = 15.0,
    fmin: float = 0.1,
    fmax: float = 0.5,
    min_coverage: float = 0.7,
    peak_prom: float = 2.0,
) -> Tuple[pd.DataFrame, str]:
    records = ibirsa.load_stream(input_path)
    if not records:
        raise SystemExit("No records loaded from input.")

    ts_vals = np.array([rec["ts"] for rec in records if "ts" in rec], dtype=float)
    ts_unit = ibirsa.detect_ts_unit(ts_vals)

    beat_t, ibi = ibirsa.reconstruct_beats(records, ts_unit)
    if beat_t.size == 0:
        raise SystemExit("No beats reconstructed (no RR data available).")

    mask_ok = ibirsa.clean_ibi(ibi, lo=0.3, hi=2.0)
    beat_t = beat_t[mask_ok]
    ibi = ibi[mask_ok]
    if beat_t.size < 5:
        raise SystemExit("Too few valid beats after cleaning.")

    t_grid, ibi_grid, valid_grid = ibirsa.to_uniform_series(beat_t, ibi, fs=fs, max_gap_s=2.0)
    if t_grid.size == 0:
        raise SystemExit("Failed to create uniform series.")

    step = 1.0 / fs
    start_aligned = ibirsa.align_grid_start(t_grid[0], win_s)
    start_idx = int(math.ceil((start_aligned - t_grid[0]) / step))
    if start_idx >= t_grid.size:
        raise SystemExit("Data too short after alignment.")
    t_grid = t_grid[start_idx:]
    ibi_grid = ibi_grid[start_idx:]
    valid_grid = valid_grid[start_idx:]

    nwin = (ibi_grid.size // int(win_s * fs))
    trim_len = nwin * int(win_s * fs)
    t_grid = t_grid[:trim_len]
    ibi_grid = ibi_grid[:trim_len]
    valid_grid = valid_grid[:trim_len]

    x_bp = ibirsa.bandpass_fft(ibi_grid, fs=fs, fmin=fmin, fmax=fmax)

    t0_idx, bpm, valid_win = _bpm_windows_fft(
        x=x_bp,
        valid_mask=valid_grid.astype(bool),
        fs=fs,
        win_s=win_s,
        fmin=fmin,
        fmax=fmax,
        min_coverage=min_coverage,
        peak_prom=peak_prom,
    )

    win_starts = t_grid[0] + t0_idx * (1.0 / fs) * int(win_s * fs)
    times_str = [ibirsa.utc_seconds_to_pdt_hms_str(float(ts)) for ts in win_starts]

    bpm_rounded = np.round(bpm.astype(float), 1)
    df = pd.DataFrame({"time": times_str, "bpm": bpm_rounded})
    df.loc[~valid_win | ~np.isfinite(df["bpm"]), "bpm"] = np.nan

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    summary = _summarize(bpm_rounded, valid_win)
    return df, summary


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Compute breathing rate at multiple window sizes from RR logs.")
    p.add_argument("--input", required=True, help="Path to JSON log (array or NDJSON).")
    p.add_argument("--output", required=True, help="Path to output CSV (used as naming template).")
    p.add_argument("--rsamp", type=float, default=4.0, help="Resampling frequency in Hz (default 4).")
    p.add_argument("--fmin", type=float, default=0.1, help="Minimum respiration frequency in Hz (default 0.1).")
    p.add_argument("--fmax", type=float, default=0.5, help="Maximum respiration frequency in Hz (default 0.5).")
    p.add_argument("--min-coverage", type=float, default=0.7, help="Minimum fraction of valid samples per window (default 0.7).")
    p.add_argument("--peak-prom", type=float, default=2.0, help="Required peak prominence factor (default 2.0).")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    win_lengths = [15, 30, 60, 120, 240]
    for win in win_lengths:
        base = Path(args.output)
        out_name = f"breathing_rate_{win}s.csv"
        out_path = base.parent / out_name

        df, summary = compute_file(
            input_path=args.input,
            output_path=str(out_path),
            fs=args.rsamp,
            win_s=win,
            fmin=args.fmin,
            fmax=args.fmax,
            min_coverage=args.min_coverage,
            peak_prom=args.peak_prom,
        )
        print(f"[{win}s] {summary} -> saved to {out_path}")


if __name__ == "__main__":
    main()
