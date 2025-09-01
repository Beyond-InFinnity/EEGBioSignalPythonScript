"""
ibirsa.py
Core utilities for reconstructing beat times from RR intervals and estimating
breathing rate via RSA using NumPy-only signal ops (no SciPy).

Pipeline:
1) load_stream(path) -> list[dict]
2) detect_ts_unit(ts_values) -> 'ms'|'s'
3) reconstruct_beats(records, ts_unit) -> (beat_times_s, ibi_s)
4) clean_ibi(ibi_s, lo=0.3, hi=2.0) -> mask filtering implausible IBIs
5) to_uniform_series(beat_times_s, ibi_s, fs=4.0, max_gap_s=2.0)
6) bandpass_fft(x, fs, fmin=0.1, fmax=0.5)
7) window_bpm_fft(x_bp, fs, win_s=15, min_coverage=0.7, fmin=0.1, fmax=0.5, peak_prom=2.0)
"""

from __future__ import annotations
import json
from typing import List, Tuple
import numpy as np
from dateutil import tz
from datetime import datetime, timezone

# --------------------
# IO / parsing
# --------------------

def load_stream(path: str) -> List[dict]:
    """
    Load a JSON file that may be:
      - a JSON array of objects, OR
      - newline-delimited JSON (one object per line).

    Returns a list of dicts containing (at least) 'ts' and optionally 'hr', 'rr'.
    """
    with open(path, "r", encoding="utf-8") as f:
        first_chunk = f.read(2048)
        f.seek(0)
        if first_chunk.strip().startswith("["):
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("Expected a JSON array at the top level.")
            return data
        else:
            # NDJSON
            records = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
            return records


def detect_ts_unit(ts_values: np.ndarray) -> str:
    """
    Heuristically determine whether timestamps are in seconds or milliseconds.
    """
    vmax = float(np.nanmax(ts_values))
    # Unix seconds are around 1.7e9 in 2025; ms are ~1.7e12.
    if vmax > 1e11:
        return "ms"
    elif vmax > 1e9:
        # could be seconds in far future, but overwhelmingly 's'
        return "s"
    else:
        # Extremely small → probably seconds (older data) or an offset
        return "s"


# --------------------
# Beat reconstruction
# --------------------

def reconstruct_beats(records: List[dict], ts_unit: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    From records containing 'ts' and 'rr' (list of ms), reconstruct beat times.

    Assumptions:
    - Each 'rr' list represents successive inter-beat intervals that END at 'ts'.
    - RR values are in milliseconds.
    - We walk backward from 'ts' for each rr item to place the corresponding beat.
    - We return arrays:
      beat_times_s: epoch seconds (float64, UTC)
      ibi_s: inter-beat interval in seconds corresponding to each beat.
    """
    beats_t = []
    beats_ibi = []

    for rec in records:
        if "ts" not in rec:
            continue
        ts_val = rec["ts"]
        if ts_unit == "ms":
            ts_s = ts_val / 1000.0
        else:
            ts_s = float(ts_val)

        if "rr" not in rec or rec["rr"] is None:
            continue
        rr_list = rec["rr"]
        if not isinstance(rr_list, list) or len(rr_list) == 0:
            continue

        # walk backward: the last RR ends AT ts_s
        cum = 0.0
        for rr_ms in reversed(rr_list):
            if rr_ms is None:
                continue
            ibi = float(rr_ms) / 1000.0
            cum += ibi
            beat_time = ts_s - cum  # time of the beat at the START of that rr interval
            beats_t.append(beat_time)
            beats_ibi.append(ibi)

    if not beats_t:
        return np.array([]), np.array([])

    beats_t = np.asarray(beats_t, dtype=np.float64)
    beats_ibi = np.asarray(beats_ibi, dtype=np.float64)

    # Sort by time
    order = np.argsort(beats_t)
    return beats_t[order], beats_ibi[order]


def clean_ibi(ibi_s: np.ndarray, lo: float = 0.3, hi: float = 2.0) -> np.ndarray:
    """
    Return a boolean mask of plausible IBI values.
    Default range 0.3–2.0 s (30–200 bpm).
    """
    return (ibi_s >= lo) & (ibi_s <= hi)


# --------------------
# Uniform resampling
# --------------------

def to_uniform_series(
    beat_times_s: np.ndarray,
    ibi_s: np.ndarray,
    fs: float = 4.0,
    max_gap_s: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Interpolate irregular IBI samples onto a uniform grid.

    Returns:
      t_grid_s: np.ndarray of time stamps (epoch seconds) at fs (uniform)
      ibi_grid: interpolated IBI series (float) with NaN where coverage poor
      valid_mask: boolean mask of where interpolation considered trustworthy

    Strategy:
    - Linear interpolation across beats.
    - Compute distance from each grid point to nearest beat; mark valid if <= max_gap_s.
    """
    if len(beat_times_s) == 0:
        return np.array([]), np.array([]), np.array([])

    # Build uniform grid aligned to a 1/fs boundary
    t0 = np.floor(beat_times_s[0] * fs) / fs
    t1 = beat_times_s[-1]
    step = 1.0 / fs
    t_grid = np.arange(t0, t1 + step * 0.5, step, dtype=np.float64)

    # Interpolate (requires strictly increasing x)
    x = beat_times_s
    y = ibi_s
    # Use last/first values for extrap edges, then we'll mask them out by validity
    y_interp = np.interp(t_grid, x, y, left=y[0], right=y[-1])

    # Validity: distance to nearest beat <= max_gap_s
    # Find insertion indices and compute distance to prev/next sample
    idx = np.searchsorted(x, t_grid, side="left")
    prev_idx = np.clip(idx - 1, 0, len(x) - 1)
    next_idx = np.clip(idx, 0, len(x) - 1)
    dist_prev = np.abs(t_grid - x[prev_idx])
    dist_next = np.abs(x[next_idx] - t_grid)
    nearest = np.minimum(dist_prev, dist_next)
    valid = nearest <= max_gap_s

    ibi_grid = y_interp.astype(np.float64)
    ibi_grid[~valid] = np.nan
    return t_grid, ibi_grid, valid


# --------------------
# Bandpass (NumPy FFT)
# --------------------

def _naninterp_to_mean(x: np.ndarray) -> Tuple[np.ndarray, float]:
    """Replace NaNs by the mean of non-NaN values; if all NaN -> zeros."""
    x = x.copy()
    if np.all(np.isnan(x)):
        return np.zeros_like(x), 0.0
    m = np.nanmean(x)
    x[np.isnan(x)] = m
    return x, m


def bandpass_fft(
    x: np.ndarray,
    fs: float,
    fmin: float = 0.1,
    fmax: float = 0.5,
) -> np.ndarray:
    """
    Simple FFT bandpass: zero out frequencies outside [fmin, fmax].
    NaNs are replaced by mean prior to FFT (and result is mean-centered).
    """
    if x.size == 0:
        return x

    xin, mean_val = _naninterp_to_mean(x)
    # Remove mean to emphasize modulation
    xin = xin - np.mean(xin)

    X = np.fft.rfft(xin)
    freqs = np.fft.rfftfreq(len(xin), d=1.0/fs)

    passband = (freqs >= fmin) & (freqs <= fmax)
    X[~passband] = 0.0

    x_bp = np.fft.irfft(X, n=len(xin))
    # Re-introduce NaNs where original coverage was missing (optional)
    x_bp[np.isnan(x)] = np.nan
    return x_bp


# --------------------
# Windowed BPM estimate
# --------------------

def window_bpm_fft(
    x_bp: np.ndarray,
    fs: float,
    win_s: float = 15.0,
    min_coverage: float = 0.7,
    fmin: float = 0.1,
    fmax: float = 0.5,
    peak_prom: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute breathing rate per window via dominant frequency in [fmin, fmax].

    Returns:
      t0_idx: array of window start indices (integers)
      bpm: breaths/minute (float) np.nan where invalid
      valid: boolean per-window validity mask

    Rules:
      - coverage >= min_coverage (fraction of non-NaN in window)
      - peak prominence: max power >= peak_prom * median(in-band power)
    """
    if x_bp.size == 0:
        return np.array([]), np.array([]), np.array([])

    win_n = int(round(win_s * fs))
    if win_n < 2:
        raise ValueError("Window too short for given sampling rate.")

    nwin = x_bp.size // win_n
    bpm = np.full(nwin, np.nan, dtype=np.float64)
    valid = np.zeros(nwin, dtype=bool)
    t0_idx = np.arange(nwin) * win_n

    freqs_full = np.fft.rfftfreq(win_n, d=1.0/fs)
    band = (freqs_full >= fmin) & (freqs_full <= fmax)

    for w in range(nwin):
        seg = x_bp[w*win_n:(w+1)*win_n]
        if seg.size != win_n:
            break

        non_nan = np.sum(~np.isnan(seg))
        coverage = non_nan / float(win_n)
        if coverage < min_coverage:
            continue

        # Replace NaN by mean of available values in this window
        s = seg.copy()
        m = np.nanmean(s)
        s[np.isnan(s)] = m
        s = s - np.mean(s)

        S = np.fft.rfft(s)
        P = (S * np.conj(S)).real  # power spectrum

        inband_power = P[band]
        if inband_power.size == 0:
            continue

        peak_idx = np.argmax(inband_power)
        peak_power = float(inband_power[peak_idx])
        med_power = float(np.median(inband_power)) if inband_power.size > 0 else 0.0

        if med_power <= 0:
            prom_ok = True  # degenerate case
        else:
            prom_ok = peak_power >= (peak_prom * med_power)

        if not prom_ok:
            continue

        peak_freq = float(freqs_full[band][peak_idx])
        bpm[w] = peak_freq * 60.0
        valid[w] = True

    return t0_idx, bpm, valid


# --------------------
# Time helpers
# --------------------

def align_grid_start(t_first_s: float, step_s: float) -> float:
    """Align a start time to the next multiple of step_s."""
    return np.ceil(t_first_s / step_s) * step_s


def utc_seconds_to_pdt_hms_str(ts_s: float) -> str:
    """Convert epoch seconds to 'HH:MM:SS' in America/Los_Angeles."""
    # Using stdlib zoneinfo would be ideal, but it's not always available on Windows py<3.9
    # dateutil works widely and respects DST.
    from_zone = timezone.utc
    to_zone = tz.gettz("America/Los_Angeles")
    dt = datetime.fromtimestamp(ts_s, tz=from_zone).astimezone(to_zone)
    return dt.strftime("%H:%M:%S")
