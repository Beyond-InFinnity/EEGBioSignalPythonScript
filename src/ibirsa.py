"""
Signal processing utilities for deriving breathing rate from RR interval logs.

Functions provided here support the main script:
- Parsing JSON logs.
- Reconstructing beat times from RR interval lists.
- Cleaning implausible inter-beat intervals.
- Interpolating to a uniform grid.
- Bandpass filtering in the respiration frequency band.
- Extracting breathing rate via FFT peak analysis.
- Time conversion utilities for reporting results in PDT.

All functions are written to be dependency-light and reproducible.
"""

from __future__ import annotations
import json
from typing import List, Tuple
import numpy as np
from dateutil import tz
from datetime import datetime, timezone

# --------------------
# IO / Parsing
# --------------------

def load_stream(path: str) -> List[dict]:
    """
    Load the input JSON file containing heart rate and RR interval data.

    The file may be formatted either as:
      - A single JSON array of objects, or
      - Newline-delimited JSON objects.

    Returns
    -------
    List of dictionaries containing at least 'ts' and optionally 'hr', 'rr'.
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
            records = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
            return records


def detect_ts_unit(ts_values: np.ndarray) -> str:
    """
    Detect whether timestamps are reported in seconds or milliseconds.

    Heuristic: timestamps above 1e11 are assumed milliseconds,
    values around 1e9 are assumed seconds.
    """
    vmax = float(np.nanmax(ts_values))
    if vmax > 1e11:
        return "ms"
    elif vmax > 1e9:
        return "s"
    else:
        return "s"

