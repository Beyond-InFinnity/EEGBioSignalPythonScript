import unittest
import numpy as np
from datetime import datetime, timezone, timedelta

from src import ibirsa

class TestSyntheticRespiration(unittest.TestCase):
    def _make_synth_records(self, duration_s=300, hr_bpm=60.0, resp_bpm=12.0, ibi_amp_s=0.08):
        """
        Create a synthetic RR stream:
          - baseline IBI = 60/hr_bpm
          - sinusoidal modulation at resp_bpm with amplitude ibi_amp_s
          - Pack into records with 'ts' and 'rr' lists (ms) that end at ts.
        """
        base_ibi = 60.0 / hr_bpm  # seconds
        resp_hz = resp_bpm / 60.0

        # Create beats iteratively
        t = 0.0
        beats_t = []
        ibis = []
        while t < duration_s:
            ibi = base_ibi + ibi_amp_s * np.sin(2*np.pi*resp_hz * t)
            ibi = float(max(0.3, min(2.0, ibi)))
            beats_t.append(t)
            ibis.append(ibi)
            t += ibi

        # Convert into fake records with rr lists (batch into chunks of ~5)
        records = []
        for i in range(0, len(beats_t), 5):
            # 'ts' is the end time of last rr (in ms since epoch)
            chunk_ibis = ibis[i:i+5]
            if not chunk_ibis:
                continue
            ts_s = beats_t[i] + sum(chunk_ibis)
            ts_ms = int((1_700_000_000 + ts_s) * 1000)  # anchor at a modern epoch
            rr_ms = [int(round(v * 1000.0)) for v in chunk_ibis]
            records.append({"ts": ts_ms, "rr": rr_ms})
        return records

    def test_recover_rate(self):
        records = self._make_synth_records(duration_s=300, hr_bpm=60, resp_bpm=12, ibi_amp_s=0.08)
        ts_unit = ibirsa.detect_ts_unit(np.array([r["ts"] for r in records], dtype=float))
        beat_t, ibi = ibirsa.reconstruct_beats(records, ts_unit)
        ok = ibirsa.clean_ibi(ibi)
        beat_t = beat_t[ok]
        ibi = ibi[ok]
        t_grid, ibi_grid, valid = ibirsa.to_uniform_series(beat_t, ibi, fs=4.0, max_gap_s=2.0)
        x_bp = ibirsa.bandpass_fft(ibi_grid, fs=4.0, fmin=0.1, fmax=0.5)
        t0_idx, bpm, v = ibirsa.window_bpm_fft(x_bp, fs=4.0, win_s=15.0, min_coverage=0.7, fmin=0.1, fmax=0.5, peak_prom=1.5)
        # Check median within 1.5 bpm of 12
        med = np.nanmedian(bpm[v])
        self.assertTrue(abs(med - 12.0) < 1.5, f"Recovered median {med}")

if __name__ == "__main__":
    unittest.main()
