"""Relative dip features: judge every pixel against its OWN recent history, never against a fixed level.

Why (user's idea): a weak field and a healthy field sit at different backscatter levels, so one absolute
threshold is wrong for one of them. A harvest is a sudden dip relative to what that same pixel showed in the
passes just before, and it is large compared with that pixel's own pass-to-pass noise. A weak crop sliding down
slowly never shows such a dip.
For each pass i the dip is: mean of passes i and i+1 minus the mean of the 3 passes before i.
Per pixel (series already smoothed over 5 x 5 pixels inside the crop mask):
  best_dip     the deepest dip anywhere in the series (dB)
  best_dip_z   the same, divided by the pixel's own noise (robust spread of its pass-to-pass changes)
  best_single  dip of the single pass at that place (shows a dip seen by one pass only)
  passes_since how many passes ago the deepest dip happened
  still_down   latest pass minus the level before the deepest dip (a rain dip recovers, a harvest stays down)
  last_dip     latest pass minus the mean of the 3 passes before it (a fresh harvest)
  last_dip_z   the same in units of the pixel's own noise
Nothing here uses an absolute level or a calendar date.
"""
from __future__ import annotations

import numpy as np

NAMES = ["best_dip", "best_dip_z", "best_single", "passes_since", "still_down", "last_dip", "last_dip_z"]


def dip_features(series: np.ndarray) -> np.ndarray:
    """series = (passes, pixels) in dB. Returns (pixels, 7) float32."""
    x = np.where(np.isfinite(series), series, np.nanmean(series, axis=0, keepdims=True)).astype("float32")
    t, n = x.shape
    change = np.diff(x, axis=0)
    noise = np.maximum(1.4826 * np.median(np.abs(change - np.median(change, axis=0)), axis=0), 0.3)
    best = np.zeros(n, "float32")
    out = {k: np.zeros(n, "float32") for k in ("single", "since", "pre")}
    for i in range(3, t):
        pre = x[i - 3:i].mean(0)
        dip = x[i:i + 2].mean(0) - pre
        better = dip < best
        best = np.where(better, dip, best)
        for name, v in (("single", x[i] - pre), ("since", np.float32(t - 1 - i)), ("pre", pre)):
            out[name] = np.where(better, v, out[name])
    still = np.where(best < 0, x[-1] - out["pre"], 0)
    last = x[-1] - x[-4:-1].mean(0)
    return np.column_stack([best, best / noise, out["single"], out["since"], still, last, last / noise]).astype("float32")
