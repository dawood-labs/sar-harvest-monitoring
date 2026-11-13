"""Describe the SHAPE of each pixel's SAR history: one sharp step (a harvest) or a slow slide (a weak crop).

Why: a harvested field and a weak, thinning crop can end at the same low backscatter. What separates them is how
they got there: harvest is one sudden fall that then stays; a weak crop slides down over many passes.
For every pixel the best single step is fitted to the series (all passes up to the report date):
  step          mean after the step minus mean before it (dB; a harvest is strongly negative)
  step_share    how much of the series' variation that one step explains (1 = a clean step, 0 = no step)
  passes_since  how many passes ago the step happened (0 = at the last pass)
  slope_before  trend of the passes before the step (dB per pass; a slide shows here, a harvest does not)
  biggest_fall  the largest fall between two neighbouring passes
  after_level   mean after the step, against the scene's standing-cane level
Positions are counted back from the report date, so the features carry over to any date.
"""
from __future__ import annotations

import numpy as np

NAMES = ["step", "step_share", "passes_since", "slope_before", "biggest_fall", "after_level"]


def step_features(series: np.ndarray) -> np.ndarray:
    """series = (passes, pixels) in dB, already shifted by the scene level. Returns (pixels, 6) float32."""
    x = np.where(np.isfinite(series), series, np.nanmean(series, axis=0, keepdims=True)).astype("float32")
    t, n = x.shape
    total, csum, csq = x.sum(0), np.cumsum(x, 0), (x ** 2).sum(0)
    sse_all = csq - total ** 2 / t
    best_sse = np.full(n, np.inf, "float32")
    best = {k: np.zeros(n, "float32") for k in ("step", "k", "before", "after")}
    for k in range(2, t):                               # step between pass k-1 and k; at least 2 passes before
        before, after = csum[k - 1] / k, (total - csum[k - 1]) / (t - k)
        sse = csq - k * before ** 2 - (t - k) * after ** 2
        better = sse < best_sse
        best_sse = np.where(better, sse, best_sse)
        for name, v in (("step", after - before), ("k", np.float32(k)), ("before", before), ("after", after)):
            best[name] = np.where(better, v, best[name])
    k = best["k"].astype(int)
    # Trend before the step: least-squares slope over passes 0..k-1, done with running sums.
    idx = np.arange(t, dtype="float32")[:, None]
    sxy, sx, sxx = np.cumsum(idx * x, 0), np.cumsum(idx, 0), np.cumsum(idx ** 2, 0)
    pick = (k - 1, np.arange(n))
    kk = k.astype("float32")
    denom = kk * np.take_along_axis(sxx, (k - 1)[None, :], 0)[0] - np.take_along_axis(sx, (k - 1)[None, :], 0)[0] ** 2
    slope = (kk * sxy[pick] - np.take_along_axis(sx, (k - 1)[None, :], 0)[0] * csum[pick]) / np.maximum(denom, 1e-6)
    share = 1 - best_sse / np.maximum(sse_all, 1e-6)
    return np.column_stack([best["step"], share, t - kk, slope, np.diff(x, axis=0).min(0), best["after"]]).astype("float32")
