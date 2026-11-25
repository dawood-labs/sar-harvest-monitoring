"""Features per LOOK DIRECTION (descending, ascending) instead of per fixed track.

Why: one track does not cover a whole large area (its swath edge crosses it), so some cane has one track only.
A neighbouring track of the same direction fills that gap. Per direction the tracks are listed in order of
preference; every pixel takes the features of the first track that covers it. Tracks are never mixed inside a
pixel's series: all features of a pixel and direction come from ONE track and are relative to that pixel's own
earlier passes, so the different viewing angle of the fill track does not matter.
Per direction (35 columns):
  14 level features, 18 rate features   as before (date_free_features, rate_features) for the chosen track
  own_noise   spread (std, dB) of the pixel's own first four passes = how much this pixel moves when nothing happens
  z_drop      the pixel's fall since its season start divided by own_noise = the fall in the pixel's OWN units,
              so no fixed dB limit is needed anywhere
  age_days    days between the direction's last pass and the report date (lets the model trust the newer pass more)
"""
from __future__ import annotations

from datetime import date

import numpy as np

import transfer_test as tt
from retrain_test import rate_features

PER_DIRECTION = 35


def one_track(track: int, direction: str, d: dict, series: dict, texture: dict, report: str) -> tuple[np.ndarray, np.ndarray]:
    """(features of every pixel from this track, pixel is covered by the track's last pass)."""
    cfg1 = {"s1_tracks": {track: direction}}
    use = np.flatnonzero(d[f"dates_{track}"] <= report)
    vh = series[("vh", track)][use]
    with np.errstate(all="ignore"):
        noise = np.nanstd(vh[:4], axis=0)
        z = (vh[-1] - np.nanmedian(vh[:3], axis=0)) / np.maximum(noise, 0.3)
    age = (date.fromisoformat(str(report)) - date.fromisoformat(str(d[f"dates_{track}"][use[-1]]))).days
    x = np.column_stack([tt.date_free_features(cfg1, d, series, report), rate_features(cfg1, d, series, texture, report),
                         noise, z, np.full(len(noise), age, "float32")]).astype("float32")
    return x, np.isfinite(vh[-1])


def direction_features(directions: dict, tracks: dict, d: dict, series: dict, texture: dict, report: str) -> np.ndarray:
    """directions = {"des": [164, 91], "asc": [26, 99]} (order = preference); tracks = {track: "DESCENDING" | ...}."""
    blocks = []
    for name in ("des", "asc"):
        out, filled = None, None
        for t in directions[name]:
            if f"dates_{t}" not in d or (d[f"dates_{t}"] <= report).sum() < 4:
                continue
            x, has = one_track(int(t), tracks[int(t)], d, series, texture, report)
            if out is None:
                out, filled = np.where(has[:, None], x, np.nan).astype("float32"), has.copy()
            else:
                take = has & ~filled
                out[take], filled = x[take], filled | take
        blocks.append(out if out is not None else np.full((len(d["rows"]), PER_DIRECTION), np.nan, "float32"))
    return np.column_stack(blocks)
