"""Features of the field and of the piece of field a pixel lies in, next to the pixel's own features.

Why (EDA of a small-field area): two thirds of the cane pixels lie within 20 m of a field edge, where a 5 x 5
mean mixes in the neighbours, and one pixel alone is too noisy (speckle about 2 dB). The mean over the whole
cane patch separates cut from standing clearly better. But a third of the cane lies in patches that are only
partly cut, so a second, smaller unit is used as well: segments = pieces of similar recent backscatter.
For each unit the mean backscatter of every pass is taken (in power, then dB) and the SAME level features as
for the pixel are computed from that mean series, so the model sees pixel, segment and patch side by side.

Used by train_local.py and predict_area.py when the config has  field_features: true.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

import transfer_test as tt

SEGMENT_SCALE = 3


def unit_mean(db: np.ndarray, unit: np.ndarray, n_units: int) -> np.ndarray:
    """Mean of every pass over each unit, given back per pixel. db = (passes, pixels) in dB, NaN = no data."""
    out = np.empty(db.shape, "float32")
    for i, row in enumerate(db):
        ok = np.isfinite(row)
        n = np.bincount(unit[ok], minlength=n_units)
        s = np.bincount(unit[ok], weights=10 ** (row[ok] / 10), minlength=n_units)
        with np.errstate(all="ignore"):
            out[i] = (10 * np.log10(s / n))[unit]
    return out


def field_features(model_cfg: dict, d: dict, series: dict, mask: np.ndarray, report: str) -> np.ndarray:
    """Level features of the pixel's segment (28) and of its cane patch (28) as of the report date, + 2 sizes."""
    rows, cols = d["rows"], d["cols"]
    patch_img, n_patch = ndimage.label(mask)
    patch = patch_img[rows, cols]
    seg = tt.segments(model_cfg, d, series, report, scale=SEGMENT_SCALE).astype("int64") * (n_patch + 1) + patch
    _, seg = np.unique(seg, return_inverse=True)                # a segment never crosses a patch border
    parts = []
    for unit, n in ((seg, int(seg.max()) + 1), (patch, n_patch + 1)):
        mean = {(b, s): unit_mean(d[f"{b}_{s}"], unit, n) for s in model_cfg["s1_tracks"] for b in ("vh", "vv")}
        parts += [tt.date_free_features(model_cfg, d, mean, report), np.log10(np.bincount(unit, minlength=n)[unit])[:, None].astype("float32")]
    return np.column_stack(parts)


def segments_by_patch(model_cfg: dict, d: dict, series: dict, mask: np.ndarray, report: str, scale: float = SEGMENT_SCALE) -> np.ndarray:
    """Segment id per crop-mask pixel, segmenting every cane patch on its own small image.

    Same idea as transfer_test.segments, but the segmentation runs only on the bounding box of each patch
    instead of the whole tile: cane is a few percent of a tile, so this is many times faster.
    """
    from skimage.segmentation import felzenszwalb
    rows, cols = d["rows"], d["cols"]
    layers = []
    for t in model_cfg["s1_tracks"]:
        for i in np.flatnonzero(d[f"dates_{t}"] <= report)[-2:]:
            img = np.full(mask.shape, -30, "float32")
            img[rows, cols] = np.nan_to_num(series[("vh", t)][i], nan=-30)
            layers.append(np.clip((img + 30) / 20, 0, 1))
    stack = np.dstack(layers)
    patch_img, n_patch = ndimage.label(mask)
    out, offset = np.zeros(mask.shape, "int64"), 0
    for k, box in enumerate(ndimage.find_objects(patch_img), 1):
        box = tuple(slice(max(s.start - 2, 0), s.stop + 2) for s in box)
        here = patch_img[box] == k
        seg = felzenszwalb(stack[box], scale=scale, sigma=0.5, min_size=10, channel_axis=-1) if here.sum() > 10 else np.zeros(here.shape, int)
        out[box][here] = seg[here] + offset
        offset += int(seg.max()) + 1
    return np.unique(out[rows, cols], return_inverse=True)[1]
