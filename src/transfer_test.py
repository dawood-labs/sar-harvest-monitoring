"""Date-transfer test: can a model trained on some report dates find harvest on a date it never saw?

Why: in production there is no report for the new date, so the model must carry over from older dates.
Steps:
  1. Labels: reference reports, but only pixels well inside a patch (>= 20 m from any class or mask edge).
  2. Features that do not name a date: the latest passes of each track counted back from the report date,
     each measured against the same pixel's early-season level and against the scene's standing-cane level.
  3. Train on all report dates but one, test on the left-out date, in spatial blocks the model never saw.
  4. Edge handling, three ideas compared: a smaller smoothing window, one decision per small SAR segment, and
     region growing (sure pixels are kept, and less sure pixels are added only where they touch a sure one).
  5. Rule: a pixel found cut on an earlier date stays cut.
Only crop-mask pixels are used.

Usage: python src/transfer_test.py <area_name>
Output: data/interim/<aoi>_transfer_scores_<area>.csv, data/interim/<aoi>_transfer_pred_<area>.npz
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage
from skimage.filters import apply_hysteresis_threshold
from skimage.segmentation import felzenszwalb
from sklearn.metrics import precision_recall_fscore_support

from compare_detectors import smoothed_series, split, tree_model
from sar_features import load_pixels

ACRE_PER_PIXEL = 100 / 4046.8564224
PASSES_BACK = 4          # passes per track counted back from the report date
SURE, MAYBE = 0.5, 0.15  # region growing: start from pixels >= SURE, grow into touching pixels >= MAYBE
log = logging.getLogger("transfer_test")


def grid_shape(d: dict) -> tuple[int, int]:
    return int(d["rows"].max()) + 1, int(d["cols"].max()) + 1


def harvested_by(d: dict, n: int) -> tuple[np.ndarray, np.ndarray]:
    """(label, known) for report n: harvested in report 1..n vs still standing; known = pixel is in a report."""
    g = d["group"]
    return (g >= 1) & (g <= n), g != 255


def near_edge(d: dict, y: np.ndarray, known: np.ndarray) -> np.ndarray:
    """True for pixels within 2 pixels (20 m) of a change of class or of the crop-mask border."""
    g = np.full(grid_shape(d), 255, "uint8")
    g[d["rows"], d["cols"]] = np.where(known, y, 255)
    return (ndimage.maximum_filter(g, 5) != ndimage.minimum_filter(g, 5))[d["rows"], d["cols"]]


def date_free_features(cfg: dict, d: dict, series: dict, report: str) -> np.ndarray:
    """Per track and band: the last passes before the report date, relative to the pixel's early-season level.

    Every pass is first shifted by the scene's standing-cane level (90th percentile over the crop mask), so
    rain or the slow seasonal drying, which move the whole scene, do not look like harvest.
    """
    cols = []
    with np.errstate(all="ignore"):
        for t in cfg["s1_tracks"]:
            use = np.flatnonzero(d[f"dates_{t}"] <= report)
            for b in ("vh", "vv"):
                s = series[(b, t)][use]
                s = s - np.nanpercentile(s, 90, axis=1)[:, None]
                base = np.nanmedian(s[:3], axis=0)
                last = s[-PASSES_BACK:]
                cols += [x - base for x in last] + [last[-1], np.nanmin(s, axis=0) - base, last[-1] - last[-2]]
    return np.column_stack(cols).astype("float32")


def segments(cfg: dict, d: dict, series: dict, report: str, scale: float) -> np.ndarray:
    """Small SAR segments (pieces of similar recent backscatter) as an id per crop-mask pixel."""
    layers = []
    for t in cfg["s1_tracks"]:
        use = np.flatnonzero(d[f"dates_{t}"] <= report)[-2:]
        for i in use:
            img = np.full(grid_shape(d), -30, "float32")
            img[d["rows"], d["cols"]] = np.nan_to_num(series[("vh", t)][i], nan=-30)
            layers.append(np.clip((img + 30) / 20, 0, 1))
    seg = felzenszwalb(np.dstack(layers), scale=scale, sigma=0.5, min_size=10, channel_axis=-1)
    return seg[d["rows"], d["cols"]]


def by_segment(prob: np.ndarray, seg: np.ndarray) -> np.ndarray:
    """Mean probability of each segment, given back to its pixels."""
    total = np.bincount(seg, weights=prob)
    return (total / np.maximum(np.bincount(seg), 1))[seg]


def grow(prob: np.ndarray, d: dict) -> np.ndarray:
    """Region growing: sure pixels plus the less sure pixels connected to them."""
    img = np.zeros(grid_shape(d), "float32")
    img[d["rows"], d["cols"]] = prob
    return apply_hysteresis_threshold(img, MAYBE, SURE)[d["rows"], d["cols"]]


def score(name: str, report: str, pred: np.ndarray, y: np.ndarray, m: np.ndarray, edge: np.ndarray, group: np.ndarray, n: int) -> dict:
    row = {"report": report, "variant": name}
    for part, sel in (("all", m), ("inside", m & ~edge), ("edge", m & edge)):
        p, r, f1, _ = precision_recall_fscore_support(y[sel], pred[sel], average="binary", zero_division=0)
        row.update({f"precision_{part}": p, f"recall_{part}": r, f"f1_{part}": f1})
    row["acres_report"] = y[m].sum() * ACRE_PER_PIXEL
    row["acres_sar"] = pred[m].sum() * ACRE_PER_PIXEL
    row["recall_newest_cut"] = pred[m & (group == n)].mean()
    log.info("%s %-28s F1 %.3f (inside %.3f, edge %.3f)  P %.3f R %.3f  acres %.0f vs report %.0f  newest-cut recall %.2f",
             report, name, row["f1_all"], row["f1_inside"], row["f1_edge"], row["precision_all"], row["recall_all"],
             row["acres_sar"], row["acres_report"], row["recall_newest_cut"])
    return row


def main(area: str) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    cfg, d = load_pixels(area)
    test, rng, group = split(d), np.random.default_rng(cfg["seed"]), d["group"]
    reports = cfg["report_dates"]
    rows, keep, final = [], {}, {}
    for size in (5, 3):
        series = smoothed_series(cfg, d, size)
        feats = {r: date_free_features(cfg, d, series, r) for r in reports}
        labels = {r: harvested_by(d, n) for n, r in enumerate(reports, 1)}
        edges = {r: near_edge(d, *labels[r]) for r in reports}
        probs = {}
        for n, held in enumerate(reports, 1):
            xs, ys = [], []
            for r in reports:
                if r == held:
                    continue
                y, known = labels[r]
                idx = np.flatnonzero(known & ~test & ~edges[r])          # clean labels, training blocks only
                idx = rng.choice(idx, min(len(idx), 300_000), replace=False)
                xs.append(feats[r][idx]); ys.append(y[idx])
            model = tree_model(cfg).fit(np.vstack(xs), np.concatenate(ys))
            probs[held] = model.predict_proba(feats[held])[:, 1]
            y, known = labels[held]
            m = known & test
            rows.append({**score(f"pixel, {size}x{size} mean", held, probs[held] >= 0.5, y, m, edges[held], group, n), "smooth": size})
            if size == 3:
                seg_prob = by_segment(probs[held], segments(cfg, d, series, held, scale=3))
                rows.append({**score("segments, 3x3 mean", held, seg_prob >= 0.5, y, m, edges[held], group, n), "smooth": size})
            else:
                keep[held] = probs[held]
                rows.append({**score("5x5 + region growing", held, grow(probs[held], d), y, m, edges[held], group, n), "smooth": size})
    earlier = np.zeros(len(group), bool)                 # once cut, stays cut (in date order)
    for n, r in enumerate(reports, 1):
        y, known = harvested_by(d, n)
        final[r] = grow(keep[r], d) | earlier
        rows.append({**score("5x5 + region growing + stays cut", r, final[r], y, known & test, near_edge(d, y, known), group, n), "smooth": 5})
        earlier = final[r]
    out = Path(cfg["paths"]["interim"])
    pd.DataFrame(rows).round(4).to_csv(out / f"{cfg['aoi']}_transfer_scores_{area}.csv", index=False)
    np.savez_compressed(out / f"{cfg['aoi']}_transfer_pred_{area}.npz", **{f"pred_{r}": final[r] for r in reports},
                        **{f"prob_{r}": keep[r].astype("float32") for r in reports})
    log.info("written scores and predictions to %s", out)


if __name__ == "__main__":
    main(sys.argv[1])
