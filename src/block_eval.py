"""Score results on BLOCKS of pixels (patches of at least half an acre), not on single pixels.

Why (user's rule): single pixels and thin strips along field edges cannot be judged at 10 m, and they are not
what a harvest report is about. A report is right or wrong field by field. So the unit here is a patch:
  original patch = connected pixels the original report calls harvested, at least MIN_ACRES
  SAR patch      = connected pixels the SAR result calls harvested, at least MIN_ACRES
A patch is "matched" when at least half of its pixels have the same class in the other map.
  found   = share of original harvested patches that SAR also calls harvested   (by count and by acres)
  correct = share of SAR harvested patches that the original report also calls harvested
Smaller patches are left out on both sides. Thin strips are removed first with a 3 x 3 opening, so a patch is
at least 30 m wide. Only test blocks are scored (the model never trained there).
Also reported: big disagreement patches (>= MIN_ACRES and >= 50 m wide) of each kind, in acres and count.

Usage: python src/block_eval.py <area_name> <predictions.npz>      keys of the npz: "<variant>|<report date>"
Output: data/interim/<aoi>_block_scores_<area>_<npz name>.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

from compare_detectors import split
from sar_features import load_pixels
from transfer_test import grid_shape, harvested_by

ACRE_PER_PIXEL = 100 / 4046.8564224
MIN_ACRES = 0.5


def to_grid(v: np.ndarray, d: dict) -> np.ndarray:
    img = np.zeros(grid_shape(d), bool)
    img[d["rows"], d["cols"]] = v
    return img


def patch_match(a: np.ndarray, b: np.ndarray, width: int = 3) -> tuple[int, int, float, float]:
    """Patches of a (>= MIN_ACRES, >= width pixels wide): how many, how many have >= half their pixels in b, and acres."""
    lab, n = ndimage.label(ndimage.binary_opening(a, np.ones((width, width), bool)))
    if n == 0:
        return 0, 0, 0.0, 0.0
    size = np.bincount(lab.ravel(), minlength=n + 1)[1:]
    hit = np.bincount(lab.ravel(), weights=b.ravel(), minlength=n + 1)[1:]
    big = size * ACRE_PER_PIXEL >= MIN_ACRES
    ok = big & (hit >= 0.5 * size)
    return int(big.sum()), int(ok.sum()), float(size[big].sum() * ACRE_PER_PIXEL), float(size[ok].sum() * ACRE_PER_PIXEL)


def block_scores(y: np.ndarray, pred: np.ndarray, use: np.ndarray, d: dict) -> dict:
    """y, pred, use = per crop-mask pixel: original harvested, SAR harvested, pixel is scored."""
    orig, sar = to_grid(y & use, d), to_grid(pred & use, d)
    n_o, found, ac_o, ac_found = patch_match(orig, sar)
    n_s, correct, ac_s, ac_correct = patch_match(sar, orig)
    recall, precision = ac_found / max(ac_o, 1e-9), ac_correct / max(ac_s, 1e-9)
    n_miss, _, ac_miss, _ = patch_match(orig & ~sar, orig, width=5)
    n_extra, _, ac_extra, _ = patch_match(sar & ~orig, sar, width=5)
    return {"original_patches": n_o, "found": found, "found_share_acres": recall, "found_share_count": found / max(n_o, 1),
            "sar_patches": n_s, "correct": correct, "correct_share_acres": precision, "correct_share_count": correct / max(n_s, 1),
            "block_f1": 2 * precision * recall / max(precision + recall, 1e-9),
            "missed_blocks": n_miss, "missed_block_acres": ac_miss, "extra_blocks": n_extra, "extra_block_acres": ac_extra}


def main(area: str, npz: str) -> None:
    cfg, d = load_pixels(area)
    z, test = np.load(npz), split(d)
    labels = {r: harvested_by(d, n) for n, r in enumerate(cfg["report_dates"], 1)}
    rows = []
    for key in z.files:
        if "|" not in key:
            continue
        variant, report = key.split("|")
        y, known = labels[report]
        rows.append({"variant": variant, "report": report, **block_scores(y, z[key], known & test, d)})
    df = pd.DataFrame(rows)
    out = Path(cfg["paths"]["interim"]) / f"{cfg['aoi']}_block_scores_{area}_{Path(npz).stem.split('_')[1]}.csv"
    df.to_csv(out, index=False)
    cols = ["found_share_acres", "correct_share_acres", "block_f1", "found_share_count", "correct_share_count",
            "missed_blocks", "missed_block_acres", "extra_blocks", "extra_block_acres"]
    print(df.groupby("variant", sort=False)[cols].mean().round(3).to_string())
    print("written", out)


if __name__ == "__main__":
    main(*sys.argv[1:3])
