"""Score results on PURE BLOCKS of pixels, not on single pixels.

Why (user's rule): a validation sample must be a group of pixels that all have the same class in the original
report - all harvested or all standing - and it must be at least half an acre. Single pixels and blocks that
straddle a field edge are not valid samples, because at 10 m nobody can say what the right answer is there.
Blocks: the area is cut into squares of 5 x 5 pixels (50 m x 50 m = 0.62 acre). A square is a validation block
only if all 25 pixels are inside the crop mask, are covered by the original report and have the same class there.
SAR answer for a block: harvested if at least half of its 25 pixels are harvested in the SAR result (majority).
  harvested blocks found  = share of the harvested blocks that SAR calls harvested
  standing blocks kept    = share of the standing blocks that SAR calls standing
  SAR harvested is right  = of the blocks SAR calls harvested, the share that is harvested in the original report
  split blocks            = share of blocks where SAR is not clear-cut (between 20 % and 80 % of the pixels harvested)
Only test blocks are scored (the model never trained there).

Usage: python src/block_eval.py <area_name> <predictions.npz>      keys of the npz: "<variant>|<report date>"
Output: data/interim/<aoi>_block_scores_<area>_<npz name>.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from compare_detectors import split
from sar_features import load_pixels
from transfer_test import grid_shape, harvested_by

SIDE = 5                                    # pixels per block side: 5 x 5 pixels = 2,500 m2 = 0.62 acre
BLOCK_ACRES = SIDE * SIDE * 100 / 4046.8564224


def pure_blocks(y: np.ndarray, use: np.ndarray, d: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(block id per pixel, ids of pure harvested blocks, ids of pure standing blocks)."""
    width = grid_shape(d)[1] // SIDE + 1
    block = (d["rows"].astype("int64") // SIDE) * width + d["cols"] // SIDE
    n_use = np.bincount(block, weights=use)
    n_cut = np.bincount(block, weights=y & use)
    full = n_use == SIDE * SIDE                                     # every pixel of the square is a scored cane pixel
    return block, np.flatnonzero(full & (n_cut == SIDE * SIDE)), np.flatnonzero(full & (n_cut == 0))


def block_scores(y: np.ndarray, pred: np.ndarray, use: np.ndarray, d: dict) -> dict:
    """y, pred, use = per crop-mask pixel: original harvested, SAR harvested, pixel may be scored."""
    block, cut, stand = pure_blocks(y, use, d)
    share = np.bincount(block, weights=pred, minlength=block.max() + 1) / (SIDE * SIDE)
    found, wrong = (share[cut] >= 0.5).sum(), (share[stand] >= 0.5).sum()
    recall, kept, right = found / max(len(cut), 1), 1 - wrong / max(len(stand), 1), found / max(found + wrong, 1)
    both = np.concatenate([share[cut], share[stand]])
    return {"harvested_blocks": len(cut), "harvested_block_acres": len(cut) * BLOCK_ACRES, "standing_blocks": len(stand),
            "harvested_found": recall, "standing_kept": kept, "sar_harvested_right": right,
            "block_f1": 2 * recall * right / max(recall + right, 1e-9),
            "missed_blocks": int(len(cut) - found), "missed_acres": (len(cut) - found) * BLOCK_ACRES,
            "wrong_blocks": int(wrong), "wrong_acres": wrong * BLOCK_ACRES,
            "split_blocks_share": float(((both > 0.2) & (both < 0.8)).mean())}


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
    print(df.groupby("report")[["harvested_blocks", "harvested_block_acres", "standing_blocks"]].first().round(0).to_string())
    cols = ["harvested_found", "standing_kept", "sar_harvested_right", "block_f1", "missed_blocks", "missed_acres", "wrong_blocks",
            "wrong_acres", "split_blocks_share"]
    print(df.groupby("variant", sort=False)[cols].mean().round(4).to_string())
    print("written", out)


if __name__ == "__main__":
    main(*sys.argv[1:3])
