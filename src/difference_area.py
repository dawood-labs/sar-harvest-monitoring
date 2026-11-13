"""Rank method variants by their DIFFERENCE AREA against the original report (the user's measure of the best model).

Difference area = acres of blocks where the SAR result and the original report disagree, counting only blocks of
at least MIN_ACRES that are at least 30 m wide (thin edge strips and stray pixels are not counted):
  missed = original report harvested, SAR standing        extra = original report standing, SAR harvested
Given per report date and added over all dates, for the whole area and for the test blocks only (the 3 km blocks
no model trained on). The smallest total wins.

Usage: python src/difference_area.py <area_name> <predictions.npz> [more.npz ...]
       npz keys: "<variant>|<report date>" or "pred_<report date>" (the first method)
Output: data/interim/<aoi>_difference_area_<area>.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from compare_detectors import split
from disagreement_eda import patches
from sar_features import load_pixels
from transfer_test import harvested_by

ACRE_PER_PIXEL = 100 / 4046.8564224
MIN_ACRES, WIDTH = 0.5, 3


def difference(y: np.ndarray, pred: np.ndarray, known: np.ndarray, d: dict) -> tuple[np.ndarray, np.ndarray]:
    """Pixels inside big missed blocks and inside big extra blocks."""
    return patches(known & y & ~pred, d, MIN_ACRES, WIDTH) > 0, patches(known & ~y & pred, d, MIN_ACRES, WIDTH) > 0


def main(area: str, *files: str) -> None:
    cfg, d = load_pixels(area)
    test, reports = split(d), cfg["report_dates"]
    labels = {r: harvested_by(d, n) for n, r in enumerate(reports, 1)}
    rows = []
    for f in files:
        z = np.load(f)
        names = sorted({k.split("|")[0] for k in z.files if "|" in k}) or ["first method (saved run)"]
        for name in names:
            row = {"variant": name, "source": Path(f).stem.split("_")[1]}
            for r in reports:
                pred = z[f"{name}|{r}"] if f"{name}|{r}" in z.files else z[f"pred_{r}"]
                missed, extra = difference(*labels[r][:1], pred, labels[r][1], d)
                row[f"missed_{r}"], row[f"extra_{r}"] = missed.sum() * ACRE_PER_PIXEL, extra.sum() * ACRE_PER_PIXEL
                row[f"test_{r}"] = ((missed | extra) & test).sum() * ACRE_PER_PIXEL
            row["missed_total"] = sum(row[f"missed_{r}"] for r in reports)
            row["extra_total"] = sum(row[f"extra_{r}"] for r in reports)
            row["difference_total"] = row["missed_total"] + row["extra_total"]
            row["difference_total_test_blocks"] = sum(row[f"test_{r}"] for r in reports)
            rows.append(row)
    df = pd.DataFrame(rows).sort_values("difference_total").round(1)
    df.to_csv(Path(cfg["paths"]["interim"]) / f"{cfg['aoi']}_difference_area_{area}.csv", index=False)
    show = ["variant", "source", "difference_total", "missed_total", "extra_total", "difference_total_test_blocks"] + \
           [f"{k}_{r}" for r in reports for k in ("missed", "extra")]
    print(df[show].to_string(index=False))


if __name__ == "__main__":
    main(*sys.argv[1:])
