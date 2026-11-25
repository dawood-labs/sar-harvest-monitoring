"""Which region-growing limits fit an area best? Re-decides harvested from the saved probability rasters.

For every pair (sure, maybe): harvested = pixels >= sure plus touching pixels >= maybe, earlier harvest stays,
then the difference area against the original reports (blocks >= 0.5 acre) is measured. No model is rerun.
Pairs run in parallel processes (about 6 GB each on a large grid).

Usage: python src/grow_limits_test.py <area> <config.yaml>
Output: data/interim/<aoi>_grow_limits_<area><output_tag>.csv
"""
from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import yaml
from skimage.filters import apply_hysteresis_threshold

from disagreement_eda import patches
from score_reports import AC, labels

PAIRS = [(0.5, 0.5), (0.6, 0.6), (0.7, 0.7), (0.8, 0.8), (0.7, 0.5), (0.9, 0.7)]          # (sure, maybe); equal = no growing


def run(args) -> list[dict]:
    cfg_path, area, sure, maybe = args
    cfg = yaml.safe_load(open(cfg_path))
    with rasterio.open(Path(cfg["paths"]["interim"]) / f"{cfg['aoi']}_cane_mask_{area}.tif") as m:
        inside = m.read(1) == 1
    rows, cols = np.nonzero(inside)
    d, out, earlier = {"rows": rows, "cols": cols}, [], np.zeros(len(rows), bool)
    folder = Path("outputs") / f"{cfg['aoi']}_{area}{cfg.get('output_tag', '')}"
    for n, r in enumerate(cfg["report_dates"], 1):
        with rasterio.open(folder / f"{cfg['aoi']}_{area}_sar_probability_{r}_v1.tif") as s:
            p = s.read(1)
        p = np.where(inside & (p != 255), p, 0).astype("float32") / 100
        pred = apply_hysteresis_threshold(p, maybe - 1e-6, sure - 1e-6)[rows, cols] | earlier
        earlier = pred
        y, known = labels(cfg, area, n, rows, cols)
        missed, extra = patches(known & y & ~pred, d, 0.5, 3) > 0, patches(known & ~y & pred, d, 0.5, 3) > 0
        out.append({"sure": sure, "maybe": maybe, "report": r, "sar_harvested_acres": (pred & known).sum() * AC,
                    "original_harvested_acres": (y & known).sum() * AC, "missed_acres": missed.sum() * AC,
                    "extra_acres": extra.sum() * AC, "difference_acres": (missed.sum() + extra.sum()) * AC})
    return out


def main(area: str, cfg_path: str) -> None:
    cfg = yaml.safe_load(open(cfg_path))
    guarantee_gb = int(os.environ.get("MEM_GUARANTEE", 16 * 1024 ** 3)) / 1024 ** 3
    with ProcessPoolExecutor(max(1, min(len(PAIRS), os.cpu_count(), int(guarantee_gb * 0.6 // 6)))) as ex:
        df = pd.DataFrame([row for res in ex.map(run, [(cfg_path, area, s, m) for s, m in PAIRS]) for row in res])
    df.to_csv(Path(cfg["paths"]["interim"]) / f"{cfg['aoi']}_grow_limits_{area}{cfg.get('output_tag', '')}.csv", index=False)
    print(df.groupby(["sure", "maybe"], sort=False)[["missed_acres", "extra_acres", "difference_acres"]].sum().round(0).to_string())


if __name__ == "__main__":
    main(*sys.argv[1:3])
