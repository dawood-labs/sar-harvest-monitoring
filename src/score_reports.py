"""Score the SAR harvest rasters of an area against its original reports (rasters from rasterize_reports.py).

For areas too big for the in-memory pixel table: works straight from the rasters.
The n-th SAR report date is compared with the n-th original report. "Harvested by report n" = class 82 in any of
the original reports 1..n (they are incremental). A pixel is scored for report n only if the original report
knows its state then: harvested earlier, or 81 / 82 in report n (cloud and image-issue pixels are not scored).
Measures per date (acres):
  difference area = blocks >= 0.5 acre and >= 30 m wide where SAR and the original report disagree
                    (missed = original harvested, SAR standing; extra = the other way round)   <- the headline
  pixel numbers   = every disagreeing pixel, for information only (they include field edges)
Dates are scored in parallel processes.

Usage: python src/score_reports.py <area> <config.yaml> [<config.yaml> ...]     (configs differ in output_tag)
Output: data/interim/<aoi>_scores_vs_original_<area>.csv
"""
from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import yaml

from disagreement_eda import patches

AC = 100 / 4046.8564224


def labels(cfg: dict, area: str, n: int, rows: np.ndarray, cols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(harvested by original report n, pixel is known in report n) per crop-mask pixel."""
    interim = Path(cfg["paths"]["interim"])
    days = [str(x) for x in cfg["original_reports"]]
    cut = np.zeros(len(rows), bool)
    for day in days[:n]:
        with rasterio.open(interim / f"{cfg['aoi']}_original_{area}_{day}.tif") as s:
            v = s.read(1)[rows, cols]
        cut |= v == 82
    return cut, cut | (v == 81)


def score(args) -> dict:
    cfg_path, area, n = args
    cfg = yaml.safe_load(open(cfg_path))
    with rasterio.open(Path(cfg["paths"]["interim"]) / f"{cfg['aoi']}_cane_mask_{area}.tif") as m:
        rows, cols = np.nonzero(m.read(1) == 1)
    d = {"rows": rows, "cols": cols}
    r = cfg["report_dates"][n - 1]
    y, known = labels(cfg, area, n, rows, cols)
    tag = cfg.get("output_tag", "")
    with rasterio.open(Path("outputs") / f"{cfg['aoi']}_{area}{tag}" / f"{cfg['aoi']}_{area}_sar_harvest_{r}_v1.tif") as s:
        pred = s.read(1)[rows, cols] == 82
    missed, extra = patches(known & y & ~pred, d, 0.5, 3) > 0, patches(known & ~y & pred, d, 0.5, 3) > 0
    h, st = (y & known).sum(), (~y & known).sum()
    test = ((rows // 300 + cols // 300) % 2).astype(bool)          # 3 km checkerboard: blocks no local model trained on
    return {"run": tag.strip("_") or "transferred model", "sar_report": r, "original_report": list(cfg["original_reports"])[n - 1],
            "scored_acres": known.sum() * AC, "original_harvested_acres": h * AC, "sar_harvested_acres": (pred & known).sum() * AC,
            "missed_acres": missed.sum() * AC, "missed_share_of_harvested": missed.sum() / h,
            "extra_acres": extra.sum() * AC, "extra_share_of_standing": extra.sum() / st,
            "difference_acres": (missed.sum() + extra.sum()) * AC, "same_share": 1 - (missed.sum() + extra.sum()) / known.sum(),
            "test_blocks_difference_acres": ((missed | extra) & test).sum() * AC, "test_blocks_same_share": 1 - ((missed | extra) & test).sum() / (known & test).sum(),
            "pixel_missed_acres": (y & ~pred & known).sum() * AC, "pixel_extra_acres": (~y & pred & known).sum() * AC}


def main(area: str, *configs: str) -> None:
    jobs = [(c, area, n) for c in configs for n in range(1, len(yaml.safe_load(open(c))["report_dates"]) + 1)]
    with ProcessPoolExecutor(min(len(jobs), 8)) as ex:
        df = pd.DataFrame(list(ex.map(score, jobs)))
    cfg = yaml.safe_load(open(configs[0]))
    df.to_csv(Path(cfg["paths"]["interim"]) / f"{cfg['aoi']}_scores_vs_original_{area}.csv", index=False)
    pd.set_option("display.width", 250)
    print(df.round(3).to_string(index=False))
    print(df.groupby("run", sort=False)[["missed_acres", "extra_acres", "difference_acres", "test_blocks_difference_acres"]].sum().round(0).to_string())


if __name__ == "__main__":
    main(*sys.argv[1:])
