"""Does SAR find harvested fields that look black (burnt) in the optical image?

Why: part of the harvested cane is burnt (before or after cutting) and shows black in Sentinel-2; the original
report counts it as harvested. This script splits the harvested pixels of each original report by how they look
in the optical image of the report date and of the optical date before it, and gives the SAR recall of each kind.
  black      = bare (NDVI < 0.35) and dark (near-infrared B8 < 2000)
  light bare = bare and bright
Only pixels inside fields (>= 20 m from a class or mask edge) are scored, so edge effects do not blur the answer.

Usage: python src/burnt_check.py <area_name>
Output: data/interim/<aoi>_burnt_check_<area>.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from sar_features import load_pixels
from transfer_test import harvested_by, near_edge

ACRE_PER_PIXEL = 100 / 4046.8564224
BARE_NDVI, GREEN_NDVI, DARK_NIR = 0.35, 0.5, 2000


def main(area: str) -> None:
    cfg, d = load_pixels(area)
    interim = Path(cfg["paths"]["interim"])
    pred = np.load(interim / f"{cfg['aoi']}_transfer_pred_{area}.npz")
    files = sorted((Path(cfg["paths"]["raw"]) / "s2").glob(f"{cfg['aoi']}_s2_*.tif"))
    dates = [f.stem[-10:] for f in files]

    def optical(date: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        with rasterio.open(files[dates.index(date)]) as r:
            red, nir, clear = r.read([3, 5, 6]).astype("float32")[:, d["rows"], d["cols"]]
        return (nir - red) / (nir + red + 1e-6), nir, clear >= 60

    rows = []
    for n, report in enumerate(cfg["report_dates"], 1):
        y, known = harvested_by(d, n)
        inside, p = ~near_edge(d, y, known), pred[f"pred_{report}"]
        ndvi, nir, _ = optical(report)
        before = dates[dates.index(report) - 1]
        ndvi0, nir0, clear0 = optical(before)
        black, light = y & (ndvi < BARE_NDVI) & (nir < DARK_NIR), y & (ndvi < BARE_NDVI) & (nir >= DARK_NIR)
        kinds = {"black, all": black & inside, "light bare, all": light & inside,
                 "black now, green on the optical date before": black & inside & clear0 & (ndvi0 > GREEN_NDVI),
                 "black now, light bare before (burnt after cutting)": black & inside & clear0 & (ndvi0 < BARE_NDVI) & (nir0 >= DARK_NIR),
                 "black now, black before": black & inside & clear0 & (ndvi0 < BARE_NDVI) & (nir0 < DARK_NIR),
                 "light bare now, green before (fresh cut, not burnt)": light & inside & clear0 & (ndvi0 > GREEN_NDVI)}
        for name, m in kinds.items():
            rows.append({"report": report, "optical_date_before": before, "kind": name, "acres_inside_fields": round(m.sum() * ACRE_PER_PIXEL, 1),
                         "sar_recall": round(float(p[m].mean()), 3) if m.any() else np.nan,
                         "harvested_acres_total": round(y.sum() * ACRE_PER_PIXEL, 1),
                         "black_share_of_harvested": round(float(black.sum() / y.sum()), 3)})
    df = pd.DataFrame(rows)
    df.to_csv(interim / f"{cfg['aoi']}_burnt_check_{area}.csv", index=False)
    print(df.drop(columns=["optical_date_before"]).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1])
