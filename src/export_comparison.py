"""Write the 4-class comparison map (SAR result against the original report) of one method variant for QGIS.

Classes: 1 both standing, 2 both harvested, 3 original report harvested / SAR standing,
         4 original report standing / SAR harvested; 0 = outside the crop mask or not covered by the original report.
One GeoTIFF per report date with a .qml style next to it (QGIS loads it by itself), plus one zip of all.

Usage: python src/export_comparison.py <area_name> <predictions.npz> "<variant name>" [version]
Output: outputs/<aoi>_<area>_sar_vs_original_<date>_<version>.tif + .qml, outputs/<aoi>_<area>_sar_vs_original_<version>.zip
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import numpy as np
import rasterio

from export_qgis import write_qml, write_raster
from sar_features import load_pixels
from transfer_test import harvested_by

ACRE_PER_PIXEL = 100 / 4046.8564224
STYLE = {1: ("#4a4a45", "both standing"), 2: ("#199e70", "both harvested"),
         3: ("#e66767", "original report harvested, SAR standing"), 4: ("#3987e5", "original report standing, SAR harvested")}


def main(area: str, npz: str, variant: str, version: str = "v2") -> None:
    cfg, d = load_pixels(area)
    out = Path("outputs")
    z = np.load(npz)
    with rasterio.open(Path(cfg["paths"]["interim"]) / f"{cfg['aoi']}_cane_mask_{area}.tif") as r:
        profile = r.profile
    files = []
    for n, report in enumerate(cfg["report_dates"], 1):
        y, known = harvested_by(d, n)
        pred = z[f"{variant}|{report}"]
        cls = np.where(~known, 0, np.where(~y & ~pred, 1, np.where(y & pred, 2, np.where(y, 3, 4))))
        p = out / f"{cfg['aoi']}_{area}_sar_vs_original_{report}_{version}.tif"
        write_raster(p, cls, d, profile, 0)
        write_qml(p, STYLE)
        files += [p, p.with_suffix(".qml")]
        print(report, {STYLE[k][1]: round(float((cls == k).sum() * ACRE_PER_PIXEL), 1) for k in STYLE})
    with zipfile.ZipFile(out / f"{cfg['aoi']}_{area}_sar_vs_original_{version}.zip", "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            zf.write(f, f.name)
    print("written", out / f"{cfg['aoi']}_{area}_sar_vs_original_{version}.zip")


if __name__ == "__main__":
    main(*sys.argv[1:5])
