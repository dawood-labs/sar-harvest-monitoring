"""QGIS rasters that compare the SAR harvest rasters of an area with its original reports, per report date.

Classes: 1 both standing, 2 both harvested, 3 missed (original harvested, SAR standing), 4 extra (original
standing, SAR harvested), 0 = not scored (outside the crop mask, cloud, image issue). Styled like the other
comparison maps. Dates are written in parallel processes; a zip with the SAR rasters is made for download.

Usage: python src/export_vs_original.py <area> <config.yaml>
Output: outputs/<aoi>_<area><output_tag>/<aoi>_<area>_sar_vs_original_<date>_v1.tif (+ .qml), ..._qgis_v1.zip
"""
from __future__ import annotations

import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import rasterio
import yaml

from export_comparison import STYLE
from export_qgis import write_qml


def one(args) -> str:
    cfg_path, area, n = args
    cfg = yaml.safe_load(open(cfg_path))
    interim, r = Path(cfg["paths"]["interim"]), cfg["report_dates"][n - 1]
    out = Path("outputs") / f"{cfg['aoi']}_{area}{cfg.get('output_tag', '')}"
    with rasterio.open(out / f"{cfg['aoi']}_{area}_sar_harvest_{r}_v1.tif") as s:
        pred, profile = s.read(1), s.profile
    cut = np.zeros(pred.shape, bool)
    for day in list(cfg["original_reports"])[:n]:
        with rasterio.open(interim / f"{cfg['aoi']}_original_{area}_{day}.tif") as s:
            v = s.read(1)
        cut |= v == 82
    known = (cut | (v == 81)) & (pred > 0)
    sar = pred == 82
    cls = np.where(~known, 0, np.where(~cut & ~sar, 1, np.where(cut & sar, 2, np.where(cut, 3, 4)))).astype("uint8")
    p = out / f"{cfg['aoi']}_{area}_sar_vs_original_{r}_v1.tif"
    with rasterio.open(p, "w", **{**profile, "dtype": "uint8", "nodata": 0}) as dst:
        dst.write(cls, 1)
    write_qml(p, STYLE)
    return str(p)


def main(area: str, cfg_path: str) -> None:
    cfg = yaml.safe_load(open(cfg_path))
    with ProcessPoolExecutor(len(cfg["report_dates"])) as ex:
        files = list(ex.map(one, [(cfg_path, area, n) for n in range(1, len(cfg["report_dates"]) + 1)]))
    out = Path(files[0]).parent
    z = out / f"{cfg['aoi']}_{area}_qgis_v1.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as f:
        for p in sorted(out.iterdir()):
            if p.suffix in (".tif", ".qml") or p.name.endswith("summary_v1.csv"):
                f.write(p, p.name)
    print("written", z, round(z.stat().st_size / 1e6, 1), "MB")


if __name__ == "__main__":
    main(*sys.argv[1:3])
