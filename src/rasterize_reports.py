"""Burn the original harvest reports of an area onto the crop-mask grid, one raster per report.

Why: scoring and training need the original report as pixels on exactly the same grid as the SAR data.
Each raster holds the report's class code per crop-mask pixel (81 remaining, 82 harvested, other codes such as
cloud or image issue are kept as they are), 0 = pixel not in that report. Reports are incremental, so later
steps build "harvested by date n" as the union of class 82 over reports 1..n.
The config lists the files:   original_reports: {"2025-12-30": path/to/report1.shp, ...}
Reports are burnt in parallel processes.

Usage: python src/rasterize_reports.py <area> <config.yaml>
Output: data/interim/<aoi>_original_<area>_<date>.tif and a table of acres per class
"""
from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import yaml
from rasterio.features import rasterize

ACRE_PER_PIXEL = 100 / 4046.8564224


def burn(args) -> tuple[str, dict]:
    day, shp, mask_file, out = args
    with rasterio.open(mask_file) as m:
        inside, profile = m.read(1) == 1, m.profile
    g = gpd.read_file(shp, engine="pyogrio").to_crs(profile["crs"])
    img = np.zeros(inside.shape, "uint16")
    for code in sorted(g.predicted.unique(), key=lambda c: c == 82):      # harvested is burnt last
        img = rasterize(((geom, int(code)) for geom in g.geometry[g.predicted == code]), out=img, transform=profile["transform"])
    img[~inside] = 0
    profile.update(dtype="uint16", nodata=0, count=1, compress="deflate", tiled=True, blockxsize=256, blockysize=256)
    tmp = out + ".tmp.tif"                       # write beside it, then rename: a reader never sees a half-written file
    with rasterio.open(tmp, "w", **profile) as dst:
        dst.write(img, 1)
    os.replace(tmp, out)
    codes, n = np.unique(img[inside], return_counts=True)
    return day, {int(c): round(float(k) * ACRE_PER_PIXEL, 1) for c, k in zip(codes, n)}


def main(area: str, cfg_path: str) -> None:
    cfg = yaml.safe_load(open(cfg_path))
    interim = Path(cfg["paths"]["interim"])
    mask_file = interim / f"{cfg['aoi']}_cane_mask_{area}.tif"
    jobs = [(str(day), shp, str(mask_file), str(interim / f"{cfg['aoi']}_original_{area}_{day}.tif")) for day, shp in cfg["original_reports"].items()]
    # about 2 GB per report on a large grid; stay well under the pod's guaranteed memory (other jobs may run too)
    guarantee_gb = int(os.environ.get("MEM_GUARANTEE", 16 * 1024 ** 3)) / 1024 ** 3
    with ProcessPoolExecutor(max(1, min(len(jobs), int(guarantee_gb * 0.4 // 2)))) as ex:
        for day, acres in ex.map(burn, jobs):
            print(day, "acres per class on the grid (0 = not in this report):", acres)


if __name__ == "__main__":
    main(*sys.argv[1:3])
