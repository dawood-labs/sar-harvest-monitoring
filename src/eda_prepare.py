"""Build the EDA table for one area: every crop-mask pixel with its report group and its SAR time series.

Why: all EDA figures and tests read this one cached file instead of re-reading 21 rasters each time.
Report groups come from the reference harvest reports (they are incremental):
  0 = still standing at the last report, 1..n = reported harvested in report 1..n, 255 = in the mask but in no report.
Only pixels inside the crop mask are read.

Usage: python src/eda_prepare.py <area_name> [configs/config.yaml]      e.g. test_tile
Output: data/interim/<aoi>_eda_pixels_<area>.npz  (rows, cols, group, dates_<track>, vv_<track>, vh_<track>; dB x 100 int16)
"""
from __future__ import annotations

import glob
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import yaml
from rasterio.features import rasterize

from resources import workers

ACRE_PER_PIXEL = 100 / 4046.8564224
log = logging.getLogger("eda_prepare")


def read_masked(path: str, inside: np.ndarray) -> np.ndarray:
    """VV and VH of one pass at the crop-mask pixels, as dB x 100 (int16, -32768 = no data)."""
    with rasterio.open(path) as r:
        a = r.read([1, 2])[:, inside]
    return np.where(np.isfinite(a), np.round(a * 100), -32768).astype("int16")


def main(area: str, cfg_path: str = "configs/config.yaml") -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    cfg = yaml.safe_load(open(cfg_path))
    interim, raw = Path(cfg["paths"]["interim"]), Path(cfg["paths"]["raw"])
    with rasterio.open(interim / f"{cfg['aoi']}_cane_mask_{area}.tif") as r:
        inside, tr, shape, crs = r.read(1) == 1, r.transform, r.shape, r.crs
    rows, cols = np.nonzero(inside)

    group = np.full(shape, 255, "uint8")
    codes = cfg["classes"]
    reports = sorted(f for f in glob.glob(str(Path(cfg["paths"]["reference_reports"]) / "*.shp")) if f != cfg["crop_mask"])
    bounds = rasterio.transform.array_bounds(*shape, tr)
    for i, f in enumerate(reports, 1):
        g = gpd.read_file(f).to_crs(crs).cx[bounds[0]:bounds[2], bounds[1]:bounds[3]]
        cut = rasterize([(x, 1) for x in g.geometry[g.predicted == codes["harvested"]]], out_shape=shape, transform=tr, dtype="uint8")
        group[(cut == 1) & (group == 255)] = i
        if i == len(reports):
            stand = rasterize([(x, 1) for x in g.geometry[g.predicted == codes["remaining_cane"]]], out_shape=shape, transform=tr, dtype="uint8")
            group[(stand == 1) & (group == 255)] = 0
    group = group[inside]
    for v in np.unique(group):
        log.info("group %3d: %9.1f acres", v, (group == v).sum() * ACRE_PER_PIXEL)

    out = {"rows": rows.astype("int16"), "cols": cols.astype("int16"), "group": group}
    for track, direction in cfg["s1_tracks"].items():
        files = sorted(glob.glob(str(raw / "s1" / f"{cfg['aoi']}_s1_ro{int(track):03d}_{direction[:3].lower()}_*.tif")))
        with ThreadPoolExecutor(workers()) as ex:      # reading + decompression releases the GIL
            stack = np.stack(list(ex.map(lambda f: read_masked(f, inside), files)))      # (dates, 2, pixels)
        out[f"dates_{track}"] = np.array([Path(f).stem[-10:] for f in files])
        out[f"vv_{track}"], out[f"vh_{track}"] = stack[:, 0], stack[:, 1]
        log.info("track %s: %d dates x %d pixels", track, *stack[:, 0].shape)
    path = interim / f"{cfg['aoi']}_eda_pixels_{area}.npz"
    np.savez_compressed(path, **out)
    log.info("written %s (%.0f MB)", path, path.stat().st_size / 1e6)


if __name__ == "__main__":
    main(*sys.argv[1:3])
