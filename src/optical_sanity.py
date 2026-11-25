"""Where no original report exists: check a SAR report against one clear optical image.

Every crop-mask pixel gets its SAR class and its optical greenness (NDVI) on a date close to the report:
  green (NDVI > 0.5) = standing cane, bare (NDVI < 0.3) = harvested, in between = unclear.
SAR harvested pixels are split by the model's own probability: strong (>= 0.5) and weak (< 0.5; these were added
by region growing or carried over from an earlier report). Tiles run in parallel processes.

Usage: python src/optical_sanity.py <area> <config.yaml> <report date> <optical date>
Output: a table on screen and data/interim/<aoi>_optical_sanity_<area>_<report>.csv
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
from rasterio.windows import Window

ACRE, TILE = 100 / 4046.8564224, 2048


def tile(args):
    cfg_path, area, report, optical, r0, c0 = args
    cfg = yaml.safe_load(open(cfg_path))
    out = Path("outputs") / f"{cfg['aoi']}_{area}{cfg.get('output_tag', '')}"
    with rasterio.open(Path(cfg["paths"]["interim"]) / f"{cfg['aoi']}_cane_mask_{area}.tif") as m:
        win = Window(c0, r0, min(TILE, m.width - c0), min(TILE, m.height - r0))
        inside = m.read(1, window=win) == 1
    if not inside.any():
        return None
    with rasterio.open(out / f"{cfg['aoi']}_{area}_sar_harvest_{report}_v1.tif") as s:
        cut = s.read(1, window=win)[inside] == 82
    with rasterio.open(out / f"{cfg['aoi']}_{area}_sar_probability_{report}_v1.tif") as s:
        prob = s.read(1, window=win)[inside].astype("float32") / 100
    with rasterio.open(Path(cfg["paths"]["raw"]) / "s2" / f"{cfg['aoi']}_s2_{optical}.tif") as s:
        red, nir, clear = s.read([3, 5, 6], window=win).astype("float32")[:, inside]
    ok = (nir > 0) & (clear >= 60)
    ndvi = (nir - red) / (nir + red + 1e-6)
    sar = np.where(cut & (prob >= 0.5), 0, np.where(cut, 1, np.where(prob >= 0.15, 2, 3)))     # strong cut, weak cut, unsure standing, standing
    opt = np.where(ndvi > 0.5, 0, np.where(ndvi < 0.3, 2, 1))
    return np.bincount((sar * 3 + opt)[ok], minlength=12), int((~ok).sum())


def main(area: str, cfg_path: str, report: str, optical: str) -> None:
    cfg = yaml.safe_load(open(cfg_path))
    with rasterio.open(Path(cfg["paths"]["interim"]) / f"{cfg['aoi']}_cane_mask_{area}.tif") as m:
        jobs = [(cfg_path, area, report, optical, r0, c0) for r0 in range(0, m.height, TILE) for c0 in range(0, m.width, TILE)]
    with ProcessPoolExecutor(os.cpu_count()) as ex:
        res = [r for r in ex.map(tile, jobs, chunksize=4) if r is not None]
    t = np.sum([r[0] for r in res], axis=0).reshape(4, 3) * ACRE
    df = pd.DataFrame(t, index=["SAR harvested, model sure (prob >= 0.5)", "SAR harvested, model NOT sure (prob < 0.5)",
                                "SAR standing, model a little unsure (prob 0.15-0.5)", "SAR standing, model sure"],
                      columns=["optical green (standing)", "optical in between", "optical bare (harvested)"])
    df["acres"] = df.sum(axis=1)
    df["green share"] = df["optical green (standing)"] / df.acres
    df["bare share"] = df["optical bare (harvested)"] / df.acres
    df.to_csv(Path(cfg["paths"]["interim"]) / f"{cfg['aoi']}_optical_sanity_{area}{cfg.get('output_tag', '')}_{report}.csv")
    print(f"SAR report {report} against optical {optical}; pixels without clear optical: {sum(r[1] for r in res) * ACRE:,.0f} ac")
    print(df.round(2).to_string())


if __name__ == "__main__":
    main(*sys.argv[1:5])
