"""Write cleaned harvest rasters from the probability rasters of predict_area.py (raster stage of postprocess.py).

Per report date, in order: harvested = probability >= the config's limits (earlier harvest stays harvested),
optional majority filter, small standing islands filled first or small harvested specks dropped first, pieces
under the smallest size take the class around them (a piece that is a whole cane patch is never changed).
The area is cleaned in tiles with a 128-pixel overlap, in parallel processes: every piece that can be changed is
smaller than the overlap, so a piece cut by a tile border is always seen whole or as clearly too big to change.
The cleaned rasters go to a folder of their own with the same file names, so every other script (scoring,
QGIS comparison, polygons) works on them through a config with that output_tag.

Usage: python src/clean_reports.py <area> <config.yaml> <smallest piece acres> <majority window px, 0 = off> <holes first 0|1> [tag]
Output: outputs/<aoi>_<area><output_tag>_clean<tag>/<aoi>_<area>_sar_harvest_<date>_v1.tif (+ .qml)
"""
from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import rasterio
import yaml
from rasterio.windows import Window

TILE, HALO = 2048, 128


def tile(args):
    import postprocess as pp
    r0, c0, cfg, area, src_dir, mmu_px, maj, holes = args
    pp.SURE, pp.MAYBE = float(cfg.get("grow_sure", 0.5)) - 1e-6, float(cfg.get("grow_maybe", 0.15)) - 1e-6
    win = Window(c0 - HALO, r0 - HALO, TILE + 2 * HALO, TILE + 2 * HALO)
    with rasterio.open(Path(cfg["paths"]["interim"]) / f"{cfg['aoi']}_cane_mask_{area}.tif") as m:
        H, W = m.height, m.width
        inside = m.read(1, window=win, boundless=True, fill_value=0) == 1
    h, w = min(TILE, H - r0), min(TILE, W - c0)
    sl = (slice(HALO, HALO + h), slice(HALO, HALO + w))
    if not inside[sl].any():
        return None
    locked, out = np.zeros(inside.shape, bool), {}
    for r in cfg["report_dates"]:
        with rasterio.open(Path(src_dir) / f"{cfg['aoi']}_{area}_sar_probability_{r}_v1.tif") as s:
            p = s.read(1, window=win, boundless=True, fill_value=255)
        raw = np.where(inside & (p != 255), p, 0).astype("float32") / 100
        cut = pp.clean_raster(raw, inside, locked, 0.0, mmu_px, maj, holes)
        locked = cut
        out[r] = np.where(inside, np.where(cut, 82, 81), 0).astype("uint8")[sl]
    return r0, c0, out


def main(area: str, cfg_path: str, mmu: str, maj: str, holes: str, tag: str = "") -> None:
    t0 = time.time()
    from export_qgis import CLASS_STYLE, write_qml
    cfg = yaml.safe_load(open(cfg_path))
    src_dir = Path("outputs") / f"{cfg['aoi']}_{area}{cfg.get('output_tag', '')}"
    out = Path(str(src_dir) + "_clean" + tag)
    out.mkdir(parents=True, exist_ok=True)
    with rasterio.open(Path(cfg["paths"]["interim"]) / f"{cfg['aoi']}_cane_mask_{area}.tif") as m:
        profile, H, W = m.profile, m.height, m.width
    profile.update(dtype="uint8", nodata=0, count=1, compress="deflate", tiled=True, blockxsize=256, blockysize=256, sparse_ok=True)
    mmu_px = max(1, int(round(float(mmu) * 4046.8564224 / 100)))
    files = {r: out / f"{cfg['aoi']}_{area}_sar_harvest_{r}_v1.tif" for r in cfg["report_dates"]}
    sinks = {r: rasterio.open(str(f) + ".tmp.tif", "w", **profile) for r, f in files.items()}
    acres = dict.fromkeys(files, 0.0)
    jobs = [(r0, c0, cfg, area, str(src_dir), mmu_px, int(maj), holes == "1") for r0 in range(0, H, TILE) for c0 in range(0, W, TILE)]
    with ProcessPoolExecutor(os.cpu_count()) as ex:
        for res in ex.map(tile, jobs):
            if res is None:
                continue
            r0, c0, per_date = res
            for r, img in per_date.items():
                sinks[r].write(img, 1, window=Window(c0, r0, img.shape[1], img.shape[0]))
                acres[r] += float((img == 82).sum()) * 100 / 4046.8564224
    for r, f in files.items():
        sinks[r].close()
        Path(str(f) + ".tmp.tif").replace(f)
        write_qml(f, CLASS_STYLE)
        print(r, "harvested acres", round(acres[r], 1))
    print(f"cleaned in {time.time() - t0:.0f} s -> {out}")


if __name__ == "__main__":
    main(*sys.argv[1:7])
