"""Build the pixel table for the EDA of a large area, straight from the rasters (no model involved).

Why: before any model, we must see how harvest looks in this area's own SAR data. The table holds a random
sample of crop-mask pixels with everything the EDA needs:
  vv_<track>, vh_<track>        raw dB of every pass (passes x pixels), NaN = no data
  vv3_, vh3_, vv5_, vh5_        the same after a 3 x 3 and a 5 x 5 in-mask mean (to judge the averaging window)
  code_<report date>            class code of the original report (81, 82, cloud, image issue; 0 = not in it)
  inner_<report date>           True if all 25 pixels of the 5 x 5 block around the pixel have the same state
                                (harvested by that report / still standing) = a pure, reliable label
  edge_m                        distance to the edge of the cane patch in metres (capped at 300 m)
  patch_acres                   size of the cane patch the pixel lies in
Only pixels inside the crop mask are read. Tiles run in parallel processes.

Usage: python src/eda_extract.py <area> <config.yaml> [share of pixels, default 0.2]
Output: data/interim/<aoi>_eda_table_<area>.npz, data/interim/<aoi>_patch_acres_<area>.tif
"""
from __future__ import annotations

import logging
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import rasterio
import yaml
from rasterio.windows import Window
from scipy import ndimage

log = logging.getLogger("eda_extract")
TILE, HALO = 2048, 64
G: dict = {}


def init(cfg: dict, area: str, files: dict, share: float) -> None:
    import predict_area
    import sar_features
    sar_features.workers = predict_area.workers = lambda *a, **k: 2
    G.update(cfg=cfg, area=area, files=files, share=share)


def tile(rc: tuple[int, int]):
    from predict_area import read_tile
    from sar_features import smooth_in_mask
    r0, c0 = rc
    cfg, area = G["cfg"], G["area"]
    interim = Path(cfg["paths"]["interim"])
    win = Window(c0 - HALO, r0 - HALO, TILE + 2 * HALO, TILE + 2 * HALO)
    with rasterio.open(interim / f"{cfg['aoi']}_cane_mask_{area}.tif") as m:
        H, W = m.height, m.width
        mask = m.read(1, window=win, boundless=True, fill_value=0) == 1
    core = np.zeros_like(mask)
    core[HALO:HALO + min(TILE, H - r0), HALO:HALO + min(TILE, W - c0)] = True
    rows, cols = np.nonzero(mask)
    pick = np.flatnonzero(core[rows, cols])
    rng = np.random.default_rng(r0 * 100003 + c0)
    pick = pick[rng.random(len(pick)) < G["share"]]
    if len(pick) == 0:
        return None
    out = {"rows": (rows[pick] + r0 - HALO).astype("int32"), "cols": (cols[pick] + c0 - HALO).astype("int32")}
    out["edge_m"] = (np.minimum(ndimage.distance_transform_edt(mask), 30) * 10)[rows[pick], cols[pick]].astype("float32")
    with rasterio.open(interim / f"{cfg['aoi']}_patch_acres_{area}.tif") as s:
        out["patch_acres"] = s.read(1, window=win, boundless=True, fill_value=0)[rows[pick], cols[pick]]
    r32, c32 = rows.astype("int32"), cols.astype("int32")
    for t, (dates, files) in G["files"].items():
        a = read_tile(list(files), win, mask)                        # passes, 2 bands, pixels
        for b, name in enumerate(("vv", "vh")):
            out[f"{name}_{t}"] = a[:, b][:, pick]
            for k in (3, 5):
                out[f"{name}{k}_{t}"] = smooth_in_mask(a[:, b], r32, c32, k)[:, pick]
    cut = np.zeros(mask.shape, bool)
    block = np.ones((5, 5), bool)
    for day in cfg["original_reports"]:
        with rasterio.open(interim / f"{cfg['aoi']}_original_{area}_{day}.tif") as s:
            v = s.read(1, window=win, boundless=True, fill_value=0)
        cut |= v == 82
        known = cut | (v == 81)
        pure = ndimage.binary_erosion(cut & known, block) | ndimage.binary_erosion(~cut & known, block)
        out[f"code_{day}"] = v[rows[pick], cols[pick]].astype("uint16")
        out[f"inner_{day}"] = pure[rows[pick], cols[pick]]
    return out


def main(area: str, cfg_path: str, share: str = "0.2") -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    t0 = time.time()
    from predict_area import track_files
    cfg = yaml.safe_load(open(cfg_path))
    cfg["original_reports"] = {str(k): v for k, v in cfg["original_reports"].items()}
    interim = Path(cfg["paths"]["interim"])
    mask_file = interim / f"{cfg['aoi']}_cane_mask_{area}.tif"
    patch_file = interim / f"{cfg['aoi']}_patch_acres_{area}.tif"
    with rasterio.open(mask_file) as m:
        H, W, profile = m.height, m.width, m.profile
        if not patch_file.exists():                                 # size of every cane patch, once for the whole grid
            lab, n = ndimage.label(m.read(1) == 1)
            acres = (np.bincount(lab.ravel()) * 100 / 4046.8564224).astype("float32")
            acres[0] = 0
            profile.update(dtype="float32", nodata=0, compress="deflate", tiled=True, blockxsize=256, blockysize=256)
            with rasterio.open(str(patch_file) + ".tmp.tif", "w", **profile) as dst:
                dst.write(acres[lab], 1)
            os.replace(str(patch_file) + ".tmp.tif", patch_file)
            log.info("%d cane patches, size raster written (%.0f s)", n, time.time() - t0)
    files = track_files(cfg, mask_file)
    jobs = [(r0, c0) for r0 in range(0, H, TILE) for c0 in range(0, W, TILE)]
    guarantee_gb = int(os.environ.get("MEM_GUARANTEE", 16 * 1024 ** 3)) / 1024 ** 3
    n_proc = max(1, min(os.cpu_count(), int(guarantee_gb * 0.8 // 2), len(jobs)))
    with ProcessPoolExecutor(n_proc, mp_context=multiprocessing.get_context("spawn"), initializer=init,
                             initargs=(cfg, area, files, float(share))) as ex:
        parts = [p for p in ex.map(tile, jobs) if p]
    table = {k: np.concatenate([p[k] for p in parts], axis=-1) for k in parts[0]}
    for t, (dates, _) in files.items():
        table[f"dates_{t}"] = dates
    table["report_days"] = np.array(list(cfg["original_reports"]))
    out = interim / f"{cfg['aoi']}_eda_table_{area}.npz"
    np.savez(str(out) + ".tmp.npz", **table)
    os.replace(str(out) + ".tmp.npz", out)
    log.info("%d pixels (%.0f acres sampled) in %.0f s -> %s", len(table["rows"]), len(table["rows"]) * 100 / 4046.8564224, time.time() - t0, out)


if __name__ == "__main__":
    main(*sys.argv[1:4])
