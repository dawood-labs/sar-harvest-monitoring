"""SAR colour composites for the quality check of a harvest report (view in QGIS as RGB).

Per report date and track: the last three passes on or before the report date, VH band, 5 x 5 in-mask mean.
  red = oldest of the three, green = middle, blue = latest (the report pass)
How to read it (bright = cane standing at that pass, dark = no canopy):
  white / light grey   standing in all three passes
  yellow               cut between the middle and the latest pass  (the FRESHEST cuts)
  red / maroon         cut between the oldest and the middle pass
  black / dark         cut before the oldest pass (or never cane)
  blue / cyan tones    backscatter went UP (rain, regrowth, ploughing) - look twice
All three bands use the same fixed stretch (LOW..HIGH dB), so colours mean the same everywhere and on every date.
Tracks are never mixed in one composite. Only crop-mask pixels are filled. The area is done in tiles, in
parallel processes, so memory stays small.

Usage: python src/sar_composite.py <area> <config.yaml>
Output: outputs/<aoi>_<area>_composites/<aoi>_<area>_sar_rgb_<report date>_track<nnn>_<3 pass dates>.tif (uint8, 0 = no data)
"""
from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import rasterio
import yaml
from rasterio.windows import Window

TILE, HALO = 2048, 8
LOW, HIGH = -24.0, -12.0          # dB shown as black .. full brightness


def tile(args):
    """One tile of one composite: three passes, 5 x 5 in-mask mean, scaled to 1..255 (0 = no data)."""
    from sar_features import smooth_in_mask
    cfg, area, files, r0, c0 = args
    win = Window(c0 - HALO, r0 - HALO, TILE + 2 * HALO, TILE + 2 * HALO)
    with rasterio.open(Path(cfg["paths"]["interim"]) / f"{cfg['aoi']}_cane_mask_{area}.tif") as m:
        H, W = m.height, m.width
        inside = m.read(1, window=win, boundless=True, fill_value=0) == 1
    h, w = min(TILE, H - r0), min(TILE, W - c0)
    sl = (slice(HALO, HALO + h), slice(HALO, HALO + w))
    if not inside[sl].any():
        return None
    rows, cols = np.nonzero(inside)
    db = []
    for f in files:
        with rasterio.open(f) as s:
            a = s.read(2, window=win, boundless=True, fill_value=s.nodata)[rows, cols]
            db.append(np.where(a == s.nodata, np.nan, a.astype("float32") / (100.0 if s.dtypes[0] == "int16" else 1.0)))
    sm = smooth_in_mask(np.array(db), rows.astype("int32"), cols.astype("int32"), 5)
    out = np.zeros((3,) + inside.shape, "uint8")
    for b, v in enumerate(sm):
        out[b][rows, cols] = np.where(np.isfinite(v), np.clip((v - LOW) / (HIGH - LOW), 0, 1) * 254 + 1, 0).astype("uint8")
    return r0, c0, out[:, sl[0], sl[1]]


def main(area: str, cfg_path: str) -> None:
    from predict_area import track_files
    cfg = yaml.safe_load(open(cfg_path))
    mask_file = Path(cfg["paths"]["interim"]) / f"{cfg['aoi']}_cane_mask_{area}.tif"
    tracks = track_files(cfg, mask_file)
    with rasterio.open(mask_file) as m:
        profile, H, W = m.profile, m.height, m.width
    profile.update(dtype="uint8", count=3, nodata=0, compress="deflate", tiled=True, blockxsize=256, blockysize=256,
                   photometric="RGB", sparse_ok=True)
    out = Path("outputs") / f"{cfg['aoi']}_{area}_composites"
    out.mkdir(parents=True, exist_ok=True)
    # Tiles in parallel processes (a tile needs well under 1 GB), one composite after another: memory stays small.
    with ProcessPoolExecutor(os.cpu_count()) as ex:
        for report in cfg["report_dates"]:
            for t, (dates, files) in tracks.items():
                use = [str(f) for d, f in zip(dates, files) if d <= str(report)][-3:]
                if len(use) < 3:
                    continue
                days = "_".join(Path(f).stem[-10:][5:] for f in use)
                p = out / f"{cfg['aoi']}_{area}_sar_rgb_{report}_track{int(t):03d}_{days}.tif"
                with rasterio.open(str(p) + ".tmp.tif", "w", **profile) as dst:
                    jobs = [(cfg, area, use, r0, c0) for r0 in range(0, H, TILE) for c0 in range(0, W, TILE)]
                    for res in ex.map(tile, jobs):
                        if res is not None:
                            r0, c0, img = res
                            dst.write(img, window=Window(c0, r0, img.shape[2], img.shape[1]))
                    for b, f in enumerate(use, 1):
                        dst.set_band_description(b, Path(f).stem[-10:])
                Path(str(p) + ".tmp.tif").replace(p)
                print("written", p, flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:3])
