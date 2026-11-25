"""Find crop-mask polygons whose result is a patchwork (part harvested, part standing, no clear pattern) and measure
where the harvested pixels come from.

Per crop-mask polygon and report date: harvested share, number of separate harvested pieces, and the share of its
harvested pixels whose own model probability is below 0.5 (they were added only by region growing or by the
'stays harvested' rule). A patchwork polygon = harvested share between 10 % and 90 % and at least 3 pieces.
The area is cut into tiles that run in parallel processes; tiles without crop are skipped. A polygon that crosses a
tile edge is counted per tile and added up (its piece count can be slightly high).

Usage: python src/mixing_check.py <area> <config.yaml>
Output: data/interim/<aoi>_mixing_<area>.csv (one row per polygon and date), a short summary on screen
"""
from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely
import yaml
from rasterio.features import rasterize
from rasterio.windows import Window
from scipy import ndimage

ACRE, TILE = 100 / 4046.8564224, 2048
G = {}


def init(cfg_path: str, area: str) -> None:
    cfg = yaml.safe_load(open(cfg_path))
    G["cfg"], G["area"] = cfg, area
    G["mask"] = Path(cfg["paths"]["interim"]) / f"{cfg['aoi']}_cane_mask_{area}.tif"
    with rasterio.open(G["mask"]) as m:
        G["tr"], crs = m.transform, m.crs
    crop = gpd.read_file(cfg["crop_mask"]).to_crs(crs)
    G["geom"] = shapely.make_valid(crop.geometry.values)
    G["tree"] = shapely.STRtree(G["geom"])


def tile(rc: tuple[int, int]) -> pd.DataFrame | None:
    r0, c0 = rc
    cfg, area = G["cfg"], G["area"]
    out = Path("outputs") / f"{cfg['aoi']}_{area}{cfg.get('output_tag', '')}"
    with rasterio.open(G["mask"]) as m:
        win = Window(c0, r0, min(TILE, m.width - c0), min(TILE, m.height - r0))
        inside = m.read(1, window=win) == 1
        if not inside.any():
            return None
        tr = m.window_transform(win)
        box = shapely.box(*m.window_bounds(win))
    hit = G["tree"].query(box, predicate="intersects")
    ids = rasterize([(G["geom"][i], int(i) + 1) for i in hit], out_shape=inside.shape, transform=tr, dtype="int32")
    ids[~inside] = 0
    n = len(G["geom"])
    rows = []
    for r in cfg["report_dates"]:
        with rasterio.open(out / f"{cfg['aoi']}_{area}_sar_harvest_{r}_v1.tif") as s:
            cut = s.read(1, window=win) == 82
        with rasterio.open(out / f"{cfg['aoi']}_{area}_sar_probability_{r}_v1.tif") as s:
            prob = s.read(1, window=win).astype("float32") / 100
        live = np.flatnonzero(np.bincount(ids.ravel(), minlength=n + 1))[1:]
        total = np.bincount(ids.ravel(), minlength=n + 1)[live]
        harv = np.bincount(ids.ravel(), weights=cut.ravel(), minlength=n + 1)[live]
        weak = np.bincount(ids.ravel(), weights=(cut & (prob < 0.5)).ravel(), minlength=n + 1)[live]
        weak3 = np.bincount(ids.ravel(), weights=(cut & (prob < 0.3)).ravel(), minlength=n + 1)[live]
        lab, k = ndimage.label(cut & (ids > 0))
        pieces = np.bincount(ndimage.minimum(ids, lab, np.arange(1, k + 1)).astype(int), minlength=n + 1)[live] if k else np.zeros(len(live))
        rows.append(pd.DataFrame({"field_id": live, "report": r, "pixels": total, "harvested": harv, "weak": weak, "weak03": weak3, "pieces": pieces}))
    return pd.concat(rows)


def main(area: str, cfg_path: str) -> None:
    cfg = yaml.safe_load(open(cfg_path))
    with rasterio.open(Path(cfg["paths"]["interim"]) / f"{cfg['aoi']}_cane_mask_{area}.tif") as m:
        tiles = [(r0, c0) for r0 in range(0, m.height, TILE) for c0 in range(0, m.width, TILE)]
    with ProcessPoolExecutor(os.cpu_count(), initializer=init, initargs=(cfg_path, area)) as ex:
        parts = [p for p in ex.map(tile, tiles, chunksize=4) if p is not None]
    df = pd.concat(parts).groupby(["field_id", "report"], as_index=False).sum()
    df["acres"] = df.pixels * ACRE
    df["harvested_share"] = df.harvested / df.pixels
    df["weak_share_of_harvested"] = df.weak / df.harvested.clip(lower=1)
    df["patchwork"] = df.harvested_share.between(0.1, 0.9) & (df.pieces >= 3)
    df.to_csv(Path(cfg["paths"]["interim"]) / f"{cfg['aoi']}_mixing_{area}{cfg.get('output_tag', '')}.csv", index=False)
    for r, g in df.groupby("report"):
        p = g[g.patchwork]
        print(f"{r}: harvested {g.harvested.sum() * ACRE:,.0f} ac; own probability < 0.5: {g.weak.sum() * ACRE:,.0f} ac ({g.weak.sum() / g.harvested.sum():.0%}), "
              f"< 0.3: {g.weak03.sum() / g.harvested.sum():.0%} | patchwork polygons {len(p):,} of {len(g):,} = {p.acres.sum():,.0f} ac of {g.acres.sum():,.0f}; "
              f"in them {p.weak.sum() / max(p.harvested.sum(), 1):.0%} of harvested is weak")


if __name__ == "__main__":
    main(*sys.argv[1:3])
