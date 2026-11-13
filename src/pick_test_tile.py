"""Pick the test tile: a square window with a lot of cane and harvest visible in every reference report.

Why: methods are tried on a small area first (time and memory are measured there) before the whole AOI.
Windows slide in steps of a third of their size. First pass: each polygon is counted in the window holding its
inner point (fast, but a very large polygon is counted whole). Second pass: the 10 best windows are measured
exactly by clipping, and the best of those is kept.

Usage: python src/pick_test_tile.py [tile_km] [configs/config.yaml]
Output: data/interim/<aoi>_test_tile_candidates.csv and data/interim/<aoi>_test_tile.gpkg (the best window)
"""
from __future__ import annotations

import glob
import logging
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import yaml
from shapely.geometry import box

ACRE_M2 = 4046.8564224
log = logging.getLogger("pick_test_tile")


def window_sums(x: np.ndarray, y: np.ndarray, w: np.ndarray, x0: np.ndarray, y0: np.ndarray, size: float) -> np.ndarray:
    """Sum of weights w of the points inside each window (lower-left corners x0, y0)."""
    return np.array([w[(x >= a) & (x < a + size) & (y >= b) & (y < b + size)].sum() for a, b in zip(x0, y0)])


def main(tile_km: float = 15.0, cfg_path: str = "configs/config.yaml") -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    cfg = yaml.safe_load(open(cfg_path))
    crs, size, step = cfg["crs"]["work"], tile_km * 1000, tile_km * 1000 / 3
    mask = gpd.read_file(cfg["crop_mask"]).to_crs(crs)
    xmin, ymin, xmax, ymax = mask.total_bounds
    gx, gy = np.meshgrid(np.arange(xmin, xmax - size + step, step), np.arange(ymin, ymax - size + step, step))
    x0, y0 = gx.ravel(), gy.ravel()

    def sums(g: gpd.GeoDataFrame) -> np.ndarray:
        c = g.geometry.representative_point()
        return window_sums(c.x.values, c.y.values, g.area.values / ACRE_M2, x0, y0, size)

    cols, layers = {"cane_acres": sums(mask)}, {"cane_acres": mask}
    codes = cfg["classes"]
    reports = sorted(f for f in glob.glob(str(Path(cfg["paths"]["reference_reports"]) / "*.shp")) if f != cfg["crop_mask"])
    for i, f in enumerate(reports, 1):
        g = gpd.read_file(f).to_crs(crs)
        for name, code in (("harvested", codes["harvested"]), ("cloud", codes["cloud"])):
            layers[f"report{i}_{name}_acres"] = g[g.predicted == code]
            cols[f"report{i}_{name}_acres"] = sums(layers[f"report{i}_{name}_acres"])
    df = pd.DataFrame({"x0": x0, "y0": y0, **cols}).round(1)
    harvested = [c for c in df if c.endswith("harvested_acres")]
    top = df.loc[df[harvested].min(axis=1).sort_values(ascending=False).index[:10]].copy()
    for i, r in top.iterrows():   # exact areas inside the window
        w = box(r.x0, r.y0, r.x0 + size, r.y0 + size)
        for name, g in layers.items():
            top.loc[i, name] = g.geometry.clip(w).area.sum() / ACRE_M2
    df = top.round(1)
    # Best = the window whose weakest report still has the most harvest (so every report can be scored there).
    df["min_harvested_acres"] = df[harvested].min(axis=1)
    df = df.sort_values("min_harvested_acres", ascending=False).reset_index(drop=True)
    out = Path(cfg["paths"]["interim"])
    df.to_csv(out / f"{cfg['aoi']}_test_tile_candidates.csv", index=False)
    best = df.iloc[0]
    tile = gpd.GeoDataFrame({"tile_acres": [round(size * size / ACRE_M2, 1)], **{k: [best[k]] for k in cols}},
                            geometry=[box(best.x0, best.y0, best.x0 + size, best.y0 + size)], crs=crs)
    tile.to_file(out / f"{cfg['aoi']}_test_tile.gpkg", driver="GPKG")
    log.info("tile %.0f acres; best window:\n%s", size * size / ACRE_M2, df.head(5).drop(columns=["x0", "y0"]).to_string())


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 15.0, *sys.argv[2:3])
