"""Measure how a delivered harvest report 'looks', so that our cleaning can be tuned to look the same.

Per report and class (81 remaining, 82 harvested): number of pieces, piece size (smallest, 5 %, median, share of
area in pieces under 0.2 / 0.5 / 1 acre), compactness (4 pi area / perimeter^2; 1 = circle, low = ragged or thin),
boundary metres per acre, vertices per piece. Reports are measured in parallel processes.

Usage: python src/report_look.py <config.yaml>          (config key original_reports, crs.work)
Output: data/interim/<aoi>_report_look.csv
"""
from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import yaml

ACRE = 4046.8564224


def look(args) -> list[dict]:
    day, f, crs = args
    g = gpd.read_file(f, engine="pyogrio").to_crs(crs).explode(index_parts=False)
    out = []
    for c in (82, 81):
        geom = g.geometry[g.predicted == c].values
        a, per = shapely.area(geom) / ACRE, shapely.length(geom)
        if len(a) == 0:
            continue
        out.append({"report": day, "class": c, "pieces": len(a), "acres": a.sum(), "smallest_ac": a.min(), "p5_ac": np.percentile(a, 5),
                    "median_ac": np.median(a), "area_share_under_0.2ac": a[a < 0.2].sum() / a.sum(), "area_share_under_0.5ac": a[a < 0.5].sum() / a.sum(),
                    "area_share_under_1ac": a[a < 1].sum() / a.sum(), "pieces_under_0.2ac": int((a < 0.2).sum()),
                    "median_compactness": float(np.median(4 * np.pi * a * ACRE / per ** 2)), "boundary_m_per_acre": per.sum() / a.sum(),
                    "median_vertices": float(np.median(shapely.get_num_coordinates(geom)))})
    return out


def main(cfg_path: str) -> None:
    cfg = yaml.safe_load(open(cfg_path))
    jobs = [(str(d), f, cfg["crs"]["work"]) for d, f in cfg["original_reports"].items()]
    with ProcessPoolExecutor(len(jobs)) as ex:
        df = pd.DataFrame([r for res in ex.map(look, jobs) for r in res])
    df.to_csv(Path(cfg["paths"]["interim"]) / f"{cfg['aoi']}_report_look.csv", index=False)
    pd.set_option("display.width", 250)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1])
