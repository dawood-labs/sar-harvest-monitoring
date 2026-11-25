"""Raster-stage cleaning trials for a large area, straight from the probability rasters of predict_area.py.

For every setting (smallest piece, majority window, order) the report dates are cleaned in order (earlier harvest stays harvested) and scored:
  difference area   acres of blocks >= 0.5 ac (>= 30 m wide) that disagree with the original report
  pieces            number of harvested pieces, and of pieces under 0.5 acre
  look              pieces, median piece size and boundary per acre of the newly harvested and of the remaining
                    cane, next to the same numbers of the original report on the same grid (the standard to match)
Smallest piece 0 = the raw model output. A piece that is a whole cane patch is never changed (see postprocess.drop_small).
Settings run in parallel processes (about 13 GB each on a large grid, measured).

Usage: python src/postprocess_trials_raster.py <area> <config.yaml>      (config: output_tag, grow_sure, grow_maybe)
Output: data/interim/<aoi>_postprocess_trials_<area><output_tag>.csv
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
from scipy import ndimage

import postprocess as pp
from disagreement_eda import patches
from score_reports import AC, labels

# (smallest piece in acres, majority window in pixels, fill standing islands before dropping harvested specks)
SETTINGS = [(0.0, 0, False), (0.2, 0, False), (0.2, 0, True), (0.2, 3, True), (0.2, 5, True), (0.1, 3, True), (0.3, 3, True)]


def look(sel: np.ndarray) -> dict:
    """Pieces of a class on the grid: number, median size, share of area in pieces under 0.2 ac, boundary per acre."""
    lab, k = ndimage.label(sel)
    size = np.bincount(lab.ravel())[1:] * AC
    edges = (sel[:, 1:] != sel[:, :-1]).sum() + (sel[1:] != sel[:-1]).sum()
    return {"pieces": int(k), "median_ac": float(np.median(size)) if k else 0.0, "area_under_0.2ac": float(size[size < 0.2].sum() / max(size.sum(), 1e-9)),
            "boundary_m_per_ac": float(edges * 10 / max(size.sum(), 1e-9))}


def run(args) -> list[dict]:
    cfg_path, area, (mmu, maj, holes) = args
    cfg = yaml.safe_load(open(cfg_path))
    pp.SURE, pp.MAYBE = float(cfg.get("grow_sure", 0.5)) - 1e-6, float(cfg.get("grow_maybe", 0.15)) - 1e-6
    with rasterio.open(Path(cfg["paths"]["interim"]) / f"{cfg['aoi']}_cane_mask_{area}.tif") as m:
        inside = m.read(1) == 1
    rows, cols = np.nonzero(inside)
    d, out = {"rows": rows, "cols": cols}, []
    folder = Path("outputs") / f"{cfg['aoi']}_{area}{cfg.get('output_tag', '')}"
    locked = np.zeros(inside.shape, bool)
    mmu_px = max(1, int(round(mmu * pp.ACRE_M2 / pp.PIXEL_M2)))
    for n, r in enumerate(cfg["report_dates"], 1):
        with rasterio.open(folder / f"{cfg['aoi']}_{area}_sar_probability_{r}_v1.tif") as s:
            p = s.read(1)
        raw = np.where(inside & (p != 255), p, 0).astype("float32") / 100
        before = ((raw >= pp.SURE) | locked) & inside if pp.MAYBE >= pp.SURE else None
        cut = pp.clean_raster(raw, inside, locked, 0.0, mmu_px, maj, holes) if mmu > 0 else (
            before if before is not None else pp.clean_raster(raw, inside, locked, 0.0, 1))
        # the look, measured like the original report: newly harvested since the last report, and remaining
        with rasterio.open(Path(cfg["paths"]["interim"]) / f"{cfg['aoi']}_original_{area}_{list(cfg['original_reports'])[n - 1]}.tif") as s:
            o = s.read(1)
        ours_new, ours_rem = look(cut & ~locked), look(inside & ~cut)
        orig_new, orig_rem = look(o == 82), look(o == 81)
        locked = cut
        pred = cut[rows, cols]
        y, known = labels(cfg, area, n, rows, cols)
        missed, extra = patches(known & y & ~pred, d, 0.5, 3) > 0, patches(known & ~y & pred, d, 0.5, 3) > 0
        lab, k = ndimage.label(cut)
        size = np.bincount(lab.ravel())[1:] * AC
        out.append({"setting": f"mmu {mmu} / majority {maj} / {'holes first' if holes else 'specks first'}", "mmu_acres": mmu, "majority": maj, "holes_first": holes, "report": r,
                    **{f"new_cut_{k}": v for k, v in ours_new.items()}, **{f"original_new_cut_{k}": v for k, v in orig_new.items()},
                    **{f"remaining_{k}": v for k, v in ours_rem.items()}, **{f"original_remaining_{k}": v for k, v in orig_rem.items()}, "harvested_acres": pred.sum() * AC, "original_harvested_acres": (y & known).sum() * AC,
                    "missed_acres": missed.sum() * AC, "extra_acres": extra.sum() * AC, "difference_acres": (missed.sum() + extra.sum()) * AC,
                    "harvested_pieces": int(k), "pieces_under_0.5ac": int((size < 0.5).sum())})
    return out


def main(area: str, cfg_path: str) -> None:
    cfg = yaml.safe_load(open(cfg_path))
    guarantee_gb = int(os.environ.get("MEM_GUARANTEE", 16 * 1024 ** 3)) / 1024 ** 3
    with ProcessPoolExecutor(max(1, min(len(SETTINGS), os.cpu_count(), int(guarantee_gb * 0.8 // 13)))) as ex:
        df = pd.DataFrame([row for res in ex.map(run, [(cfg_path, area, s) for s in SETTINGS]) for row in res])
    df.to_csv(Path(cfg["paths"]["interim"]) / f"{cfg['aoi']}_postprocess_trials_{area}{cfg.get('output_tag', '')}.csv", index=False)
    pd.set_option("display.width", 220)
    show = ["missed_acres", "extra_acres", "difference_acres", "new_cut_pieces", "original_new_cut_pieces", "remaining_pieces", "original_remaining_pieces"]
    print(df.groupby("setting", sort=False)[show].sum().round(0).to_string())
    mean = ["new_cut_median_ac", "original_new_cut_median_ac", "new_cut_boundary_m_per_ac", "original_new_cut_boundary_m_per_ac",
            "remaining_boundary_m_per_ac", "original_remaining_boundary_m_per_ac", "new_cut_area_under_0.2ac"]
    print(df.groupby("setting", sort=False)[mean].mean().round(2).to_string())


if __name__ == "__main__":
    main(*sys.argv[1:3])
