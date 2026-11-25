"""Trials for the raster stage of postprocess.py: how much smoothing, and what smallest piece size?

For every setting the three report dates are cleaned in order (earlier harvest stays harvested) and scored on:
  difference area   acres of blocks >= 0.5 ac (>= 30 m wide) that disagree with the original report
  pieces            number of harvested / remaining pieces, and pieces smaller than 0.5 acre
  boundary          length of the harvested / remaining boundary in metres per acre of cane (lower = cleaner)
  sure flips        share of pixels the model was sure about (>= 0.9 or <= 0.1) whose class the cleaning changed
  harvested change  harvested acres after cleaning against before
The original report is scored the same way (pieces, boundary) as a reference for 'clean'.
Settings are chosen on 10 + 15 Jan; 31 Dec is reported separately.

Usage: python src/postprocess_trials.py <area> <config.yaml> <pred.npz> <prob key prefix>
Output: data/interim/<aoi>_postprocess_trials_<area>.csv
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import yaml
from scipy import ndimage

from difference_area import difference
from postprocess import ACRE_M2, PIXEL_M2, clean_raster
from transfer_test import harvested_by
from skimage.filters import apply_hysteresis_threshold

log = logging.getLogger("postprocess_trials")
AC = PIXEL_M2 / ACRE_M2


def shape_stats(cut: np.ndarray, inside: np.ndarray, known: np.ndarray) -> dict:
    """Pieces and boundary length of a harvested map, counted where the original report has an answer."""
    use = inside & known
    out = {}
    for name, v in (("harvested", True), ("remaining", False)):
        lab, n = ndimage.label((cut == v) & use)
        size = np.bincount(lab.ravel())[1:] * AC
        out[f"{name}_pieces"], out[f"{name}_pieces_under_0.5ac"] = int(n), int((size < 0.5).sum())
    edges = ((cut[:, 1:] != cut[:, :-1]) & use[:, 1:] & use[:, :-1]).sum() + ((cut[1:] != cut[:-1]) & use[1:] & use[:-1]).sum()
    out["boundary_m_per_acre"] = float(edges * 10.0 / max(use.sum() * AC, 1))
    return out


def load(area: str, cfg_path: str):
    cfg = yaml.safe_load(open(cfg_path))
    interim = Path(cfg["paths"]["interim"])
    z = np.load(interim / f"{cfg['aoi']}_eda_pixels_{area}.npz")
    d = {k: z[k] for k in z.files if k in ("rows", "cols", "group") or k.startswith("cloud_")}
    with rasterio.open(interim / f"{cfg['aoi']}_cane_mask_{area}.tif") as m:
        inside = m.read(1) == 1
    return cfg, d, inside


def run_setting(args) -> list[dict]:
    """All report dates of one setting, in order (earlier harvest stays harvested)."""
    (name, tv, mmu), cfg_path, area, npz, key = args
    cfg, d, inside = load(area, cfg_path)
    rows, cols, p, out = d["rows"], d["cols"], np.load(npz), []
    locked = np.zeros(inside.shape, bool)
    for n, r in enumerate(cfg["report_dates"], 1):
        raw = np.zeros(inside.shape, "float32")
        raw[rows, cols] = p[f"{key}{r}"]
        before = (apply_hysteresis_threshold(raw, 0.15, 0.5) | locked) & inside
        cut = before if name == "raw model" else clean_raster(raw, inside, locked, tv, max(1, int(round(mmu * ACRE_M2 / PIXEL_M2))))
        y, known = harvested_by(d, n)
        missed, extra = difference(y, cut[rows, cols], known, d)
        known_grid = np.zeros(inside.shape, bool)
        known_grid[rows, cols] = known
        sure = ((raw >= 0.9) | (raw <= 0.1)) & inside & ~locked
        flips = sure & (cut != (raw >= 0.5))
        out.append({"setting": name, "tv": tv, "mmu": mmu, "report": r, "missed_acres": missed.sum() * AC,
                    "extra_acres": extra.sum() * AC, "difference_acres": (missed.sum() + extra.sum()) * AC,
                    **shape_stats(cut, inside, known_grid), "sure_flip_share": float(flips.sum() / max(sure.sum(), 1)),
                    "harvested_change_share": float(cut.sum() / max(before.sum(), 1) - 1)})
        locked = cut
    return out


def main(area: str, cfg_path: str, npz: str, key: str) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    cfg = yaml.safe_load(open(cfg_path))
    interim = Path(cfg["paths"]["interim"])
    z = np.load(interim / f"{cfg['aoi']}_eda_pixels_{area}.npz")
    d = {k: z[k] for k in z.files if k in ("rows", "cols", "group") or k.startswith("cloud_")}
    rows, cols = d["rows"], d["cols"]
    p = np.load(npz)
    with rasterio.open(interim / f"{cfg['aoi']}_cane_mask_{area}.tif") as m:
        inside = m.read(1) == 1
    reports = cfg["report_dates"]

    def grid(v: np.ndarray, fill=False) -> np.ndarray:
        g = np.full(inside.shape, fill, dtype=v.dtype)
        g[rows, cols] = v
        return g

    rows_out = []
    # reference: the original report itself
    for n, r in enumerate(reports, 1):
        y, known = harvested_by(d, n)
        rows_out.append({"setting": "original report", "report": r, **shape_stats(grid(y), inside, grid(known))})
    settings = [("raw model", 0.0, 0.0)] + [(f"tv {tv} / mmu {mmu}", tv, mmu) for tv in (0.0, 0.05, 0.1, 0.2, 0.3) for mmu in (0.25, 0.5, 1.0)]
    # Settings are independent: run them in parallel processes. Each needs ~2.5 GB (measured), so the
    # number of processes is set by the memory the pod guarantees as well as by the cores.
    import os
    from concurrent.futures import ProcessPoolExecutor
    guarantee_gb = int(os.environ.get("MEM_GUARANTEE", 32 * 1024 ** 3)) / 1024 ** 3
    n_proc = max(1, min(len(settings), int(guarantee_gb * 0.8 // 2.6), os.cpu_count()))
    os.environ["TRIAL_THREADS"] = str(max(1, os.cpu_count() // n_proc))
    log.info("%d settings, %d at a time", len(settings), n_proc)
    with ProcessPoolExecutor(n_proc) as ex:
        for res in ex.map(run_setting, [(s_, cfg_path, area, npz, key) for s_ in settings]):
            rows_out += res
            log.info("%-20s done", res[0]["setting"])
            pd.DataFrame(rows_out).to_csv(interim / f"{cfg['aoi']}_postprocess_trials_{area}.csv", index=False)
    log.info("finished")


if __name__ == "__main__":
    main(*sys.argv[1:5])
