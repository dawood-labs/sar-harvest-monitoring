"""Run the best method on one area and score it against the original reports: SAR only, and SAR + optical.

Method (fixed on the test tile):
  1. 5 x 5 in-mask mean of every Sentinel-1 pass (VV, VH, two tracks), crop-mask pixels only.
  2. 64 date-free features per pixel: levels of the last 4 passes against the pixel's own early-season level,
     plus rate-of-change features (first and second derivative, slopes, spread, texture).
  3. LightGBM, trained on the original reports of the OTHER report dates, only on training blocks (3 km
     checkerboard) and only on pixels at least 20 m from an edge. So every score is on a date the model never saw.
  4. Region growing (sure >= 0.5, grow into touching pixels >= 0.15).
  5. Optional optical cross-check with the Sentinel-2 image of the report day, only where the model is unsure:
       SAR standing but probability >= 0.05 and optical bare (NDVI < 0.40)      -> harvested
       SAR harvested with probability < 0.8 and optical clearly green (> 0.50) -> standing
     Cloudy optical pixels change nothing.
  6. Once harvested, always harvested in later reports.
Scores per report date: difference area (blocks >= 0.5 acre where SAR and the original report disagree), as acres
and as a share of the harvested / standing area, the same on test blocks only, per-pixel numbers, pure-block scores.
Outputs for QGIS: per date and variant a 4-class comparison raster and a harvest raster (81 / 82), with styles.
A final model trained on all report dates is saved in runs/ for use on new dates.

Usage: python src/run_area.py <area_name> <config.yaml>
Output: data/interim/<aoi>_area_scores_<area>.csv, ..._area_pred_<area>.npz, outputs/<area>/*.tif + .qml + zip,
        runs/<today>_<area>_final/ (model, feature list, settings)
"""
from __future__ import annotations

import json
import logging
import sys
import time
import zipfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

import transfer_test as tt
from block_eval import block_scores
from compare_detectors import smoothed_series, split, tree_model
from difference_area import ACRE_PER_PIXEL, difference
from disagreement_figures import ndvi_series
from export_comparison import STYLE as DIFF_STYLE
from export_qgis import CLASS_STYLE, write_qml, write_raster
from retrain_test import rate_features
from sar_features import load_pixels, local_std_in_mask

log = logging.getLogger("run_area")
LOW, SURE, BARE_NDVI, GREEN_NDVI, SAMPLE = 0.05, 0.8, 0.40, 0.50, 300_000


def main(area: str, cfg_path: str) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    t0 = time.time()
    cfg, d = load_pixels(area, cfg_path)
    interim, out = Path(cfg["paths"]["interim"]), Path("outputs") / area
    out.mkdir(parents=True, exist_ok=True)
    test, reports, rng = split(d), cfg["report_dates"], np.random.default_rng(cfg["seed"])
    log.info("%s: %.1f acres of crop mask, %d report dates", area, len(test) * ACRE_PER_PIXEL, len(reports))

    series = smoothed_series(cfg, d, 5)
    texture = {t: local_std_in_mask(d[f"vh_{t}"], d["rows"], d["cols"], 5) for t in cfg["s1_tracks"]}
    labels = {r: tt.harvested_by(d, n) for n, r in enumerate(reports, 1)}
    edges = {r: tt.near_edge(d, *labels[r]) for r in reports}
    feats = {r: np.column_stack([tt.date_free_features(cfg, d, series, r), rate_features(cfg, d, series, texture, r)]) for r in reports}
    dates, ndvi = ndvi_series(cfg, d)
    log.info("features ready (%d per pixel) after %.0f s; optical dates %s", feats[reports[0]].shape[1], time.time() - t0, dates)

    def training_set(use: list[str], blocks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xs, ys = [], []
        for r in use:
            pool = np.flatnonzero(labels[r][1] & blocks & ~edges[r])
            idx = rng.choice(pool, min(len(pool), SAMPLE), replace=False)
            xs.append(feats[r][idx]); ys.append(labels[r][0][idx])
        return np.vstack(xs), np.concatenate(ys)

    prob = {}
    for held in reports:                                   # a date is always scored by a model that never saw it
        prob[held] = tree_model(cfg).fit(*training_set([r for r in reports if r != held], ~test)).predict_proba(feats[held])[:, 1].astype("float32")
        log.info("model for %s done", held)
    own = {r: tt.grow(prob[r], d) for r in reports}

    variants = {"SAR only": False, "SAR + optical where unsure": True}
    with rasterio.open(interim / f"{cfg['aoi']}_cane_mask_{area}.tif") as src:
        profile = src.profile
    rows, keep, files = [], {}, []
    for name, optical in variants.items():
        earlier = np.zeros(len(test), bool)
        tag = "sar_only" if not optical else "sar_optical"
        for n, r in enumerate(reports, 1):
            pred, looked = own[r].copy(), np.zeros(len(test), bool)
            if optical:
                v = ndvi[dates.index(r)]
                doubt = ~pred & (prob[r] >= LOW) & ~earlier
                unsure_cut = own[r] & (prob[r] < SURE) & ~earlier
                pred = (pred | (doubt & (v < BARE_NDVI))) & ~(unsure_cut & (v > GREEN_NDVI))
                looked = doubt | unsure_cut
            pred = pred | earlier
            earlier = pred
            keep[f"{name}|{r}"] = pred
            y, known = labels[r]
            missed, extra = difference(y, pred, known, d)
            h, s, k = (y & known).sum(), (~y & known).sum(), known.sum()
            pure = block_scores(y, pred, known & test, d)
            row = {"variant": name, "report": r, "cane_acres": k * ACRE_PER_PIXEL, "original_harvested_acres": h * ACRE_PER_PIXEL,
                   "original_standing_acres": s * ACRE_PER_PIXEL, "sar_harvested_acres": (pred & known).sum() * ACRE_PER_PIXEL,
                   "missed_acres": missed.sum() * ACRE_PER_PIXEL, "missed_share_of_harvested": missed.sum() / h,
                   "extra_acres": extra.sum() * ACRE_PER_PIXEL, "extra_share_of_standing": extra.sum() / s,
                   "same_share": 1 - (missed.sum() + extra.sum()) / k,
                   "test_missed_share_of_harvested": (missed & test).sum() / (y & known & test).sum(),
                   "test_extra_share_of_standing": (extra & test).sum() / (~y & known & test).sum(),
                   "test_same_share": 1 - ((missed | extra) & test).sum() / (known & test).sum(),
                   "pixel_missed_acres": (y & ~pred & known).sum() * ACRE_PER_PIXEL, "pixel_extra_acres": (~y & pred & known).sum() * ACRE_PER_PIXEL,
                   "pixel_same_share": ((y == pred) & known).sum() / k,
                   "pure_harvested_blocks_found": pure["harvested_found"], "pure_standing_blocks_kept": pure["standing_kept"],
                   "optical_looked_acres": looked.sum() * ACRE_PER_PIXEL, "optical_looked_share": looked.sum() / len(test),
                   "optical_clear_share": float(np.isfinite(ndvi[dates.index(r)]).mean())}
            rows.append(row)
            log.info("%-28s %s: missed %7.0f ac (%.1f%% of harvested), extra %7.0f ac (%.1f%% of standing), same %.1f%%",
                     name, r, row["missed_acres"], 100 * row["missed_share_of_harvested"], row["extra_acres"],
                     100 * row["extra_share_of_standing"], 100 * row["same_share"])
            cls = np.where(~known, 0, np.where(~y & ~pred, 1, np.where(y & pred, 2, np.where(y, 3, 4))))
            for stem, values, style in ((f"{tag}_vs_original", cls, DIFF_STYLE), (f"{tag}_harvest", np.where(pred, 82, 81), CLASS_STYLE)):
                p = out / f"{cfg['aoi']}_{area}_{stem}_{r}_v1.tif"
                write_raster(p, values, d, profile, 0)
                write_qml(p, style)
                files += [p, p.with_suffix(".qml")]
        pd.DataFrame(rows).to_csv(interim / f"{cfg['aoi']}_area_scores_{area}.csv", index=False)
    np.savez_compressed(interim / f"{cfg['aoi']}_area_pred_{area}.npz", **keep, **{f"prob_{r}": prob[r] for r in reports})
    with zipfile.ZipFile(out / f"{cfg['aoi']}_{area}_qgis_v1.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, f.name)

    # Final model for new dates: all report dates, all blocks (the scores above come from the held-out models).
    run = Path("runs") / f"{date.today()}_{area}_final"
    run.mkdir(parents=True, exist_ok=True)
    final = tree_model(cfg).fit(*training_set(reports, np.ones(len(test), bool)))
    final.booster_.save_model(str(run / "model.txt"))
    (run / "settings.json").write_text(json.dumps({
        "area": area, "trained_on_reports": reports, "tracks": {str(k): v for k, v in cfg["s1_tracks"].items()}, "season_start": cfg["season"]["start"],
        "smooth": 5, "passes_back": tt.PASSES_BACK, "features": feats[reports[0]].shape[1], "sample_per_date": SAMPLE,
        "grow": [tt.SURE, tt.MAYBE], "optical_check": {"low": LOW, "sure": SURE, "bare_ndvi": BARE_NDVI, "green_ndvi": GREEN_NDVI},
        "seed": cfg["seed"], "lightgbm": final.get_params()}, indent=1, default=str))
    with open("runs/experiments.md", "a") as f:
        f.write(f"- {date.today()} | {run.name} | level + rate features (64), 5x5 mean, trained on {len(reports)} report dates of area '{area}' | "
                f"held-out difference area: see data/interim/{cfg['aoi']}_area_scores_{area}.csv\n")
    log.info("finished in %.0f s; QGIS files in %s, final model in %s", time.time() - t0, out, run)


if __name__ == "__main__":
    main(*sys.argv[1:3])
