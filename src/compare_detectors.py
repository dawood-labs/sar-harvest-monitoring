"""Compare three ways of telling harvested from standing cane on each report date, using SAR only.

  rgb3  : the user's idea - VH of the 3 latest passes of the main track (what a 3-date colour composite shows)
  drop  : one number - latest VH minus the same pixel's early-season VH (first 3 passes), one threshold
  tree  : small LightGBM on every pass available up to the report date, both tracks, VV and VH
Truth = the reference reports (harvested by the report date vs still standing). Train and test pixels come from
different 3 km blocks (checkerboard), because neighbouring pixels are near copies of each other.
Only passes on or before the report date are used, as in production.

Usage: python src/compare_detectors.py <area_name> [smooth_size]
Output: data/interim/<aoi>_detector_scores_<area>.csv
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

from resources import workers
from sar_features import load_pixels, smooth_in_mask

ACRE_PER_PIXEL = 100 / 4046.8564224
log = logging.getLogger("compare_detectors")


def best_threshold(x: np.ndarray, y: np.ndarray) -> float:
    """Threshold on x (harvested = x below it) with the best F1 on the training pixels."""
    grid = np.quantile(x, np.linspace(0.02, 0.98, 97))
    f1 = [precision_recall_fscore_support(y, x < t, average="binary", zero_division=0)[2] for t in grid]
    return float(grid[int(np.argmax(f1))])


def smoothed_series(cfg: dict, d: dict, size: int) -> dict:
    """VH and VV of every track, smoothed inside the crop mask."""
    return {(b, t): smooth_in_mask(d[f"{b}_{t}"], d["rows"], d["cols"], size) for t in cfg["s1_tracks"] for b in ("vh", "vv")}


def detector_features(cfg: dict, d: dict, series: dict, report: str) -> dict:
    """Inputs of the three detectors, built only from passes on or before the report date."""
    main_track = next(iter(cfg["s1_tracks"]))
    use = {t: d[f"dates_{t}"] <= report for t in cfg["s1_tracks"]}
    vh = series[("vh", main_track)][use[main_track]]
    with np.errstate(all="ignore"):
        return {
            "rgb3": vh[-3:].T,
            "drop": (vh[-1] - np.nanmedian(vh[:3], axis=0))[:, None],
            "tree": np.column_stack([series[(b, t)][use[t]].T - np.nanmedian(series[(b, t)][:3], axis=0)[:, None]
                                     for t in cfg["s1_tracks"] for b in ("vh", "vv")]),
        }


def tree_model(cfg: dict) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(n_estimators=200, num_leaves=31, learning_rate=0.08, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.8, n_jobs=workers(keep_free_cores=0), random_state=cfg["seed"], verbose=-1)


def split(d: dict) -> np.ndarray:
    """True for test pixels: 3 km blocks in a checkerboard."""
    return ((d["rows"] // 300 + d["cols"] // 300) % 2).astype(bool)


def main(area: str, size: int = 5) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    cfg, d = load_pixels(area)
    main_track = next(iter(cfg["s1_tracks"]))
    series = smoothed_series(cfg, d, size)
    group, rng, test = d["group"], np.random.default_rng(cfg["seed"]), split(d)
    rows = []
    for n, report in enumerate(cfg["report_dates"], 1):
        known = group != 255
        y = (group >= 1) & (group <= n)
        feats = detector_features(cfg, d, series, report)
        last_pass = d[f"dates_{main_track}"][d[f"dates_{main_track}"] <= report][-1]
        tr = np.flatnonzero(known & ~test)
        tr = rng.choice(tr, min(len(tr), 400_000), replace=False)
        te = np.flatnonzero(known & test)
        for name, x in feats.items():
            if name == "drop":
                thr = best_threshold(x[tr, 0], y[tr])
                score, pred = -x[te, 0], x[te, 0] < thr
            else:
                m = tree_model(cfg)
                m.fit(x[tr], y[tr])
                score = m.predict_proba(x[te])[:, 1]
                pred = score >= 0.5
            p, r, f1, _ = precision_recall_fscore_support(y[te], pred, average="binary")
            rows.append({"report": report, "last_pass": last_pass, "detector": name,
                         "auc": roc_auc_score(y[te], score), "precision": p, "recall": r, "f1": f1,
                         "harvested_acres_report": y[te].sum() * ACRE_PER_PIXEL, "harvested_acres_sar": pred.sum() * ACRE_PER_PIXEL,
                         "threshold_db": thr if name == "drop" else np.nan, "smooth": size})
            log.info("%s %-5s AUC %.3f  P %.3f  R %.3f  F1 %.3f", report, name, rows[-1]["auc"], p, r, f1)
    out = Path(cfg["paths"]["interim"]) / f"{cfg['aoi']}_detector_scores_{area}_smooth{size}.csv"
    pd.DataFrame(rows).round(3).to_csv(out, index=False)
    log.info("written %s", out)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 5)
