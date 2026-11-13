"""Try to make the method right on the hard cases found in the review, instead of only flagging them.

Hard cases: weak / thin standing cane that slides to a low backscatter and gets called "harvested" too early.
Variants compared (all: 5x5 in-mask mean, trained on other report dates, 3 km block split, region growing,
once-cut-stays-cut):
  baseline      the date-free level features used so far
  + shape       baseline plus the step-shape features of step_features.py (VH and VV, both tracks)
  shape only    only the step-shape features
  rule          no model: harvested = a clean, big step on either track (limits tuned on the training blocks)
  dips only     model on the relative dip features of dip_features.py only (each pixel against its own history)
  baseline AND own dip <= X dB
                the user's idea as a gate: the baseline says harvested AND the pixel itself shows a sudden dip of
                at least X dB against its own previous passes (dip measured on a 9 x 9 mean, VH, either track)
Scores: overall F1 on test blocks; precision and recall inside fields; acres of big in-field patches that are
extra (SAR cut, original standing) or missed; and what each variant says on the user-reviewed patches.

Usage: python src/hard_cases_test.py <area_name>
Output: data/interim/<aoi>_hard_cases_scores_<area>.csv, ..._hard_cases_reviewed_<area>.csv, ..._hard_cases_pred_<area>.npz
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from sklearn.metrics import precision_recall_fscore_support

from compare_detectors import smoothed_series, split, tree_model
from dip_features import dip_features
from disagreement_eda import patches
from sar_features import load_pixels
from step_features import step_features
from transfer_test import ACRE_PER_PIXEL, date_free_features, grid_shape, grow, harvested_by, near_edge

log = logging.getLogger("hard_cases_test")


def shape_features(cfg: dict, d: dict, series: dict, report: str) -> np.ndarray:
    cols = []
    with np.errstate(all="ignore"):
        for t in cfg["s1_tracks"]:
            use = d[f"dates_{t}"] <= report
            for b in ("vh", "vv"):
                s = series[(b, t)][use]
                cols.append(step_features(s - np.nanpercentile(s, 90, axis=1)[:, None]))
    return np.column_stack(cols)


def rule_score(x: np.ndarray) -> np.ndarray:
    """Step size where the step is clean (explains >= 40 % of the variation), on the VH of either track."""
    vh = [x[:, i * 6:(i + 1) * 6] for i in (0, 2)]                       # VH blocks of track 1 and track 2
    return np.minimum.reduce([np.where(v[:, 1] >= 0.4, v[:, 0], 0) for v in vh])


def main(area: str) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    cfg, d = load_pixels(area)
    test, rng, reports = split(d), np.random.default_rng(cfg["seed"]), cfg["report_dates"]
    series = smoothed_series(cfg, d, 5)
    level = {r: date_free_features(cfg, d, series, r) for r in reports}
    shape = {r: shape_features(cfg, d, series, r) for r in reports}
    labels = {r: harvested_by(d, n) for n, r in enumerate(reports, 1)}
    edges = {r: near_edge(d, *labels[r]) for r in reports}
    dips = {r: np.column_stack([dip_features(series[(b, t)][d[f"dates_{t}"] <= r]) for t in cfg["s1_tracks"] for b in ("vh", "vv")]) for r in reports}
    wide = smoothed_series(cfg, d, 9)                   # the pixel's own history with less speckle, for the gate
    own_dip = {r: np.minimum.reduce([dip_features(wide[("vh", t)][d[f"dates_{t}"] <= r])[:, 0] for t in cfg["s1_tracks"]]) for r in reports}
    feats = {"baseline": level, "+ shape": {r: np.column_stack([level[r], shape[r]]) for r in reports}, "shape only": shape,
             "dips only": dips}
    gates = {"baseline AND own dip <= -2.5 dB": -2.5, "baseline AND own dip <= -3.0 dB": -3.0}
    own = {}                                            # baseline result of each date before the stays-cut rule

    interim = Path(cfg["paths"]["interim"])
    with rasterio.open(interim / f"{cfg['aoi']}_cane_mask_{area}.tif") as src:
        tr, crs = src.transform, src.crs
    reviewed = gpd.read_file(f"outputs/review_patches/{cfg['aoi']}_{area}_review_patches_v2.gpkg").to_crs(crs)
    reviewed_mask = {p.patch_id: rasterize([(p.geometry, 1)], out_shape=grid_shape(d), transform=tr, dtype="uint8")[d["rows"], d["cols"]] == 1
                     for _, p in reviewed.iterrows()}

    rows, review_rows, keep = [], [], {}
    for name in ("baseline", "+ shape", "shape only", "rule", "dips only", *gates):
        earlier = np.zeros(len(d["group"]), bool)
        for n, held in enumerate(reports, 1):
            others = [r for r in reports if r != held]
            if name in gates:
                pred = own[held] & (own_dip[held] <= gates[name])
            elif name == "rule":                          # one threshold, tuned on the training blocks of the other dates
                xs = np.concatenate([rule_score(shape[r])[labels[r][1] & ~test & ~edges[r]] for r in others])
                ys = np.concatenate([labels[r][0][labels[r][1] & ~test & ~edges[r]] for r in others])
                grid = np.arange(-8, -1, 0.25)
                f1 = [precision_recall_fscore_support(ys, xs <= g, average="binary", zero_division=0)[2] for g in grid]
                pred = rule_score(shape[held]) <= grid[int(np.argmax(f1))]
            else:
                xs, ys = [], []
                for r in others:
                    idx = np.flatnonzero(labels[r][1] & ~test & ~edges[r])
                    idx = rng.choice(idx, min(len(idx), 300_000), replace=False)
                    xs.append(feats[name][r][idx]); ys.append(labels[r][0][idx])
                prob = tree_model(cfg).fit(np.vstack(xs), np.concatenate(ys)).predict_proba(feats[name][held])[:, 1]
                pred = grow(prob, d)
            if name == "baseline":
                own[held] = pred
            pred = pred | earlier
            earlier = pred
            keep[f"{name}|{held}"] = pred
            y, known = labels[held]
            m, inside = known & test, known & test & ~edges[held]
            p, r_, f1_, _ = precision_recall_fscore_support(y[m], pred[m], average="binary")
            pi, ri, _, _ = precision_recall_fscore_support(y[inside], pred[inside], average="binary")
            extra, missed = patches(known & ~y & pred, d) > 0, patches(known & y & ~pred, d) > 0
            rows.append({"variant": name, "report": held, "f1": f1_, "precision": p, "recall": r_, "precision_inside": pi,
                         "recall_inside": ri, "big_extra_acres": extra.sum() * ACRE_PER_PIXEL, "big_missed_acres": missed.sum() * ACRE_PER_PIXEL,
                         "sar_acres": pred[m].sum() * ACRE_PER_PIXEL, "original_acres": y[m].sum() * ACRE_PER_PIXEL})
            log.info("%-32s %s F1 %.3f P %.3f R %.3f | inside P %.3f R %.3f | big extra %.1f ac, big missed %.1f ac",
                     name, held, f1_, p, r_, pi, ri, rows[-1]["big_extra_acres"], rows[-1]["big_missed_acres"])
            for pid, pm in reviewed_mask.items():
                review_rows.append({"variant": name, "report": held, "patch_id": pid, "sar_cut_share": round(float(pred[pm].mean()), 2),
                                    "original_cut_share": round(float(y[pm & known].mean()), 2) if (pm & known).any() else np.nan})
    pd.DataFrame(rows).round(3).to_csv(interim / f"{cfg['aoi']}_hard_cases_scores_{area}.csv", index=False)
    rv = pd.DataFrame(review_rows)
    rv.to_csv(interim / f"{cfg['aoi']}_hard_cases_reviewed_{area}.csv", index=False)
    np.savez_compressed(interim / f"{cfg['aoi']}_hard_cases_pred_{area}.npz", **keep)
    print(rv.pivot_table(index=["patch_id", "report"], columns="variant", values="sar_cut_share").join(
        rv.drop_duplicates(["patch_id", "report"]).set_index(["patch_id", "report"]).original_cut_share).to_string())


if __name__ == "__main__":
    main(sys.argv[1])
