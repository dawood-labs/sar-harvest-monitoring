"""Combine the two fixes that each helped on their own: extra weight on hard examples and the own-dip gate.

  hard weights : standing pixels that a first model calls harvested on the training dates get 10 times the
                 weight in a second training round (best overall score so far), 6 passes back
  own-dip gate : a pixel is kept as harvested only if it shows a sudden dip against its own previous passes
                 (9 x 9 mean, VH, either track) - the user's relative idea (best on weak cane so far)
  soft gate    : the gate is skipped where the model is very sure (probability >= 0.9)
All variants: 5x5 in-mask mean, trained on other report dates, 3 km block split, region growing, stays-cut.

Usage: python src/combine_test.py <area_name>
Output: data/interim/<aoi>_combine_scores_<area>.csv, ..._combine_pred_<area>.npz
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

import transfer_test as tt
from compare_detectors import smoothed_series, split, tree_model
from dip_features import dip_features
from disagreement_eda import patches
from sar_features import load_pixels

log = logging.getLogger("combine_test")
WEAK = ["P02", "P03", "P04", "P06"]


def main(area: str) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    cfg, d = load_pixels(area)
    test, reports = split(d), cfg["report_dates"]
    labels = {r: tt.harvested_by(d, n) for n, r in enumerate(reports, 1)}
    edges = {r: tt.near_edge(d, *labels[r]) for r in reports}
    interim = Path(cfg["paths"]["interim"])
    with rasterio.open(interim / f"{cfg['aoi']}_cane_mask_{area}.tif") as src:
        tr, crs = src.transform, src.crs
    rv = gpd.read_file(f"outputs/review_patches/{cfg['aoi']}_{area}_review_patches_v2.gpkg").to_crs(crs)
    weak = np.zeros(len(d["group"]), bool)
    for _, p in rv.iterrows():
        if p.patch_id in WEAK:
            weak |= rasterize([(p.geometry, 1)], out_shape=tt.grid_shape(d), transform=tr, dtype="uint8")[d["rows"], d["cols"]] == 1

    series, wide = smoothed_series(cfg, d, 5), smoothed_series(cfg, d, 9)
    own_dip = {r: np.minimum.reduce([dip_features(wide[("vh", t)][d[f"dates_{t}"] <= r])[:, 0] for t in cfg["s1_tracks"]]) for r in reports}
    prob = {}
    for name, back, hard in (("current", 4, 1), ("hard", 6, 10)):
        tt.PASSES_BACK = back
        feats = {r: tt.date_free_features(cfg, d, series, r) for r in reports}
        rng = np.random.default_rng(cfg["seed"])
        for held in reports:
            idx = {r: rng.choice(np.flatnonzero(labels[r][1] & ~test & ~edges[r]), 300_000, replace=False) for r in reports if r != held}
            x = np.vstack([feats[r][i] for r, i in idx.items()])
            y_tr = np.concatenate([labels[r][0][i] for r, i in idx.items()])
            model = tree_model(cfg).fit(x, y_tr)
            if hard > 1:
                model = tree_model(cfg).fit(x, y_tr, sample_weight=np.where(~y_tr & (model.predict_proba(x)[:, 1] >= 0.3), hard, 1.0))
            prob[(name, held)] = model.predict_proba(feats[held])[:, 1]
        log.info("%s models done", name)

    variants = {
        "current method": lambda r: tt.grow(prob[("current", r)], d),
        "current + own dip <= -2.5": lambda r: tt.grow(prob[("current", r)], d) & (own_dip[r] <= -2.5),
        "hard weights": lambda r: tt.grow(prob[("hard", r)], d),
        "hard weights + own dip <= -2.0": lambda r: tt.grow(prob[("hard", r)], d) & (own_dip[r] <= -2.0),
        "hard weights + own dip <= -2.5": lambda r: tt.grow(prob[("hard", r)], d) & (own_dip[r] <= -2.5),
        "hard weights + soft gate -2.5": lambda r: tt.grow(prob[("hard", r)], d) & ((own_dip[r] <= -2.5) | (prob[("hard", r)] >= 0.9)),
        "hard weights + soft gate -3.0": lambda r: tt.grow(prob[("hard", r)], d) & ((own_dip[r] <= -3.0) | (prob[("hard", r)] >= 0.9)),
    }
    rows, keep = [], {}
    for name, fn in variants.items():
        earlier = np.zeros(len(weak), bool)
        for held in reports:
            pred = fn(held) | earlier
            earlier = pred
            keep[f"{name}|{held}"] = pred
            y, known = labels[held]
            m, inside = known & test, known & test & ~edges[held]
            p, r_, f1, _ = precision_recall_fscore_support(y[m], pred[m], average="binary")
            pi, ri, _, _ = precision_recall_fscore_support(y[inside], pred[inside], average="binary")
            rows.append({"variant": name, "report": held, "f1": f1, "precision": p, "recall": r_, "precision_inside": pi, "recall_inside": ri,
                         "big_extra_acres": (patches(known & ~y & pred, d) > 0).sum() * tt.ACRE_PER_PIXEL,
                         "big_missed_acres": (patches(known & y & ~pred, d) > 0).sum() * tt.ACRE_PER_PIXEL,
                         "sar_vs_original_area": pred[m].sum() / y[m].sum() - 1, "weak_called_cut": float(pred[weak & ~y].mean())})
        log.info("%s scored", name)
        pd.DataFrame(rows).to_csv(interim / f"{cfg['aoi']}_combine_scores_{area}.csv", index=False)
    np.savez_compressed(interim / f"{cfg['aoi']}_combine_pred_{area}.npz", **keep)
    print(pd.DataFrame(rows).groupby("variant", sort=False)[["f1", "precision", "recall", "precision_inside", "recall_inside", "big_extra_acres",
                                                            "big_missed_acres", "sar_vs_original_area", "weak_called_cut"]].mean().round(3).to_string())


if __name__ == "__main__":
    main(sys.argv[1])
