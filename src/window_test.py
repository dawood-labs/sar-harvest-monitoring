"""Does a bigger window help? Two meanings are tested, with and without extra weight on hard examples.

  space: the in-mask mean is taken over 5x5, 7x7, 9x9 or 11x11 pixels before the model sees the data
  time : the model looks 6 passes back from the report date instead of 4 (the second track has no more than 6
         passes before the first report date, so 6 is the most that can be tested here)
Hard examples (x10): standing pixels that a first model calls harvested on the training dates get 10 times the
weight in a second training round.
All variants: trained on other report dates, 3 km block split, region growing, once-cut-stays-cut.

Usage: python src/window_test.py <area_name>
Output: data/interim/<aoi>_window_scores_<area>.csv
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
from disagreement_eda import patches
from sar_features import load_pixels

log = logging.getLogger("window_test")
WEAK = ["P02", "P03", "P04", "P06"]          # reviewed patches: standing weak cane (P03 until the last report)


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
    weak = {p.patch_id: rasterize([(p.geometry, 1)], out_shape=tt.grid_shape(d), transform=tr, dtype="uint8")[d["rows"], d["cols"]] == 1
            for _, p in rv.iterrows() if p.patch_id in WEAK}

    rows = []
    for size, back in ((5, 4), (7, 4), (9, 4), (11, 4), (5, 6), (9, 6)):
        series = smoothed_series(cfg, d, size)
        tt.PASSES_BACK = back
        feats = {r: tt.date_free_features(cfg, d, series, r) for r in reports}
        for hard in (1, 10):
            rng, earlier, weak_share = np.random.default_rng(cfg["seed"]), np.zeros(len(d["group"]), bool), []
            for n, held in enumerate(reports, 1):
                idx = {r: rng.choice(np.flatnonzero(labels[r][1] & ~test & ~edges[r]), 300_000, replace=False) for r in reports if r != held}
                x = np.vstack([feats[r][i] for r, i in idx.items()])
                y_tr = np.concatenate([labels[r][0][i] for r, i in idx.items()])
                model = tree_model(cfg).fit(x, y_tr)
                if hard > 1:                               # second round: more weight where the first model was fooled
                    w = np.where(~y_tr & (model.predict_proba(x)[:, 1] >= 0.3), hard, 1.0)
                    model = tree_model(cfg).fit(x, y_tr, sample_weight=w)
                pred = tt.grow(model.predict_proba(feats[held])[:, 1], d) | earlier
                earlier = pred
                y, known = labels[held]
                m, inside, edge = known & test, known & test & ~edges[held], known & test & edges[held]
                f1 = precision_recall_fscore_support(y[m], pred[m], average="binary")[2]
                pi, ri, _, _ = precision_recall_fscore_support(y[inside], pred[inside], average="binary")
                fe = precision_recall_fscore_support(y[edge], pred[edge], average="binary")[2]
                if n < len(reports):
                    weak_share.append(np.mean([pred[v].mean() for v in weak.values()]))
                rows.append({"smooth": f"{size}x{size}", "passes_back": back, "hard_weight": hard, "report": held, "f1": f1, "f1_edge": fe,
                             "precision_inside": pi, "recall_inside": ri,
                             "big_extra_acres": (patches(known & ~y & pred, d) > 0).sum() * tt.ACRE_PER_PIXEL,
                             "big_missed_acres": (patches(known & y & ~pred, d) > 0).sum() * tt.ACRE_PER_PIXEL,
                             "sar_vs_original_area": pred[m].sum() / y[m].sum() - 1, "weak_called_cut": np.nan})
            for r in rows[-len(reports):]:
                r["weak_called_cut"] = float(np.mean(weak_share))
            log.info("%sx%s, %d passes back, hard x%d done", size, size, back, hard)
            pd.DataFrame(rows).to_csv(interim / f"{cfg['aoi']}_window_scores_{area}.csv", index=False)   # keep what is done
    df = pd.DataFrame(rows)
    print(df.groupby(["smooth", "passes_back", "hard_weight"], sort=False)[
        ["f1", "f1_edge", "precision_inside", "recall_inside", "big_extra_acres", "big_missed_acres", "sar_vs_original_area", "weak_called_cut"]].mean().round(3).to_string())


if __name__ == "__main__":
    main(sys.argv[1])
