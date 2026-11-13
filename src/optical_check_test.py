"""SAR first, optical only where SAR is unsure (the user's idea), scored by difference area.

Model: the best so far (level + rate features, trained on inside pixels of the other report dates, region growing).
Doubtful pixels = SAR says standing, but the model's harvest probability is at least LOW (it is not sure).
Only those pixels are looked up in an optical image: NDVI below BARE_NDVI means bare soil -> harvested.
Optionally the other way too: SAR says harvested with probability below 0.8 but optical is clearly green -> standing.
Cloudy optical pixels change nothing. Then once-cut-stays-cut as always.
Two cases for the optical image: the one of the report date (best case), and the clear one before it (3-5 days
older; the case when the report day is cloudy).
Reported: difference area (blocks >= 0.5 acre) and how many acres had to be looked up in optical.
Caution: the original report was itself made from the report-date optical image, so the best case agrees with it
partly by construction. The older-image case is the fairer one.

Usage: python src/optical_check_test.py <area_name>
Output: data/interim/<aoi>_optical_check_scores_<area>.csv, ..._optical_check_pred_<area>.npz
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import transfer_test as tt
from compare_detectors import smoothed_series, split, tree_model
from difference_area import ACRE_PER_PIXEL, difference
from disagreement_figures import ndvi_series
from retrain_test import rate_features
from sar_features import load_pixels, local_std_in_mask

log = logging.getLogger("optical_check_test")
BARE_NDVI, GREEN_NDVI, SURE = 0.40, 0.50, 0.8


def main(area: str) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    cfg, d = load_pixels(area)
    test, reports, rng = split(d), cfg["report_dates"], np.random.default_rng(cfg["seed"])
    series = smoothed_series(cfg, d, 5)
    texture = {t: local_std_in_mask(d[f"vh_{t}"], d["rows"], d["cols"], 5) for t in cfg["s1_tracks"]}
    labels = {r: tt.harvested_by(d, n) for n, r in enumerate(reports, 1)}
    edges = {r: tt.near_edge(d, *labels[r]) for r in reports}
    feats = {r: np.column_stack([tt.date_free_features(cfg, d, series, r), rate_features(cfg, d, series, texture, r)]) for r in reports}
    dates, ndvi = ndvi_series(cfg, d)
    clear_share = {x: float(np.isfinite(v).mean()) for x, v in zip(dates, ndvi)}

    prob = {}
    for held in reports:
        idx = {r: rng.choice(np.flatnonzero(labels[r][1] & ~test & ~edges[r]), 300_000, replace=False) for r in reports if r != held}
        model = tree_model(cfg).fit(np.vstack([feats[r][i] for r, i in idx.items()]), np.concatenate([labels[r][0][i] for r, i in idx.items()]))
        prob[held] = model.predict_proba(feats[held])[:, 1]
    own = {r: tt.grow(prob[r], d) for r in reports}
    optical = {"report date": {r: r for r in reports},
               "clear image before": {r: max(x for x in dates if x < r and clear_share[x] > 0.9) for r in reports}}
    log.info("optical dates used: %s", optical)

    variants = [("SAR only", None, None, False)]
    for which in optical:
        for low in (0.2, 0.1, 0.05, 0.02):
            variants.append((f"optical ({which}) where prob >= {low}", which, low, False))
        variants.append((f"optical ({which}) where prob >= 0.05, both ways", which, 0.05, True))
    rows, keep = [], {}
    for name, which, low, both in variants:
        earlier, row = np.zeros(len(test), bool), {"variant": name}
        checked = 0.0
        for n, r in enumerate(reports, 1):
            pred = own[r].copy()
            if which:
                v = ndvi[dates.index(optical[which][r])]
                doubt = ~pred & (prob[r] >= low) & ~earlier
                pred |= doubt & (v < BARE_NDVI)
                looked = doubt
                if both:
                    unsure_cut = own[r] & (prob[r] < SURE) & ~earlier
                    pred &= ~(unsure_cut & (v > GREEN_NDVI))
                    looked = looked | unsure_cut
                checked += looked.sum() * ACRE_PER_PIXEL
                row[f"checked_{r}"] = looked.sum() * ACRE_PER_PIXEL
            pred = pred | earlier
            earlier = pred
            keep[f"{name}|{r}"] = pred
            y, known = labels[r]
            missed, extra = difference(y, pred, known, d)
            row[f"missed_{r}"], row[f"extra_{r}"] = missed.sum() * ACRE_PER_PIXEL, extra.sum() * ACRE_PER_PIXEL
            row[f"test_{r}"] = ((missed | extra) & test).sum() * ACRE_PER_PIXEL
        row["missed_total"] = sum(row[f"missed_{r}"] for r in reports)
        row["extra_total"] = sum(row[f"extra_{r}"] for r in reports)
        row["difference_total"] = row["missed_total"] + row["extra_total"]
        row["test_difference_total"] = sum(row[f"test_{r}"] for r in reports)
        row["optical_checked_acres_per_date"] = checked / len(reports)
        row["optical_checked_share_of_cane"] = checked / len(reports) / (len(test) * ACRE_PER_PIXEL)
        rows.append(row)
        log.info("%-52s difference %6.0f (missed %5.0f, extra %5.0f) | test blocks %5.0f | optical looked at %6.0f ac per date (%.1f%%)",
                 name, row["difference_total"], row["missed_total"], row["extra_total"], row["test_difference_total"],
                 row["optical_checked_acres_per_date"], 100 * row["optical_checked_share_of_cane"])
    interim = Path(cfg["paths"]["interim"])
    pd.DataFrame(rows).round(3).to_csv(interim / f"{cfg['aoi']}_optical_check_scores_{area}.csv", index=False)
    np.savez_compressed(interim / f"{cfg['aoi']}_optical_check_pred_{area}.npz", **keep,
                        **{f"prob_{r}": prob[r].astype("float32") for r in reports})
    log.info("finished")


if __name__ == "__main__":
    main(sys.argv[1])
