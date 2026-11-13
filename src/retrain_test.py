"""Two ideas from the user, tested on the difference area: rate-of-change features, and teaching the mistakes.

Features
  level   the 28 date-free level features used so far
  + rate  level plus, per track and band: the 3 changes between the last 4 passes (first derivative), the 2
          changes of those changes (second derivative), the slope over the last 3 and the last 4 passes, the
          spread of the last 4 passes; and per track the texture (spread inside 5 x 5 pixels) of VH at the last
          two passes
Training pixels
  inside              only pixels at least 20 m from any class or mask edge (as so far)
  all                 every pixel of the original report, edges included
  inside + wrong blocks xN
                      inside, plus every pixel of the BLOCKS (>= 0.5 acre) where a first model and the original
                      report disagree, with N times the weight - both kinds, missed harvest and false cut.
                      The first model's mistakes are found honestly: the training blocks are split in two halves
                      and each half is judged by a model trained on the other half.
First run (kept in ..._retrain_scores_<area>_run1_edge_pixels.csv): weighting single wrong PIXELS on the "all" set
was a disaster (thousands of acres of false cut) because edge pixels have unreliable labels. Mistakes are therefore
taken as whole blocks, which is what a wrong field is.
Nothing from the test blocks or from the left-out report date is used for training.
All variants: 5x5 in-mask mean, region growing, once-cut-stays-cut. Score = difference area (blocks >= 0.5 acre).

Usage: python src/retrain_test.py <area_name>
Output: data/interim/<aoi>_retrain_scores_<area>.csv, ..._retrain_pred_<area>.npz
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
from sar_features import load_pixels, local_std_in_mask

log = logging.getLogger("retrain_test")
SAMPLE = 300_000


def rate_features(cfg: dict, d: dict, series: dict, texture: dict, report: str) -> np.ndarray:
    cols, steps = [], np.arange(4, dtype="float32")
    with np.errstate(all="ignore"):
        for t in cfg["s1_tracks"]:
            use = np.flatnonzero(d[f"dates_{t}"] <= report)
            for b in ("vh", "vv"):
                s = series[(b, t)][use]
                last = np.nan_to_num(s[-4:] - np.nanpercentile(s[-4:], 90, axis=1)[:, None])     # scene level removed
                first = np.diff(last, axis=0)
                slope4 = ((steps - 1.5)[:, None] * last).sum(0) / 5.0                              # least-squares slope
                slope3 = (last[3] - last[1]) / 2.0
                cols += [*first, *np.diff(first, axis=0), slope3, slope4, last.std(0)]
            cols += [*np.nan_to_num(texture[t][use][-2:])]
    return np.column_stack(cols).astype("float32")


def main(area: str) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    cfg, d = load_pixels(area)
    test, reports = split(d), cfg["report_dates"]
    fold = ((d["rows"] // 300) % 2).astype(bool)                       # two halves of the training blocks
    series = smoothed_series(cfg, d, 5)
    texture = {t: local_std_in_mask(d[f"vh_{t}"], d["rows"], d["cols"], 5) for t in cfg["s1_tracks"]}
    labels = {r: tt.harvested_by(d, n) for n, r in enumerate(reports, 1)}
    edges = {r: tt.near_edge(d, *labels[r]) for r in reports}
    level = {r: tt.date_free_features(cfg, d, series, r) for r in reports}
    feats = {"level": level, "+ rate": {r: np.column_stack([level[r], rate_features(cfg, d, series, texture, r)]) for r in reports}}
    log.info("features: level %d, + rate %d", level[reports[0]].shape[1], feats["+ rate"][reports[0]].shape[1])

    interim = Path(cfg["paths"]["interim"])
    rows, keep = [], {}
    for fname, f in feats.items():
        for tname, inside_only, hard in (("inside", True, 1), ("all", False, 1), ("inside + wrong blocks x3", True, 3),
                                         ("inside + wrong blocks x10", True, 10), ("inside + wrong blocks x30", True, 30)):
            name = f"{fname} | {tname}"
            rng, earlier, row = np.random.default_rng(cfg["seed"]), np.zeros(len(test), bool), {"variant": name}
            for n, held in enumerate(reports, 1):
                xs, ys, ws = [], [], []
                for r in reports:
                    if r == held:
                        continue
                    y, known = labels[r]
                    pool = np.flatnonzero(known & ~test & (~edges[r] if inside_only else True))
                    idx = rng.choice(pool, min(len(pool), SAMPLE), replace=False)
                    w = np.ones(len(idx), "float32")
                    if hard > 1:                               # wrong blocks of each half, judged by the other half's model
                        wrong = np.zeros(len(test), bool)
                        for side in (True, False):
                            a = idx[fold[idx] == side]
                            first = tt.grow(tree_model(cfg).fit(f[r][a], y[a]).predict_proba(f[r])[:, 1], d)
                            missed, extra = difference(y, first, known, d)
                            wrong |= (missed | extra) & (fold != side) & ~test
                        add = np.setdiff1d(np.flatnonzero(wrong), idx)
                        w = np.concatenate([np.where(wrong[idx], hard, 1.0), np.full(len(add), hard)]).astype("float32")
                        idx = np.concatenate([idx, add])
                    xs.append(f[r][idx]); ys.append(y[idx]); ws.append(w)
                model = tree_model(cfg).fit(np.vstack(xs), np.concatenate(ys), sample_weight=np.concatenate(ws))
                pred = tt.grow(model.predict_proba(f[held])[:, 1], d) | earlier
                earlier = pred
                keep[f"{name}|{held}"] = pred
                y, known = labels[held]
                missed, extra = difference(y, pred, known, d)
                row[f"missed_{held}"], row[f"extra_{held}"] = missed.sum() * ACRE_PER_PIXEL, extra.sum() * ACRE_PER_PIXEL
                row[f"test_missed_{held}"], row[f"test_extra_{held}"] = (missed & test).sum() * ACRE_PER_PIXEL, (extra & test).sum() * ACRE_PER_PIXEL
            for k in ("missed", "extra", "test_missed", "test_extra"):
                row[f"{k}_total"] = sum(row[f"{k}_{r}"] for r in reports)
            row["difference_total"], row["test_difference_total"] = row["missed_total"] + row["extra_total"], row["test_missed_total"] + row["test_extra_total"]
            rows.append(row)
            pd.DataFrame(rows).round(1).to_csv(interim / f"{cfg['aoi']}_retrain_scores_{area}.csv", index=False)
            log.info("%-28s test blocks: %6.0f ac (missed %5.0f, extra %5.0f) | whole tile: %6.0f (missed %5.0f, extra %5.0f)", name,
                     row["test_difference_total"], row["test_missed_total"], row["test_extra_total"],
                     row["difference_total"], row["missed_total"], row["extra_total"])
    np.savez_compressed(interim / f"{cfg['aoi']}_retrain_pred_{area}.npz", **keep)
    log.info("finished")


if __name__ == "__main__":
    main(sys.argv[1])
