"""What if the SAR report waits for the next pass? And can the fresh cuts be picked out on the report date?

Part A - delayed report. For each original report date the SAR result is made twice: with the passes up to the
report date (as now), and with the passes up to the next pass of the main track (about 6 days later).
The delayed result is compared with the original report of the report date. Its extra area is then split with the
NEXT original report: blocks that report calls harvested were cut after the report date (not errors).
Part B - picking out fresh cuts on the report date itself. Fresh cuts = pure harvested blocks (5 x 5 pixels) that
the on-time SAR result misses and the delayed result finds. Two scores are tried as a flag on blocks SAR calls
standing: the model's harvest probability, and the latest dip of the block against its own previous 3 passes.
For several flag levels: share of the fresh-cut blocks caught, and how many standing blocks get flagged with them.
Model: date-free tree, trained on the other report dates, region growing; no stays-cut here, to see each date alone.

Usage: python src/delayed_report_test.py <area_name>
Output: data/interim/<aoi>_delayed_report_<area>.csv, ..._fresh_cut_flag_<area>.csv, ..._delayed_pred_<area>.npz
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import transfer_test as tt
from block_eval import SIDE, pure_blocks
from compare_detectors import smoothed_series, split, tree_model
from difference_area import ACRE_PER_PIXEL, difference
from dip_features import dip_features
from sar_features import load_pixels

log = logging.getLogger("delayed_report_test")
BLOCK_ACRES = SIDE * SIDE * ACRE_PER_PIXEL


def main(area: str) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    cfg, d = load_pixels(area)
    test, reports, rng = split(d), cfg["report_dates"], np.random.default_rng(cfg["seed"])
    main_track = next(iter(cfg["s1_tracks"]))
    passes = d[f"dates_{main_track}"]
    later = {r: str(passes[passes > r][0]) for r in reports}               # next pass of the main track after the report
    series = smoothed_series(cfg, d, 5)
    labels = {r: tt.harvested_by(d, n) for n, r in enumerate(reports, 1)}
    edges = {r: tt.near_edge(d, *labels[r]) for r in reports}
    feats = {r: tt.date_free_features(cfg, d, series, r) for r in reports}
    rows, flags, keep = [], [], {}
    for n, held in enumerate(reports, 1):
        idx = {r: rng.choice(np.flatnonzero(labels[r][1] & ~test & ~edges[r]), 300_000, replace=False) for r in reports if r != held}
        model = tree_model(cfg).fit(np.vstack([feats[r][i] for r, i in idx.items()]), np.concatenate([labels[r][0][i] for r, i in idx.items()]))
        prob_now = model.predict_proba(feats[held])[:, 1]
        prob_later = model.predict_proba(tt.date_free_features(cfg, d, series, later[held]))[:, 1]
        now, delayed = tt.grow(prob_now, d), tt.grow(prob_later, d)
        keep[f"on time|{held}"], keep[f"delayed|{held}"] = now, delayed
        y, known = labels[held]
        for name, pred in (("on time", now), ("delayed to the next pass", delayed)):
            missed, extra = difference(y, pred, known, d)
            row = {"report": held, "sar_made": name, "last_pass_used": held if name == "on time" else later[held],
                   "missed_acres": missed.sum() * ACRE_PER_PIXEL, "extra_acres": extra.sum() * ACRE_PER_PIXEL}
            if n < len(reports):                               # extra blocks that the next original report calls harvested
                cut_next = labels[reports[n]][0]
                row["extra_cut_by_next_original_report"] = (extra & cut_next).sum() * ACRE_PER_PIXEL
            rows.append(row)

        # Part B: flag scores on pure blocks that the on-time result calls standing.
        block, cut, stand = pure_blocks(y, known, d)
        share = lambda v: np.bincount(block, weights=v, minlength=block.max() + 1) / (SIDE * SIDE)
        sar_now, sar_later, p_block = share(now) >= 0.5, share(delayed) >= 0.5, share(prob_now)
        with np.errstate(all="ignore"):
            dip = np.minimum.reduce([dip_features(series[("vh", t)][d[f"dates_{t}"] <= held])[:, 5] for t in cfg["s1_tracks"]])
        dip_block = share(np.nan_to_num(dip))
        fresh = cut[~sar_now[cut] & sar_later[cut]]            # missed now, found one pass later
        never = cut[~sar_now[cut] & ~sar_later[cut]]
        standing = stand[~sar_now[stand]]
        for score, values, levels in (("harvest probability >=", p_block, (0.02, 0.05, 0.1, 0.2, 0.3)),
                                      ("latest dip (dB) <=", -dip_block, (1.0, 1.5, 2.0, 2.5, 3.0))):
            for level in levels:
                flags.append({"report": held, "flag": f"{score} {level if 'prob' in score else -level}",
                              "fresh_cut_blocks": len(fresh), "fresh_caught": float((values[fresh] >= level).mean()) if len(fresh) else np.nan,
                              "never_found_blocks": len(never), "never_found_caught": float((values[never] >= level).mean()) if len(never) else np.nan,
                              "standing_blocks": len(standing), "standing_flagged": float((values[standing] >= level).mean()),
                              "standing_flagged_acres": float((values[standing] >= level).sum() * BLOCK_ACRES),
                              "fresh_caught_acres": float((values[fresh] >= level).sum() * BLOCK_ACRES)})
        log.info("%s done (delayed uses passes up to %s)", held, later[held])
    interim = Path(cfg["paths"]["interim"])
    a, b = pd.DataFrame(rows).round(1), pd.DataFrame(flags)
    a.to_csv(interim / f"{cfg['aoi']}_delayed_report_{area}.csv", index=False)
    b.to_csv(interim / f"{cfg['aoi']}_fresh_cut_flag_{area}.csv", index=False)
    np.savez_compressed(interim / f"{cfg['aoi']}_delayed_pred_{area}.npz", **keep)
    print(a.to_string(index=False))
    print(b.groupby("flag", sort=False)[["fresh_cut_blocks", "fresh_caught", "fresh_caught_acres", "never_found_blocks", "never_found_caught",
                                         "standing_blocks", "standing_flagged", "standing_flagged_acres"]].mean().round(3).to_string())


if __name__ == "__main__":
    main(sys.argv[1])
