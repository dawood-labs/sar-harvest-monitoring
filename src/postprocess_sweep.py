"""Which post-processing rule adds or removes difference area? Same model probabilities, different rules.

Rules tried on the saved probabilities of the first method: plain threshold, region growing with several limits,
"once cut stays cut", and a softer version that keeps an earlier cut only while the pixel still looks a little cut.
Difference area as in difference_area.py (blocks >= 0.5 acre, >= 30 m wide), whole area, all report dates added.

Usage: python src/postprocess_sweep.py <area_name>
Output: data/interim/<aoi>_postprocess_sweep_<area>.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from skimage.filters import apply_hysteresis_threshold

from difference_area import ACRE_PER_PIXEL, difference
from sar_features import load_pixels
from transfer_test import grid_shape, harvested_by


def main(area: str) -> None:
    cfg, d = load_pixels(area)
    interim, reports = Path(cfg["paths"]["interim"]), cfg["report_dates"]
    z = np.load(interim / f"{cfg['aoi']}_transfer_pred_{area}.npz")
    prob = {r: z[f"prob_{r}"] for r in reports}

    def grow(p: np.ndarray, low: float, high: float) -> np.ndarray:
        img = np.zeros(grid_shape(d), "float32")
        img[d["rows"], d["cols"]] = p
        return apply_hysteresis_threshold(img, low, high)[d["rows"], d["cols"]]

    rules = {"threshold 0.5 only": (lambda r: prob[r] >= 0.5, None)}
    for high, low in ((0.5, 0.15), (0.5, 0.1), (0.4, 0.15), (0.3, 0.1), (0.3, 0.05)):
        rules[f"growing {high}/{low}"] = (lambda r, h=high, l=low: grow(prob[r], l, h), None)
        rules[f"growing {high}/{low} + stays cut"] = (lambda r, h=high, l=low: grow(prob[r], l, h), 0.0)
    for keep in (0.05, 0.1, 0.2):
        rules[f"growing 0.5/0.15 + stays cut while prob >= {keep}"] = (lambda r: grow(prob[r], 0.15, 0.5), keep)

    rows = []
    for name, (own, keep) in rules.items():
        earlier, row = None, {"rule": name}
        for n, r in enumerate(reports, 1):
            pred = own(r)
            if keep is not None and earlier is not None:
                pred = pred | (earlier & (prob[r] >= keep))
            earlier = pred
            y, known = harvested_by(d, n)
            missed, extra = difference(y, pred, known, d)
            row[f"missed_{r}"], row[f"extra_{r}"] = missed.sum() * ACRE_PER_PIXEL, extra.sum() * ACRE_PER_PIXEL
        row["missed_total"] = sum(v for k, v in row.items() if k.startswith("missed_"))
        row["extra_total"] = sum(v for k, v in row.items() if k.startswith("extra_"))
        row["difference_total"] = row["missed_total"] + row["extra_total"]
        rows.append(row)
    df = pd.DataFrame(rows).round(1)
    df.to_csv(interim / f"{cfg['aoi']}_postprocess_sweep_{area}.csv", index=False)
    print(df[["rule", "difference_total", "missed_total", "extra_total"]].to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1])
