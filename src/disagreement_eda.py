"""Find the big disagreements between the SAR result and the original report, and describe each one.

Why: thin strips along field edges are a known limit of SAR; what matters are whole patches inside fields where
SAR and the original report say different things. For each such patch this script collects the evidence needed
to say who is right and why: the optical image (NDVI) on every check date and the SAR curve of the patch.

A patch = disagreement pixels that survive a 5 x 5 opening (so strips thinner than 50 m vanish) and cover at
least MIN_ACRES. Two kinds: "missed" (original report: harvested, SAR: standing) and "extra" (original report:
standing, SAR: harvested).

Usage: python src/disagreement_eda.py <area_name>
Output: data/interim/<aoi>_disagreement_patches_<area>.csv and ..._labels_<area>.npz (patch id per pixel, per date)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage

from compare_detectors import split
from sar_features import load_pixels
from transfer_test import grid_shape, harvested_by

ACRE_PER_PIXEL = 100 / 4046.8564224
MIN_ACRES = 1.0


def ndvi_by_date(cfg: dict, d: dict) -> dict:
    out = {}
    for date in cfg["report_dates"]:
        with rasterio.open(Path(cfg["paths"]["raw"]) / "s2" / f"{cfg['aoi']}_s2_{date}.tif") as r:
            red, nir = r.read([3, 5]).astype("float32")[:, d["rows"], d["cols"]]
        out[date] = (nir - red) / (nir + red + 1e-6)
    return out


def patches(sel: np.ndarray, d: dict) -> np.ndarray:
    """Patch id per crop-mask pixel (0 = not in a big patch)."""
    img = np.zeros(grid_shape(d), bool)
    img[d["rows"], d["cols"]] = sel
    lab, n = ndimage.label(ndimage.binary_opening(img, np.ones((5, 5), bool)))
    size = np.bincount(lab.ravel())
    small = np.flatnonzero(size * ACRE_PER_PIXEL < MIN_ACRES)
    lab[np.isin(lab, small)] = 0
    return lab[d["rows"], d["cols"]]


def main(area: str) -> None:
    cfg, d = load_pixels(area)
    interim = Path(cfg["paths"]["interim"])
    z = np.load(interim / f"{cfg['aoi']}_transfer_pred_{area}.npz")
    ndvi, test, reports = ndvi_by_date(cfg, d), split(d), cfg["report_dates"]
    tracks = list(cfg["s1_tracks"])
    rows, labels = [], {}
    earlier = np.zeros(len(d["group"]), bool)
    for n, report in enumerate(reports, 1):
        y, known = harvested_by(d, n)
        pred = z[f"pred_{report}"]
        own = pred & ~earlier                                  # found on this date itself, not carried over
        standing = known & ~y & ~pred                          # both agree: standing -> the reference level per pass
        for kind, sel in (("missed", known & y & ~pred), ("extra", known & ~y & pred)):
            lab = patches(sel, d)
            labels[f"{kind}_{report}"] = lab.astype("int32")
            for pid in np.unique(lab)[1:]:
                m = lab == pid
                row = {"report": report, "kind": kind, "patch": int(pid), "acres": round(m.sum() * ACRE_PER_PIXEL, 2),
                       "row": int(d["rows"][m].mean()), "col": int(d["cols"][m].mean()), "test_share": round(test[m].mean(), 2),
                       "prob": round(float(z[f"prob_{report}"][m].mean()), 2), "carried_over_share": round(float((~own[m]).mean()), 2),
                       "original_group": int(np.bincount(d["group"][m]).argmax())}
                for date in reports:
                    row[f"ndvi_{date}"] = round(float(np.median(ndvi[date][m])), 2)
                for t in tracks:                               # patch VH minus standing cane's VH, per pass
                    dates = d[f"dates_{t}"]
                    rel = np.nanmedian(d[f"vh_{t}"][:, m], axis=1) - np.nanmedian(d[f"vh_{t}"][:, standing], axis=1)
                    before = dates <= report
                    row[f"vh{t}_last"] = round(float(rel[before][-1]), 1)
                    row[f"vh{t}_min_before"] = round(float(rel[before].min()), 1)
                    row[f"vh{t}_next"] = round(float(rel[~before][0]), 1) if (~before).any() else np.nan
                    row[f"vh{t}_first_low"] = next((x for x, v in zip(dates, rel) if v <= -2.5), "")
                rows.append(row)
        earlier = pred
    df = pd.DataFrame(rows)
    df.to_csv(interim / f"{cfg['aoi']}_disagreement_patches_{area}.csv", index=False)
    np.savez_compressed(interim / f"{cfg['aoi']}_disagreement_labels_{area}.npz", **labels)
    print(df.groupby(["report", "kind"]).acres.agg(["count", "sum", "median", "max"]).round(1))


if __name__ == "__main__":
    main(sys.argv[1])
