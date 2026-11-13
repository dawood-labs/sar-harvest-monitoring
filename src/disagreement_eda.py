"""Find the big disagreements between the SAR result and the original report, and describe each one.

Why: thin strips along field edges are a known limit of SAR; what matters are whole patches inside fields where
SAR and the original report say different things. For each such patch this script collects the evidence needed
to say who is right and why: the optical image (NDVI) on every check date and the SAR curve of the patch.

A patch = disagreement pixels that survive a 5 x 5 opening (so strips thinner than 50 m vanish) and cover at
least MIN_ACRES. Two kinds: "missed" (original report: harvested, SAR: standing) and "extra" (original report:
standing, SAR: harvested).

Usage: python src/disagreement_eda.py <area_name> [predictions.npz "<variant>" <tag> <min_acres> <width_pixels>]
       Without the extra arguments the first method's predictions are used (patches >= 1 acre, >= 50 m wide).
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


def patches(sel: np.ndarray, d: dict, min_acres: float = MIN_ACRES, width: int = 5) -> np.ndarray:
    """Patch id per crop-mask pixel (0 = not in a big patch). width = opening size in pixels (thinner strips vanish)."""
    img = np.zeros(grid_shape(d), bool)
    img[d["rows"], d["cols"]] = sel
    lab, n = ndimage.label(ndimage.binary_opening(img, np.ones((width, width), bool)))
    size = np.bincount(lab.ravel())
    small = np.flatnonzero(size * ACRE_PER_PIXEL < min_acres)
    lab[np.isin(lab, small)] = 0
    return lab[d["rows"], d["cols"]]


def main(area: str, npz: str = "", variant: str = "", tag: str = "", min_acres: float = MIN_ACRES, width: int = 5) -> None:
    cfg, d = load_pixels(area)
    interim = Path(cfg["paths"]["interim"])
    z = np.load(npz or interim / f"{cfg['aoi']}_transfer_pred_{area}.npz")
    key = (lambda r: f"{variant}|{r}") if variant else (lambda r: f"pred_{r}")
    ndvi, test, reports = ndvi_by_date(cfg, d), split(d), cfg["report_dates"]
    tracks = list(cfg["s1_tracks"])
    rows, labels = [], {}
    earlier = np.zeros(len(d["group"]), bool)
    for n, report in enumerate(reports, 1):
        y, known = harvested_by(d, n)
        pred = z[key(report)]
        own = pred & ~earlier                                  # found on this date itself, not carried over
        standing = known & ~y & ~pred                          # both agree: standing -> the reference level per pass
        for kind, sel in (("missed", known & y & ~pred), ("extra", known & ~y & pred)):
            lab = patches(sel, d, float(min_acres), int(width))
            labels[f"{kind}_{report}"] = lab.astype("int32")
            idx = np.flatnonzero(lab > 0)
            if len(idx) == 0:
                continue
            ids, inv = np.unique(lab[idx], return_inverse=True)
            count = np.bincount(inv)

            def patch_mean(v: np.ndarray) -> np.ndarray:           # mean of v inside every patch, all patches at once
                x = v[idx].astype("float64")
                ok = np.isfinite(x)
                return np.bincount(inv, weights=np.where(ok, x, 0)) / np.maximum(np.bincount(inv, weights=ok), 1)

            part = pd.DataFrame({"report": report, "kind": kind, "patch": ids, "acres": (count * ACRE_PER_PIXEL).round(2),
                                 "row": patch_mean(d["rows"]).astype(int), "col": patch_mean(d["cols"]).astype(int),
                                 "test_share": patch_mean(test).round(2),
                                 "prob": patch_mean(z[f"prob_{report}"]).round(2) if f"prob_{report}" in z.files else np.nan,
                                 "carried_over_share": patch_mean(~own).round(2),
                                 "original_group": np.argmax([patch_mean(d["group"] == g) for g in range(len(reports) + 1)], axis=0)})
            img = np.zeros(grid_shape(d), "int32")                # widest place of each patch: tells a strip from a field
            img[d["rows"], d["cols"]] = lab
            part["widest_m"] = (ndimage.maximum(ndimage.distance_transform_edt(img > 0), img, ids) * 20).round(0)
            for date in reports:
                part[f"ndvi_{date}"] = patch_mean(ndvi[date]).round(2)
            for t in tracks:                                       # patch VH minus standing cane's VH, per pass
                dates = d[f"dates_{t}"]
                rel = np.array([patch_mean(v) for v in d[f"vh_{t}"]]) - np.nanmedian(d[f"vh_{t}"][:, standing], axis=1)[:, None]
                before = dates <= report
                part[f"vh{t}_last"] = rel[before][-1].round(1)
                part[f"vh{t}_min_before"] = rel[before].min(0).round(1)
                part[f"vh{t}_next"] = rel[~before][0].round(1) if (~before).any() else np.nan
                low = rel <= -2.5
                part[f"vh{t}_first_low"] = np.where(low.any(0), dates[np.argmax(low, axis=0)], "")
            rows.append(part)
        earlier = pred
    df = pd.concat(rows, ignore_index=True)
    df.to_csv(interim / f"{cfg['aoi']}_disagreement_patches_{area}{tag}.csv", index=False)
    np.savez_compressed(interim / f"{cfg['aoi']}_disagreement_labels_{area}{tag}.npz", **labels)
    print(df.groupby(["report", "kind"]).acres.agg(["count", "sum", "median", "max"]).round(1))


if __name__ == "__main__":
    main(*sys.argv[1:7])
