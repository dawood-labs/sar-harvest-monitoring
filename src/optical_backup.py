"""Optical backup for the SAR harvest report: use a clear Sentinel-2 image only where SAR is not sure.

Why: SAR misses fresh cuts and some cut fields that keep a high backscatter. A clear optical image near the
report date sees those. But a fixed NDVI limit is wrong here (weak cane is not green), so the optical rule is
LEARNED AGAIN FOR EVERY REPORT from the pixels where SAR is sure:
  teachers   harvested = SAR probability >= SURE_CUT (or harvested in an earlier report), standing = <= SURE_STAND
  features   per pixel, the image near the report AND the pixel's own change since a clear early-season image
             (NDVI now, NDVI change, red now and its ratio to before, NIR change, red-edge, brightness change)
  model      a small LightGBM per report date -> optical probability of 'harvested' for every clear pixel
So the limits adapt to the area, the week and the health of the cane. Cloudy pixels are never touched.
Variants scored against the original reports (difference area, blocks >= 0.5 acre):
  sar only        probability >= 0.5, earlier harvest stays
  add             SAR unsure-standing (LOW..0.5) and optical says harvested          -> harvested
  add + remove    also: SAR unsure-harvested (0.5..SURE_CUT) and optical says clearly standing -> standing
  add + blind     also: SAR sure-standing but optical is very sure of harvest        -> harvested
  all three
The optical image may be a few days older than the SAR pass, so 'remove' is the risky one (a field cut after
the optical image would be flipped back).

Config key optical_tuned: path of the json written by optical_tune.py; its limits and teachers replace the
hand-set ones and the outputs get the suffix _tuned.

Usage: python src/optical_backup.py <area> <report config.yaml> <reference S2 date> <report date=S2 date> [...]
       e.g. ... full configs/<report config>.yaml 2025-11-23 2025-12-31=2025-12-28 2026-01-10=2026-01-07 2026-01-13=2026-01-12
Output: data/interim/<aoi>_optical_backup_<area><output_tag>.csv; rasters of the best variant in
        outputs/<aoi>_<area><output_tag>_optical/
"""
from __future__ import annotations

import sys
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import rasterio
import yaml

from disagreement_eda import patches
from score_reports import AC, labels

warnings.filterwarnings("ignore")
SURE_CUT, SURE_STAND, LOW, OPT_CUT, OPT_STAND, OPT_VERY = 0.9, 0.05, 0.05, 0.5, 0.2, 0.9
REM_TOP = SURE_CUT                  # top of the 'unsure harvested' band that optical may remove
PER_CLASS = 200_000


def s2_pixels(f: Path, rows: np.ndarray, cols: np.ndarray) -> dict:
    """Blue, green, red, red-edge, NIR reflectance and the clear flag at the crop-mask pixels."""
    def band(b):                                   # every thread opens the file itself (a shared handle is not thread-safe)
        with rasterio.open(f) as s:
            return s.read(b)[rows, cols]
    with ThreadPoolExecutor(3) as ex:
        a = list(ex.map(band, range(1, 7)))
    v = {k: x.astype("float32") / 10000 for k, x in zip(("blue", "green", "red", "edge", "nir"), a[:5])}
    v["clear"] = (a[4] > 0) & (a[5] >= 60)
    v["ndvi"] = (v["nir"] - v["red"]) / (v["nir"] + v["red"] + 1e-6)
    v["bright"] = (v["blue"] + v["green"] + v["red"]) / 3
    return v


def features(now: dict, ref: dict) -> np.ndarray:
    with np.errstate(all="ignore"):
        cols = [now["ndvi"], now["ndvi"] - ref["ndvi"], ref["ndvi"], now["red"], now["red"] / (ref["red"] + 1e-3), now["nir"] - ref["nir"],
                now["edge"], now["bright"] - ref["bright"], now["bright"], (now["nir"] - now["edge"]) / (now["nir"] + now["edge"] + 1e-6)]
    return np.column_stack(cols).astype("float32")


def main(area: str, cfg_path: str, ref_day: str, *pairs: str) -> None:
    global SURE_CUT, SURE_STAND, LOW, OPT_CUT, OPT_STAND, OPT_VERY, REM_TOP
    cfg = yaml.safe_load(open(cfg_path))
    aoi, interim, tag = cfg["aoi"], Path(cfg["paths"]["interim"]), cfg.get("output_tag", "")
    if cfg.get("optical_tuned"):                     # limits found by optical_tune.py (json), instead of the hand-set ones
        import json
        tuned = json.load(open(cfg["optical_tuned"]))
        lim = tuned["limits"]
        LOW, OPT_CUT, REM_TOP, OPT_STAND, OPT_VERY = lim["low"], lim["opt_cut"], lim["rem_top"], lim["opt_stand"], lim["opt_very"]
        SURE_CUT, SURE_STAND = tuned["teachers"]
        tag += "_tuned"
        print("tuned limits:", lim, "teachers:", tuned["teachers"], flush=True)
    s2_of = dict(p.split("=") for p in pairs)
    with rasterio.open(interim / f"{aoi}_cane_mask_{area}.tif") as m:
        inside, profile = m.read(1) == 1, m.profile
    rows, cols = np.nonzero(inside)
    d = {"rows": rows, "cols": cols}
    s2 = Path(cfg["paths"]["raw"]) / "s2"
    ref = s2_pixels(s2 / f"{aoi}_s2_{ref_day}.tif", rows, cols)
    folder = Path("outputs") / f"{aoi}_{area}{cfg.get('output_tag', '')}"
    out_dir = Path(str(folder) + "_optical" + ("_tuned" if cfg.get("optical_tuned") else ""))
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg["seed"])
    names = ["sar only", "add", "add + remove", "add + blind", "add + remove + blind"]
    earlier = {n: np.zeros(len(rows), bool) for n in names}
    res, keep = [], {}
    for n, r in enumerate(cfg["report_dates"], 1):
        with rasterio.open(folder / f"{aoi}_{area}_sar_probability_{r}_v1.tif") as s:
            p = s.read(1)[rows, cols].astype("float32") / 100
        now = s2_pixels(s2 / f"{aoi}_s2_{s2_of[str(r)]}.tif", rows, cols)
        ok = now["clear"] & ref["clear"]
        x = features(now, ref)
        cut_t = np.flatnonzero(ok & ((p >= SURE_CUT) | earlier["sar only"]))          # teachers: where SAR is sure
        stand_t = np.flatnonzero(ok & (p <= SURE_STAND) & ~earlier["sar only"])
        idx = np.r_[rng.choice(cut_t, min(len(cut_t), PER_CLASS), replace=False), rng.choice(stand_t, min(len(stand_t), PER_CLASS), replace=False)]
        y = np.r_[np.ones(min(len(cut_t), PER_CLASS)), np.zeros(min(len(stand_t), PER_CLASS))]
        model = lgb.LGBMClassifier(n_estimators=150, num_leaves=15, learning_rate=0.08, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                   n_jobs=-1, random_state=cfg["seed"], verbose=-1).fit(x[idx], y)
        po = np.where(ok, model.predict_proba(x)[:, 1], np.nan)
        sar = p >= 0.5
        add = ok & (p >= LOW) & (p < 0.5) & (po >= OPT_CUT)
        remove = ok & (p >= 0.5) & (p < REM_TOP) & (po <= OPT_STAND)
        blind = ok & (p < LOW) & (po >= OPT_VERY)
        rules = {"sar only": sar, "add": sar | add, "add + remove": (sar | add) & ~remove, "add + blind": sar | add | blind,
                 "add + remove + blind": (sar | add | blind) & ~remove}
        n_add, n_rem, n_blind = (float((v & ~earlier["sar only"]).sum() * AC) for v in (add, remove, blind))
        yv, known = labels(cfg, area, n, rows, cols)
        for name in names:
            pred = rules[name] | earlier[name]
            earlier[name] = pred
            keep[(name, r)] = pred
            missed, extra = patches(known & yv & ~pred, d, 0.5, 3) > 0, patches(known & ~yv & pred, d, 0.5, 3) > 0
            res.append({"variant": name, "report": r, "optical_image": s2_of[str(r)], "clear_share_of_cane": float(ok.mean()),
                        "teachers_cut": len(cut_t), "teachers_standing": len(stand_t),
                        "added_acres": n_add if "add" in name else 0.0, "removed_acres": n_rem if "remove" in name else 0.0,
                        "blind_added_acres": n_blind if "blind" in name else 0.0,
                        "missed_acres": missed.sum() * AC, "extra_acres": extra.sum() * AC, "difference_acres": (missed.sum() + extra.sum()) * AC})
        print(r, "done; optical clear on", f"{ok.mean():.0%}", "of the cane", flush=True)
    df = pd.DataFrame(res)
    df.to_csv(interim / f"{aoi}_optical_backup_{area}{tag}.csv", index=False)
    total = df.groupby("variant", sort=False)[["missed_acres", "extra_acres", "difference_acres", "added_acres", "removed_acres", "blind_added_acres"]].sum().round(0)
    pd.set_option("display.width", 200)
    print(total.to_string())
    print(df.pivot(index="variant", columns="report", values="difference_acres").loc[names].round(0).to_string())
    best = total.difference_acres.idxmin()
    profile.update(dtype="uint8", nodata=0, count=1, compress="deflate", tiled=True, blockxsize=256, blockysize=256)
    for r in cfg["report_dates"]:
        img = np.zeros(inside.shape, "uint8")
        img[rows, cols] = np.where(keep[(best if not cfg.get("optical_tuned") else "add + remove + blind", r)], 82, 81)
        with rasterio.open(out_dir / f"{aoi}_{area}_sar_harvest_{r}_v1.tif", "w", **profile) as dst:
            dst.write(img, 1)
    print("best variant:", best, "-> rasters in", out_dir)


if __name__ == "__main__":
    main(*sys.argv[1:])
