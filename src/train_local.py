"""Teach the model a NEW area from clear optical images, when no original harvest reports exist there.

Why: a model trained in one area is unsure in another (other tracks, soils, crop condition). Optical images are
only needed here, for teaching; the weekly reports are still made from SAR alone.
Labels from each clear, full-cover optical date (crop-mask pixels only):
  harvested = NDVI below BARE (bare soil), standing = NDVI above GREEN, pixels in between are not used.
Features: the same 64 date-free SAR features as always, taken as of the first SAR pass on or after the optical date,
so a field cut just before the optical image has a SAR pass after it.
Test: leave one optical date out AND use only the test blocks of a 3 km checkerboard. The old (transferred) model
is scored on exactly the same pixels. Then a final local model is trained on everything and saved.
Tiles run in parallel processes; each returns a random sample of its labelled pixels, both classes at the same
rate, so the sample keeps the real share of harvested cane. The sample is cached in data/interim and reused.
Config key extra_training: an .npz (X, y) with samples of another area, added to every training set.

The same script also trains on the area's OWN original reports when they exist (word "original" instead of dates):
labels = harvested by report n / still standing, pixels at least 20 m inside their class, features as of the
n-th SAR report date. A model without each date is saved too, so that date can be scored fairly on the whole area.

Usage: python src/train_local.py <area> <config.yaml> <optical date> [<optical date> ...]
       python src/train_local.py <area> <config.yaml> original
Output: runs/<today>_<aoi>_local/ or ..._own_reports/: model.txt, model_without_<date>.txt, settings.json, scores.csv
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import rasterio
import yaml
from rasterio.windows import Window
from scipy import ndimage
from sklearn.metrics import precision_recall_fscore_support

log = logging.getLogger("train_local")
BARE, GREEN, TILE, HALO, SHARE = 0.25, 0.38, 2048, 64, 0.15       # SHARE of the labelled pixels is sampled
G = {}


def init(cfg_path: str, area: str, files: dict, asof: dict, original: bool) -> None:
    import predict_area
    import sar_features
    sar_features.workers = predict_area.workers = lambda *a, **k: 2      # the processes are the parallel level here
    cfg = yaml.safe_load(open(cfg_path))
    # look-direction mode keeps every track under its own number; otherwise tracks map to the model's slots
    slots = {int(t): int(t) for t in cfg["s1_tracks"]} if cfg.get("directions") else {int(k): int(v) for k, v in cfg["model_tracks"].items()}
    G.update(cfg=cfg, area=area, files=files, asof=asof, slots=slots, original=original,
             model_cfg={"s1_tracks": {slots[int(t)]: d for t, d in cfg["s1_tracks"].items()}})


def tile(rc: tuple[int, int]):
    import transfer_test as tt
    from predict_area import read_tile
    from retrain_test import rate_features
    from sar_features import local_std_in_mask, smooth_in_mask
    r0, c0 = rc
    cfg, area, slots, model_cfg = G["cfg"], G["area"], G["slots"], G["model_cfg"]
    with rasterio.open(Path(cfg["paths"]["interim"]) / f"{cfg['aoi']}_cane_mask_{area}.tif") as m:
        H, W = m.height, m.width
        win = Window(c0 - HALO, r0 - HALO, TILE + 2 * HALO, TILE + 2 * HALO)
        mask = m.read(1, window=win, boundless=True, fill_value=0) == 1
    core = np.zeros_like(mask)
    core[HALO:HALO + min(TILE, H - r0), HALO:HALO + min(TILE, W - c0)] = True
    if not (mask & core).any():
        return None
    rows, cols = np.nonzero(mask)
    d = {"rows": rows.astype("int32"), "cols": cols.astype("int32")}
    for t, (dates, files) in G["files"].items():
        a = read_tile(list(files), win, mask)
        d[f"dates_{slots[t]}"], d[f"vv_{slots[t]}"], d[f"vh_{slots[t]}"] = dates, a[:, 0], a[:, 1]
    series = {(b, s): smooth_in_mask(d[f"{b}_{s}"], d["rows"], d["cols"], int(cfg.get("smooth", 5))) for s in model_cfg["s1_tracks"] for b in ("vh", "vv")}
    texture = {s: local_std_in_mask(d[f"vh_{s}"], d["rows"], d["cols"], 5) for s in model_cfg["s1_tracks"]}
    keep = core[rows, cols]
    test = (((rows + r0 - HALO) // 300 + (cols + c0 - HALO) // 300) % 2).astype(bool)
    rng = np.random.default_rng(r0 * 100003 + c0)
    out = []
    cut_before = np.zeros(mask.shape, bool)
    for k, (opt, asof) in enumerate(G["asof"].items()):
        if G["original"]:                                    # labels = the area's own original reports (incremental)
            with rasterio.open(Path(cfg["paths"]["interim"]) / f"{cfg['aoi']}_original_{area}_{opt}.tif") as s:
                v = s.read(1, window=win, boundless=True, fill_value=0)
            cut_before |= v == 82
            known = cut_before | (v == 81)
            inner = np.ones((5, 5), bool)                    # keep pixels at least 20 m inside their own class
            is_cut = ndimage.binary_erosion(cut_before & known, inner)[rows, cols] & keep
            is_standing = ndimage.binary_erosion(~cut_before & known, inner)[rows, cols] & keep
            if cfg.get("train_on_all_pixels"):               # edge pixels too (their labels are less sure)
                is_cut, is_standing = (cut_before & known)[rows, cols] & keep, (~cut_before & known)[rows, cols] & keep
        else:                                                # labels = a clear optical image
            with rasterio.open(Path(cfg["paths"]["raw"]) / "s2" / f"{cfg['aoi']}_s2_{opt}.tif") as s:
                red, nir, clear = s.read([3, 5, 6], window=win, boundless=True, fill_value=0).astype("float32")[:, mask]
            ndvi = (nir - red) / (nir + red + 1e-6)
            ok = keep & (nir > 0) & (clear >= 60)
            is_cut, is_standing = ok & (ndvi < BARE), ok & (ndvi > GREEN)
        # Both classes are sampled at the SAME rate, so the sample keeps the real share of harvested cane.
        # (Equal numbers per class teach the model that harvest is common, and it then over-calls it.)
        idx = np.flatnonzero(is_cut | is_standing)
        idx = idx[rng.random(len(idx)) < SHARE]
        if len(idx):
            if cfg.get("directions"):                        # 35 features per look direction, first covering track wins
                from direction_features import direction_features
                x = [direction_features(cfg["directions"], model_cfg["s1_tracks"], d, series, texture, asof)]
            else:
                x = [tt.date_free_features(model_cfg, d, series, asof), rate_features(model_cfg, d, series, texture, asof)]
            if cfg.get("field_features"):                    # + the same level features of the pixel's segment and patch
                from field_features import field_features
                x.append(field_features(model_cfg, d, series, mask, asof))
            x = np.column_stack(x)
            out.append((x[idx], is_cut[idx].astype("int8"), np.full(len(idx), k, "int8"), test[idx], len(idx)))
    return out


def main(area: str, cfg_path: str, *optical: str) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    t0 = time.time()
    from predict_area import track_files
    cfg = yaml.safe_load(open(cfg_path))
    mask_file = Path(cfg["paths"]["interim"]) / f"{cfg['aoi']}_cane_mask_{area}.tif"
    files = track_files(cfg, mask_file)
    every = sorted({str(x) for dates, _ in files.values() for x in dates})
    original = optical == ("original",)
    if original:      # label = n-th original report, features as of the n-th SAR report date (last pass on or before it)
        optical = tuple(str(x) for x in cfg["original_reports"])
        asof = dict(zip(optical, cfg["report_dates"]))
    else:
        asof = {o: min(x for x in every if x >= o) for o in optical}
    log.info("label date -> SAR as-of date: %s", asof)
    smooth_tag = "" if int(cfg.get("smooth", 5)) == 5 else f"_smooth{cfg['smooth']}"     # averaging window in pixels (config key smooth)
    smooth_tag += ("_fields" if cfg.get("field_features") else "") + ("_allpixels" if cfg.get("train_on_all_pixels") else "")
    smooth_tag += "" if len(asof) == 3 else f"_{len(asof)}dates"
    smooth_tag += "_directions" if cfg.get("directions") else ""
    cache = Path(cfg["paths"]["interim"]) / f"{cfg['aoi']}_train_sample_{area}_{'original' if original else 'optical'}{smooth_tag}.npz"
    if cache.exists():                                       # the sample is costly to build: reuse it
        z = np.load(cache)
        X, y, k, test = z["X"], z["y"], z["k"], z["test"]
    else:
        with rasterio.open(mask_file) as m:
            jobs = [(r0, c0) for r0 in range(0, m.height, TILE) for c0 in range(0, m.width, TILE)]
        with ProcessPoolExecutor(os.cpu_count() // 2, initializer=init, initargs=(cfg_path, area, files, asof, original)) as ex:
            parts = [p for res in ex.map(tile, jobs) if res for p in res]
        X = np.vstack([p[0] for p in parts]); y = np.concatenate([p[1] for p in parts])
        k = np.concatenate([p[2] for p in parts]); test = np.concatenate([p[3] for p in parts])
        np.savez(cache, X=X, y=y, k=k, test=test)
    # Optional samples of another area (same 64 features), added to every training set but never to a test set.
    other = np.load(cfg["extra_training"]) if cfg.get("extra_training") else None
    if other is not None:
        log.info("plus %d samples of another area (%d harvested)", len(other["y"]), int(other["y"].sum()))

    def fit(sel: np.ndarray) -> lgb.LGBMClassifier:
        Xa, ya = (X[sel], y[sel]) if other is None else (np.vstack([X[sel], other["X"]]), np.concatenate([y[sel], other["y"]]))
        return lgb.LGBMClassifier(**params).fit(Xa, ya)

    log.info("samples %d (harvested %d = %.0f %%, standing %d) after %.0f s", len(y), int(y.sum()), 100 * y.mean(), int((y == 0).sum()), time.time() - t0)

    params = dict(n_estimators=300, num_leaves=31, learning_rate=0.08, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                  n_jobs=os.cpu_count(), random_state=cfg["seed"], verbose=-1)
    old = lgb.Booster(model_file=cfg["model_file"])
    run = Path("runs") / f"{date.today()}_{cfg['aoi']}_{'own_reports' if original else 'local'}{'_plus_other_area' if other is not None else ''}{smooth_tag}"
    run.mkdir(parents=True, exist_ok=True)
    rows = []
    for j, opt in enumerate(optical):                        # leave one optical date out, score on test blocks only
        tr, te = (k != j) & ~test, (k == j) & test
        m = fit(tr)
        m.booster_.save_model(str(run / f"model_without_{opt}.txt"))      # for a fair whole-area score of that date
        if cfg.get("directions"):                            # also one model per look direction (first / second half of the columns)
            half = X.shape[1] // 2
            for name, cols in (("des", slice(0, half)), ("asc", slice(half, None))):
                lgb.LGBMClassifier(**params).fit(X[tr][:, cols], y[tr]).booster_.save_model(str(run / f"model_without_{opt}_{name}.txt"))
        both = [("local model (other dates, other blocks)", m.predict_proba(X[te])[:, 1])]
        if old.num_feature() == X.shape[1]:
            both.append(("transferred model", old.predict(X[te])))
        for name, p in both:
            pr, rc, f1, _ = precision_recall_fscore_support(y[te], p >= 0.5, average="binary")
            mid = float(((p > 0.1) & (p < 0.9)).mean())
            rows.append({"held_out_optical_date": opt, "model": name, "test_pixels": int(te.sum()), "precision": pr, "recall": rc, "f1": f1,
                         "unsure_share_0.1_to_0.9": mid})
            log.info("%s %-42s P %.3f R %.3f F1 %.3f unsure %.1f%%", opt, name, pr, rc, f1, 100 * mid)
    pd.DataFrame(rows).to_csv(run / "scores.csv", index=False)
    final = fit(np.ones(len(y), bool))
    final.booster_.save_model(str(run / "model.txt"))
    if cfg.get("directions"):
        half = X.shape[1] // 2
        for name, cols in (("des", slice(0, half)), ("asc", slice(half, None))):
            lgb.LGBMClassifier(**params).fit(X[:, cols], y).booster_.save_model(str(run / f"model_{name}.txt"))
    (run / "settings.json").write_text(json.dumps({"area": area, "aoi": cfg["aoi"], "optical_dates": asof, "labels": "original reports" if original else "optical", "bare_ndvi": BARE, "green_ndvi": GREEN,
                                                   "samples": int(len(y)), "sample_share": SHARE, "extra_training": cfg.get("extra_training"), "features": int(X.shape[1]), "tracks": cfg.get("model_tracks") or cfg.get("directions"),
                                                   "lightgbm": params}, indent=1, default=str))
    with open("runs/experiments.md", "a") as f:
        f.write(f"- {date.today()} | {run.name} | local model, labels from {'original reports' if original else 'optical'} ({', '.join(optical)}), {len(y)} samples | scores in {run}/scores.csv\n")
    log.info("final model saved in %s (%.0f s)", run, time.time() - t0)


if __name__ == "__main__":
    main(*sys.argv[1:])
