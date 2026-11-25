"""Tune the limits of the optical backup (optical_backup.py) in a way that cannot fool us.

The backup has five hand-set limits. Here they are searched, but:
  - the search only looks at the TRAINING half of the area (3 km checkerboard); the other half is used once, at the
    end, to report what the chosen limits are worth on land the search never saw;
  - the search is a coordinate descent (one limit at a time, all its values in parallel, keep the best, two rounds),
    so a few dozen settings are tried, not thousands - little room for chance hits;
  - every setting is scored with the difference area: difference area against the original reports, all dates.
Limits (SAR probability p, optical probability po):
  low        SAR 'unsure standing' band starts here:      low <= p < 0.5 and po >= opt_cut   -> harvested
  opt_cut    optical probability needed to add harvest
  rem_top    SAR 'unsure harvested' band ends here:       0.5 <= p < rem_top and po <= opt_stand -> standing
  opt_stand  optical probability under which harvest is removed
  opt_very   optical probability needed where SAR is sure of standing (p < low)            -> harvested
Teachers of the optical model (SAR-sure pixels) are a second, outer choice: (sure harvested, sure standing).

Usage: python src/optical_tune.py <area> <report config.yaml> <reference S2 date> <report date=S2 date> [...]
Output: data/interim/<aoi>_optical_tune_<area><output_tag>.csv (every setting tried), ..._optical_tuned.json,
        rasters of the tuned result in outputs/<aoi>_<area><output_tag>_optical_tuned/
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import yaml

warnings.filterwarnings("ignore")
DEFAULT = {"low": 0.05, "opt_cut": 0.5, "rem_top": 0.9, "opt_stand": 0.2, "opt_very": 0.9}
GRID = {"low": [0.01, 0.02, 0.05, 0.10, 0.20], "opt_cut": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        "rem_top": [0.5, 0.7, 0.8, 0.9, 0.97, 1.01], "opt_stand": [0.05, 0.1, 0.2, 0.3, 0.4],
        "opt_very": [0.6, 0.7, 0.8, 0.9, 0.95, 1.01]}                      # rem_top 0.5 = never remove, opt_very 1.01 = never
TEACHERS = [(0.9, 0.05), (0.8, 0.10), (0.97, 0.02)]
G: dict = {}


def prepare(area: str, cfg_path: str, ref_day: str, pairs: tuple, cache: Path) -> None:
    """SAR probability, optical probability (per teacher choice), labels: once, saved for the worker processes."""
    import lightgbm as lgb
    from optical_backup import PER_CLASS, features, s2_pixels
    from score_reports import labels
    cfg = yaml.safe_load(open(cfg_path))
    aoi, interim = cfg["aoi"], Path(cfg["paths"]["interim"])
    s2_of = dict(p.split("=") for p in pairs)
    with rasterio.open(interim / f"{aoi}_cane_mask_{area}.tif") as m:
        rows, cols = np.nonzero(m.read(1) == 1)
    s2 = Path(cfg["paths"]["raw"]) / "s2"
    ref = s2_pixels(s2 / f"{aoi}_s2_{ref_day}.tif", rows, cols)
    folder = Path("outputs") / f"{aoi}_{area}{cfg.get('output_tag', '')}"
    out = {"rows": rows.astype("int32"), "cols": cols.astype("int32")}
    earlier = np.zeros(len(rows), bool)
    for n, r in enumerate(cfg["report_dates"], 1):
        with rasterio.open(folder / f"{aoi}_{area}_sar_probability_{r}_v1.tif") as s:
            p = s.read(1)[rows, cols].astype("float32") / 100
        now = s2_pixels(s2 / f"{aoi}_s2_{s2_of[str(r)]}.tif", rows, cols)
        ok = now["clear"] & ref["clear"]
        x = features(now, ref)
        for k, (sure_cut, sure_stand) in enumerate(TEACHERS):
            rng = np.random.default_rng(cfg["seed"])
            cut_t = np.flatnonzero(ok & ((p >= sure_cut) | earlier))
            stand_t = np.flatnonzero(ok & (p <= sure_stand) & ~earlier)
            idx = np.r_[rng.choice(cut_t, min(len(cut_t), PER_CLASS), replace=False), rng.choice(stand_t, min(len(stand_t), PER_CLASS), replace=False)]
            y = np.r_[np.ones(min(len(cut_t), PER_CLASS)), np.zeros(min(len(stand_t), PER_CLASS))]
            model = lgb.LGBMClassifier(n_estimators=150, num_leaves=15, learning_rate=0.08, subsample=0.8, subsample_freq=1,
                                       colsample_bytree=0.8, n_jobs=-1, random_state=cfg["seed"], verbose=-1).fit(x[idx], y)
            out[f"po{k}_{n}"] = np.where(ok, np.round(model.predict_proba(x)[:, 1] * 100), 255).astype("uint8")
        yv, known = labels(cfg, area, n, rows, cols)
        out[f"p_{n}"], out[f"y_{n}"], out[f"known_{n}"] = np.round(p * 100).astype("uint8"), yv, known
        earlier |= p >= 0.5
        print(r, "prepared", flush=True)
    np.savez(str(cache) + ".tmp.npz", **out)
    os.replace(str(cache) + ".tmp.npz", cache)


def init(cache: str, n_dates: int) -> None:
    z = np.load(cache)
    G.update({k: z[k] for k in z.files}, n=n_dates)
    G["test"] = ((G["rows"] // 300 + G["cols"] // 300) % 2).astype(bool)


def predictions(par: dict, teacher: int) -> list[np.ndarray]:
    preds, earlier = [], np.zeros(len(G["rows"]), bool)
    for n in range(1, G["n"] + 1):
        p, po8 = G[f"p_{n}"].astype("float32") / 100, G[f"po{teacher}_{n}"]
        ok, po = po8 != 255, po8.astype("float32") / 100
        sar = p >= 0.5
        add = ok & (p >= par["low"]) & (p < 0.5) & (po >= par["opt_cut"])
        remove = ok & (p >= 0.5) & (p < par["rem_top"]) & (po <= par["opt_stand"])
        blind = ok & (p < par["low"]) & (po >= par["opt_very"])
        earlier = ((sar | add | blind) & ~remove) | earlier
        preds.append(earlier)
    return preds


def evaluate(job: tuple) -> dict:
    from disagreement_eda import patches
    par, teacher, name = job
    d, test, AC = {"rows": G["rows"], "cols": G["cols"]}, G["test"], 100 / 4046.8564224
    row = {"setting": name, "teacher": teacher, **par, "train": 0.0, "test": 0.0, "missed": 0.0, "extra": 0.0}
    for n, pred in enumerate(predictions(par, teacher), 1):
        y, known = G[f"y_{n}"], G[f"known_{n}"]
        missed, extra = patches(known & y & ~pred, d, 0.5, 3) > 0, patches(known & ~y & pred, d, 0.5, 3) > 0
        bad = missed | extra
        row[f"date{n}"] = bad.sum() * AC
        row["train"] += (bad & ~test).sum() * AC; row["test"] += (bad & test).sum() * AC
        row["missed"] += missed.sum() * AC; row["extra"] += extra.sum() * AC
    row["total"] = row["train"] + row["test"]
    return row


def main(area: str, cfg_path: str, ref_day: str, *pairs: str) -> None:
    t0 = time.time()
    cfg = yaml.safe_load(open(cfg_path))
    aoi, interim, tag = cfg["aoi"], Path(cfg["paths"]["interim"]), cfg.get("output_tag", "")
    cache = interim / f"{aoi}_optical_tune_cache_{area}{tag}.npz"
    if not cache.exists():
        prepare(area, cfg_path, ref_day, pairs, cache)
    print(f"prepared in {time.time() - t0:.0f} s", flush=True)
    guarantee_gb = int(os.environ.get("MEM_GUARANTEE", 16 * 1024 ** 3)) / 1024 ** 3
    n_proc = max(1, min(os.cpu_count(), int(guarantee_gb * 0.75 // 4)))             # one scoring needs about 4 GB
    log = []
    with ProcessPoolExecutor(n_proc, initializer=init, initargs=(str(cache), len(cfg["report_dates"]))) as ex:
        def run(jobs):
            res = list(ex.map(evaluate, jobs))
            log.extend(res)
            pd.DataFrame(log).to_csv(interim / f"{aoi}_optical_tune_{area}{tag}.csv", index=False)
            return res
        base = run([(dict(DEFAULT), 0, "hand-set limits"), ({"low": 0.5, "opt_cut": 2, "rem_top": 0.5, "opt_stand": -1, "opt_very": 2}, 0, "SAR only")])
        best, best_train, teacher = dict(DEFAULT), base[0]["train"], 0
        for rnd in (1, 2):
            for key, values in GRID.items():
                res = run([({**best, key: v}, teacher, f"round {rnd}: {key}={v}") for v in values if v != best[key]])
                top = min(res, key=lambda r: r["train"])
                if top["train"] < best_train:
                    best, best_train = {k: top[k] for k in DEFAULT}, top["train"]
                print(f"round {rnd} {key}: best so far {best} train {best_train:,.0f} ac ({time.time() - t0:.0f} s)", flush=True)
            if rnd == 1:                                                           # outer choice: the optical model's teachers
                res = run([(dict(best), k, f"teachers {TEACHERS[k]}") for k in (1, 2)])
                top = min(res, key=lambda r: r["train"])
                if top["train"] < best_train:
                    teacher, best_train = top["teacher"], top["train"]
                print(f"teachers: {TEACHERS[teacher]} train {best_train:,.0f} ac", flush=True)
        final = run([(dict(best), teacher, "TUNED")])[0]
    df = pd.DataFrame(log)
    show = df[df.setting.isin(["SAR only", "hand-set limits", "TUNED"])][["setting", "train", "test", "total", "missed", "extra", "date1", "date2", "date3"]]
    print(show.round(0).to_string(index=False))
    print("tuned limits:", best, "teachers (sure harvested, sure standing):", TEACHERS[teacher])
    (interim / f"{aoi}_optical_tuned_{area}{tag}.json").write_text(json.dumps({"limits": best, "teachers": TEACHERS[teacher],
        "train_blocks_acres": final["train"], "test_blocks_acres": final["test"], "hand_set_test_blocks_acres": base[0]["test"],
        "sar_only_test_blocks_acres": base[1]["test"], "settings_tried": len(df)}, indent=1))
    # rasters of the tuned result
    init(str(cache), len(cfg["report_dates"]))
    with rasterio.open(interim / f"{aoi}_cane_mask_{area}.tif") as m:
        profile, shape = m.profile, (m.height, m.width)
    profile.update(dtype="uint8", nodata=0, count=1, compress="deflate", tiled=True, blockxsize=256, blockysize=256)
    out_dir = Path("outputs") / f"{aoi}_{area}{tag}_optical_tuned"
    out_dir.mkdir(parents=True, exist_ok=True)
    for r, pred in zip(cfg["report_dates"], predictions(best, teacher)):
        img = np.zeros(shape, "uint8")
        img[G["rows"], G["cols"]] = np.where(pred, 82, 81)
        with rasterio.open(out_dir / f"{aoi}_{area}_sar_harvest_{r}_v1.tif", "w", **profile) as dst:
            dst.write(img, 1)
    print(f"finished in {time.time() - t0:.0f} s; rasters in {out_dir}")


if __name__ == "__main__":
    main(*sys.argv[1:])
