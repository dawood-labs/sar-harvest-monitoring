"""Make SAR harvest reports for a NEW area with an already trained model (no original reports needed).

Why: a trained model must be usable on any area and any report date. The area is handled in tiles that run in
parallel processes, so any size fits in memory and every core is used; every tile is read straight from the
downloaded rasters, crop-mask pixels only.
Per tile and report date: 5 x 5 in-mask mean of every pass -> the same 64 date-free features as in training ->
harvest probability -> region growing -> once harvested, harvested in later reports.
Track mapping: the model was trained on two tracks (descending first, ascending second). `model_tracks` in the
config says which new track takes which slot, e.g. {164: 62, 26: 172}.

Usage: python src/predict_area.py <area_name> <config.yaml>
Output: outputs/<aoi>_<area>/<aoi>_<area>_sar_harvest_<date>_v1.tif (81 / 82, 0 = outside) + .qml,
        ..._sar_probability_<date>_v1.tif (0-100), ..._sar_harvest_<date>_v1.gpkg (predicted, area_ac),
        ..._summary_v1.csv, and a zip of all
"""
from __future__ import annotations

import logging
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor
import time
import zipfile
from pathlib import Path

import geopandas as gpd
import lightgbm as lgb
import numpy as np
import pandas as pd
import rasterio
import yaml
from rasterio.features import shapes
from rasterio.windows import Window
from shapely.geometry import shape

import transfer_test as tt
from export_qgis import CLASS_STYLE, write_qml
from resources import workers
from retrain_test import rate_features
from sar_features import local_std_in_mask, smooth_in_mask

log = logging.getLogger("predict_area")
ACRE_M2 = 4046.8564224
TILE, HALO = 2048, 64                     # tile size and overlap in pixels (the overlap keeps 5 x 5 means and growing seamless)


def coverage(f: Path, mask_file: Path) -> float:
    """Share of the crop mask that a pass covers (read at 1/8 resolution, enough to see a missing frame)."""
    from rasterio.enums import Resampling
    with rasterio.open(mask_file) as m:
        shape = (m.height // 8, m.width // 8)
        mk = m.read(1, out_shape=shape, resampling=Resampling.nearest) == 1
    with rasterio.open(f) as r:
        a = r.read(2, out_shape=shape, resampling=Resampling.nearest)
        return float((a[mk] != r.nodata).mean()) if mk.any() else 0.0


def track_files(cfg: dict, mask_file: Path) -> dict:
    """Downloaded passes per track: {track: (dates, files)}, only passes inside the season.

    Passes that cover much less of the crop than the track's best pass are dropped (e.g. one satellite of a track
    that images only a strip of the area): for most pixels they are empty, and an empty 'latest pass' would look
    like missing data to the model. The model then sees the regular series of the full passes.
    """
    from concurrent.futures import ThreadPoolExecutor
    out = {}
    for track, direction in cfg["s1_tracks"].items():
        files = sorted(Path(cfg["paths"]["raw"], "s1").glob(f"{cfg['aoi']}_s1_ro{int(track):03d}_{direction[:3].lower()}_*.tif"))
        files = [f for f in files if cfg["season"]["start"] <= f.stem[-10:] < cfg["season"]["end"]]
        with ThreadPoolExecutor(workers(keep_free_cores=0)) as ex:
            cov = list(ex.map(lambda f: coverage(f, mask_file), files))
        best = max(cov) if cov else 0
        keep = [f for f, c in zip(files, cov) if c >= 0.5 * best]
        for f, c in zip(files, cov):
            if c < 0.5 * best:
                log.info("track %03d: pass %s dropped, covers %.1f %% of the crop (best pass %.1f %%)", int(track), f.stem[-10:], 100 * c, 100 * best)
        out[int(track)] = (np.array([f.stem[-10:] for f in keep]), keep)
    return out


def read_tile(files: list[Path], win: Window, inside: np.ndarray) -> np.ndarray:
    """VV and VH of every pass at the tile's crop-mask pixels, dB, NaN = no data. Shape (passes, 2, pixels)."""
    def one(f: Path) -> np.ndarray:
        with rasterio.open(f) as r:
            a = r.read([1, 2], window=win, boundless=True, fill_value=r.nodata)[:, inside].astype("float32")
            nodata, scale = r.nodata, 100.0 if r.dtypes[0] == "int16" else 1.0
        a[a == nodata] = np.nan
        return a / scale

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(workers(keep_free_cores=0)) as ex:
        return np.stack(list(ex.map(one, files)))


_G: dict = {}


def _init(cfg: dict, area: str, model_cfg: dict, slots: dict, tracks: dict, mask_file: str, sure: float, maybe: float) -> None:
    """Per process: load the model once; inner thread pools stay small because the processes are the parallel level."""
    import sar_features
    global workers
    sar_features.workers = workers = lambda *a, **k: 2
    tt.SURE, tt.MAYBE = sure, maybe
    _G.update(cfg=cfg, area=area, model_cfg=model_cfg, slots=slots, tracks=tracks, mask_file=mask_file,
              booster=lgb.Booster(model_file=cfg["model_file"]),
              # optional: another model for single report dates (used to score a date with a model that never saw it)
              boosters={str(r): lgb.Booster(model_file=f) for r, f in cfg.get("model_files", {}).items()})
    if cfg.get("direction_mode", "one") != "one":            # one model per look direction, combined in _tile
        def per_direction(f: str) -> dict:
            return {n: lgb.Booster(model_file=f.replace(".txt", f"_{n}.txt")) for n in ("des", "asc")}
        _G["dir_booster"] = per_direction(cfg["model_file"])
        _G["dir_boosters"] = {str(r): per_direction(f) for r, f in cfg.get("model_files", {}).items()}


def _tile(rc: tuple[int, int]):
    """One tile: features, probability, growing and the stays-harvested rule for every report date."""
    r0, c0 = rc
    cfg, model_cfg, slots = _G["cfg"], _G["model_cfg"], _G["slots"]
    with rasterio.open(_G["mask_file"]) as m:
        H, W = m.height, m.width
        win = Window(c0 - HALO, r0 - HALO, TILE + 2 * HALO, TILE + 2 * HALO)
        mask = m.read(1, window=win, boundless=True, fill_value=0) == 1
    h, w = min(TILE, H - r0), min(TILE, W - c0)
    core = np.zeros_like(mask)
    core[HALO:HALO + h, HALO:HALO + w] = True
    if not (mask & core).any():
        return None
    rows, cols = np.nonzero(mask)
    d = {"rows": rows.astype("int32"), "cols": cols.astype("int32")}
    for t, (dates, files) in _G["tracks"].items():
        a = read_tile(list(files), win, mask)
        d[f"dates_{slots[t]}"], d[f"vv_{slots[t]}"], d[f"vh_{slots[t]}"] = dates, a[:, 0], a[:, 1]
    series = {(b, s): smooth_in_mask(d[f"{b}_{s}"], d["rows"], d["cols"], int(cfg.get("smooth", 5))) for s in model_cfg["s1_tracks"] for b in ("vh", "vv")}
    texture = {s: local_std_in_mask(d[f"vh_{s}"], d["rows"], d["cols"], 5) for s in model_cfg["s1_tracks"]}
    earlier, keep, out = np.zeros(len(rows), bool), core[rows, cols], {}
    sl = (slice(HALO, HALO + h), slice(HALO, HALO + w))
    for r in cfg["report_dates"]:
        if cfg.get("directions"):
            from direction_features import direction_features
            x = [direction_features(cfg["directions"], model_cfg["s1_tracks"], d, series, texture, r)]
        else:
            x = [tt.date_free_features(model_cfg, d, series, r), rate_features(model_cfg, d, series, texture, r)]
        if cfg.get("field_features"):                        # + level features of the pixel's segment and cane patch
            from field_features import field_features
            x.append(field_features(model_cfg, d, series, mask, r))
        x = np.column_stack(x)
        mode = cfg.get("direction_mode", "one")
        if mode == "one":
            prob = _G["boosters"].get(str(r), _G["booster"]).predict(x, num_threads=2)
        else:
            # Each look direction gives its own answer; "max" takes the higher one, "recency" weights the answers by
            # how new each direction's last pass is (half the weight every `recency_days` days of age).
            half = x.shape[1] // 2
            b = _G["dir_boosters"].get(str(r), _G["dir_booster"])
            pd_, pa = b["des"].predict(x[:, :half], num_threads=2), b["asc"].predict(x[:, half:], num_threads=2)
            has_d, has_a = np.isfinite(x[:, half - 1]), np.isfinite(x[:, -1])
            if mode == "max":
                prob = np.fmax(np.where(has_d, pd_, np.nan), np.where(has_a, pa, np.nan))
            else:
                tau = float(cfg.get("recency_days", 6.0))
                wd = np.where(has_d, 0.5 ** (np.nan_to_num(x[:, half - 1]) / tau), 0.0)
                wa = np.where(has_a, 0.5 ** (np.nan_to_num(x[:, -1]) / tau), 0.0)
                prob = (wd * pd_ + wa * pa) / np.maximum(wd + wa, 1e-9)
            prob = np.nan_to_num(prob)
        pred = tt.grow(prob, d) | earlier
        earlier = pred
        nodata = ~np.isfinite(np.column_stack([series[("vh", s)][d[f"dates_{s}"] <= r][-1] for s in model_cfg["s1_tracks"]])).any(axis=1)
        img = np.zeros(mask.shape, "uint8")
        img[rows, cols] = np.where(pred, 82, 81)
        pimg = np.full(mask.shape, 255, "uint8")
        pimg[rows, cols] = np.round(prob * 100).astype("uint8")
        out[r] = (img[sl], pimg[sl], int((pred & keep).sum()), int((~pred & keep).sum()), int((nodata & keep).sum()))
    return rc, out


def _polygons(args) -> tuple[str, int]:
    """Raster of one report date -> polygons (predicted, area_ac) in its own GeoPackage."""
    out, aoi, area, r, crs, deliver = args
    with rasterio.open(Path(out) / f"{aoi}_{area}_sar_harvest_{r}_v1.tif") as src:
        img = src.read(1)
        geoms = [(shape(g), int(v)) for g, v in shapes(img, mask=img > 0, transform=src.transform)]
    polys = gpd.GeoDataFrame({"predicted": [v for _, v in geoms], "report_date": r}, geometry=[g for g, _ in geoms], crs=crs)
    polys["area_ac"] = (polys.area / ACRE_M2).round(3)
    f = Path(out) / f"{aoi}_{area}_sar_harvest_{r}_v1.gpkg"
    f.unlink(missing_ok=True)
    polys.to_crs(deliver).to_file(f, layer=f"sar_harvest_{r}", driver="GPKG", engine="pyogrio")
    return r, int((polys.predicted == 82).sum())


def main(area: str, cfg_path: str) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    t0 = time.time()
    cfg = yaml.safe_load(open(cfg_path))
    aoi, reports = cfg["aoi"], cfg["report_dates"]
    slots = ({int(t): int(t) for t in cfg["s1_tracks"]} if cfg.get("directions")     # look-direction mode: tracks keep their numbers
             else {int(k): int(v) for k, v in cfg["model_tracks"].items()})          # new track -> model slot
    model_cfg = {"s1_tracks": {slots[int(t)]: d for t, d in cfg["s1_tracks"].items()}}
    order = list(model_cfg["s1_tracks"])                                       # slot order the model expects
    model_cfg["s1_tracks"] = {s: model_cfg["s1_tracks"][s] for s in sorted(order, key=lambda s: list(slots.values()).index(s))}
    booster = lgb.Booster(model_file=cfg["model_file"])
    tracks = track_files(cfg, Path(cfg["paths"]["interim"]) / f"{aoi}_cane_mask_{area}.tif")
    for t, (dates, _) in tracks.items():
        log.info("track %03d -> model slot %d: %d passes %s .. %s", t, slots[t], len(dates), dates[0], dates[-1])
        for r in reports:
            used = dates[dates <= r]
            if len(used) < 4 and not cfg.get("directions"):
                sys.exit(f"track {t}: only {len(used)} passes up to {r}; the model needs at least 4")

    # Region-growing limits can be set per area: a model used outside the area it was trained on is less sure,
    # so its middle probabilities deserve less trust (grow_sure / grow_maybe in the config).
    tt.SURE, tt.MAYBE = float(cfg.get("grow_sure", tt.SURE)), float(cfg.get("grow_maybe", tt.MAYBE))
    out = Path("outputs") / f"{aoi}_{area}{cfg.get('output_tag', '')}"
    out.mkdir(parents=True, exist_ok=True)
    mask_file = Path(cfg["paths"]["interim"]) / f"{aoi}_cane_mask_{area}.tif"
    with rasterio.open(mask_file) as m:
        profile, H, W = m.profile, m.height, m.width
    prof = {**profile, "dtype": "uint8", "nodata": 0, "count": 1, "compress": "deflate", "tiled": True,
            "blockxsize": 256, "blockysize": 256, "sparse_ok": True}
    sinks = {r: rasterio.open(out / f"{aoi}_{area}_sar_harvest_{r}_v1.tif", "w", **prof) for r in reports}
    psinks = {r: rasterio.open(out / f"{aoi}_{area}_sar_probability_{r}_v1.tif", "w", **{**prof, "nodata": 255}) for r in reports}
    counts = {r: {"standing": 0, "harvested": 0, "no_data": 0} for r in reports}
    # Tiles are independent: one process per tile, as many at a time as cores and guaranteed memory allow
    # (a tile needs about 1.5 GB). The main process only writes the finished tiles.
    jobs = [(r0, c0) for r0 in range(0, H, TILE) for c0 in range(0, W, TILE)]
    guarantee_gb = int(os.environ.get("MEM_GUARANTEE", 16 * 1024 ** 3)) / 1024 ** 3
    n_proc = max(1, min(os.cpu_count(), int(guarantee_gb * 0.8 // 1.5), len(jobs)))
    log.info("%d tiles of %d px, %d processes", len(jobs), TILE, n_proc)
    done = 0
    # 'spawn': a forked child hangs when it loads the model, because the parent has already used the model library.
    with ProcessPoolExecutor(n_proc, mp_context=multiprocessing.get_context("spawn"), initializer=_init, initargs=(cfg, area, model_cfg, slots, tracks, str(mask_file), tt.SURE, tt.MAYBE)) as ex:
        for res in ex.map(_tile, jobs):
            if res is None:
                continue
            (r0, c0), per_date = res
            for r, (img, pimg, n_cut, n_stand, n_nodata) in per_date.items():
                w_out = Window(c0, r0, img.shape[1], img.shape[0])
                sinks[r].write(img, 1, window=w_out)
                psinks[r].write(pimg, 1, window=w_out)
                counts[r]["harvested"] += n_cut; counts[r]["standing"] += n_stand; counts[r]["no_data"] += n_nodata
            done += 1
    for f in list(sinks.values()) + list(psinks.values()):
        f.close()
    log.info("%d tiles with crop predicted in %.0f s", done, time.time() - t0)

    # Polygons per date (area in the metric CRS, delivered in WGS84) + summary. Dates run in parallel processes;
    # each writes its own GeoPackage layer file, which are then listed in the summary.
    with ProcessPoolExecutor(min(len(reports), os.cpu_count()), mp_context=multiprocessing.get_context("spawn")) as ex:
        n_poly = dict(ex.map(_polygons, [(str(out), aoi, area, r, str(profile["crs"]), cfg["crs"]["deliver"]) for r in reports]))
    summary = []
    for r in reports:
        write_qml(out / f"{aoi}_{area}_sar_harvest_{r}_v1.tif", CLASS_STYLE)
        last = {t: str(dates[dates <= r][-1]) for t, (dates, _) in tracks.items()}
        c = counts[r]
        summary.append({"report_date": r, "last_pass_" + "_".join(f"{t:03d}" for t in last): " / ".join(last.values()),
                        "cane_acres": round((c["harvested"] + c["standing"]) * 100 / ACRE_M2, 3),
                        "harvested_acres": round(c["harvested"] * 100 / ACRE_M2, 3), "standing_acres": round(c["standing"] * 100 / ACRE_M2, 3),
                        "harvested_share": round(c["harvested"] / max(c["harvested"] + c["standing"], 1), 4),
                        "acres_without_a_recent_pass": round(c["no_data"] * 100 / ACRE_M2, 3),
                        "polygons_82": n_poly[r]})
        log.info("%s: %s", r, summary[-1])
    pd.DataFrame(summary).to_csv(out / f"{aoi}_{area}_summary_v1.csv", index=False)
    with zipfile.ZipFile(out / f"{aoi}_{area}_sar_reports_v1.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(out.iterdir()):
            if f.suffix != ".zip" and "probability" not in f.name:
                z.write(f, f.name)
    log.info("finished in %.0f s; files in %s", time.time() - t0, out)


if __name__ == "__main__":
    main(*sys.argv[1:3])
