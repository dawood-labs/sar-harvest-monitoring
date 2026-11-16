"""Make SAR harvest reports for a NEW area with an already trained model (no original reports needed).

Why: a trained model must be usable on any mill area and any report date. The area is handled in tiles, so any
size fits in memory; every tile is read straight from the downloaded rasters, crop-mask pixels only.
Per tile and report date: 5 x 5 in-mask mean of every pass -> the same 64 date-free features as in training ->
harvest probability -> region growing -> once harvested, harvested in later reports.
Track mapping: the model was trained on two tracks (descending first, ascending second). `model_tracks` in the
config says which new track takes which slot, e.g. {164: 62, 26: 172}.

Usage: python src/predict_area.py <area_name> <config.yaml>
Output: outputs/<aoi>_<area>/<aoi>_<area>_sar_harvest_<date>_v1.tif (81 / 82, 0 = outside) + .qml,
        ..._sar_probability_<date>_v1.tif (0-100), ..._sar_harvest_v1.gpkg (one layer per date: predicted, area_ac),
        ..._summary_v1.csv, and a zip of all
"""
from __future__ import annotations

import logging
import sys
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
TILE, HALO = 4096, 64                     # tile size and overlap in pixels (the overlap keeps 5 x 5 means and growing seamless)


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


def main(area: str, cfg_path: str) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    t0 = time.time()
    cfg = yaml.safe_load(open(cfg_path))
    aoi, reports = cfg["aoi"], cfg["report_dates"]
    slots = {int(k): int(v) for k, v in cfg["model_tracks"].items()}          # new track -> model slot
    model_cfg = {"s1_tracks": {slots[int(t)]: d for t, d in cfg["s1_tracks"].items()}}
    order = list(model_cfg["s1_tracks"])                                       # slot order the model expects
    model_cfg["s1_tracks"] = {s: model_cfg["s1_tracks"][s] for s in sorted(order, key=lambda s: list(slots.values()).index(s))}
    booster = lgb.Booster(model_file=cfg["model_file"])
    tracks = track_files(cfg, Path(cfg["paths"]["interim"]) / f"{aoi}_cane_mask_{area}.tif")
    for t, (dates, _) in tracks.items():
        log.info("track %03d -> model slot %d: %d passes %s .. %s", t, slots[t], len(dates), dates[0], dates[-1])
        for r in reports:
            used = dates[dates <= r]
            if len(used) < 4:
                sys.exit(f"track {t}: only {len(used)} passes up to {r}; the model needs at least 4")

    out = Path("outputs") / f"{aoi}_{area}"
    out.mkdir(parents=True, exist_ok=True)
    with rasterio.open(Path(cfg["paths"]["interim"]) / f"{aoi}_cane_mask_{area}.tif") as m:
        profile, H, W, transform = m.profile, m.height, m.width, m.transform
        prof = {**profile, "dtype": "uint8", "nodata": 0, "count": 1, "compress": "deflate", "tiled": True,
                "blockxsize": 256, "blockysize": 256, "sparse_ok": True}
        sinks = {r: rasterio.open(out / f"{aoi}_{area}_sar_harvest_{r}_v1.tif", "w", **prof) for r in reports}
        psinks = {r: rasterio.open(out / f"{aoi}_{area}_sar_probability_{r}_v1.tif", "w", **{**prof, "nodata": 255}) for r in reports}
        counts = {r: {"standing": 0, "harvested": 0, "no_data": 0} for r in reports}
        for r0 in range(0, H, TILE):
            for c0 in range(0, W, TILE):
                win = Window(c0 - HALO, r0 - HALO, TILE + 2 * HALO, TILE + 2 * HALO)
                mask = m.read(1, window=win, boundless=True, fill_value=0) == 1
                core = np.zeros_like(mask)
                core[HALO:HALO + min(TILE, H - r0), HALO:HALO + min(TILE, W - c0)] = True
                if not (mask & core).any():
                    continue
                rows, cols = np.nonzero(mask)
                d = {"rows": rows.astype("int32"), "cols": cols.astype("int32")}
                for t, (dates, files) in tracks.items():
                    a = read_tile(list(files), win, mask)
                    s = slots[t]
                    d[f"dates_{s}"], d[f"vv_{s}"], d[f"vh_{s}"] = dates, a[:, 0], a[:, 1]
                series = {(b, s): smooth_in_mask(d[f"{b}_{s}"], d["rows"], d["cols"], 5) for s in model_cfg["s1_tracks"] for b in ("vh", "vv")}
                texture = {s: local_std_in_mask(d[f"vh_{s}"], d["rows"], d["cols"], 5) for s in model_cfg["s1_tracks"]}
                earlier = np.zeros(len(rows), bool)
                keep = core[rows, cols]
                for r in reports:
                    x = np.column_stack([tt.date_free_features(model_cfg, d, series, r), rate_features(model_cfg, d, series, texture, r)])
                    prob = booster.predict(x, num_threads=workers(keep_free_cores=0))
                    pred = tt.grow(prob, d) | earlier
                    earlier = pred
                    nodata = ~np.isfinite(np.column_stack([series[("vh", s)][d[f"dates_{s}"] <= r][-1] for s in model_cfg["s1_tracks"]])).any(axis=1)
                    img = np.zeros(mask.shape, "uint8")
                    img[rows, cols] = np.where(pred, 82, 81)
                    pimg = np.full(mask.shape, 255, "uint8")
                    pimg[rows, cols] = np.round(prob * 100).astype("uint8")
                    sl = (slice(HALO, HALO + min(TILE, H - r0)), slice(HALO, HALO + min(TILE, W - c0)))
                    w_out = Window(c0, r0, min(TILE, W - c0), min(TILE, H - r0))
                    sinks[r].write(img[sl], 1, window=w_out)
                    psinks[r].write(pimg[sl], 1, window=w_out)
                    counts[r]["harvested"] += int((pred & keep).sum())
                    counts[r]["standing"] += int((~pred & keep).sum())
                    counts[r]["no_data"] += int((nodata & keep).sum())
                log.info("tile row %d col %d: %d crop pixels (%.0f s so far)", r0, c0, int(keep.sum()), time.time() - t0)
        for f in list(sinks.values()) + list(psinks.values()):
            f.close()

    # Polygons per date (area in the metric CRS, delivered in WGS84) + summary.
    vector = out / f"{aoi}_{area}_sar_harvest_v1.gpkg"
    vector.unlink(missing_ok=True)
    summary = []
    for r in reports:
        p = out / f"{aoi}_{area}_sar_harvest_{r}_v1.tif"
        write_qml(p, CLASS_STYLE)
        with rasterio.open(p) as src:
            img = src.read(1)
            geoms = [(shape(g), int(v)) for g, v in shapes(img, mask=img > 0, transform=src.transform)]
        polys = gpd.GeoDataFrame({"predicted": [v for _, v in geoms], "report_date": r}, geometry=[g for g, _ in geoms], crs=profile["crs"])
        polys["area_ac"] = (polys.area / ACRE_M2).round(3)
        polys.to_crs(cfg["crs"]["deliver"]).to_file(vector, layer=f"sar_harvest_{r}", driver="GPKG")
        last = {t: str(dates[dates <= r][-1]) for t, (dates, _) in tracks.items()}
        c = counts[r]
        summary.append({"report_date": r, "last_pass_" + "_".join(f"{t:03d}" for t in last): " / ".join(last.values()),
                        "cane_acres": round((c["harvested"] + c["standing"]) * 100 / ACRE_M2, 3),
                        "harvested_acres": round(c["harvested"] * 100 / ACRE_M2, 3), "standing_acres": round(c["standing"] * 100 / ACRE_M2, 3),
                        "harvested_share": round(c["harvested"] / max(c["harvested"] + c["standing"], 1), 4),
                        "acres_without_a_recent_pass": round(c["no_data"] * 100 / ACRE_M2, 3),
                        "polygons_82": int((polys.predicted == 82).sum())})
        log.info("%s: %s", r, summary[-1])
    pd.DataFrame(summary).to_csv(out / f"{aoi}_{area}_summary_v1.csv", index=False)
    with zipfile.ZipFile(out / f"{aoi}_{area}_sar_reports_v1.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(out.iterdir()):
            if f.suffix != ".zip" and "probability" not in f.name:
                z.write(f, f.name)
    log.info("finished in %.0f s; files in %s", time.time() - t0, out)


if __name__ == "__main__":
    main(*sys.argv[1:3])
