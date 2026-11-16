"""Download Sentinel-1 (every pass of the chosen tracks) and Sentinel-2 (report dates only) for one area.

Why: the SAR method needs each pass as its own image, kept per track; optical is fetched only to check results.
Pixels are read straight from Earth Engine in 1024 x 1024 blocks onto one fixed 10 m grid, so pixel (row, col)
is the same place in every file. A finished file is skipped on a re-run (resume).

Usage: python src/download.py <area.gpkg> [configs/config.yaml] [s2_start s2_end]
       With s2_start and s2_end (YYYY-MM-DD) every Sentinel-2 date in that range that is at least half clear is
       fetched too (extra check images); files that already exist are never downloaded again.
Only pixels inside the crop mask are kept for SAR; nothing outside it is processed anywhere in the project.

Output: data/raw/s1/<aoi>_s1_ro<track>_<asc|des>_<date>.tif   VV, VH (dB), angle (deg); float32, NaN = outside the mask
        data/interim/<aoi>_cane_mask_<area>.tif                 1 = pixel inside the crop mask
        data/raw/s2/<aoi>_s2_<date>.tif                        reflectance x 10000 + cs_cdf x 100 (clear score); uint16
        data/interim/<aoi>_download_manifest.csv
"""
from __future__ import annotations

import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import ee
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import yaml

from resources import workers
from s1_passes import init_ee

RES, BLOCK = 10, 1024      # BLOCK is replaced by `download_block` from the config when given
log = logging.getLogger("download")


def make_grid(area_file: str, crs: str) -> dict:
    """10 m grid that covers the area's bounding box, snapped to whole 10 m coordinates."""
    xmin, ymin, xmax, ymax = gpd.read_file(area_file).to_crs(crs).total_bounds
    x0, y0 = np.floor(xmin / RES) * RES, np.ceil(ymax / RES) * RES
    return {"crs": crs, "x0": float(x0), "y0": float(y0),
            "width": int(np.ceil((xmax - x0) / RES)), "height": int(np.ceil((y0 - ymin) / RES))}


def cane_mask(grid: dict, cfg: dict, out: Path) -> np.ndarray:
    """True for pixels whose centre lies inside the crop mask; saved once so every later step uses the same mask."""
    from rasterio.features import rasterize
    tr = rasterio.transform.from_origin(grid["x0"], grid["y0"], RES, RES)
    w = gpd.GeoSeries.from_wkt([f"POLYGON(({grid['x0']} {grid['y0']}, {grid['x0'] + grid['width'] * RES} {grid['y0']}, "
                                f"{grid['x0'] + grid['width'] * RES} {grid['y0'] - grid['height'] * RES}, "
                                f"{grid['x0']} {grid['y0'] - grid['height'] * RES}, {grid['x0']} {grid['y0']}))"], crs=grid["crs"])[0]
    polys = gpd.read_file(cfg["crop_mask"]).to_crs(grid["crs"]).geometry.make_valid()   # repaired in memory only
    polys = polys[polys.intersects(w)]
    inside = rasterize([(g, 1) for g in polys], out_shape=(grid["height"], grid["width"]), transform=tr, dtype="uint8")
    with rasterio.open(out, "w", driver="GTiff", width=grid["width"], height=grid["height"], count=1, dtype="uint8",
                       crs=grid["crs"], transform=tr, nodata=0, compress="deflate", tiled=True) as dst:
        dst.write(inside, 1)
    return inside.astype(bool)


def fetch_to_file(image: ee.Image, job: dict, grid: dict, inside: np.ndarray, threads: int) -> None:
    """Read `image` block by block and write each block straight into the output file.

    Why: memory stays small for any area size (no whole-grid array). Blocks without a crop-mask pixel are not
    requested at all and are left empty in the file (sparse GeoTIFF, read back as no data). SAR keeps only the
    crop-mask pixels; optical images stay whole inside the blocks that are fetched.
    """
    import threading
    bands, dtype, fill = job["bands"], job["dtype"], job["fill"]
    H, W = grid["height"], grid["width"]
    send = -9999 if np.isnan(fill) else fill      # Earth Engine cannot take NaN in a request
    img = image.select(bands).unmask(send)
    tmp = job["out"].with_suffix(".tmp")
    prof = dict(driver="GTiff", width=W, height=H, count=len(bands), dtype=dtype, crs=grid["crs"],
                transform=rasterio.transform.from_origin(grid["x0"], grid["y0"], RES, RES), nodata=fill, compress="deflate",
                predictor=2 if np.dtype(dtype).kind in "ui" else 3, tiled=True, blockxsize=256, blockysize=256, sparse_ok=True)
    blocks = [(r0, c0) for r0 in range(0, H, BLOCK) for c0 in range(0, W, BLOCK)
              if inside[r0:r0 + BLOCK, c0:c0 + BLOCK].any()]
    lock = threading.Lock()
    with rasterio.open(tmp, "w", **prof) as dst:
        def one(rc: tuple[int, int]) -> None:
            r0, c0 = rc
            h, w = min(BLOCK, H - r0), min(BLOCK, W - c0)
            req = {"expression": img, "fileFormat": "NUMPY_NDARRAY",
                   "grid": {"dimensions": {"width": w, "height": h}, "crsCode": grid["crs"],
                            "affineTransform": {"scaleX": RES, "shearX": 0, "translateX": grid["x0"] + c0 * RES,
                                                "shearY": 0, "scaleY": -RES, "translateY": grid["y0"] - r0 * RES}}}
            for attempt in range(6):
                try:
                    arr = ee.data.computePixels(req)
                    break
                except Exception:
                    if attempt == 5:
                        raise
                    time.sleep(2 ** attempt + np.random.rand())
            block = np.stack([arr[b] for b in bands]).astype(dtype)
            if np.isnan(fill):
                block[block == send] = np.nan
            if job["sensor"] == "s1":
                block[:, ~inside[r0:r0 + h, c0:c0 + w]] = fill
            with lock:                                 # one writer at a time
                dst.write(block, window=rasterio.windows.Window(c0, r0, w, h))

        with ThreadPoolExecutor(threads) as ex:
            list(ex.map(one, blocks))
        dst.descriptions = tuple(bands)
        dst.update_tags(scenes=",".join(job["ids"]))
    tmp.replace(job["out"])


def write_tif(path: Path, arr: np.ndarray, bands: list[str], nodata: float, grid: dict, tags: dict) -> None:
    """Write to *.tmp and rename, so a half-written file is never taken as finished."""
    tmp = path.with_suffix(".tmp")
    prof = dict(driver="GTiff", width=grid["width"], height=grid["height"], count=len(bands), dtype=arr.dtype,
                crs=grid["crs"], transform=rasterio.transform.from_origin(grid["x0"], grid["y0"], RES, RES),
                nodata=nodata, compress="deflate", predictor=2 if arr.dtype.kind in "ui" else 3, tiled=True)
    with rasterio.open(tmp, "w", **prof) as dst:
        dst.write(arr)
        dst.descriptions = tuple(bands)
        dst.update_tags(**tags)
    tmp.replace(path)


def s1_jobs(cfg: dict, box: ee.Geometry, raw: Path) -> list[dict]:
    jobs = []
    for track, direction in cfg["s1_tracks"].items():
        col = (ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(box)
               .filterDate(cfg["season"]["start"], cfg["season"]["end"])
               .filter(ee.Filter.eq("instrumentMode", "IW"))
               .filter(ee.Filter.eq("relativeOrbitNumber_start", int(track)))
               .filter(ee.Filter.eq("orbitProperties_pass", direction))
               .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH")))
        ids = col.aggregate_array("system:index").getInfo()          # S1A_IW_GRDH_1SDV_YYYYMMDDT...
        for day in sorted({i.split("_")[4][:8] for i in ids}):
            same = [i for i in ids if i.split("_")[4][:8] == day]
            date = f"{day[:4]}-{day[4:6]}-{day[6:]}"
            name = f"{cfg['aoi']}_s1_ro{int(track):03d}_{direction[:3].lower()}_{date}.tif"
            jobs.append({"sensor": "s1", "date": date, "track": int(track), "satellite": same[0][:3], "ids": same,
                         "out": raw / "s1" / name,
                         # Large areas: only VV and VH, sent and stored as dB x 100 in int16 (half the transfer and disk).
                         **({"bands": ["VV", "VH"], "dtype": "int16", "fill": -32768, "scale": 100} if cfg.get("s1_int16")
                            else {"bands": ["VV", "VH", "angle"], "dtype": "float32", "fill": np.nan})})
    return jobs


def s2_jobs(cfg: dict, box: ee.Geometry, raw: Path) -> list[dict]:
    """For each report date: the clearest Sentinel-2 date within 3 days of it."""
    cs = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
    jobs = []
    for report in cfg["report_dates"]:
        d = pd.Timestamp(report)
        col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(box)
               .filterDate(str((d - pd.Timedelta(days=3)).date()), str((d + pd.Timedelta(days=4)).date())))
        ids = col.aggregate_array("system:index").getInfo()          # YYYYMMDDT..._Txxxxx
        clear = {}
        for day in sorted({i[:8] for i in ids}):
            same = [i for i in ids if i[:8] == day]
            score = ee.ImageCollection([cs.filter(ee.Filter.eq("system:index", i)).first() for i in same]).mosaic()
            clear[day] = score.select("cs_cdf").gte(0.6).unmask(0).reduceRegion(ee.Reducer.mean(), box, 100).get("cs_cdf").getInfo() or 0
        log.info("report %s: clear share of the area per Sentinel-2 date %s", report, {k: round(v, 2) for k, v in clear.items()})
        day = max(clear, key=lambda k: (round(clear[k], 2), -abs((pd.Timestamp(k) - d).days)))
        if cfg.get("s2_exact_report_dates") and d.strftime("%Y%m%d") in clear:
            day = d.strftime("%Y%m%d")                 # the report day itself, however cloudy (cross-check needs that day)
        date = f"{day[:4]}-{day[4:6]}-{day[6:]}"
        jobs.append({"sensor": "s2", "date": date, "report": report, "ids": [i for i in ids if i[:8] == day],
                     "clear_share": round(clear[day], 3), "out": raw / "s2" / f"{cfg['aoi']}_s2_{date}.tif",
                     "bands": cfg["s2_bands"] + ["cs_cdf"], "dtype": "uint16", "fill": 0})
    return jobs


def s2_range_jobs(cfg: dict, box: ee.Geometry, raw: Path, start: str, end: str) -> list[dict]:
    """Every Sentinel-2 date between start and end whose clear share over the area is at least 0.5."""
    cs = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
    ids = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(box).filterDate(start, end)
           .aggregate_array("system:index").getInfo())

    def clear_share(day: str) -> float:
        same = [i for i in ids if i[:8] == day]
        score = ee.ImageCollection([cs.filter(ee.Filter.eq("system:index", i)).first() for i in same]).mosaic()
        return score.select("cs_cdf").gte(0.6).unmask(0).reduceRegion(ee.Reducer.mean(), box, 100).get("cs_cdf").getInfo() or 0

    days = sorted({i[:8] for i in ids})
    with ThreadPoolExecutor(min(len(days), 16)) as ex:
        clear = dict(zip(days, ex.map(clear_share, days)))
    log.info("clear share per Sentinel-2 date: %s", {k: round(v, 2) for k, v in clear.items()})
    return [{"sensor": "s2", "date": f"{day[:4]}-{day[4:6]}-{day[6:]}", "report": "", "ids": [i for i in ids if i[:8] == day],
             "clear_share": round(clear[day], 3), "out": raw / "s2" / f"{cfg['aoi']}_s2_{day[:4]}-{day[4:6]}-{day[6:]}.tif",
             "bands": cfg["s2_bands"] + ["cs_cdf"], "dtype": "uint16", "fill": 0} for day in days if clear[day] >= 0.5]


def run(job: dict, grid: dict, inside: np.ndarray) -> dict:
    """Fetch one image into its file (skipped when the file exists: resume)."""
    t0 = time.time()
    if job["out"].with_suffix(".tmp").exists():        # another download process is writing this file right now
        log.info("skip %s: being written by another process", job["out"].name)
        return {"sensor": job["sensor"], "date": job["date"], "file": job["out"].name, "mb": 0, "seconds": 0}
    if not job["out"].exists():
        if job["sensor"] == "s1":
            img = ee.ImageCollection([ee.Image(f"COPERNICUS/S1_GRD/{i}") for i in job["ids"]]).mosaic()
            if job.get("scale"):
                img = img.select(job["bands"]).multiply(job["scale"]).round().toInt16()
        else:
            cs = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
            img = ee.ImageCollection([
                ee.Image(f"COPERNICUS/S2_SR_HARMONIZED/{i}").addBands(
                    cs.filter(ee.Filter.eq("system:index", i)).first().select("cs_cdf").multiply(100))
                for i in job["ids"]]).mosaic().toUint16()
        fetch_to_file(img, job, grid, inside, job.get("block_threads", 1))
    row = {k: v for k, v in job.items() if k in ("sensor", "date", "track", "satellite", "report", "clear_share")}
    return {**row, "file": job["out"].name, "mb": round(job["out"].stat().st_size / 1e6, 1), "seconds": round(time.time() - t0, 1)}


def main(area_file: str, cfg_path: str = "configs/config.yaml", s2_start: str = "", s2_end: str = "") -> None:
    global BLOCK
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    cfg = yaml.safe_load(open(cfg_path))
    BLOCK = int(cfg.get("download_block", BLOCK))      # small blocks skip more empty ground when the crop is scattered
    init_ee(cfg["gee"], high_volume=True)
    ee.data.setDefaultWorkloadTag(cfg["gee_workload_tag"])
    grid = make_grid(area_file, cfg["crs"]["work"])
    box = ee.Geometry.Rectangle(list(gpd.read_file(area_file).to_crs(4326).total_bounds))
    raw = Path(cfg["paths"]["raw"])
    mask_file = Path(cfg["paths"]["interim"]) / f"{cfg['aoi']}_cane_mask_{Path(area_file).stem.split('_', 1)[-1]}.tif"
    if mask_file.exists():                             # same mask on a re-run (resume)
        with rasterio.open(mask_file) as r:
            inside = r.read(1) == 1
    else:
        inside = cane_mask(grid, cfg, mask_file)
    for sub in ("s1", "s2"):
        (raw / sub).mkdir(parents=True, exist_ok=True)
    # Extra optical dates only when a range is given; the SAR job list is always complete (finished files are skipped).
    optical = [] if not cfg.get("s2", True) else (s2_range_jobs(cfg, box, raw, s2_start, s2_end) if s2_start else s2_jobs(cfg, box, raw))
    jobs = [j for j in s1_jobs(cfg, box, raw) + optical if not j["out"].exists()]
    # Network-bound: Earth Engine serves about 40 requests at once, so files x blocks-per-file stays near 40.
    # Few files at a time, oldest first: early report dates can start while later passes still download.
    jobs.sort(key=lambda j: (j["sensor"] != "s1", j["date"]), reverse=bool(cfg.get("download_newest_first")))
    n = max(1, min(len(jobs), int(cfg.get("download_files_at_once", 20))))
    for j in jobs:
        j["block_threads"] = max(1, int(cfg.get("download_concurrency", 40)) // n)
    log.info("crop mask inside the grid: %.1f acres (%.0f%% of the pixels)", inside.sum() * RES * RES / 4046.8564224, 100 * inside.mean())
    log.info("grid %d x %d px; %d files to fetch, %d at a time, %d blocks each at a time", grid["width"], grid["height"], len(jobs), n,
             jobs[0]["block_threads"] if jobs else 0)
    t0 = time.time()
    with ThreadPoolExecutor(n) as ex:
        rows = list(ex.map(lambda j: run(j, grid, inside), jobs))
    out = Path(cfg["paths"]["interim"]) / f"{Path(area_file).stem}_download_manifest{'_s2_extra' if s2_start else ''}.csv"
    df = pd.DataFrame(rows, columns=["sensor", "date", "file", "mb", "seconds"] if not rows else None).sort_values(["sensor", "date"])
    df.to_csv(out, index=False)
    log.info("done: %d files, %.0f MB, %.0f s wall; manifest %s", len(df), df.mb.sum(), time.time() - t0, out)


if __name__ == "__main__":
    main(*sys.argv[1:5])
