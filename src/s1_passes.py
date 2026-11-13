"""List every Sentinel-1 pass over the crop mask: date, satellite, orbit track, and how much of the mask it covers.

Why: the revisit time per track and the number of usable tracks decide how fresh a SAR harvest report can be.
Reads only image metadata from Earth Engine (no pixels, no cost).

Usage: python src/s1_passes.py [configs/config.yaml]
Output: data/interim/<aoi>_s1_passes_<start>_to_<end>.csv
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import ee
import geopandas as gpd
import pandas as pd
import yaml
from shapely.geometry import shape

ACRE_M2 = 4046.8564224
log = logging.getLogger("s1_passes")


def init_ee(gee: dict, high_volume: bool = False) -> None:
    """Start Earth Engine with the service-account key named in the config (high_volume = many pixel requests)."""
    import json
    email = json.load(open(gee["key_file"]))["client_email"]
    url = "https://earthengine-highvolume.googleapis.com" if high_volume else None
    ee.Initialize(ee.ServiceAccountCredentials(email, gee["key_file"]), project=gee["project"], opt_url=url)


def main(cfg_path: str = "configs/config.yaml") -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    cfg = yaml.safe_load(open(cfg_path))
    init_ee(cfg["gee"])
    mask = gpd.read_file(cfg["crop_mask"]).to_crs(cfg["crs"]["work"])
    total_ac = mask.area.sum() / ACRE_M2
    box = ee.Geometry.Rectangle(list(gpd.GeoSeries(mask.geometry, crs=mask.crs).to_crs(4326).total_bounds))

    start, end = cfg["season"]["start"], cfg["season"]["end"]
    col = (ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(box).filterDate(start, end)
           .filter(ee.Filter.eq("instrumentMode", "IW"))
           .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH")))
    # One request: id, track, pass direction and footprint of every image.
    feats = col.map(lambda i: ee.Feature(i.geometry(), {
        "id": i.get("system:index"), "track": i.get("relativeOrbitNumber_start"),
        "pass": i.get("orbitProperties_pass")})).getInfo()["features"]
    df = gpd.GeoDataFrame(
        [{"id": f["properties"]["id"], "track": int(f["properties"]["track"]), "pass": f["properties"]["pass"][:3],
          "geometry": shape(f["geometry"])} for f in feats], crs=4326).to_crs(mask.crs)
    df["satellite"] = df["id"].str[:3]
    df["date"] = pd.to_datetime(df["id"].str.split("_").str[4].str[:8])

    # Frames of the same pass are joined, then the cane area under the pass is measured.
    passes = df.dissolve(by=["date", "track", "pass", "satellite"]).reset_index()
    centroids = mask.geometry.centroid
    areas = mask.area.values / ACRE_M2
    passes["cane_acres"] = [areas[centroids.within(g).values].sum().round(1) for g in passes.geometry]
    passes["cane_share"] = (passes["cane_acres"] / total_ac).round(3)
    out = Path(cfg["paths"]["interim"]) / f"{cfg['aoi']}_s1_passes_{start}_to_{end}.csv"
    passes.drop(columns=["geometry", "id"]).sort_values("date").to_csv(out, index=False)
    log.info("crop mask %.1f acres; %d passes written to %s", total_ac, len(passes), out)

    for (track, direction), g in passes.groupby(["track", "pass"]):
        gaps = g["date"].sort_values().diff().dt.days.dropna()
        log.info("track %03d %s: %d passes, satellites %s, covers %.0f%% of the cane, days between passes: %s",
                 track, direction, len(g), "+".join(sorted(g["satellite"].unique())),
                 100 * g["cane_share"].median(), gaps.value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main(*sys.argv[1:2])
