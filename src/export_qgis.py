"""Write the SAR harvest results of one area as files that open directly in QGIS, with ready styles.

Why: the final check of a result is looking at it on top of the satellite image.
Per report date:
  <aoi>_<area>_sar_harvest_<date>_v1.tif        81 = remaining cane, 82 = harvested, 0 = outside the crop mask
  <aoi>_<area>_sar_vs_report_<date>_v1.tif      1 both standing, 2 both harvested, 3 report cut / SAR standing,
                                                4 report standing / SAR cut
  <aoi>_<area>_sar_probability_<date>_v1.tif    model's harvest probability, 0-100 (255 = outside the mask)
Once: <aoi>_<area>_sar_harvest_v1.gpkg (polygons per date: predicted, area_ac; EPSG:4326)
      <aoi>_<area>_blocks_v1.gpkg (which 3 km blocks were used for training and which for testing)
Each raster gets a .qml style file next to it; QGIS loads it by itself.

Usage: python src/export_qgis.py <area_name>
"""
from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import shapes
from shapely.geometry import box, shape

from sar_features import load_pixels
from transfer_test import harvested_by

ACRE_M2 = 4046.8564224
CLASS_STYLE = {81: ("#454B1B", "81 Remaining Cane"), 82: ("#F89508", "82 Harvested")}
DIFF_STYLE = {1: ("#4a4a45", "both standing"), 2: ("#199e70", "both harvested"),
              3: ("#e66767", "report cut, SAR standing"), 4: ("#3987e5", "report standing, SAR cut")}


def write_qml(path: Path, style: dict) -> None:
    """Minimal QGIS style: one colour and label per raster value."""
    entries = "".join(f'<paletteEntry value="{v}" color="{c}" label="{name}" alpha="255"/>' for v, (c, name) in style.items())
    path.with_suffix(".qml").write_text(
        '<!DOCTYPE qgis><qgis><pipe><rasterrenderer type="paletted" band="1" opacity="1" alphaBand="-1">'
        f"<colorPalette>{entries}</colorPalette></rasterrenderer></pipe></qgis>")


def write_raster(path: Path, values: np.ndarray, d: dict, profile: dict, nodata: int) -> np.ndarray:
    img = np.full((profile["height"], profile["width"]), nodata, "uint8")
    img[d["rows"], d["cols"]] = values
    with rasterio.open(path, "w", **{**profile, "dtype": "uint8", "nodata": nodata, "count": 1, "compress": "deflate"}) as dst:
        dst.write(img, 1)
    return img


def main(area: str) -> None:
    cfg, d = load_pixels(area)
    interim, out = Path(cfg["paths"]["interim"]), Path("outputs")
    out.mkdir(exist_ok=True)
    stem = f"{cfg['aoi']}_{area}"
    with rasterio.open(interim / f"{cfg['aoi']}_cane_mask_{area}.tif") as r:
        profile, tr = r.profile, r.transform
    z = np.load(interim / f"{cfg['aoi']}_transfer_pred_{area}.npz")
    codes = cfg["classes"]
    vector = out / f"{stem}_sar_harvest_v1.gpkg"
    vector.unlink(missing_ok=True)
    for n, report in enumerate(cfg["report_dates"], 1):
        pred, (y, known) = z[f"pred_{report}"], harvested_by(d, n)
        p = out / f"{stem}_sar_harvest_{report}_v1.tif"
        img = write_raster(p, np.where(pred, codes["harvested"], codes["remaining_cane"]), d, profile, 0)
        write_qml(p, CLASS_STYLE)
        p = out / f"{stem}_sar_vs_report_{report}_v1.tif"
        write_raster(p, np.where(~known, 0, np.where(~y & ~pred, 1, np.where(y & pred, 2, np.where(y, 3, 4)))), d, profile, 0)
        write_qml(p, DIFF_STYLE)
        write_raster(out / f"{stem}_sar_probability_{report}_v1.tif", np.round(z[f"prob_{report}"] * 100), d, profile, 255)

        geoms = [(shape(g), int(v)) for g, v in shapes(img, mask=img > 0, transform=tr)]
        polys = gpd.GeoDataFrame({"predicted": [v for _, v in geoms]}, geometry=[g for g, _ in geoms], crs=profile["crs"])
        polys["area_ac"] = (polys.area / ACRE_M2).round(3)        # area in the metric CRS, before the move to WGS84
        polys.to_crs(cfg["crs"]["deliver"]).to_file(vector, layer=f"sar_harvest_{report}", driver="GPKG")
        print(report, polys.groupby("predicted").area_ac.agg(["count", "sum"]).round(1).to_dict("index"))

    # 3 km blocks: test blocks were never used for training (on any date).
    h, w = profile["height"], profile["width"]
    blocks = [{"role": "test" if (i + j) % 2 else "train",
               "geometry": box(tr.c + j * 3000, tr.f - min((i + 1) * 3000, h * 10), tr.c + min((j + 1) * 3000, w * 10), tr.f - i * 3000)}
              for i in range(-(-h // 300)) for j in range(-(-w // 300))]
    gpd.GeoDataFrame(blocks, crs=profile["crs"]).to_crs(cfg["crs"]["deliver"]).to_file(out / f"{stem}_blocks_v1.gpkg", driver="GPKG")
    print("written to", out)


if __name__ == "__main__":
    main(sys.argv[1])
