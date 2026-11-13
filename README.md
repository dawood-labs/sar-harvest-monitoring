# sar-harvest-monitoring

Sugarcane harvest monitoring from Sentinel-1 SAR time series.

Given a crop mask (polygons of standing sugarcane at the start of the season), the goal is to say, for any report
date, which part of the cane is **still standing** and which part is **harvested**, and roughly **when** it was cut,
without needing a cloud-free optical image.

## Status

Research stage. On a first test area the harvest signal is clear in Sentinel-1 (VH falls about 4 dB when a field is
cut) and a model trained on some report dates also works on a date it never saw. Next: the whole area.

## Why SAR

The usual way uses one cloud-free optical image per report date. Clouds block it for days or weeks in the harvest
season. Radar sees through cloud, and cutting a cane field changes the radar signal strongly because the tall canopy
disappears.

## Plan

| Step | What it does | Output |
|---|---|---|
| 1. Data | Sentinel-1 (VV, VH) for every pass over the area, kept per orbit track | time series inside the crop mask |
| 2. EDA | How standing and harvested cane look in each track, and what can be confused with harvest | HTML report |
| 3. Detection | Compare detectors: multi-date colour composite, per-pixel drop rule, small tree model | harvest status + date window per pixel |
| 4. Check | Compare with existing harvest reports and optical images of the same dates | error maps and scores |
| 5. Polygons | Clean the raster, build polygons, add class code and area, check that the classes add up to the crop mask | report file |

## Steps and commands

Run from the project folder. `<area>` is the name of an area file, e.g. `test_tile`.

| Step | Command | Output |
|---|---|---|
| List the Sentinel-1 passes (free, metadata only) | `python src/s1_passes.py` | `data/interim/<aoi>_s1_passes_*.csv` |
| Pick a small test area | `python src/pick_test_tile.py` | `data/interim/<aoi>_test_tile.gpkg` |
| Download Sentinel-1 (+ Sentinel-2 for checking) | `python src/download.py data/interim/<aoi>_test_tile.gpkg` | `data/raw/s1`, `data/raw/s2`, crop-mask raster |
| Cache the crop-mask pixels and their time series | `python src/eda_prepare.py <area>` | `data/interim/<aoi>_eda_pixels_<area>.npz` |
| Compare three detectors on each report date | `python src/compare_detectors.py <area> 5` | scores csv |
| EDA figures and report | `python src/eda_figures.py <area>` then `python src/embed_figures.py <template> <out.html>` | `reports/` |
| Test on a date the model never saw | `python src/transfer_test.py <area>` then `python src/transfer_figures.py <area>` | scores csv, predictions, figures |
| Files for QGIS | `python src/export_qgis.py <area>` | `outputs/` rasters with styles + polygons |

Rules built into the code: only pixels inside the crop mask are processed; only passes on or before the report
date are used; each orbit track is compared with its own earlier passes, tracks are never mixed in one image;
train and test pixels come from different 3 km blocks.

## Folders

```
configs/    config.example.yaml (copy to config.yaml and fill in)
src/        all scripts; resources.py decides workers and chunk sizes from the machine
data/       raw (read-only), interim, final      - not in git
runs/ outputs/ reports/ qc/                      - not in git
```

## Setup

1. Python 3.10+ with geopandas, rasterio, numpy, pandas, scipy, scikit-learn, scikit-image, lightgbm, matplotlib,
   earthengine-api and an Earth Engine service-account key.
2. `cp configs/config.example.yaml configs/config.yaml` and fill in the area name, dates, UTM zone and paths.
3. Put the crop mask and any older harvest reports in `data/raw/reference_reports/`.
4. `python src/resources.py` prints the cores and memory the code will use on this machine.

All settings live in `configs/config.yaml`; nothing is hardcoded. File names follow `NAMING.md`.

## Output classes

| `predicted` | meaning |
|---|---|
| 81 | remaining (standing) cane |
| 82 | harvested |

Area is reported in acres (3 decimals), computed in the local UTM zone; files are delivered in EPSG:4326.

## License

MIT
