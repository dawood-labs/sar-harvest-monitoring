# sar-harvest-monitoring

Sugarcane harvest monitoring from Sentinel-1 SAR time series.

Given a crop mask (polygons of standing sugarcane at the start of the season), the goal is to say, for any report
date, which part of the cane is **still standing** and which part is **harvested**, and roughly **when** it was cut,
without needing a cloud-free optical image.

## Status

Research stage. The project structure, settings and the machine resource helper are in place. The SAR method is
being developed and tested; code for each step is added here once it has been run and checked.

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

## Folders

```
configs/    config.example.yaml (copy to config.yaml and fill in)
src/        reusable code; resources.py decides workers and chunk sizes from the machine
data/       raw (read-only), interim, final      - not in git
runs/ outputs/ reports/ qc/                      - not in git
```

## Setup

1. Python 3.10+ with the usual geospatial stack (geopandas, rasterio, numpy, pandas).
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
