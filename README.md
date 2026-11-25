# sar-harvest-monitoring

Research code: find out which sugarcane fields have been cut, using Sentinel-1 radar images.
Study areas: Sakhon Nakhon and Phetchabun provinces, Thailand. Season 2025-26.

## What the project does

A sugar mill needs to know, on a given report date, which cane is still standing and which is already harvested.
This code answers that for every pixel (10 m) inside a crop mask. The result is a raster with two classes:
`81` = remaining cane, `82` = harvested. Area is reported in acres.

## Why radar

The usual method uses one cloud-free Sentinel-2 (optical) image per report date, plus hand-drawn samples and a
Random Forest. Clouds block it. In the old reports one date had 8,325 acres hidden by cloud. Another date had
65,172 acres that could not be used because of an image problem.

Radar sees through cloud. When cane is cut, the tall canopy disappears and the radar VH signal falls by about 4 dB.

## How it works

1. **Data per orbit track.** We use Sentinel-1 GRD images (VV and VH). Each orbit track is kept separate. Tracks are
   never mixed, because each one looks at the ground from a different angle. In Sakhon Nakhon the main tracks revisit
   every 6 and 12 days. 81 % of the cane is seen by 2 tracks and 19 % by 3.
2. **Mask.** Only pixels inside the crop mask are downloaded and processed, in every step.
3. **Features.** Each pass is smoothed with a 5 x 5 mean inside the mask. Each pixel gets 64 features that do not
   depend on the calendar date: 28 level features and 36 rate / texture features.
4. **Model.** A LightGBM classifier gives the probability that a pixel is harvested. Region growing then joins
   uncertain pixels that touch sure ones.
5. **"Once harvested stays harvested."** A pixel that is harvested on one report date is harvested on all later dates.
6. **Optical backup (optional).** Radar can be unsure, for example for a fresh cut. Where radar is unsure and a clear
   Sentinel-2 image exists, a small optical model decides. This model is taught by the pixels where radar is sure,
   so it needs no hand-drawn samples. Optical is used only there, never as the main method.
7. **Block-based scoring.** We score only on blocks of at least 0.5 acre where the delivered optical report has one
   class. Single pixels and field edges are not scored. One report date is held out each time, and training and test
   pixels come from different 3 km checkerboard blocks.

## Results

Sakhon Nakhon, cane area 100,792 acres. The model was trained on 7 report dates. Scores are against the delivered
optical reports.

| | Agreement | Difference area (3 dates) |
|---|---|---|
| Radar alone | 96.7 %, 95.5 %, 95.8 % on the 3 scored dates (overall 96.2 %) | 8,869 acres |
| Radar + optical backup | about 99.0 % | 2,387 acres |

Difference area = acres of blocks (0.5 acre or more) where radar and the optical report disagree.

One date in detail (13 Jan): the optical report has 31,049 acres harvested and 69,743 acres standing. Radar alone
gives 33,139 and 67,654.

Phetchabun (first area): cane area 261,458 acres, whole-area run. The total difference area is 11,049 acres with
radar alone and 5,550 acres with the optical cross-check.

## Honest limits

- About half of the missed harvest is fresh cuts that show up one radar pass late.
- 6-10 % of cut pixels are "radar-blind": they are bare in the optical image, but the radar stays at the standing level.
- The reference reports are themselves made from optical images, so they are not perfect truth.

## Repository layout

All scripts are in `src/`. Run them from the project folder. `<area>` is a name for the part of the area you run
(for example `test_tile` or `full`). `<aoi>` is the area name in the config.

| Step | Script | What it does |
|---|---|---|
| Download | `s1_passes.py` | Lists every Sentinel-1 pass over the crop mask (metadata only, free). |
| | `pick_test_tile.py` | Picks a small test window with much cane and harvest in every report. |
| | `download.py` | Downloads Sentinel-1 per pass and Sentinel-2 for the report dates, from Earth Engine. |
| | `rasterize_reports.py` | Burns the delivered optical reports onto the crop-mask grid. |
| EDA | `eda_prepare.py` | Builds the pixel table (SAR time series and report group) of a small area. |
| | `eda_extract.py` | Same for a large area, straight from the rasters (random sample). |
| | `eda_figures.py`, `embed_figures.py` | EDA figures, and one self-contained HTML report. |
| | `eda_ideas_test.py` | Two ideas tested on the large-area table. |
| | `compare_detectors.py` | Compares three ways to tell harvested from standing cane. |
| | `track_figure.py`, `track_map_figure.py` | Which tracks cover the area and how often. |
| | `burnt_check.py`, `step_features.py`, `dip_features.py` | Checks and features about burnt cane and slow slides versus sudden dips. |
| Training | `sar_features.py`, `direction_features.py`, `field_features.py` | Feature code shared by the other scripts. |
| | `transfer_test.py`, `transfer_figures.py` | Tests a model on a report date it never saw. |
| | `retrain_test.py`, `window_test.py`, `hard_cases_test.py`, `combine_test.py`, `cluster_test.py` | Tests of ideas: rate features, window size, hard examples, clustering. |
| | `run_area.py` | Trains and scores the best method on one area, SAR only and SAR + optical. |
| | `train_local.py` | Teaches a model on a new area from clear optical images when no reports exist. |
| | `export_training_sample.py` | Saves a training sample for use in another area. |
| Prediction | `predict_area.py` | Makes SAR harvest reports for a new area with a trained model, in tiles. |
| Scoring | `block_eval.py` | Scores on pure blocks, not single pixels. |
| | `difference_area.py`, `difference_figures.py` | Ranks variants by difference area; causes by block shape. |
| | `score_reports.py` | Scores SAR rasters of a large area against the original reports. |
| | `disagreement_eda.py`, `disagreement_blocks.py`, `disagreement_figures.py` | Describes and explains the big disagreement patches. |
| | `delayed_report_test.py` | Tests what a SAR report one pass later would gain. |
| Optical backup | `optical_check_test.py` | First test of "radar first, optical only where unsure". |
| | `optical_backup.py` | The optical backup, learned again for every report date. |
| | `optical_tune.py` | Tunes the backup limits on training blocks only. |
| | `optical_sanity.py` | Checks a SAR report against one clear optical image where no report exists. |
| Cleaning / post-processing | `postprocess.py` | Turns probabilities into report polygons that add up to the field polygons. |
| | `clean_reports.py` | Writes cleaned harvest rasters (majority filter, small pieces) in tiles. |
| | `postprocess_sweep.py`, `postprocess_trials.py`, `postprocess_trials_raster.py`, `grow_limits_test.py` | Trials of cleaning and region-growing settings. |
| | `report_look.py`, `mixing_check.py` | Measures how a report looks; finds patchwork fields. |
| Export and figures | `export_qgis.py`, `export_comparison.py`, `export_vs_original.py`, `export_review.py` | Rasters and files for QGIS with styles. |
| | `sar_composite.py` | SAR colour composites (last 3 passes as RGB) for visual checks. |
| | `inspect_pixel.py`, `inspect_points.py` | Show the evidence at one place. |
| | `area_report_figures.py` | Overview figures of a whole-area run. |
| Helpers | `resources.py` | Decides workers and chunk sizes from the machine's cores and memory. |

There are no automated tests in this repository.

## How to run the main steps

Placeholders: `<aoi>` area name, `<area>` run name, `<report config>` a copy of the config for that run.

```
python src/s1_passes.py configs/config.yaml
python src/pick_test_tile.py 15 configs/config.yaml                  # tile size in km
python src/download.py data/interim/<aoi>_test_tile.gpkg configs/config.yaml
python src/rasterize_reports.py <area> configs/config.yaml
python src/eda_prepare.py <area> configs/config.yaml
python src/eda_figures.py <area>
python src/run_area.py <area> configs/config.yaml                    # train + score, SAR and SAR + optical
python src/predict_area.py <area> configs/config.yaml                # new area, trained model
python src/clean_reports.py <area> configs/config.yaml <smallest piece acres> <majority window px, 0 = off> <holes first 0|1> [tag]
python src/score_reports.py <area> configs/config.yaml
python src/optical_tune.py <area> <report config> <reference S2 date> <report date>=<S2 date> [...]
python src/optical_backup.py <area> <report config> <reference S2 date> <report date>=<S2 date> [...]
python src/postprocess.py <area> <report config> <pred.npz> <prob key prefix> <out_tag> [tv_weight] [mmu_acres] [edge_snap_m]
python src/export_qgis.py <area>
```

Each script's docstring (top of the file) explains its inputs and outputs. `python src/resources.py` prints the
cores and memory the code will use.

## Configuration

Copy `configs/config.example.yaml` to `configs/config.yaml` and fill it in. Nothing about an area is hard-coded.
Required keys: `aoi`, `season`, `crs`, `classes`, `report_dates`, `paths` (`raw`, `reference_reports`, `interim`),
`seed`, `gee` (`key_file`, `project`), `crop_mask`, `s1_tracks` (orbit track to `DESCENDING` or `ASCENDING`),
`s2_bands`, `gee_workload_tag`. Optional keys (download speed, model files, track mapping, region growing,
`output_tag`, `original_reports`) are listed as comments in the example file. File names follow `NAMING.md`.

Setup: Python 3.10+ with geopandas, rasterio, numpy, pandas, scipy, scikit-learn, scikit-image, lightgbm,
matplotlib, pyyaml and earthengine-api, plus an Earth Engine service-account key. Put the crop mask and the older
reports under `data/raw/` (the `data/`, `runs/`, `outputs/` and `reports/` folders are not in git).

## Polygon tools

Turning a classified raster into remaining-cane and harvested polygons is a separate tool, now in its own
repository: https://github.com/dawood-labs/harvest-polygon-tools (classified raster -> remaining cane / harvested
polygons).

## License

MIT
