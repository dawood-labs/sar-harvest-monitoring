"""EDA of the big blocks where the SAR report and the original report disagree (large area, from rasters).

For one SAR report date every disagreement block of at least MIN_ACRES (and >= 30 m wide) is described:
  what    missed (original harvested, SAR standing) or extra (original standing, SAR harvested); acres; centre
  model   mean harvest probability of the block
  SAR     mean VH of the block per pass of each track: level at season start, at the report, at the passes
          AFTER the report (does a missed block fall later? = harvest seen late)
  optical NDVI of the clear Sentinel-2 images near the report (who is right: green = standing, bare = cut)
  reports in which original report the block is harvested, earlier or LATER than this one
  place   size of the cane patch, share of the block without the main track
From these, each block gets a likely cause (rules in `cause`), and the causes are summed in acres.
Only crop-mask pixels are read. Files are read in parallel threads.

Usage: python src/disagreement_blocks.py <area> <report config.yaml> <season config.yaml> <report number 1..n> [min acres]
       report config = the scored run (output_tag, 3 report dates); season config = all passes and all reports
Output: data/interim/<aoi>_disagreement_blocks_<area>_<report date>.csv (one row per block) + summary on screen
"""
from __future__ import annotations

import sys
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import yaml
from pyproj import Transformer

from disagreement_eda import patches
from predict_area import track_files

warnings.filterwarnings("ignore")
AC = 100 / 4046.8564224


def at(f, rows, cols, band=1):
    with rasterio.open(f) as s:
        return s.read(band)[rows, cols], s.nodata


def block_mean(v: np.ndarray, block: np.ndarray, n: int) -> np.ndarray:
    ok = np.isfinite(v)
    return np.bincount(block[ok], weights=v[ok], minlength=n) / np.maximum(np.bincount(block[ok], minlength=n), 1)


def main(area: str, cfg_path: str, season_path: str, number: str, min_acres: str = "1.0") -> None:
    cfg, season = yaml.safe_load(open(cfg_path)), yaml.safe_load(open(season_path))
    n, interim = int(number), Path(cfg["paths"]["interim"])
    r, aoi = cfg["report_dates"][n - 1], cfg["aoi"]
    mask_file = interim / f"{aoi}_cane_mask_{area}.tif"
    with rasterio.open(mask_file) as m:
        rows, cols = np.nonzero(m.read(1) == 1)
        transform, crs = m.transform, m.crs
    d = {"rows": rows, "cols": cols}
    folder = Path("outputs") / f"{aoi}_{area}{cfg.get('output_tag', '')}"
    pred = at(folder / f"{aoi}_{area}_sar_harvest_{r}_v1.tif", rows, cols)[0] == 82
    prob = at(folder / f"{aoi}_{area}_sar_probability_{r}_v1.tif", rows, cols)[0].astype("float32") / 100
    days = [str(x) for x in season["original_reports"]]
    with ThreadPoolExecutor(4) as ex:
        codes = dict(zip(days, ex.map(lambda day: at(interim / f"{aoi}_original_{area}_{day}.tif", rows, cols)[0], days)))
    cut_by = np.zeros(len(rows), bool)
    first_cut = np.zeros(len(rows), "int8")                      # number of the original report that first shows the cut
    for k, day in enumerate(days, 1):
        new = (codes[day] == 82) & (first_cut == 0)
        first_cut[new] = k
        if k <= n:
            cut_by |= codes[day] == 82
    known = cut_by | (codes[days[n - 1]] == 81)
    missed = patches(known & cut_by & ~pred, d, float(min_acres), 3)
    extra = patches(known & ~cut_by & pred, d, float(min_acres), 3)
    block = np.where(missed > 0, missed, np.where(extra > 0, extra + missed.max(), 0))
    nb = int(block.max()) + 1
    kind = np.array([""] + ["missed"] * int(missed.max()) + ["extra"] * int(extra.max()))
    size = np.bincount(block, minlength=nb)
    print(f"{r}: blocks >= {min_acres} ac: missed {len(np.unique(missed)) - 1:,} = {(missed > 0).sum() * AC:,.0f} ac | extra {len(np.unique(extra)) - 1:,} = {(extra > 0).sum() * AC:,.0f} ac")

    df = pd.DataFrame({"block": np.arange(nb), "kind": kind, "acres": size * AC, "probability": block_mean(prob, block, nb)})
    x, y = rasterio.transform.xy(transform, block_mean(rows.astype(float), block, nb), block_mean(cols.astype(float), block, nb))
    lon, lat = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform(np.array(x), np.array(y))
    df["lat"], df["lon"] = np.round(lat, 6), np.round(lon, 6)
    with rasterio.open(interim / f"{aoi}_patch_acres_{area}.tif") as s:
        df["patch_acres"] = block_mean(s.read(1)[rows, cols].astype(float), block, nb)
    # original reports: earliest report number in which most of the block is harvested (0 = never by the last report)
    for k, day in enumerate(days, 1):
        df[f"cut_share_by_report_{k}"] = block_mean((first_cut > 0) & (first_cut <= k), block, nb)
    df["cloud_or_issue_in_report_1"] = block_mean(np.isin(codes[days[0]], (232, 282)), block, nb)
    # optical
    s2 = sorted(Path(cfg["paths"]["raw"], "s2").glob(f"{aoi}_s2_*.tif"))
    for f in s2:
        with rasterio.open(f) as s:
            red, nir, clear = (s.read(b)[rows, cols].astype("float32") for b in (3, 5, 6))
        v = np.where((nir > 0) & (clear >= 60), (nir - red) / (nir + red + 1e-6), np.nan)
        df[f"ndvi_{f.stem[-10:]}"] = np.where(block_mean(np.isfinite(v), block, nb) >= 0.5, block_mean(v, block, nb), np.nan)
    # SAR: VH per pass (block mean in power)
    for t, (dates, files) in track_files(season, mask_file).items():
        def vh(f):
            a, nodata = at(f, rows, cols, 2)
            with rasterio.open(f) as s:
                scale = 100.0 if s.dtypes[0] == "int16" else 1.0
            v = np.where(a == nodata, np.nan, a.astype("float32") / scale)
            return 10 * np.log10(np.maximum(block_mean(10 ** (v / 10), block, nb), 1e-9)), block_mean(np.isfinite(v), block, nb)
        with ThreadPoolExecutor(4) as ex:
            res = list(ex.map(vh, files))
        s_ = np.array([a for a, _ in res]); cover = np.array([c for _, c in res])
        s_[cover < 0.5] = np.nan
        before, after = np.flatnonzero(dates <= r), np.flatnonzero(dates > r)
        df[f"t{t}_covered"] = cover[before[-1]]
        df[f"t{t}_start"] = np.nanmean(s_[:3], axis=0)
        df[f"t{t}_at_report"] = s_[before[-1]]
        df[f"t{t}_drop_at_report"] = s_[before[-1]] - np.nanmean(s_[:3], axis=0)
        df[f"t{t}_previous_pass"] = s_[before[-2]]
        for j, i in enumerate(after[:3], 1):
            df[f"t{t}_after_{j}_{dates[i][5:]}"] = s_[i] - s_[before[-1]]
        df[f"t{t}_last_pass_date"] = dates[before[-1]]
    df = df[(df.block > 0) & (df.acres > 0)].copy()             # block ids have gaps (small patches were dropped)

    # likely cause per block (first rule that fits)
    main_t = [c for c in df.columns if c.endswith("_drop_at_report")]
    drop = df[main_t].min(axis=1)                                 # deepest fall since season start on either track
    later = df[[c for c in df.columns if "_after_1_" in c or "_after_2_" in c]].min(axis=1)   # further fall in the next two passes
    nd = df[[c for c in df.columns if c.startswith("ndvi_") and c[-10:] <= days[n - 1]]].iloc[:, -1]   # last clear optical on or before the report
    nxt = df[f"cut_share_by_report_{n + 1}"] if n < len(days) else pd.Series(0.0, index=df.index)
    cause = pd.Series("unclear", index=df.index)
    m, e = df.kind == "missed", df.kind == "extra"
    cause[m & (nd > 0.40)] = "missed: optical is GREEN -> original report doubtful"
    cause[m & (cause == "unclear") & (later < -2.5)] = "missed: SAR falls in the next passes -> fresh cut, seen late"
    cause[m & (cause == "unclear") & (drop < -3.0)] = "missed: SAR did fall >= 3 dB, model still said standing"
    cause[m & (cause == "unclear") & (nd < 0.28)] = "missed: optical bare, SAR shows NO fall -> SAR cannot see it"
    cause[m & (cause == "unclear") & nd.isna()] = "missed: no clear optical, no SAR fall"
    cause[e & (nxt >= 0.5)] = "extra: harvested in the NEXT original report -> SAR earlier / date gap"
    cause[e & (cause == "unclear") & (nd < 0.28)] = "extra: optical is BARE -> original report doubtful"
    cause[e & (cause == "unclear") & (nd > 0.40)] = "extra: optical green -> SAR wrong (standing cane with low VH)"
    cause[e & (cause == "unclear") & nd.isna()] = "extra: no clear optical"
    cause[e & (cause == "unclear")] = "extra: optical in between (0.28-0.40)"
    cause[m & (cause == "unclear")] = "missed: optical in between, no SAR fall"
    df["cause"], df["deepest_drop_db"], df["further_fall_db"], df["ndvi_at_report"] = cause, drop, later, nd
    out = interim / f"{aoi}_disagreement_blocks_{area}_{r}.csv"
    df.round(3).to_csv(out, index=False)
    pd.set_option("display.width", 220); pd.set_option("display.max_colwidth", 80)
    g = df.groupby(["kind", "cause"]).agg(blocks=("acres", "size"), acres=("acres", "sum"), median_block_ac=("acres", "median"),
                                           probability=("probability", "median"), drop_db=("deepest_drop_db", "median"),
                                           further_fall_db=("further_fall_db", "median"), ndvi=("ndvi_at_report", "median"),
                                           patch_ac=("patch_acres", "median")).round(2)
    g["share_of_kind"] = (g.acres / g.groupby(level=0).acres.transform("sum")).round(2)
    print(g.sort_values(["kind", "acres"], ascending=[True, False]).to_string())
    print("written", out)


if __name__ == "__main__":
    main(*sys.argv[1:6])
