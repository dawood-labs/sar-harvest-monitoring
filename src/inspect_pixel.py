"""Look at one place: what the original report says, what our result says, and the SAR and optical history there.

Why: when a pixel or field looks wrong in QGIS, this shows the evidence behind it in a few seconds.
Reads straight from the rasters (no big cache): the pixel itself and the mean of the 5 x 5 pixels around it that
lie inside the crop mask.

Usage: python src/inspect_pixel.py <lon> <lat> [area_name] [config.yaml]      (copy lon / lat from QGIS, EPSG:4326)
       or open notebooks/inspect_places.ipynb, paste coordinates and run it.
Output: a table on screen and reports/figures/<aoi>_<area>_pixel_<lon>_<lat>.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import yaml
from pyproj import Transformer
from rasterio.windows import Window

import transfer_figures as tf            # dark-theme settings

CLASS = {0: "outside the crop mask / no original answer", 1: "both standing", 2: "both harvested",
         3: "original harvested, ours standing", 4: "original standing, ours harvested"}


def main(lon: str, lat: str, area: str = "full", cfg_path: str = "configs/config_full.yaml") -> Path | None:
    cfg = yaml.safe_load(open(cfg_path))
    raw, interim, out = Path(cfg["paths"]["raw"]), Path(cfg["paths"]["interim"]), Path("outputs") / area
    with rasterio.open(interim / f"{cfg['aoi']}_cane_mask_{area}.tif") as r:
        x, y = Transformer.from_crs(4326, r.crs, always_xy=True).transform(float(lon), float(lat))
        row, col = r.index(x, y)
        if not (0 <= row < r.height and 0 <= col < r.width):
            print(f"lon {lon}, lat {lat}: this point is outside the area.")
            return None
        win = Window(max(col - 2, 0), max(row - 2, 0), 5, 5)
        mask = r.read(1, window=win) == 1
        inside = r.read(1, window=Window(col, row, 1, 1))[0, 0] == 1
    print(f"point lon {lon}, lat {lat} -> row {row}, col {col}; inside the crop mask: {'yes' if inside else 'NO'}; "
          f"{mask.sum()} of 25 neighbours inside the mask")

    for r_date in cfg["report_dates"]:
        line = [r_date]
        for tag, name in (("sar_only", "SAR only"), ("sar_optical", "SAR + optical")):
            f = out / f"{cfg['aoi']}_{area}_{tag}_vs_original_{r_date}_v1.tif"
            if f.exists():
                with rasterio.open(f) as r:
                    line.append(f"{name}: {CLASS[int(r.read(1, window=Window(col, row, 1, 1))[0, 0])]}")
        # Areas without original reports (e.g. a new mill): our result and the model's harvest probability
        new = Path("outputs") / f"{cfg['aoi']}_{area}"
        h, pr = new / f"{cfg['aoi']}_{area}_sar_harvest_{r_date}_v1.tif", new / f"{cfg['aoi']}_{area}_sar_probability_{r_date}_v1.tif"
        if h.exists():
            with rasterio.open(h) as r:
                v = int(r.read(1, window=Window(col, row, 1, 1))[0, 0])
            line.append("SAR: " + {81: "standing (81)", 82: "harvested (82)"}.get(v, "outside the crop mask"))
            if pr.exists():
                with rasterio.open(pr) as r:
                    q = int(r.read(1, window=Window(col, row, 1, 1))[0, 0])
                line.append(f"harvest probability {q} %" if q <= 100 else "")
        print("  " + " | ".join(x for x in line if x))

    rows = []
    for track, direction in cfg["s1_tracks"].items():
        for f in sorted((raw / "s1").glob(f"{cfg['aoi']}_s1_ro{int(track):03d}_{direction[:3].lower()}_*.tif")):
            with rasterio.open(f) as r:
                a = r.read([1, 2], window=win).astype("float32")
                scale = 100.0 if r.dtypes[0] == "int16" else 1.0
                a[a == r.nodata] = np.nan
            a /= scale
            power = np.where(mask & np.isfinite(a), 10 ** (a / 10), np.nan)        # mean on power, in-mask pixels only
            mean = 10 * np.log10(np.nanmean(power, axis=(1, 2))) if mask.any() else [np.nan, np.nan]
            rows.append({"date": f.stem[-10:], "track": int(track), "VV_pixel": a[0, 2, 2], "VH_pixel": a[1, 2, 2],
                         "VV_5x5": mean[0], "VH_5x5": mean[1]})
    sar = pd.DataFrame(rows).sort_values(["track", "date"]).astype({k: "float64" for k in ("VV_pixel", "VH_pixel", "VV_5x5", "VH_5x5")})
    opt = []
    for f in sorted((raw / "s2").glob(f"{cfg['aoi']}_s2_*.tif")):
        with rasterio.open(f) as r:
            b = r.read([3, 5, 6], window=Window(col, row, 1, 1)).astype("float32")[:, 0, 0]
        opt.append({"date": f.stem[-10:], "NDVI": (b[1] - b[0]) / (b[1] + b[0] + 1e-6) if b[1] > 0 else np.nan, "clear_score": b[2]})
    opt = pd.DataFrame(opt)
    print(sar.round(1).to_string(index=False))
    print(opt.round(2).to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(13, 3.8), gridspec_kw={"width_ratios": [2, 1]})
    for track, color in zip(cfg["s1_tracks"], ("#3987e5", "#c98500")):
        s = sar[sar.track == int(track)]
        axes[0].plot(pd.to_datetime(s.date), s.VH_5x5, color=color, lw=2, marker="o", ms=4, label=f"VH, 5 x 5 mean, track {int(track):03d}")
        axes[0].plot(pd.to_datetime(s.date), s.VH_pixel, color=color, lw=1, ls=(0, (2, 3)), label=f"VH, this pixel, track {int(track):03d}")
    axes[1].plot(pd.to_datetime(opt.date), opt.NDVI, color="#199e70", lw=2, marker="o", ms=5)
    for ax in axes:
        for r_date in cfg["report_dates"]:
            ax.axvline(pd.Timestamp(r_date), color=tf.MUTED, lw=1, ls=(0, (3, 3)))
        ax.tick_params(axis="x", labelrotation=25, labelsize=8)
    axes[0].set_title(f"SAR at lon {lon}, lat {lat}  (dashed lines = report dates)", fontsize=10); axes[0].set_ylabel("VH (dB)")
    axes[0].legend(fontsize=8, ncol=2)
    axes[1].set_title("Optical greenness (NDVI)", fontsize=10); axes[1].set_ylim(0, 0.9)
    p = Path("reports/figures") / f"{cfg['aoi']}_{area}_pixel_{lon}_{lat}.png"
    fig.savefig(p); plt.close(fig)
    print("figure:", p)
    return p


if __name__ == "__main__":
    main(*sys.argv[1:5])
