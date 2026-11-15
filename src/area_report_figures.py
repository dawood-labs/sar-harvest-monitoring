"""Figures of a whole-area run (dark theme): missed / extra per report date for each variant, and an overview map.

Usage: python src/area_report_figures.py <area_name> <config.yaml>
Output: reports/figures/<aoi>_<area>_area_1_scores.png, ..._area_2_overview.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import yaml
from rasterio.enums import Resampling

import transfer_figures as tf            # dark-theme settings

COLORS = {"SAR only": "#3987e5", "SAR + optical where unsure": "#199e70"}


def main(area: str, cfg_path: str) -> None:
    cfg = yaml.safe_load(open(cfg_path))
    figs, out = Path("reports/figures"), Path("outputs") / area
    s = pd.read_csv(Path(cfg["paths"]["interim"]) / f"{cfg['aoi']}_area_scores_{area}.csv")
    reports = list(s.report.unique())
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    for ax, col, title in zip(axes, ("missed_share_of_harvested", "extra_share_of_standing"),
                              ("Harvested in the original report, standing in ours\n(% of the harvested area)",
                               "Standing in the original report, harvested in ours\n(% of the standing area)")):
        for i, (name, color) in enumerate(COLORS.items()):
            v = s[s.variant == name].set_index("report").loc[reports, col].values * 100
            x = np.arange(len(reports)) + (i - 0.5) * 0.38
            ax.bar(x, v, 0.34, color=color, label=name)
            for a, b in zip(x, v):
                ax.text(a, b, f"{b:.1f}%", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(range(len(reports)), [pd.Timestamp(r).strftime("%-d %b") for r in reports])
        ax.set_title(title, fontsize=10); ax.grid(axis="x", visible=False); ax.set_ylim(0, 9)
    axes[0].legend(fontsize=9, loc="upper right")
    fig.savefig(figs / f"{cfg['aoi']}_{area}_area_1_scores.png"); plt.close(fig)

    last = reports[-1]
    fig, axes = plt.subplots(1, 2, figsize=(11, 7.6))
    paint = {1: "#4a4a45", 2: "#199e70", 3: "#e66767", 4: "#3987e5"}
    for ax, tag, title in zip(axes, ("sar_only", "sar_optical"), ("SAR only", "SAR + optical where unsure")):
        with rasterio.open(out / f"{cfg['aoi']}_{area}_{tag}_vs_original_{last}_v1.tif") as r:
            a = r.read(1, out_shape=(r.height // 12, r.width // 12), resampling=Resampling.mode)
        img = np.zeros(a.shape + (3,), "float32") + plt.matplotlib.colors.to_rgb(tf.SURFACE)
        for k, c in paint.items():
            img[a == k] = plt.matplotlib.colors.to_rgb(c)
        ax.imshow(img, interpolation="nearest"); ax.set_title(f"{title}, {last}", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for c, name in zip(paint.values(), ("both standing", "both harvested", "original harvested, ours standing", "original standing, ours harvested")):
        axes[0].plot([], [], "s", color=c, ms=8, label=name)
    axes[0].legend(loc="upper center", bbox_to_anchor=(1.0, -0.01), fontsize=9, ncol=4)
    fig.savefig(figs / f"{cfg['aoi']}_{area}_area_2_overview.png"); plt.close(fig)
    print("written", figs)


if __name__ == "__main__":
    main(*sys.argv[1:3])
