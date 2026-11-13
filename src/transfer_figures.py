"""Figures of the date-transfer test (dark theme): scores, harvested area, and a map of where SAR and report differ.

Usage: python src/transfer_figures.py <area_name>
Output: reports/figures/<aoi>_<area>_transfer_*.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio

from sar_features import load_pixels
from transfer_test import harvested_by

SURFACE, INK, MUTED, GRID = "#1a1a19", "#f0efec", "#c3c2b7", "#3a3a37"
SERIES = {"same-date model": "#3987e5", "pixel, 5x5 mean": "#d95926", "5x5 + region growing": "#199e70",
          "5x5 + region growing + stays cut": "#c98500"}
LABEL = {"same-date model": "trained on the same date (easy test)", "pixel, 5x5 mean": "trained on other dates",
         "5x5 + region growing": "other dates + region growing", "5x5 + region growing + stays cut": "+ once cut stays cut"}
plt.rcParams.update({"font.size": 10, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "text.color": INK, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.axisbelow": True,
                     "grid.color": GRID, "grid.linewidth": 0.8, "axes.titlecolor": INK, "axes.titlesize": 11, "axes.titleweight": "bold",
                     "axes.titlelocation": "left", "legend.frameon": False, "savefig.dpi": 140, "savefig.bbox": "tight"})


def bars(ax, table: pd.DataFrame, value: str, fmt: str, extra: dict | None = None) -> None:
    reports = list(table.report.unique())
    names = list(extra or {}) + list(SERIES)
    width = 0.8 / len(names)
    for i, name in enumerate(names):
        x = np.arange(len(reports)) + (i - (len(names) - 1) / 2) * width
        v = extra[name] if extra and name in extra else table[table.variant == name].set_index("report").loc[reports, value].values
        ax.bar(x, v, width * 0.9, color=SERIES.get(name, "#8a897f"), label=LABEL.get(name, name))
        for a, b in zip(x, v):
            ax.text(a, b, fmt.format(b), ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(range(len(reports)), [pd.Timestamp(r).strftime("report %-d %b") for r in reports])
    ax.grid(axis="x", visible=False)


def main(area: str) -> None:
    cfg, d = load_pixels(area)
    interim, figs = Path(cfg["paths"]["interim"]), Path("reports/figures")
    stem = f"{cfg['aoi']}_{area}_transfer"
    t = pd.read_csv(interim / f"{cfg['aoi']}_transfer_scores_{area}.csv")
    same = pd.read_csv(interim / f"{cfg['aoi']}_detector_scores_{area}_smooth5.csv").query("detector == 'tree'")
    same = same.assign(variant="same-date model", f1_all=same.f1, acres_sar=same.harvested_acres_sar)
    table = pd.concat([same[["report", "variant", "f1_all", "acres_sar"]], t[t.variant.isin(SERIES)]])

    fig, ax = plt.subplots(figsize=(9.5, 3.8))
    bars(ax, table, "f1_all", "{:.2f}")
    ax.set_ylim(0, 1.08); ax.set_ylabel("F1 of the harvested class")
    ax.legend(fontsize=9, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.savefig(figs / f"{stem}_1_scores.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.5, 3.8))
    report_acres = t[t.variant == "pixel, 5x5 mean"].set_index("report").loc[list(table.report.unique()), "acres_report"].values
    bars(ax, table, "acres_sar", "{:,.0f}", {"reference report": report_acres})
    ax.set_ylabel("harvested acres (test blocks)")
    ax.legend(fontsize=9, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.savefig(figs / f"{stem}_2_acres.png"); plt.close(fig)

    # Map of the last report date in the same zoom window as the EDA report.
    import json
    win = json.loads((figs / f"{cfg['aoi']}_{area}_numbers.json").read_text())["map"]
    r0, c0, n, report = win["row0"], win["col0"], len(cfg["report_dates"]), cfg["report_dates"][-1]
    sl = (slice(r0, r0 + 300), slice(c0, c0 + 300))
    shape = (int(d["rows"].max()) + 1, int(d["cols"].max()) + 1)
    y, known = harvested_by(d, n)
    pred = np.load(interim / f"{cfg['aoi']}_transfer_pred_{area}.npz")[f"pred_{report}"]

    def to_grid(v):
        g = np.full(shape, -1, "int8"); g[d["rows"], d["cols"]] = v
        return g[sl]

    def paint(g, colors):
        img = np.zeros(g.shape + (3,), "float32") + matplotlib.colors.to_rgb(SURFACE)
        for k, c in colors.items():
            img[g == k] = matplotlib.colors.to_rgb(c)
        return img

    with rasterio.open(Path(cfg["paths"]["raw"]) / "s2" / f"{cfg['aoi']}_s2_{report}.tif") as r:
        s2 = r.read([4, 2, 1]).astype("float32")[:, sl[0], sl[1]]
    optical = np.dstack([np.clip((b - np.percentile(b, 2)) / (np.percentile(b, 98) - np.percentile(b, 2)), 0, 1) for b in s2])
    classes = {0: "#454B1B", 1: "#F89508"}
    diff = {0: "#4a4a45", 1: "#199e70", 2: "#e66767", 3: "#3987e5"}
    agree = np.where(~known, -1, np.where(y & pred, 1, np.where(~y & ~pred, 0, np.where(y & ~pred, 2, 3))))
    panels = [(optical, f"Sentinel-2, {report}\n(B5, B3, B2; check only)", None),
              (paint(to_grid(np.where(known, y, -1)), classes), f"Reference report\n{report}", {"81 remaining cane": "#454B1B", "82 harvested": "#F89508"}),
              (paint(to_grid(np.where(known, pred, -1)), classes), "SAR, trained on other dates\n+ region growing + stays cut", None),
              (paint(to_grid(agree), diff), "Where they differ", {"both standing": diff[0], "both harvested": diff[1],
                                                                "report cut, SAR standing": diff[2], "report standing, SAR cut": diff[3]})]
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.9))
    for ax, (img, title, legend) in zip(axes, panels):
        ax.imshow(img, interpolation="nearest"); ax.set_title(title, fontsize=9.5)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        if legend:
            for name, c in legend.items():
                ax.plot([], [], "s", color=c, ms=8, label=name)
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), fontsize=8, ncol=2)
    fig.savefig(figs / f"{stem}_3_maps.png"); plt.close(fig)
    print("written", figs / stem)


if __name__ == "__main__":
    main(sys.argv[1])
