"""Figures for the difference-area analysis (dark theme): ranking of all variants, and causes by block shape.

Usage: python src/difference_figures.py <area_name> <tag of the analysed predictions, e.g. _base05>
Output: reports/figures/<aoi>_<area>_difference_1_ranking.png, ..._2_causes_by_shape.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

import transfer_figures as tf            # dark-theme settings

MISSED, EXTRA = "#e66767", "#3987e5"
CAUSE_COLOR = {"fresh cut": "#3987e5", "never falls": "#d95926", "SAR saw it first": "#199e70", "low all season": "#c98500", "other": "#8a897f",
               "optical green": "#9085e9"}


def main(area: str, tag: str) -> None:
    cfg = yaml.safe_load(open("configs/config.yaml"))
    interim, figs = Path(cfg["paths"]["interim"]), Path("reports/figures")
    stem = f"{cfg['aoi']}_{area}_difference"
    rank = pd.read_csv(interim / f"{cfg['aoi']}_difference_area_{area}.csv").drop_duplicates(["missed_total", "extra_total"])
    rank = rank.sort_values("difference_total", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 0.42 * len(rank) + 1.4))
    ax.barh(range(len(rank)), rank.missed_total, color=MISSED, edgecolor=tf.SURFACE, lw=2, label="original report harvested, SAR standing")
    ax.barh(range(len(rank)), rank.extra_total, left=rank.missed_total, color=EXTRA, edgecolor=tf.SURFACE, lw=2, label="original report standing, SAR harvested")
    for i, v in enumerate(rank.difference_total):
        ax.text(v, i, f"  {v:,.0f} ac", va="center", fontsize=9)
    ax.set_yticks(range(len(rank)), rank.variant, fontsize=9)
    ax.set_xlabel("difference area, 3 report dates added (acres; blocks >= 0.5 acre)")
    ax.set_xlim(0, rank.difference_total.max() * 1.15); ax.grid(axis="y", visible=False)
    ax.legend(fontsize=9, ncol=2, loc="upper center", bbox_to_anchor=(0.4, -0.12))
    fig.savefig(figs / f"{stem}_1_ranking.png"); plt.close(fig)

    c = pd.read_csv(interim / f"{cfg['aoi']}_disagreement_causes_{area}{tag}.csv")
    c["shape"] = pd.cut(c.widest_m, [0, 45, 65, 10_000], labels=["strip 30-40 m wide", "50-60 m wide", "70 m wide or more"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
    for ax, kind, title in zip(axes, ("missed", "extra"), ("Original report harvested, SAR standing", "Original report standing, SAR harvested")):
        t = c[c.kind == kind].pivot_table(index="shape", columns="cause", values="acres", aggfunc="sum", fill_value=0, observed=False)
        left = np.zeros(len(t))
        for cause in [k for k in CAUSE_COLOR if k in t.columns]:
            ax.barh(range(len(t)), t[cause], left=left, color=CAUSE_COLOR[cause], edgecolor=tf.SURFACE, lw=2, label=cause, height=0.6)
            left += t[cause].values
        for i, v in enumerate(left):
            ax.text(v, i, f"  {v:,.0f} ac", va="center", fontsize=9)
        ax.set_yticks(range(len(t)), t.index); ax.set_title(title, fontsize=10)
        ax.set_xlabel("acres, 3 report dates added"); ax.set_xlim(0, 520); ax.grid(axis="y", visible=False)
        ax.legend(fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    fig.savefig(figs / f"{stem}_2_causes_by_shape.png"); plt.close(fig)
    print(c.pivot_table(index=["kind", "cause"], columns="shape", values="acres", aggfunc="sum", fill_value=0, observed=False).round(0).to_string())


if __name__ == "__main__":
    main(*sys.argv[1:3])
