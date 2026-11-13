"""Name the likely cause of each big disagreement patch and draw examples (dark theme).

Causes are decided from evidence that does not depend on either map:
  missed (original report: harvested, SAR: standing)
    fresh cut      optical shows bare soil on the report date, SAR falls only at the next pass
    never falls    optical shows bare soil, SAR stays high afterwards too
    optical green  optical still shows green cane on the report date
  extra (original report: standing, SAR: harvested)
    SAR saw it first   SAR falls at the last pass and keeps falling, optical still green on the report date
    low all season     SAR was already low long before the harvest season
Each example row: the optical image on the three check dates (B5, B3, B2) with the patch outlined, and the VH
curve of the patch against standing cane.

All optical dates found in data/raw/s2 are also used to see when (and whether) each patch was really cut.

Usage: python src/disagreement_figures.py <area_name> [tag]
Output: data/interim/<aoi>_disagreement_causes_<area>.csv, reports/figures/<aoi>_<area>_disagreement_*.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio

import transfer_figures as tf            # dark-theme settings and colours
from sar_features import load_pixels
from transfer_test import grid_shape, harvested_by

BARE, GREEN, FALL = 0.42, 0.50, -2.5     # NDVI limits and the VH fall (dB below standing cane) that counts as "cut"
CAUSE_COLOR = {"fresh cut": "#3987e5", "never falls": "#d95926", "optical green": "#199e70", "other": "#8a897f",
               "SAR saw it first": "#3987e5", "low all season": "#d95926"}


def cause(r: pd.Series) -> str:
    ndvi = r[f"ndvi_{r.report}"]
    nxt = np.nanmin([r.vh62_next, r.vh172_next]) if not (np.isnan(r.vh62_next) and np.isnan(r.vh172_next)) else np.nan
    if r.kind == "missed":
        if ndvi > GREEN:
            return "optical green"
        if ndvi <= BARE:
            return "fresh cut" if nxt <= FALL else "never falls"
        return "other"
    first = min([str(x) for x in (r.vh62_first_low, r.vh172_first_low) if isinstance(x, str) and x] or ["9999"])
    if first < "2025-12-15" and not (r.prob >= 0.5 and nxt <= -4):
        return "low all season"
    if ndvi > BARE and r.vh62_last <= FALL and nxt <= -4:
        return "SAR saw it first"
    return "other"


def examples(cfg, d, df, labels, kind, out, n_rows=6):
    reports, shape = cfg["report_dates"], grid_shape(d)
    s2 = {}
    for date in reports:
        with rasterio.open(Path(cfg["paths"]["raw"]) / "s2" / f"{cfg['aoi']}_s2_{date}.tif") as r:
            a = r.read([4, 2, 1]).astype("float32")
        s2[date] = np.dstack([np.clip((b - np.percentile(b, 2)) / (np.percentile(b, 98) - np.percentile(b, 2)), 0, 1) for b in a])
    pick = (df[df.kind == kind].sort_values("acres", ascending=False).groupby("cause", sort=False).head(3)
            .sort_values(["cause", "acres"], ascending=[True, False]).head(n_rows))
    fig, axes = plt.subplots(len(pick), 4, figsize=(15, 3.1 * len(pick)), gridspec_kw={"width_ratios": [1, 1, 1, 1.9]})
    half = 35
    for row, (_, p) in zip(np.atleast_2d(axes), pick.iterrows()):
        lab = np.zeros(shape, bool)
        m = labels[f"{kind}_{p.report}"] == p.patch
        lab[d["rows"][m], d["cols"][m]] = True
        r0, c0 = int(np.clip(p.row - half, 0, shape[0] - 2 * half)), int(np.clip(p.col - half, 0, shape[1] - 2 * half))
        sl = (slice(r0, r0 + 2 * half), slice(c0, c0 + 2 * half))
        for ax, date in zip(row[:3], reports):
            ax.imshow(s2[date][sl], interpolation="nearest")
            ax.contour(lab[sl], [0.5], colors=["#ffffff"], linewidths=1.6)
            ax.set_title(f"optical {date}" + ("  (report date)" if date == p.report else ""), fontsize=9)
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        ax = row[3]
        y, known = harvested_by(d, reports.index(p.report) + 1)
        standing = d["group"] == 0                     # standing through the last original report
        for t, color in zip(cfg["s1_tracks"], ("#3987e5", "#c98500")):
            dates = pd.to_datetime(d[f"dates_{t}"])
            ax.plot(dates, np.nanmedian(d[f"vh_{t}"][:, m], axis=1), color=color, lw=2, marker="o", ms=4, label=f"patch, track {int(t):03d}")
            ax.plot(dates, np.nanmedian(d[f"vh_{t}"][:, standing], axis=1), color=color, lw=1.2, ls=(0, (4, 3)), label=f"standing cane, track {int(t):03d}")
        ax.axvline(pd.Timestamp(p.report), color=tf.INK, lw=1)
        ax.text(pd.Timestamp(p.report), ax.get_ylim()[1], " report date", color=tf.MUTED, fontsize=8, va="top")
        ax.set_title(f"{p.cause}  |  {p.acres:.1f} acres  |  optical NDVI on report date {p[f'ndvi_{p.report}']:.2f}", fontsize=9.5)
        ax.set_ylabel("VH (dB)"); ax.tick_params(axis="x", labelrotation=25, labelsize=8)
    np.atleast_2d(axes)[0, 3].legend(fontsize=8, ncol=2, loc="lower left")
    fig.tight_layout()
    fig.savefig(out); plt.close(fig)


def ndvi_series(cfg: dict, d: dict) -> tuple[list[str], np.ndarray]:
    """NDVI of every downloaded optical date at the crop-mask pixels; cloudy pixels (clear score < 60) are NaN."""
    import glob
    files = sorted(glob.glob(str(Path(cfg["paths"]["raw"]) / "s2" / f"{cfg['aoi']}_s2_*.tif")))
    out = []
    for f in files:
        with rasterio.open(f) as r:
            red, nir, clear = r.read([3, 5, 6]).astype("float32")[:, d["rows"], d["cols"]]
        v = (nir - red) / (nir + red + 1e-6)
        v[(clear < 60) | (nir == 0)] = np.nan
        out.append(v)
    return [f[-14:-4] for f in files], np.array(out)


def optical_history(cfg, d, df, labels, out):
    """What the optical record says about each patch over two months, and the figure of it per cause."""
    dates, ndvi = ndvi_series(cfg, d)
    when = pd.to_datetime(dates)
    verdict = []
    with np.errstate(all="ignore"):
        for _, p in df.iterrows():
            v = np.nanmedian(ndvi[:, labels[f"{p.kind}_{p.report}"] == p.patch], axis=1)
            ok = np.isfinite(v)
            step = np.diff(v[ok])
            i = int(np.argmin(step)) if len(step) else 0
            sharp = len(step) > 0 and step[i] <= -0.15 and np.nanmin(v) <= 0.32
            verdict.append({"optical_cut_seen": bool(sharp), "optical_cut_date": str(when[ok][i + 1].date()) if sharp else "",
                            "ndvi_min": round(float(np.nanmin(v)), 2)})
        df = pd.concat([df.reset_index(drop=True), pd.DataFrame(verdict)], axis=1)
        fig, ax = plt.subplots(figsize=(10, 4))
        standing = np.nanmedian(ndvi[:, d["group"] == 0], axis=1)
        ax.plot(when, standing, color=tf.MUTED, lw=1.5, ls=(0, (4, 3)), label="standing cane")
        lines = {("missed", "fresh cut"): "#3987e5", ("missed", "never falls"): "#d95926",
                 ("extra", "SAR saw it first"): "#199e70", ("extra", "low all season"): "#c98500"}
        last = cfg["report_dates"][-1]
        for (kind, cause_name), color in lines.items():
            sel = df[(df.kind == kind) & (df.cause == cause_name) & (df.report == (cfg["report_dates"][1] if kind == "missed" else last))]
            m = np.isin(labels[f"{kind}_{sel.report.iloc[0]}"], sel.patch.values) if len(sel) else None
            if m is not None:
                ax.plot(when, np.nanmedian(ndvi[:, m], axis=1), color=color, lw=2, marker="o", ms=4,
                        label=f"{cause_name} ({pd.Timestamp(sel.report.iloc[0]).strftime('%-d %b')} report)")
    for r in cfg["report_dates"]:
        ax.axvline(pd.Timestamp(r), color=tf.GRID, lw=1.2)
    ax.set_ylabel("NDVI (optical greenness)"); ax.set_ylim(0, 0.85)
    ax.legend(fontsize=8.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    fig.savefig(out); plt.close(fig)
    return df


def main(area: str, tag: str = "") -> None:
    cfg, d = load_pixels(area)
    interim, figs = Path(cfg["paths"]["interim"]), Path("reports/figures")
    area = area + tag                                  # tag = which predictions were analysed (see disagreement_eda.py)
    df = pd.read_csv(interim / f"{cfg['aoi']}_disagreement_patches_{area}.csv")
    df["cause"] = df.apply(cause, axis=1)
    labels = np.load(interim / f"{cfg['aoi']}_disagreement_labels_{area}.npz")
    stem = f"{cfg['aoi']}_{area}_disagreement"
    df = optical_history(cfg, d, df, labels, figs / f"{stem}_4_optical_history.png")
    df.to_csv(interim / f"{cfg['aoi']}_disagreement_causes_{area}.csv", index=False)
    print(df.groupby(["kind", "cause", "optical_cut_seen"]).acres.agg(["count", "sum"]).round(1))
    summary = df.groupby(["kind", "report", "cause"]).acres.agg(["count", "sum"]).round(1)
    print(summary)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1.5, 1]})
    for ax, kind, title in zip(axes, ("missed", "extra"), ("Original report: harvested, SAR: standing", "Original report: standing, SAR: harvested")):
        t = df[df.kind == kind].pivot_table(index="report", columns="cause", values="acres", aggfunc="sum", fill_value=0)
        left = np.zeros(len(t))
        for c in [c for c in CAUSE_COLOR if c in t.columns]:
            ax.barh(range(len(t)), t[c], left=left, color=CAUSE_COLOR[c], edgecolor=tf.SURFACE, lw=2, label=c, height=0.6)
            left += t[c].values
        for i, v in enumerate(left):
            ax.text(v, i, f"  {v:,.0f} ac", va="center", fontsize=9)
        ax.set_yticks(range(len(t)), [pd.Timestamp(r).strftime("%-d %b") for r in t.index]); ax.invert_yaxis()
        ax.set_title(title, fontsize=10); ax.set_xlabel("acres in big patches"); ax.grid(axis="y", visible=False)
        ax.set_xlim(0, left.max() * 1.25); ax.legend(fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    fig.savefig(figs / f"{stem}_1_causes.png"); plt.close(fig)
    examples(cfg, d, df, labels, "missed", figs / f"{stem}_2_missed.png")
    examples(cfg, d, df, labels, "extra", figs / f"{stem}_3_extra.png", n_rows=5)
    print("written", figs / stem)


if __name__ == "__main__":
    main(*sys.argv[1:3])
