"""EDA figures for one area: how harvest looks in Sentinel-1, and how the three detectors behave.

Every figure is saved as PNG in reports/figures/ and the numbers behind the text go to one JSON file, so the
report can be rebuilt and each figure checked on its own. Only crop-mask pixels are used for SAR.

Usage: python src/eda_figures.py <area_name>
Output: reports/figures/<aoi>_<area>_*.png, reports/figures/<aoi>_<area>_numbers.json
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import f1_score

from compare_detectors import detector_features, smoothed_series, split, tree_model
from sar_features import load_pixels

ACRE_PER_PIXEL = 100 / 4046.8564224
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e6e5df"
GROUP_COLOR = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]          # fixed order: standing, report 1, 2, 3
DETECTOR_COLOR = {"rgb3": "#6250d6", "drop": "#e87ba4", "tree": "#008300"}
DETECTOR_NAME = {"rgb3": "3 latest passes (colour composite)", "drop": "VH drop, one threshold", "tree": "tree model, all passes"}
log = logging.getLogger("eda_figures")
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.axisbelow": True,
                     "grid.color": GRID, "grid.linewidth": 0.8, "axes.titlecolor": INK, "axes.titlesize": 11,
                     "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False, "savefig.dpi": 140,
                     "savefig.bbox": "tight", "figure.facecolor": "white"})


def group_names(cfg: dict) -> list[str]:
    r = [pd.Timestamp(x).strftime("%-d %b") for x in cfg["report_dates"]]
    return ["standing on " + r[-1], "cut by " + r[0]] + [f"cut {a} to {b}" for a, b in zip(r[:-1], r[1:])]


def fig_curves(cfg, d, names, out):
    """Median VH and VV of each report group through time, per track (band = middle half of the pixels)."""
    tracks = list(cfg["s1_tracks"])
    fig, axes = plt.subplots(2, len(tracks), figsize=(6.2 * len(tracks), 6.6), sharex=True, sharey="row")
    for j, t in enumerate(tracks):
        dates = pd.to_datetime(d[f"dates_{t}"])
        for i, b in enumerate(("vh", "vv")):
            ax = axes[i, j]
            for k, name in enumerate(names):
                q = np.nanpercentile(d[f"{b}_{t}"][:, d["group"] == k], [25, 50, 75], axis=1)
                ax.fill_between(dates, q[0], q[2], color=GROUP_COLOR[k], alpha=0.10, lw=0)
                ax.plot(dates, q[1], color=GROUP_COLOR[k], lw=2, marker="o", ms=4, label=name)
            for r in cfg["report_dates"]:
                ax.axvline(pd.Timestamp(r), color=MUTED, lw=1, ls=(0, (3, 3)))
                if i == 0:
                    ax.text(pd.Timestamp(r), ax.get_ylim()[1], pd.Timestamp(r).strftime(" %-d %b"), color=MUTED, fontsize=8, rotation=90, va="top", ha="right")
            ax.set_title(f"{b.upper()}, track {int(t):03d} ({cfg['s1_tracks'][t][:4].lower()}.)")
            ax.set_ylabel("backscatter (dB)")
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%-d %b"))
    axes[0, 0].legend(loc="lower left", fontsize=9)
    fig.savefig(out)
    plt.close(fig)


def fig_speckle(cfg, d, series1, series5, names, out):
    """Change of VH over the pass pair that brackets report 2's harvest: one pixel vs the 5 x 5 mean."""
    t = next(iter(cfg["s1_tracks"]))
    dates = list(d[f"dates_{t}"])
    before = max(x for x in dates if x <= cfg["report_dates"][0])
    after = max(x for x in dates if x <= cfg["report_dates"][1])
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    bins = np.linspace(-12, 8, 81)
    res = {}
    for ax, (label, s) in zip(axes, (("one pixel (10 m)", series1), ("5 x 5 pixel mean (50 m)", series5))):
        step = s[("vh", t)][dates.index(after)] - s[("vh", t)][dates.index(before)]
        for k in (0, 2):
            v = step[d["group"] == k]
            ax.hist(v, bins, density=True, histtype="stepfilled", color=GROUP_COLOR[k], alpha=0.35, lw=0)
            ax.hist(v, bins, density=True, histtype="step", color=GROUP_COLOR[k], lw=2, label=names[k])
            res[f"{label}|{names[k]}"] = [float(x) for x in np.nanpercentile(v, [10, 50, 90])]
        ax.set_title(label)
        ax.set_xlabel(f"VH change from {before[5:]} to {after[5:]} (dB)")
    axes[0].set_ylabel("share of pixels")
    axes[0].legend(fontsize=9)
    fig.savefig(out)
    plt.close(fig)
    return res


def stretch(a: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip((a - lo) / (hi - lo), 0, 1)


def fig_maps(cfg, d, series, pred, names, out):
    """One zoom window on report 2's date: optical view, the user's 3-pass SAR composite, the report, the SAR result."""
    t = next(iter(cfg["s1_tracks"]))
    n, report = 2, cfg["report_dates"][1]
    rows, cols, group = d["rows"], d["cols"], d["group"]
    shape = (int(rows.max()) + 1, int(cols.max()) + 1)
    win = 300
    count = ndimage.uniform_filter(np.bincount(rows[group == n].astype("int64") * shape[1] + cols[group == n], minlength=shape[0] * shape[1])
                                   .reshape(shape).astype("float32"), win)
    r0, c0 = np.unravel_index(np.argmax(count), shape)
    r0, c0 = int(np.clip(r0 - win // 2, 0, shape[0] - win)), int(np.clip(c0 - win // 2, 0, shape[1] - win))
    sl = (slice(r0, r0 + win), slice(c0, c0 + win))

    def to_grid(v, fill=np.nan):
        g = np.full(shape, fill, "float32")
        g[rows, cols] = v
        return g[sl]

    with rasterio.open(Path(cfg["paths"]["raw"]) / "s2" / f"{cfg['aoi']}_s2_{report}.tif") as r:
        s2 = r.read([4, 2, 1]).astype("float32")[:, sl[0], sl[1]]           # B5, B3, B2 = the user's usual view
    optical = np.dstack([stretch(b, *np.percentile(b, [2, 98])) for b in s2])
    dates = d[f"dates_{t}"][d[f"dates_{t}"] <= report][-3:]
    idx = [list(d[f"dates_{t}"]).index(x) for x in dates]
    comp = np.dstack([stretch(to_grid(series[("vh", t)][i]), -24, -12) for i in idx])
    comp[np.isnan(comp)] = 1.0                                              # outside the crop mask: white, not processed
    truth = to_grid(np.where(group == 255, np.nan, np.where((group >= 1) & (group <= n), 1, 0)))
    sar = to_grid(pred.astype("float32"))

    def classes(g):
        img = np.ones(g.shape + (3,), "float32")
        img[g == 0] = matplotlib.colors.to_rgb("#454B1B")                  # official 81 Remaining Cane
        img[g == 1] = matplotlib.colors.to_rgb("#F89508")                  # official 82 Harvested
        return img

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.6))
    titles = [f"Sentinel-2, {report}\n(B5, B3, B2; check only)",
              f"SAR VH composite\nred {dates[0][5:]}, green {dates[1][5:]}, blue {dates[2][5:]}",
              f"Reference report\n{report}", f"SAR tree model\nlast pass {dates[2]}"]
    for ax, img, title in zip(axes, (optical, comp, classes(truth), classes(sar)), titles):
        ax.imshow(img, interpolation="nearest")
        ax.set_title(title, fontsize=9.5)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for ax in axes[2:]:
        for color, label in (("#454B1B", "81 remaining cane"), ("#F89508", "82 harvested")):
            ax.plot([], [], "s", color=color, ms=9, label=label)
        ax.legend(loc="lower left", fontsize=8, frameon=True, facecolor="white", edgecolor=GRID)
    fig.savefig(out)
    plt.close(fig)
    return {"window_acres": round(win * win * ACRE_PER_PIXEL, 1), "row0": r0, "col0": c0}


def fig_scores(scores: pd.DataFrame, out):
    """F1 of each detector on each report date (truth = reference report, test blocks only)."""
    fig, ax = plt.subplots(figsize=(8, 3.6))
    reports = list(scores.report.unique())
    for i, name in enumerate(DETECTOR_NAME):
        s = scores[scores.detector == name].set_index("report").loc[reports]
        x = np.arange(len(reports)) + (i - 1) * 0.27
        ax.bar(x, s.f1, 0.24, color=DETECTOR_COLOR[name], label=DETECTOR_NAME[name])
        for a, v in zip(x, s.f1):
            ax.text(a, v + 0.012, f"{v:.2f}", ha="center", fontsize=9, color=INK)
    ax.set_xticks(range(len(reports)), [f"report {pd.Timestamp(r).strftime('%-d %b')}" for r in reports])
    ax.set_ylim(0, 1.05); ax.set_ylabel("F1 of the harvested class"); ax.grid(axis="x", visible=False)
    ax.legend(fontsize=9, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.savefig(out)
    plt.close(fig)


def fig_clusters(cfg, d, series, names, out):
    """Unsupervised view: 6 k-means clusters of the VH series (main track), their mean curves and what they hold."""
    t = next(iter(cfg["s1_tracks"]))
    x = series[("vh", t)].T
    ok = np.isfinite(x).all(axis=1)
    rng = np.random.default_rng(cfg["seed"])
    km = MiniBatchKMeans(6, random_state=cfg["seed"], n_init=3, batch_size=20000).fit(x[rng.choice(np.flatnonzero(ok), 300_000, replace=False)])
    lab = np.full(len(x), -1)
    lab[ok] = km.predict(x[ok])
    order = np.argsort([-(lab == k).sum() for k in range(6)])
    dates = pd.to_datetime(d[f"dates_{t}"])
    fig, axes = plt.subplots(2, 3, figsize=(13, 5.6), sharex=True, sharey=True)
    table = []
    for ax, k in zip(axes.ravel(), order):
        m = lab == k
        share = [float(((d["group"] == g) & m).sum() / m.sum()) for g in range(4)]
        table.append({"acres": round(m.sum() * ACRE_PER_PIXEL, 1), **{names[g]: round(share[g], 3) for g in range(4)}})
        ax.plot(dates, km.cluster_centers_[k], color=INK, lw=2, marker="o", ms=3.5)
        for r in cfg["report_dates"]:
            ax.axvline(pd.Timestamp(r), color=MUTED, lw=1, ls=(0, (3, 3)))
        ax.set_title(f"{m.sum() * ACRE_PER_PIXEL:,.0f} acres", fontsize=10)
        left = 0.0
        inset = ax.inset_axes([0.03, 0.06, 0.6, 0.07])
        for g in range(4):
            inset.barh(0, share[g], left=left, color=GROUP_COLOR[g], edgecolor="white", lw=1)
            left += share[g]
        inset.set_xlim(0, 1); inset.axis("off")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%-d %b")); ax.tick_params(axis="x", labelrotation=30)
    handles = [plt.Line2D([], [], marker="s", ls="", color=GROUP_COLOR[g], label=names[g]) for g in range(4)]
    fig.legend(handles=handles, fontsize=9, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 0.0), title="bar in each panel = what the cluster holds")
    axes[0, 0].set_ylabel("VH (dB)"); axes[1, 0].set_ylabel("VH (dB)")
    fig.savefig(out)
    plt.close(fig)
    return table


def fig_drop_date(cfg, d, series, names, out):
    """For pixels of each harvested group: at which pass the biggest VH fall happens (main track)."""
    t = next(iter(cfg["s1_tracks"]))
    vh = series[("vh", t)]
    dates = d[f"dates_{t}"]
    fall = np.diff(vh, axis=0)
    when = np.nanargmin(np.where(np.isfinite(fall), fall, 0), axis=0) + 1          # index of the pass after the fall
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4), sharey=True)
    res = {}
    for ax, g in zip(axes, (1, 2, 3)):
        share = np.bincount(when[d["group"] == g], minlength=len(dates))[1:] / (d["group"] == g).sum()
        ax.bar(range(1, len(dates)), share, 0.7, color=GROUP_COLOR[g])
        ax.set_xticks(range(1, len(dates)), [x[5:] for x in dates[1:]], rotation=60, fontsize=8)
        ax.set_title(names[g]); ax.grid(axis="x", visible=False)
        res[names[g]] = {x[5:]: round(float(s), 3) for x, s in zip(dates[1:], share)}
    axes[0].set_ylabel("share of the group's pixels")
    fig.supxlabel("pass at which the biggest single fall of VH is seen", fontsize=9, color=MUTED)
    fig.savefig(out)
    plt.close(fig)
    return res


def edge_and_optical_check(cfg, d, pred, test, n):
    """Where the SAR result and report n disagree: is it at edges, and what does the optical image show?"""
    rows, cols, group = d["rows"], d["cols"], d["group"]
    known = group != 255
    y = (group >= 1) & (group <= n)
    shape = (int(rows.max()) + 1, int(cols.max()) + 1)
    g = np.full(shape, 255, "uint8")
    g[rows, cols] = np.where(known, y, 255)
    edge = (ndimage.maximum_filter(g, 5) != ndimage.minimum_filter(g, 5))[rows, cols]
    with rasterio.open(Path(cfg["paths"]["raw"]) / "s2" / f"{cfg['aoi']}_s2_{cfg['report_dates'][n - 1]}.tif") as r:
        red, nir = r.read([3, 5]).astype("float32")[:, rows, cols]
    ndvi = (nir - red) / (nir + red + 1e-6)
    m = known & test
    out = {"edge_share_pixels": float(edge[m].mean()), "edge_share_errors": float(edge[m & (pred != y)].mean()),
           "f1_all": float(f1_score(y[m], pred[m])), "f1_inside": float(f1_score(y[m & ~edge], pred[m & ~edge])),
           "f1_edge": float(f1_score(y[m & edge], pred[m & edge]))}
    for name, sel in (("agree_harvested", pred & y), ("agree_standing", ~pred & ~y), ("report_harvested_sar_standing", ~pred & y),
                      ("report_standing_sar_harvested", pred & ~y)):
        v = ndvi[m & sel]
        out[name] = {"acres": round(float(v.size * ACRE_PER_PIXEL), 1), "ndvi_median": round(float(np.median(v)), 2),
                     "bare_share": round(float((v < 0.35).mean()), 2), "green_share": round(float((v > 0.5).mean()), 2)}
    out["recall_by_group"] = {int(k): round(float(pred[m & (group == k)].mean()), 3) for k in range(1, n + 1)}
    return out


def main(area: str) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    cfg, d = load_pixels(area)
    figs = Path("reports/figures"); figs.mkdir(parents=True, exist_ok=True)
    stem = f"{cfg['aoi']}_{area}"
    names = group_names(cfg)
    series1, series5 = smoothed_series(cfg, d, 1), smoothed_series(cfg, d, 5)
    numbers = {"groups_acres": {names[k]: round(float((d["group"] == k).sum() * ACRE_PER_PIXEL), 1) for k in range(4)}}

    fig_curves(cfg, d, names, figs / f"{stem}_1_curves.png")
    numbers["step"] = fig_speckle(cfg, d, series1, series5, names, figs / f"{stem}_2_speckle.png")
    numbers["clusters"] = fig_clusters(cfg, d, series5, names, figs / f"{stem}_3_clusters.png")
    numbers["drop_date"] = fig_drop_date(cfg, d, series5, names, figs / f"{stem}_4_drop_date.png")

    test, rng = split(d), np.random.default_rng(cfg["seed"])
    preds = {}
    for n, report in enumerate(cfg["report_dates"], 1):       # tree model, trained on the train blocks, applied everywhere
        x = detector_features(cfg, d, series5, report)["tree"]
        y = (d["group"] >= 1) & (d["group"] <= n)
        tr = np.flatnonzero((d["group"] != 255) & ~test)
        tr = rng.choice(tr, min(len(tr), 400_000), replace=False)
        preds[n] = tree_model(cfg).fit(x[tr], y[tr]).predict_proba(x)[:, 1] >= 0.5
        numbers[f"check_report_{n}"] = edge_and_optical_check(cfg, d, preds[n], test, n)
    numbers["map"] = fig_maps(cfg, d, series5, preds[2], names, figs / f"{stem}_5_maps.png")
    scores = pd.read_csv(Path(cfg["paths"]["interim"]) / f"{cfg['aoi']}_detector_scores_{area}_smooth5.csv")
    fig_scores(scores, figs / f"{stem}_6_scores.png")
    (figs / f"{stem}_numbers.json").write_text(json.dumps(numbers, indent=1))
    log.info("figures and numbers written to %s", figs)


if __name__ == "__main__":
    main(sys.argv[1])
