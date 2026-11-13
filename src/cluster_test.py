"""Clustering idea: split the pixels the model calls "harvested" by the SHAPE of their curve, drop the slides.

Why (user's idea): one pixel's curve is noisy, so a rule on single pixels cannot tell a slow slide (weak cane)
from a real harvest. The mean curve of a group of similar pixels is clean. So:
  1. take every pixel the current method calls harvested on a report date;
  2. group them (k-means) by their VH curves on both tracks, each curve measured against the pixel's own
     early-season level, so a weak field and a healthy field are compared on their own scale;
  3. look at each group's mean curve: a harvest group shows one sudden dip, a slide group does not;
  4. keep the harvest groups, drop the slide groups.
No labels are used for the grouping or for the decision; the original report is used only to score the result.

Usage: python src/cluster_test.py <area_name>
Output: data/interim/<aoi>_cluster_scores_<area>.csv, ..._cluster_groups_<area>.csv, reports/figures/<aoi>_<area>_clusters_cut_<date>.png
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import precision_recall_fscore_support

import transfer_figures as tf            # dark-theme settings
import transfer_test as tt
from compare_detectors import smoothed_series, split, tree_model
from dip_features import dip_features
from disagreement_eda import patches
from sar_features import load_pixels

log = logging.getLogger("cluster_test")
K = 10
WEAK = ["P02", "P03", "P04", "P06"]


def curves(cfg: dict, d: dict, series: dict, report: str) -> tuple[np.ndarray, list[tuple[int, np.ndarray]]]:
    """Per pixel: VH of every pass up to the report date on each track, minus the pixel's own first-3-pass mean."""
    parts, layout = [], []
    with np.errstate(all="ignore"):
        for t in cfg["s1_tracks"]:
            use = d[f"dates_{t}"] <= report
            s = series[("vh", t)][use]
            s = s - np.nanpercentile(s, 90, axis=1)[:, None]            # take out scene-wide changes (rain, drying)
            s = np.nan_to_num(s - np.nanmean(s[:3], axis=0), nan=0.0)
            parts.append(s.T)
            layout.append((t, d[f"dates_{t}"][use]))
    return np.hstack(parts).astype("float32"), layout


def main(area: str) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    cfg, d = load_pixels(area)
    test, reports, rng = split(d), cfg["report_dates"], np.random.default_rng(cfg["seed"])
    series = smoothed_series(cfg, d, 5)
    feats = {r: tt.date_free_features(cfg, d, series, r) for r in reports}
    labels = {r: tt.harvested_by(d, n) for n, r in enumerate(reports, 1)}
    edges = {r: tt.near_edge(d, *labels[r]) for r in reports}
    interim, figs = Path(cfg["paths"]["interim"]), Path("reports/figures")
    with rasterio.open(interim / f"{cfg['aoi']}_cane_mask_{area}.tif") as src:
        tr, crs = src.transform, src.crs
    rv = gpd.read_file(f"outputs/review_patches/{cfg['aoi']}_{area}_review_patches_v2.gpkg").to_crs(crs)
    weak = np.zeros(len(d["group"]), bool)
    for _, p in rv.iterrows():
        if p.patch_id in WEAK:
            weak |= rasterize([(p.geometry, 1)], out_shape=tt.grid_shape(d), transform=tr, dtype="uint8")[d["rows"], d["cols"]] == 1

    limits = (-2.0, -2.5, -3.0, -3.5)
    rows, groups = [], []
    earlier = {"current method": np.zeros(len(weak), bool), **{g: np.zeros(len(weak), bool) for g in limits}}
    for n, held in enumerate(reports, 1):
        idx = {r: rng.choice(np.flatnonzero(labels[r][1] & ~test & ~edges[r]), 300_000, replace=False) for r in reports if r != held}
        model = tree_model(cfg).fit(np.vstack([feats[r][i] for r, i in idx.items()]), np.concatenate([labels[r][0][i] for r, i in idx.items()]))
        own = tt.grow(model.predict_proba(feats[held])[:, 1], d)          # the current method, this date only

        x, layout = curves(cfg, d, series, held)
        cut = np.flatnonzero(own)
        km = MiniBatchKMeans(K, random_state=cfg["seed"], n_init=3, batch_size=20_000).fit(x[rng.choice(cut, min(len(cut), 200_000), replace=False)])
        lab = km.predict(x[cut])
        # Decide per group from its clean mean curve: deepest sudden dip on either track.
        dips, start = np.zeros(K), 0
        best = np.zeros(K)
        for t, dates in layout:
            c = km.cluster_centers_[:, start:start + len(dates)].T          # (passes, groups)
            best = np.minimum(best, dip_features(c)[:, 0])
            start += len(dates)
        y, known = labels[held]
        for k in range(K):
            m = cut[lab == k]
            groups.append({"report": held, "group": k, "acres": round(len(m) * tt.ACRE_PER_PIXEL, 1), "deepest_dip_db": round(float(best[k]), 2),
                           "original_harvested_share": round(float(y[m][known[m]].mean()), 3), "weak_patch_pixels": int(weak[m].sum())})

        # Figure: mean curve of every group (main track), coloured by the dip it shows.
        t0, dates0 = layout[0]
        fig, axes = plt.subplots(2, K // 2, figsize=(3.1 * K // 2, 5.2), sharex=True, sharey=True)
        for k, ax in zip(np.argsort(best), axes.ravel()):
            ax.plot(pd.to_datetime(dates0), km.cluster_centers_[k, :len(dates0)], color="#3987e5" if best[k] <= -3 else "#d95926", lw=2, marker="o", ms=3)
            g = groups[-K + k]
            ax.set_title(f"{g['acres']:,.0f} ac | dip {g['deepest_dip_db']:.1f} dB\noriginal harvested {g['original_harvested_share']:.0%}", fontsize=8.5)
            ax.tick_params(axis="x", labelrotation=45, labelsize=7)
        fig.supylabel("VH against the pixel's own early level (dB)", fontsize=9, color=tf.MUTED)
        fig.tight_layout()
        fig.savefig(figs / f"{cfg['aoi']}_{area}_clusters_cut_{held}.png"); plt.close(fig)

        variants = {"current method": own, **{g: own & np.isin(np.arange(len(own)), cut[np.isin(lab, np.flatnonzero(best <= g))]) for g in limits}}
        for name, pred in variants.items():
            pred = pred | earlier[name]
            earlier[name] = pred
            m, inside = known & test, known & test & ~edges[held]
            f1 = precision_recall_fscore_support(y[m], pred[m], average="binary")[2]
            pi, ri, _, _ = precision_recall_fscore_support(y[inside], pred[inside], average="binary")
            rows.append({"variant": name if isinstance(name, str) else f"keep groups with dip <= {name} dB", "report": held, "f1": f1,
                         "precision_inside": pi, "recall_inside": ri,
                         "big_extra_acres": (patches(known & ~y & pred, d) > 0).sum() * tt.ACRE_PER_PIXEL,
                         "big_missed_acres": (patches(known & y & ~pred, d) > 0).sum() * tt.ACRE_PER_PIXEL,
                         "sar_vs_original_area": pred[m].sum() / y[m].sum() - 1,
                         "weak_called_cut": float(pred[weak & ~y].mean())})
        log.info("%s done", held)
    df, gr = pd.DataFrame(rows), pd.DataFrame(groups)
    df.to_csv(interim / f"{cfg['aoi']}_cluster_scores_{area}.csv", index=False)
    gr.to_csv(interim / f"{cfg['aoi']}_cluster_groups_{area}.csv", index=False)
    print(gr.sort_values(["report", "deepest_dip_db"]).to_string(index=False))
    print(df.groupby("variant", sort=False)[["f1", "precision_inside", "recall_inside", "big_extra_acres", "big_missed_acres",
                                              "sar_vs_original_area", "weak_called_cut"]].mean().round(3).to_string())


if __name__ == "__main__":
    main(sys.argv[1])
