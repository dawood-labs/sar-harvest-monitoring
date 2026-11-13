"""Package the doubtful disagreement patches for a manual review in QGIS.

Why: for some patches neither the optical record nor SAR settles who is right; a person has to look.
Chosen patches: (a) original report says harvested, SAR never falls and the optical record shows no cut;
(b) original report says standing, SAR has been low since early in the season.
The same place on several report dates becomes one patch.

Output in outputs/review_patches/ (everything opens styled in QGIS):
  <aoi>_<area>_review_patches_v1.gpkg + .qml   polygons with id, question, evidence, and two empty fields
                                               (user_verdict, user_note) to fill in
  <aoi>_<area>_review_s2_<date>_v1.tif + .qml  every optical date, cut to 500 m around the patches, shown as B5, B3, B2
  <aoi>_<area>_review_sheet_v1.png             per patch: SAR curve and optical greenness curve
  <aoi>_<area>_review_patches_v1.zip           all of the above

Usage: python src/export_review.py <area_name>
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape
from shapely.ops import unary_union

import transfer_figures as tf
from disagreement_figures import ndvi_series
from sar_features import load_pixels
from transfer_test import grid_shape

ACRE_M2 = 4046.8564224
QUESTION = {"never falls": "Original report: harvested. SAR never falls and optical shows no clear cut. Was it really harvested?",
            "low all season": "Original report: standing. SAR low since early season. Is this cane, and was it cut?"}


def vector_qml(path: Path) -> None:
    path.with_suffix(".qml").write_text(
        '<!DOCTYPE qgis><qgis><renderer-v2 type="singleSymbol"><symbols><symbol type="fill" name="0" alpha="1">'
        '<layer class="SimpleFill"><prop k="style" v="no"/><prop k="outline_color" v="255,255,0,255"/>'
        '<prop k="outline_width" v="0.8"/><prop k="outline_style" v="solid"/></layer></symbol></symbols></renderer-v2>'
        '<labeling type="simple"><settings><text-style fieldName="patch_id" fontSize="10" textColor="255,255,0,255">'
        '<text-buffer bufferDraw="1" bufferSize="1" bufferColor="0,0,0,255"/></text-style></settings></labeling>'
        '<labelsEnabled>1</labelsEnabled></qgis>')


def optical_qml(path: Path, limits: list[tuple[float, float]]) -> None:
    """Show bands 4, 2, 1 of the file (= B5, B3, B2) with one fixed stretch, so all dates look alike."""
    parts = "".join(f"<{c}ContrastEnhancement><minValue>{lo:.0f}</minValue><maxValue>{hi:.0f}</maxValue>"
                    f"<algorithm>StretchToMinimumMaximum</algorithm></{c}ContrastEnhancement>"
                    for c, (lo, hi) in zip(("red", "green", "blue"), limits))
    path.with_suffix(".qml").write_text(
        '<!DOCTYPE qgis><qgis><pipe><rasterrenderer type="multibandcolor" redBand="4" greenBand="2" blueBand="1" '
        f'opacity="1" alphaBand="-1">{parts}</rasterrenderer></pipe></qgis>')


def main(area: str) -> None:
    cfg, d = load_pixels(area)
    interim, out = Path(cfg["paths"]["interim"]), Path("outputs/review_patches")
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{cfg['aoi']}_{area}_review"
    df = pd.read_csv(interim / f"{cfg['aoi']}_disagreement_causes_{area}.csv")
    labels = np.load(interim / f"{cfg['aoi']}_disagreement_labels_{area}.npz")
    pick = df[((df.cause == "never falls") & ~df.optical_cut_seen) | (df.cause == "low all season")]
    with rasterio.open(interim / f"{cfg['aoi']}_cane_mask_{area}.tif") as r:
        tr, crs = r.transform, r.crs
    shape2d = grid_shape(d)

    # One polygon per place: patches of different report dates that overlap are merged.
    geoms = []
    for _, p in pick.iterrows():
        img = np.zeros(shape2d, "uint8")
        m = labels[f"{p.kind}_{p.report}"] == p.patch
        img[d["rows"][m], d["cols"][m]] = 1
        geoms.append(unary_union([shape(g) for g, v in shapes(img, mask=img == 1, transform=tr)]))
    pick = gpd.GeoDataFrame(pick.reset_index(drop=True), geometry=geoms, crs=crs)
    merged = gpd.GeoDataFrame(geometry=list(unary_union(pick.geometry.buffer(1)).geoms), crs=crs)
    merged["geometry"] = merged.buffer(-1)
    rows = []
    merged = merged.iloc[np.argsort(-merged.area.values)]
    for i, g in enumerate(merged.geometry, 1):
        part = pick[pick.intersects(g)].sort_values("report")
        first = part.iloc[0]
        rows.append({"patch_id": f"P{i:02d}", "cause": first.cause, "question": QUESTION[first.cause],
                     "report_dates": ", ".join(part.report), "area_ac": round(g.area / ACRE_M2, 3),
                     "original_says": "harvested" if first.kind == "missed" else "standing",
                     "sar_says": "standing" if first.kind == "missed" else "harvested",
                     "ndvi_min": float(part.ndvi_min.min()), "optical_cut_seen": bool(part.optical_cut_seen.any()),
                     "optical_cut_date": ", ".join(sorted({x for x in part.optical_cut_date.fillna("") if x})),
                     "vh_first_low_062": str(first.vh62_first_low) if isinstance(first.vh62_first_low, str) else "",
                     "user_verdict": "", "user_note": "", "geometry": g})
    review = gpd.GeoDataFrame(rows, crs=crs)
    vector = out / f"{stem}_patches_v1.gpkg"
    vector.unlink(missing_ok=True)
    review.to_crs(cfg["crs"]["deliver"]).to_file(vector, layer="review_patches", driver="GPKG")
    vector_qml(vector)

    # Optical dates, kept only within 500 m of a patch (small files), one fixed colour stretch for all dates.
    near = np.zeros(shape2d, bool)
    for g in review.geometry:
        x0, y0, x1, y1 = g.buffer(500).bounds
        (r0, r1), (c0, c1) = sorted([int((tr.f - y1) / 10), int((tr.f - y0) / 10)]), sorted([int((x0 - tr.c) / 10), int((x1 - tr.c) / 10)])
        near[max(r0, 0):r1, max(c0, 0):c1] = True
    files = sorted((Path(cfg["paths"]["raw"]) / "s2").glob(f"{cfg['aoi']}_s2_*.tif"))
    limits = None
    for f in files:
        with rasterio.open(f) as r:
            a, prof = r.read(), r.profile
        if limits is None:
            limits = [tuple(np.percentile(a[b - 1][near & (a[b - 1] > 0)], [2, 98])) for b in (4, 2, 1)]
        a[:, ~near] = 0
        p = out / f"{stem}_s2_{f.stem[-10:]}_v1.tif"
        with rasterio.open(p, "w", **{**prof, "compress": "deflate", "predictor": 2}) as dst:
            dst.write(a)
            dst.descriptions = ("B2", "B3", "B4", "B5", "B8", "clear_score")
        optical_qml(p, limits)

    # Review sheet: SAR and optical greenness of each patch against standing cane.
    dates, ndvi = ndvi_series(cfg, d)
    standing = d["group"] == 0
    fig, axes = plt.subplots(len(review), 2, figsize=(13, 2.6 * len(review)), squeeze=False)
    with np.errstate(all="ignore"):
        for (ax1, ax2), (_, p) in zip(axes, review.iterrows()):
            img = rasterio.features.rasterize([(p.geometry, 1)], out_shape=shape2d, transform=tr, dtype="uint8")
            m = img[d["rows"], d["cols"]] == 1
            for t, color in zip(cfg["s1_tracks"], ("#3987e5", "#c98500")):
                when = pd.to_datetime(d[f"dates_{t}"])
                ax1.plot(when, np.nanmedian(d[f"vh_{t}"][:, m], axis=1), color=color, lw=2, marker="o", ms=3.5, label=f"patch, track {int(t):03d}")
                ax1.plot(when, np.nanmedian(d[f"vh_{t}"][:, standing], axis=1), color=color, lw=1.2, ls=(0, (4, 3)), label=f"standing cane, track {int(t):03d}")
            ax2.plot(pd.to_datetime(dates), np.nanmedian(ndvi[:, m], axis=1), color="#199e70", lw=2, marker="o", ms=3.5, label="patch")
            ax2.plot(pd.to_datetime(dates), np.nanmedian(ndvi[:, standing], axis=1), color=tf.MUTED, lw=1.2, ls=(0, (4, 3)), label="standing cane")
            ax1.set_title(f"{p.patch_id}  |  {p.area_ac:.1f} acres  |  original: {p.original_says}, SAR: {p.sar_says}", fontsize=10)
            ax2.set_title(f"{p.patch_id}  |  optical greenness (NDVI)", fontsize=10)
            ax1.set_ylabel("VH (dB)"); ax2.set_ylim(0, 0.85)
            for ax in (ax1, ax2):
                for r in cfg["report_dates"]:
                    ax.axvline(pd.Timestamp(r), color=tf.GRID, lw=1.2)
                ax.tick_params(axis="x", labelsize=8)
    axes[0, 0].legend(fontsize=7.5, ncol=2); axes[0, 1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / f"{stem}_sheet_v1.png"); plt.close(fig)

    with zipfile.ZipFile(out / f"{stem}_patches_v1.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(out.iterdir()):
            if f.suffix != ".zip":
                z.write(f, f.name)
    print(review.drop(columns=["geometry", "question", "user_verdict", "user_note"]).to_string(index=False))
    print("zip MB", round((out / f"{stem}_patches_v1.zip").stat().st_size / 1e6, 1))


if __name__ == "__main__":
    main(sys.argv[1])
