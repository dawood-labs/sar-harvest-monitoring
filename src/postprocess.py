"""Turn the raw model output into clean report polygons whose total area is exactly the crop-mask area.

Inputs: the crop-mask polygons and the model's harvest probability per crop-mask pixel and report date.
Output: per report date, polygons with `predicted` 81 (remaining cane) / 82 (harvested) that split every crop-mask
polygon exactly: harvested + remaining = the crop-mask polygon, no gaps, no overlaps.

Raster stage (crop-mask pixels only)
  1. Smooth the probability map with total-variation denoising (TV_WEIGHT; 0 = off): the map stays close to the
     model but small ups and downs vanish and boundaries get short and straight, while real sharp field edges stay.
  2. Harvested = the same region growing as the model (sure >= 0.5, grown into touching pixels >= 0.15).
  3. Pieces smaller than MMU_ACRES (either class) take the class around them, unless the model was sure about them
     (mean raw probability >= 0.9 for harvested specks, <= 0.1 for standing islands).
  Everything harvested in an earlier report stays harvested and is never touched by steps 1-3.
Vector stage (per crop-mask polygon)
  4. Harvested pixels -> polygons, clipped to the crop-mask polygon. Where harvested land stops within EDGE_SNAP_M of
     the crop-mask edge, it is extended to the edge (thin remaining strips along edges are pixel artefacts).
  5. Remaining = crop-mask polygon minus harvested.
  6. Pieces smaller than MMU_ACRES switch to the other class inside the same crop-mask polygon (area is kept),
     except harvested pieces that were harvested in an earlier report.
  7. A crop-mask polygon that ends up one class keeps its exact original geometry. Polygons without any crop-mask
     pixel take the class of the harvested pixels touching them, else remaining, with review = "no_pixels".
     Split pieces smaller than 2 x MMU_ACRES get review = "small_clipped".
  8. Checks: per crop-mask polygon, harvested + remaining = its area (to 0.001 acre); geometries valid.

Usage: python src/postprocess.py <area> <config.yaml> <pred.npz> <prob key prefix> <out_tag> [tv_weight] [mmu_acres] [edge_snap_m]
       e.g. python src/postprocess.py full configs/<report config>.yaml data/interim/<aoi>_area_pred_full.npz prob_ v1 0 0.5 10
Output: outputs/<aoi>_<area>_clean/<aoi>_<area>_harvest_<date>_<tag>.gpkg (+ .qml), ..._clean_raster_<date>_<tag>.tif,
        ..._postprocess_checks_<tag>.csv
"""
from __future__ import annotations

import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely
import yaml
from rasterio.features import shapes
from scipy import ndimage
from skimage.filters import apply_hysteresis_threshold
from skimage.restoration import denoise_tv_chambolle

from resources import workers

log = logging.getLogger("postprocess")
ACRE_M2, PIXEL_M2 = 4046.8564224, 100.0
SURE, MAYBE, CONFIDENT = 0.5, 0.15, 0.9
TILE, HALO = 2048, 32


# ------------------------------------------------------------------ raster stage
def tv_smooth(prob: np.ndarray, inside: np.ndarray, weight: float) -> np.ndarray:
    """Total-variation smoothing inside the crop mask, tile by tile. Outside pixels copy their nearest inside value,
    so the crop-mask edge does not pull the inside values towards anything."""
    if weight <= 0:
        return prob
    idx = ndimage.distance_transform_edt(~inside, return_distances=False, return_indices=True)
    filled = prob[idx[0], idx[1]]
    out = prob.copy()
    H, W = prob.shape
    jobs = [(r0, c0) for r0 in range(0, H, TILE) for c0 in range(0, W, TILE) if inside[r0:r0 + TILE, c0:c0 + TILE].any()]

    def one(rc):
        r0, c0 = rc
        a0, b0 = max(r0 - HALO, 0), max(c0 - HALO, 0)
        a1, b1 = min(r0 + TILE + HALO, H), min(c0 + TILE + HALO, W)
        sm = denoise_tv_chambolle(filled[a0:a1, b0:b1], weight=weight, max_num_iter=100)
        return r0, c0, sm[r0 - a0:r0 - a0 + min(TILE, H - r0), c0 - b0:c0 - b0 + min(TILE, W - c0)]

    from concurrent.futures import ThreadPoolExecutor
    import os
    with ThreadPoolExecutor(int(os.environ.get("TRIAL_THREADS", 0)) or workers(keep_free_cores=0)) as ex:
        for r0, c0, sm in ex.map(one, jobs):
            out[r0:r0 + TILE, c0:c0 + TILE] = np.where(inside[r0:r0 + TILE, c0:c0 + TILE], sm, out[r0:r0 + TILE, c0:c0 + TILE])
    return out


def drop_small(cut: np.ndarray, inside: np.ndarray, raw: np.ndarray, locked: np.ndarray, mmu_px: int,
               holes_first: bool = False) -> np.ndarray:
    """Pieces smaller than mmu_px take the class around them, unless the model was sure or they are locked.

    A piece that is a whole cane patch by itself is never changed: it has no other class around it, and in areas
    with small fields a whole field is often smaller than the smallest piece size.
    """
    cut = cut.copy()
    patch, n_patch = ndimage.label(inside)
    patch_size = np.bincount(patch.ravel(), minlength=n_patch + 1)
    for _ in range(2):                                 # a second pass catches pieces that the first pass created
        # holes_first: small standing islands inside harvested land are filled first, which joins harvested
        # fragments into whole pieces before the small harvested pieces are judged
        for value in ((False, True) if holes_first else (True, False)):
            lab, n = ndimage.label((cut == value) & inside)
            if n == 0:
                continue
            size = np.bincount(lab.ravel(), minlength=n + 1)
            mean = ndimage.mean(raw, lab, np.arange(n + 1))
            of_patch = np.zeros(n + 1, "int64")
            of_patch[lab[inside]] = patch[inside]                    # the cane patch every piece lies in
            small = (size < mmu_px) & (np.arange(n + 1) > 0) & (size < patch_size[of_patch])
            sure = (mean >= CONFIDENT) if value else (mean <= 1 - CONFIDENT)
            flip = small & ~sure
            change = flip[lab] & ~(locked & cut)       # earlier harvest is never undone
            cut[change] = not value
    return cut & inside


def majority(cut: np.ndarray, inside: np.ndarray, k: int) -> np.ndarray:
    """Each crop-mask pixel takes the class of most crop-mask pixels in its k x k window (removes salt and pepper)."""
    n = ndimage.uniform_filter(inside.astype("float32"), k)
    c = ndimage.uniform_filter((cut & inside).astype("float32"), k)
    return (c > 0.5 * n) & inside


def clean_raster(raw: np.ndarray, inside: np.ndarray, locked: np.ndarray, tv: float, mmu_px: int,
                 majority_k: int = 0, holes_first: bool = False) -> np.ndarray:
    sm = tv_smooth(raw, inside, tv)
    sm = np.where(locked, 1.0, sm)
    cut = apply_hysteresis_threshold(np.where(inside, sm, 0), MAYBE, SURE) | locked
    if majority_k > 1:
        cut = majority(cut, inside, majority_k) | locked
    return drop_small(cut, inside, raw, locked, mmu_px, holes_first) | locked


# ------------------------------------------------------------------ vector stage
def split_polygon(args):
    """One crop-mask polygon -> list of (geometry, class, review)."""
    i, poly, harvested, earlier, mmu_m2, snap = args
    if (harvested is None or harvested.is_empty) and earlier is None:
        return [(i, poly, 81, "")]
    h = shapely.intersection(harvested, poly) if harvested is not None else shapely.Polygon()
    if snap > 0 and not h.is_empty:                    # extend harvested to the crop-mask edge where it stops just short
        band = shapely.difference(poly, shapely.buffer(poly, -snap))
        h = shapely.union(h, shapely.intersection(band, shapely.buffer(h, snap)))
    if earlier is not None:                            # harvested in an earlier report stays harvested
        h = shapely.union(h, earlier)
    r = shapely.difference(poly, h)
    pieces = [(g, 82) for g in shapely.get_parts(h) if g.area > 0] + [(g, 81) for g in shapely.get_parts(r) if g.area > 0]
    for _ in range(3):                                 # small pieces switch class inside the same crop-mask polygon
        small = [k for k, (g, c) in enumerate(pieces) if g.area < mmu_m2 and not (c == 82 and earlier is not None
                                                                                   and shapely.intersection(g, earlier).area > 0.5 * g.area)]
        if not small or len(pieces) == 1:
            break
        for k in small:
            pieces[k] = (pieces[k][0], 163 - pieces[k][1])        # 81 <-> 82
        merged = {c: shapely.union_all([g for g, cc in pieces if cc == c]) for c in (81, 82)}
        pieces = [(g, c) for c in (82, 81) for g in shapely.get_parts(merged[c]) if g.area > 0]
    classes = {c for _, c in pieces}
    if len(classes) == 1:                              # one class: the exact original geometry
        return [(i, poly, classes.pop(), "")]
    return [(i, shapely.make_valid(g), c, "small_clipped" if g.area < 2 * mmu_m2 else "") for g, c in pieces]


def main(area: str, cfg_path: str, npz: str, key: str, tag: str, tv: str = "0.1", mmu: str = "0.5", snap: str = "10") -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    t0 = time.time()
    tv_w, mmu_ac, snap_m = float(tv), float(mmu), float(snap)
    cfg = yaml.safe_load(open(cfg_path))
    interim = Path(cfg["paths"]["interim"])
    out = Path("outputs") / f"{cfg['aoi']}_{area}_clean"
    out.mkdir(parents=True, exist_ok=True)
    z = np.load(interim / f"{cfg['aoi']}_eda_pixels_{area}.npz")
    rows, cols = z["rows"], z["cols"]
    p = np.load(npz)
    with rasterio.open(interim / f"{cfg['aoi']}_cane_mask_{area}.tif") as m:
        profile, inside, transform, crs = m.profile, m.read(1) == 1, m.transform, m.crs
    crop = gpd.read_file(cfg["crop_mask"]).to_crs(crs)
    original_geom = crop.geometry.values
    crop_geom = shapely.make_valid(original_geom)
    tree = shapely.STRtree(crop_geom)
    mmu_px, mmu_m2 = max(1, int(round(mmu_ac * ACRE_M2 / PIXEL_M2))), mmu_ac * ACRE_M2
    locked = np.zeros(inside.shape, bool)
    earlier_poly = {}
    from rasterio.features import rasterize
    ids = rasterize([(geom, k + 1) for k, geom in enumerate(crop_geom)], out_shape=inside.shape, transform=transform, dtype="int32")
    no_pixels = np.flatnonzero(np.bincount(ids[inside], minlength=len(crop_geom) + 1)[1:] == 0)
    log.info("%d crop-mask polygons have no crop-mask pixel (too small or thin)", len(no_pixels))
    checks = []
    for r in cfg["report_dates"]:
        raw = np.zeros(inside.shape, "float32")
        raw[rows, cols] = p[f"{key}{r}"]
        cut = clean_raster(raw, inside, locked, tv_w, mmu_px)
        with rasterio.open(out / f"{cfg['aoi']}_{area}_clean_raster_{r}_{tag}.tif", "w",
                           **{**profile, "dtype": "uint8", "nodata": 0, "compress": "deflate"}) as dst:
            dst.write(np.where(inside, np.where(cut, 82, 81), 0).astype("uint8"), 1)
        locked = cut.copy()
        # harvested pixels -> polygons, then each crop-mask polygon is split
        harvested = shapely.union_all([shapely.geometry.shape(g) for g, v in shapes(cut.astype("uint8"), mask=cut, transform=transform)])
        parts = np.array(shapely.get_parts(harvested))
        part_tree = shapely.STRtree(parts)
        pairs = part_tree.query(crop_geom, predicate="intersects")
        by_poly = pd.Series(pairs[1]).groupby(pairs[0]).apply(list).to_dict()
        args = [(i, crop_geom[i], shapely.union_all(parts[by_poly[i]]) if i in by_poly else None,
                 earlier_poly.get(i), mmu_m2, snap_m) for i in range(len(crop_geom))]
        with ProcessPoolExecutor(workers(keep_free_cores=0)) as ex:
            res = [x for chunk in ex.map(split_polygon, args, chunksize=64) for x in chunk]
        g = gpd.GeoDataFrame([{"field_id": i, "predicted": c, "review": rv} for i, _, c, rv in res],
                             geometry=[gg for _, gg, _, _ in res], crs=crs)
        whole = g.groupby("field_id").size() == 1
        keep_original = g.field_id.map(whole)
        g.loc[keep_original, "geometry"] = original_geom[g.loc[keep_original, "field_id"].values]   # exact original geometry
        g.loc[g.field_id.isin(no_pixels), "review"] = "no_pixels"
        g["area_ac"] = (g.area / ACRE_M2).round(3)
        g["report_date"] = r
        # checks: area per crop-mask polygon
        orig_ac = pd.Series(shapely.area(original_geom) / ACRE_M2)
        got = g.groupby("field_id").area_ac.sum()
        bad = (got - orig_ac.loc[got.index]).abs() > 0.01
        checks.append({"report": r, "field_acres": float(orig_ac.sum()), "output_acres": float(g.area_ac.sum()),
                       "harvested_acres": float(g.loc[g.predicted == 82, "area_ac"].sum()),
                       "remaining_acres": float(g.loc[g.predicted == 81, "area_ac"].sum()), "polygons": len(g),
                       "polygons_harvested": int((g.predicted == 82).sum()), "field_polygons_off_by_more_than_0.01_ac": int(bad.sum()),
                       "invalid": int((~g.is_valid).sum()), "small_clipped": int((g.review == "small_clipped").sum()),
                       "tv_weight": tv_w, "mmu_acres": mmu_ac, "edge_snap_m": snap_m})
        log.info("%s: %s", r, checks[-1])
        f = out / f"{cfg['aoi']}_{area}_harvest_{r}_{tag}.gpkg"
        f.unlink(missing_ok=True)
        g.to_crs(cfg["crs"]["deliver"]).to_file(f, driver="GPKG")
        cut82 = g[g.predicted == 82]
        earlier_poly = {i: shapely.union_all(x.values) for i, x in cut82.groupby("field_id").geometry}
    pd.DataFrame(checks).to_csv(out / f"{cfg['aoi']}_{area}_postprocess_checks_{tag}.csv", index=False)
    log.info("finished in %.0f s", time.time() - t0)


if __name__ == "__main__":
    main(*sys.argv[1:9])
