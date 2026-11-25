"""Remaining cane (81) and harvested (82) polygons from the checked remaining-cane layer and L1.

Fast replacement of the post-processing model (Clip, Erase, Select by Location, sliver rule,
Merge, Dissolve, area). Dependencies: the same as preprocess_classified.py (it is imported
from the same folder): pip install numpy rasterio shapely pyogrio pyproj   (optional: psutil)

Inputs
  FINAL : remaining-cane polygons after the analyst's check (may be edited by hand: new
          polygons, overlaps, invalid shapes are all accepted).
  L1    : the area that had cane before this report (any fields; every polygon counts).
Steps (same result as the model)
  1. Remaining = FINAL clipped to L1                                   -> class 81.
  2. Harvested pieces = every L1 polygon minus FINAL, split into single parts. The pieces are
     made per L1 polygon (a harvested area that spans two L1 polygons gives two pieces).
  3. A piece "touches remaining" when it intersects the remaining cane (edge or corner contact,
     within the default XY tolerance of the model: 0.001 m, or 8.98e-9 degrees).
  4. Piece area in acres = m2 / 4046.85642 in the chosen projected CRS (default: UTM zone of the
     data centre). The value is compared as the model stores it (12 significant digits).
  5. Pieces that touch remaining AND have area <= --sliver-acres (default 0.2) become 81
     (switch off with --no-sliver-rule); all other pieces are 82.
  6. Dissolve by class and split into single parts: neighbouring pieces of one class that
     share an edge (not just a corner) or overlap become one polygon.
  Every point of the union of L1 ends up in exactly one class: where L1 polygons overlap and
  the model would give both classes, 81 is kept (the model's class-81 layer is unchanged) and
  82 is cut back.
  7. Output: EPSG:4326, fields predicted (81 / 82) and area_ac (acres, 3 decimals). Pieces
     below 0.01 m2 (numerical noise) are dropped. Check: acres 81 + acres 82 = acres of the
     union of L1 (warning above 0.01 % difference).

How it scales: the machine (cores, free memory) is read first; the L1 polygons are cut into
spatial chunks sized from the memory budget (available memory minus a 1.5-2 GB buffer) and
processed by a pool of worker processes. Each finished chunk is saved in <output stem>_work/,
so an interrupted run resumes where it stopped. Stage caches: changing only the sliver rule,
its size or the projection reruns only the classification stage; changing only the classes to
write reruns only the writing. The work folder holds a marker file; the script only ever
deletes its own files.

Run without arguments (or with --ask) for a question-and-answer mode; all choices are also
available as flags:
    python postprocess_remaining.py --final qc.shp --l1 l1.shp --out result.shp [--sliver-acres 0.2]
        [--no-sliver-rule] [--classes both|81|82] [--utm 47N | EPSG:32647] [--allow-non-utm]
        [--assume-crs] [--workers N] [--max-memory-gb G] [--memory-buffer-gb B] [--clean] [--summary-line]
"""
import argparse
import json
import math
import os
import re
import shutil
import sys
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import preprocess_classified as pc  # noqa: E402  (checks numpy, shapely, pyogrio, pyproj)

import numpy as np  # noqa: E402
import pyogrio  # noqa: E402
import pyogrio.raw  # noqa: E402
import pyproj  # noqa: E402
import shapely  # noqa: E402
from shapely import STRtree  # noqa: E402

CACHE_VERSION = 1                     # bump whenever saved chunk results change meaning
ACRE_M2 = 4046.85642
REMAIN, HARVEST = 81, 82
NOISE_M2 = 0.01                       # pieces below this area are numerical noise, always dropped
TOL_DEG = 8.983152841195215e-09       # default XY tolerance of the model for geographic data
TOL_M = 0.001                         # default XY tolerance of the model for projected data (metres)
BYTES_PER_VERTEX = 400                # planning figure for the overlay work of one chunk
WORKER_BASE_BYTES = pc.WORKER_BASE_BYTES
MARKER = "postprocess_remaining.workfolder"
LOCK = pc.LOCK
LAST_RUN = Path.home() / ".postprocess_remaining" / "last_run.json"
CLASS_CHOICES = {"both": (REMAIN, HARVEST), "81": (REMAIN,), "82": (HARVEST,)}
_OURS = re.compile(r"^(manifest\.json|chunks\.bin|clusters\.bin|classified\.bin|"
                   r"(l1|final)_valid(\.tmp)?\.gpkg(-wal|-shm|-journal)?)(\.tmp)?$")
_CHUNK = re.compile(r"^c\d{6}\.s[12](\.tmp)?$")
UserError = pc.UserError
DiskSpaceError = pc.DiskSpaceError
tick = pc.tick


# ---------------------------------------------------------------- small helpers
def tolerance_of(crs):
    """XY tolerance of the model in the units of the working CRS."""
    crs = pyproj.CRS(crs)
    if crs.is_geographic:
        return TOL_DEG
    f = crs.axis_info[0].unit_conversion_factor if crs.axis_info else 1.0
    return TOL_M / (f or 1.0)


def dbf_value(acres):
    """The area as the model stores it in a shapefile Double field (19.11 = 12 significant digits)."""
    return np.array([float("%.11e" % v) for v in np.asarray(acres, dtype=float)])


def empty():
    return np.array([], dtype=object)


def _overlay(op, a, b, tol):
    """op(a, b); on a topology error retry on repaired inputs snapped to a grid of tol / 10."""
    try:
        return op(a, b)
    except shapely.errors.GEOSException:
        return op(shapely.make_valid(a), shapely.make_valid(b), grid_size=tol / 10)


def _union(geoms, tol, snap=False):
    geoms = np.asarray(geoms, dtype=object)
    if len(geoms) == 1 and not snap:
        return geoms[0]
    try:
        return shapely.union_all(geoms, grid_size=tol / 10 if snap else None)
    except shapely.errors.GEOSException:
        return shapely.union_all(shapely.make_valid(geoms), grid_size=tol / 10)


def hilbert_order(x, y, bits=16):
    """Order of points along a Hilbert curve (spatially compact chunks)."""
    n = 1 << bits
    x0, x1, y0, y1 = np.nanmin(x), np.nanmax(x), np.nanmin(y), np.nanmax(y)
    xi = ((x - x0) / ((x1 - x0) or 1) * (n - 1)).astype(np.int64)
    yi = ((y - y0) / ((y1 - y0) or 1) * (n - 1)).astype(np.int64)
    d = np.zeros(len(x), dtype=np.int64)
    s = n >> 1
    while s > 0:
        rx = (xi & s) > 0
        ry = (yi & s) > 0
        d += s * s * ((3 * rx) ^ ry)
        flip = ~ry
        swap_x = np.where(flip & rx, n - 1 - xi, xi)
        swap_y = np.where(flip & rx, n - 1 - yi, yi)
        xi, yi = np.where(flip, swap_y, swap_x), np.where(flip, swap_x, swap_y)
        s >>= 1
    return np.argsort(d, kind="stable")


def area_m2(geoms, work_crs, area_crs):
    if len(geoms) == 0:
        return np.zeros(0)
    return pc.area_m2(geoms, work_crs, area_crs, 1)


# ---------------------------------------------------------------- inputs
def layer_crs(path):
    return pyogrio.read_info(str(path)).get("crs")


def working_crs(final_path, l1_path, assume_crs=False):
    """Working CRS = the CRS of L1 (else of FINAL when L1 has none and --assume-crs is given)."""
    lc, fc = layer_crs(l1_path), layer_crs(final_path)
    if lc and fc:
        return pyproj.CRS(lc)
    if not lc and not fc:
        raise UserError("neither the L1 layer nor the FINAL layer has a coordinate system (.prj missing)")
    if not assume_crs:
        which = "L1 layer" if not lc else "FINAL layer"
        raise UserError(f"the {which} has no coordinate system (.prj missing). Add one, or add --assume-crs to "
                        f"treat it as being in the coordinate system of the other layer")
    return pyproj.CRS(lc or fc)


def write_valid(path, geoms, crs, lid=False):
    tmp = Path(str(path).replace(".gpkg", ".tmp.gpkg"))
    for q in pc.file_set(tmp, pc.GPKG_SIDECARS).values():
        if q.exists():
            q.unlink()
    fd, fields = ([np.arange(len(geoms), dtype=np.int64)], ["lid"]) if lid else ([], [])
    pyogrio.raw.write(str(tmp), geometry=pc.to_wkb(geoms), field_data=fd, fields=fields, geometry_type="Polygon",
                      crs=pyproj.CRS(crs).to_wkt(), driver="GPKG", layer="v")
    os.replace(tmp, path)


def read_valid(path, bbox=None, lid=False):
    kw = {"bbox": tuple(float(v) for v in bbox)} if bbox is not None else {}
    _, _, wkb, fd = pyogrio.raw.read(str(path), columns=["lid"] if lid else [], **kw)
    g = shapely.from_wkb(wkb) if len(wkb) else empty()
    if lid:
        return g, (np.asarray(fd[0], dtype=np.int64) if len(wkb) else np.zeros(0, dtype=np.int64))
    return g


def plan_chunks(l1, fg, workers, budget, max_per_chunk=None):
    """Split the L1 polygons into spatially compact chunks whose overlay work fits one worker.
    Returns (list of lid arrays, list of bboxes, per-worker byte estimate of the biggest chunk)."""
    n = len(l1)
    if n == 0:
        return [], [], 0
    b = shapely.bounds(l1)
    w = shapely.get_num_coordinates(l1).astype(np.int64)
    if len(fg):
        ftree = STRtree(fg)
        li, fi = ftree.query(shapely.box(b[:, 0], b[:, 1], b[:, 2], b[:, 3]))
        np.add.at(w, li, shapely.get_num_coordinates(fg)[fi])
    order = hilbert_order((b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2)
    per_worker = max(budget / max(workers, 1) - WORKER_BASE_BYTES, 32 * 1024 ** 2)
    cap = max(1, int(per_worker / BYTES_PER_VERTEX))
    cap = min(cap, max(2000, int(w.sum() / (max(workers, 1) * 8))))   # several chunks per worker
    chunks, cur, s = [], [], 0
    for i in order:
        if cur and (s + w[i] > cap or (max_per_chunk and len(cur) >= max_per_chunk)):
            chunks.append(np.array(cur, dtype=np.int64))
            cur, s = [], 0
        cur.append(int(i))
        s += int(w[i])
    if cur:
        chunks.append(np.array(cur, dtype=np.int64))
    boxes = [(float(b[c, 0].min()), float(b[c, 1].min()), float(b[c, 2].max()), float(b[c, 3].max())) for c in chunks]
    biggest = max(int(w[c].sum()) for c in chunks)
    return chunks, boxes, WORKER_BASE_BYTES + biggest * BYTES_PER_VERTEX


# ---------------------------------------------------------------- stage 1: overlay (runs in workers)
def chunk_overlay(task):
    try:
        return _chunk_overlay(task)
    except MemoryError:
        return "memory"


def _chunk_overlay(task):
    """Per L1 polygon: remaining part (L1 intersected with FINAL) and harvested pieces (L1 minus FINAL,
    single parts) with the flags 'touches remaining' and 'near another L1 polygon' (seam)."""
    tol = task["tol"]
    x0, y0, x1, y1 = task["bbox"]
    m = 4 * tol
    lg, llid = read_valid(task["l1"], (x0 - m, y0 - m, x1 + m, y1 + m), lid=True)
    fg = read_valid(task["final"], (x0 - m, y0 - m, x1 + m, y1 + m))
    pos = {int(v): k for k, v in enumerate(llid)}
    shapely.prepare(fg)
    ftree, ltree = STRtree(fg), STRtree(lg)
    r_out, rl_out, rs_out, h_out, hl_out, hs_out, hsel_out, overlaps = [], [], [], [], [], [], [], []
    for lid in task["lids"]:
        k = pos[int(lid)]
        L = lg[k]
        fi = ftree.query(L, predicate="intersects")
        if len(fi) == 0:
            rp, dp = empty(), np.array([L], dtype=object)
        else:
            uf = _union(fg[fi], tol)
            rp = pc.polys_only(np.array([_overlay(shapely.intersection, L, uf, tol)], dtype=object))
            dp = pc.polys_only(np.array([_overlay(shapely.difference, L, uf, tol)], dtype=object))
        sel = np.zeros(len(dp), dtype=bool)
        if len(dp) and len(rp):
            rm = rp[0] if len(rp) == 1 else shapely.multipolygons(rp)
            shapely.prepare(rm)
            sel = shapely.dwithin(dp, rm, tol)
        seam_d, seam_r = np.zeros(len(dp), dtype=bool), np.zeros(len(rp), dtype=bool)
        nb = ltree.query(L, predicate="dwithin", distance=tol)
        for j in nb[nb != k]:
            lj = lg[j]
            near = shapely.dwithin(dp, lj, tol) if len(dp) else np.zeros(0, dtype=bool)
            seam_d |= near
            if len(rp):
                seam_r |= shapely.dwithin(rp, lj, tol)
            if shapely.relate_pattern(L, lj, "T********"):
                overlaps.append((int(lid), int(llid[j])))
            need = near & ~sel
            if need.any():   # remaining cane of the neighbour polygon next to these pieces
                bb = shapely.bounds(dp[need])
                box = shapely.box(bb[:, 0].min() - 2 * tol, bb[:, 1].min() - 2 * tol,
                                  bb[:, 2].max() + 2 * tol, bb[:, 3].max() + 2 * tol)
                fj = ftree.query(box, predicate="intersects")
                if len(fj):
                    lc = _overlay(shapely.intersection, lj, box, tol)
                    rj = pc.polys_only(np.array([_overlay(shapely.intersection, lc, _union(fg[fj], tol), tol)],
                                                dtype=object))
                    if len(rj):
                        rjm = rj[0] if len(rj) == 1 else shapely.multipolygons(rj)
                        sel[need] |= shapely.dwithin(dp[need], rjm, tol)
        r_out.append(rp)
        rl_out.append(np.full(len(rp), lid, dtype=np.int64))
        rs_out.append(seam_r)
        h_out.append(dp)
        hl_out.append(np.full(len(dp), lid, dtype=np.int64))
        hs_out.append(seam_d)
        hsel_out.append(sel)
    cat = lambda xs, dt: np.concatenate(xs) if xs else np.zeros(0, dtype=dt)  # noqa: E731
    r, h = cat(r_out, object), cat(h_out, object)
    res = {"r_wkb": pc.to_wkb(r), "r_lid": cat(rl_out, np.int64), "r_seam": cat(rs_out, bool),
           "h_wkb": pc.to_wkb(h), "h_lid": cat(hl_out, np.int64), "h_seam": cat(hs_out, bool),
           "h_sel": cat(hsel_out, bool), "overlaps": overlaps, "lids": np.asarray(task["lids"], dtype=np.int64)}
    pc._save(res, task["path"])
    return "ok"


# ---------------------------------------------------------------- stage 2: classes (runs in workers)
def chunk_classes(task):
    try:
        return _chunk_classes(task)
    except MemoryError:
        return "memory"


def _chunk_classes(task):
    """Sliver rule, per-L1-polygon union of class 81, areas of the pieces that need no dissolve."""
    d = pc._load(task["s1"])
    tol, wcrs, acrs = task["tol"], pyproj.CRS(task["work_crs"]), pyproj.CRS(task["area_crs"])
    h = shapely.from_wkb(d["h_wkb"]) if len(d["h_wkb"]) else empty()
    r = shapely.from_wkb(d["r_wkb"]) if len(d["r_wkb"]) else empty()
    hl, hs, sel = d["h_lid"], d["h_seam"], d["h_sel"]
    rl, rs = d["r_lid"], d["r_seam"]
    # pieces below 0.01 m2 are numerical noise of the overlay: dropped here, so they never change
    # class and never bridge two polygons that only touch at a corner
    hm2, rm2 = area_m2(h, wcrs, acrs), area_m2(r, wcrs, acrs)
    hk, rk = hm2 >= NOISE_M2, rm2 >= NOISE_M2
    st = {"pieces": int(hk.sum()), "touching": 0, "absorbed": 0, "absorbed_acres": 0.0, "touching_acres": 0.0,
          "noise_overlay": int((~hk).sum() + (~rk).sum()), "noise_overlay_m2": float(hm2[~hk].sum() + rm2[~rk].sum())}
    h, hl, hs, sel, hm2 = h[hk], hl[hk], hs[hk], sel[hk], hm2[hk]
    r, rl, rs = r[rk], rl[rk], rs[rk]
    absorbed = np.zeros(len(h), dtype=bool)
    st["touching"] = int(sel.sum())
    if sel.any():
        ac = hm2[sel] / ACRE_M2
        st["touching_acres"] = float(ac.sum())
        if task["rule"]:
            small = dbf_value(ac) <= task["sliver_acres"]
            absorbed[np.flatnonzero(sel)[small]] = True
            st["absorbed"], st["absorbed_acres"] = int(small.sum()), float(ac[small].sum())
    g81, s81, l81 = [], [], []
    ab_lids = np.unique(hl[absorbed])
    keep_r = ~np.isin(rl, ab_lids)
    g81.append(r[keep_r])
    s81.append(rs[keep_r])
    l81.append(rl[keep_r])
    for lid in ab_lids:
        mr, mh = rl == lid, absorbed & (hl == lid)
        parts = pc.polys_only(np.array([_union(np.concatenate([r[mr], h[mh]]), tol)], dtype=object))
        g81.append(parts)
        s81.append(np.full(len(parts), bool(rs[mr].any() or hs[mh].any())))
        l81.append(np.full(len(parts), lid, dtype=np.int64))
    g81, s81, l81 = np.concatenate(g81), np.concatenate(s81), np.concatenate(l81)
    g82, s82, l82 = h[~absorbed], hs[~absorbed], hl[~absorbed]
    geoms = np.concatenate([g81, g82])
    cls = np.concatenate([np.full(len(g81), REMAIN, dtype=np.int32), np.full(len(g82), HARVEST, dtype=np.int32)])
    seam = np.concatenate([s81, s82])
    lids = np.concatenate([l81, l82])
    fin = geoms[~seam]
    lg, llid = read_valid(task["l1"], task["bbox_pad"], lid=True)
    l1m2 = float(area_m2(lg[np.isin(llid, d["lids"])], wcrs, acrs).sum())
    res = {"fin_wkb": pc.to_wkb(fin), "fin_cls": cls[~seam], "fin_m2": area_m2(fin, wcrs, acrs),
           "seam_wkb": pc.to_wkb(geoms[seam]), "seam_cls": cls[seam], "seam_lid": lids[seam],
           "l1_m2": l1m2, "stats": st}
    pc._save(res, task["path"])
    return "ok"


# ---------------------------------------------------------------- dissolve of seam pieces (main process)
def _edge_pairs(geoms, lids, tol, workers):
    """Pairs (a, b) of pieces of different L1 polygons that share an edge or overlap. Pieces that are
    only within the tolerance of each other count when their boundaries run together for more than
    10 x tolerance (not a corner contact). Returns (a, b, tolerance_only flags)."""
    tree = STRtree(geoms)
    a, b = tree.query(geoms, predicate="dwithin", distance=tol)
    m = (a < b) & (lids[a] != lids[b])
    a, b = a[m], b[m]
    if len(a) == 0:
        return a, b, np.zeros(0, dtype=bool)

    def rel(ix):
        ga, gb = geoms[a[ix]], geoms[b[ix]]
        return shapely.relate_pattern(ga, gb, "T********") | shapely.relate_pattern(ga, gb, "****1****")

    parts = np.array_split(np.arange(len(a)), max(1, min(len(a) // 2000, workers * 4)))
    with ThreadPoolExecutor(max(1, workers)) as ex:
        exact = np.concatenate(list(ex.map(rel, parts)))
    near = np.flatnonzero(~exact)
    loose = np.zeros(len(a), dtype=bool)
    if len(near):
        ba, bb = shapely.bounds(geoms[a[near]]), shapely.bounds(geoms[b[near]])
        box = shapely.box(np.maximum(ba[:, 0], bb[:, 0]) - 2 * tol, np.maximum(ba[:, 1], bb[:, 1]) - 2 * tol,
                          np.minimum(ba[:, 2], bb[:, 2]) + 2 * tol, np.minimum(ba[:, 3], bb[:, 3]) + 2 * tol)
        la = shapely.intersection(shapely.boundary(geoms[a[near]]), box)
        lb = shapely.intersection(shapely.boundary(geoms[b[near]]), box)
        shared = shapely.length(shapely.intersection(la, shapely.buffer(lb, tol, quad_segs=2)))
        loose[near] = shared > 10 * tol
    keep = exact | loose
    return a[keep], b[keep], loose[keep]


def dissolve(geoms, lids, tol, workers, label, budget=None):
    """Union pieces that share an edge or overlap (connected groups only, never one union over
    everything), split to single parts."""
    n = len(geoms)
    if n == 0:
        return empty()
    a, b, loose = _edge_pairs(geoms, lids, tol, workers)
    parent = np.arange(n)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for x, y in zip(a.tolist(), b.tolist()):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx
    roots = np.fromiter((find(i) for i in range(n)), dtype=np.int64, count=n)
    snap_roots = set(roots[a[loose]].tolist()) if loose.any() else set()
    order = np.argsort(roots, kind="stable")
    groups = np.split(order, np.flatnonzero(np.diff(roots[order])) + 1)
    single = np.array([g[0] for g in groups if len(g) == 1], dtype=np.int64)
    multi = sorted((g for g in groups if len(g) > 1), key=len, reverse=True)
    out = [geoms[single]]
    total, done, last = len(multi), [0], [time.time()]

    def report():
        if time.time() - last[0] > 10:
            print(f"  dissolve {label}: {done[0]:,} of {total:,} groups", flush=True)
            last[0] = time.time()

    def union_groups(gs):
        res = []
        for g in gs:
            res.append(_union(geoms[g], tol, roots[g[0]] in snap_roots))
            done[0] += 1
        return res

    res = []
    for g in [g for g in multi if len(g) > 256]:   # a very large polygon: parallel tree reduction
        nv = int(shapely.get_num_coordinates(geoms[g]).sum())
        if budget and nv * 16 * 8 > budget:
            raise UserError(f"one output polygon is too large for the memory budget ({nv:,} vertices); "
                            f"raise --max-memory-gb")
        snap = roots[g[0]] in snap_roots
        cur = geoms[g][pc._spatial_order(geoms[g])]
        with ThreadPoolExecutor(max(1, workers)) as ex:
            while len(cur) > 1:
                cur = np.array(list(ex.map(lambda c: _union(c, tol, snap), [cur[i:i + 16] for i in range(0, len(cur), 16)])),
                               dtype=object)
                report()
        res.append(cur[0])
        done[0] += 1
    small = [g for g in multi if len(g) <= 256]
    if small:
        k = max(1, workers * 8)
        with ThreadPoolExecutor(max(1, workers)) as ex:
            futs = [ex.submit(union_groups, small[i::k]) for i in range(k)]
            while True:
                _, nd = wait(futs, timeout=2)
                report()
                if not nd:
                    break
            for f in futs:
                res.extend(f.result())
    if res:
        out.append(pc.polys_only(np.array(res, dtype=object)))
    return np.concatenate(out)


def trim_harvested(g81, g82, l82, tol):
    """82 minus 81 where they overlap (only possible where L1 polygons overlap). Returns (pieces, lids, n cut)."""
    if len(g81) == 0 or len(g82) == 0:
        return g82, l82, 0
    tree = STRtree(g81)
    i82, i81 = tree.query(g82, predicate="intersects")
    if len(i82):
        ov = shapely.relate_pattern(g82[i82], g81[i81], "T********")
        i82, i81 = i82[ov], i81[ov]
    if len(i82) == 0:
        return g82, l82, 0
    hit = np.unique(i82)
    keep = np.ones(len(g82), dtype=bool)
    keep[hit] = False
    gs, ls = [g82[keep]], [l82[keep]]
    for k in hit:
        parts = pc.polys_only(np.array([_overlay(shapely.difference, g82[k], _union(g81[i81[i82 == k]], tol), tol)],
                                       dtype=object))
        gs.append(parts)
        ls.append(np.full(len(parts), l82[k], dtype=np.int64))
    return np.concatenate(gs), np.concatenate(ls), len(hit)


# ---------------------------------------------------------------- work folder
def open_work(work):
    work = Path(work)
    if work.exists():
        if not (work / MARKER).exists():
            raise UserError(f"the folder {work} already exists and was not made by this script. "
                            f"Rename or move it, or choose another output name.")
        return
    work.mkdir(parents=True)
    (work / MARKER).write_text("work folder of postprocess_remaining.py; safe to delete when no run is active\n")


def _clean_dir(d):
    d = Path(d)
    if d.is_dir():
        for f in list(d.iterdir()):
            if f.is_file() and _CHUNK.match(f.name):
                f.unlink()
        try:
            d.rmdir()
        except OSError:
            pass


def clean_work(work, stage=1, remove_folder=False):
    """Delete only this script's own files: stage 1 = everything, 2 = classification and later."""
    work = Path(work)
    if not (work / MARKER).exists():
        return
    if stage <= 2:
        _clean_dir(work / "s2")
        if (work / "classified.bin").exists():
            (work / "classified.bin").unlink()
    if stage <= 1:
        _clean_dir(work / "s1")
        for f in list(work.iterdir()):
            if f.is_file() and _OURS.match(f.name):
                f.unlink()
    if remove_folder:
        try:
            left = [f.name for f in work.iterdir()]
        except OSError:
            left = ["?"]
        if left == [MARKER]:
            (work / MARKER).unlink()
            try:
                work.rmdir()
            except OSError:
                pass


# ---------------------------------------------------------------- pool
def run_pool(fn, tasks, workers, buffer, label, on_progress=None):
    """Run the tasks whose result file is missing; resume-safe; a dead worker restarts the pool."""
    todo = [t for t in tasks if not Path(t["path"]).exists()]
    ndone = len(tasks) - len(todo)
    if ndone:
        print(f"resuming: {ndone} of {len(tasks)} {label} chunks already done", flush=True)
    stop_after = int(os.environ.get("POSTPROCESS_TEST_STOP_AFTER", "0") or 0)
    queue, fails, finished, last = deque(todo), {}, 0, time.time()
    workers = max(1, min(workers, len(todo) or 1))
    while queue:
        ex, live, held, retry = pc._make_pool(workers), {}, 0.0, False
        try:
            while (queue and not retry) or live:
                while queue and not retry and len(live) < workers:
                    if live and pc.available_memory() < buffer:
                        if time.time() - held > 30:
                            print("  waiting: free memory is below the buffer", flush=True)
                            held = time.time()
                        break
                    t = queue.popleft()
                    live[ex.submit(fn, t)] = t
                dn, _ = wait(list(live), timeout=2, return_when=FIRST_COMPLETED)
                for f in dn:
                    t = live.pop(f)
                    if f.result() == "memory":
                        fails[t["path"]] = fails.get(t["path"], 0) + 1
                        if fails[t["path"]] >= 2:
                            raise UserError(f"a {label} chunk does not fit in memory even alone; free memory or "
                                            f"raise --max-memory-gb")
                        queue.append(t)
                        retry = True
                        continue
                    finished += 1
                    if stop_after and label == "overlay" and finished >= stop_after:
                        raise KeyboardInterrupt("test stop")
                if time.time() - last > 10:
                    print(f"  {label}: {ndone + finished} of {len(tasks)} chunks", flush=True)
                    if on_progress:
                        on_progress(f"{label}: {ndone + finished} of {len(tasks)} chunks")
                    last = time.time()
        except BrokenProcessPool:
            lost = [t for t in live.values() if not Path(t["path"]).exists()]
            finished += len(live) - len(lost)
            for t in lost:
                fails[t["path"]] = fails.get(t["path"], 0) + 1
                if fails[t["path"]] >= 3:
                    raise UserError(f"a worker died three times on the same {label} chunk (out of memory?); "
                                    f"free memory or raise --max-memory-gb")
            queue = deque(lost + list(queue))
            retry = True
            print(f"a worker died (probably out of memory): {len(lost)} chunk(s) are retried", flush=True)
        finally:
            pc._kill_pool(ex)
        if retry and workers > 1:
            workers = max(1, workers // 2)
            print(f"  continuing with {workers} worker(s)", flush=True)


# ---------------------------------------------------------------- last run / cache keys
def save_last_run(info):
    try:
        LAST_RUN.parent.mkdir(parents=True, exist_ok=True)
        pc._write_json(LAST_RUN, info)
    except Exception:
        pass


def load_last_run():
    try:
        d = json.loads(LAST_RUN.read_text())
        return d if d.get("final") and d.get("l1") and d.get("out") else None
    except Exception:
        return None


def stage_keys(final_path, l1_path, out, area_crs, a):
    """Cache keys: overlay = inputs only; classes = + projection and sliver rule; write = + classes, output."""
    k1 = {"version": CACHE_VERSION, "final": pc._file_sig(final_path, "l1"), "l1": pc._file_sig(l1_path, "l1"),
          "assume_crs": bool(a["assume_crs"])}
    k2 = {"overlay": pc._hash(k1), "epsg": area_crs.to_epsg() or area_crs.to_wkt(), "rule": bool(a["rule"]),
          "sliver_acres": a["sliver_acres"] if a["rule"] else None}
    k3 = {"classes": a["classes"], "out": str(Path(out).resolve()), "classes_key": pc._hash(k2)}
    return k1, k2, k3


def describe_changes(old, new):
    names = {"final": "FINAL file", "l1": "L1 file", "version": "script cache version", "assume_crs": "assume CRS",
             "epsg": "projection", "rule": "sliver rule", "sliver_acres": "sliver size", "overlay": "overlay stage"}
    return ", ".join(f"{names.get(k, k)}" for k in sorted(set(old) | set(new)) if old.get(k) != new.get(k))


# ---------------------------------------------------------------- run
def run(final, l1_path, out, sliver_acres=0.2, rule=True, utm=None, classes="both", workers=None,
        max_memory_gb=None, memory_buffer_gb=None, clean=False, allow_non_utm=False, assume_crs=False,
        max_chunk_polygons=None, summary_line=False):
    final, l1_path, out = Path(final).resolve(), Path(l1_path).resolve(), Path(out).resolve()
    pc._T0[0] = t_start = time.time()
    if out.suffix.lower() not in (".shp", ".gpkg"):
        raise UserError("the output must end with .shp or .gpkg")
    if not out.parent.is_dir():
        raise UserError(f"output folder does not exist: {out.parent}")
    if not pc.folder_writable(out.parent):
        raise UserError(f"cannot write in the output folder {out.parent}")
    if classes not in CLASS_CHOICES:
        raise UserError("classes must be both, 81 or 82")
    for p, what in ((final, "FINAL"), (l1_path, "L1")):
        try:
            pyogrio.read_info(str(p))
        except Exception as e:
            raise UserError(f"cannot read the {what} layer {p}: {e}")
    wcrs = working_crs(final, l1_path, assume_crs)
    _, lb = pyogrio.read_bounds(str(l1_path))
    lb = np.asarray(lb).T
    lb = lb[np.isfinite(lb).all(axis=1)]
    if len(lb) == 0:
        raise UserError("the L1 layer has no polygons")
    bounds = (lb[:, 0].min(), lb[:, 1].min(), lb[:, 2].max(), lb[:, 3].max())
    lcrs = layer_crs(l1_path) or wcrs
    area_crs = pc.resolve_area_crs(utm, lcrs, bounds, allow_non_utm)
    bad = pc.locked_outputs(out)
    if bad:
        raise UserError(f"the output {out.name} is open in another program ({', '.join(bad)}). "
                        f"Close it in ArcMap/QGIS and run again, or choose another output name.")
    ans = {"rule": bool(rule), "sliver_acres": float(sliver_acres), "classes": classes, "assume_crs": bool(assume_crs)}
    work = out.parent / (out.stem + "_work")
    last = {"final": str(final), "l1": str(l1_path), "out": str(out),
            "utm": f"EPSG:{area_crs.to_epsg()}" if area_crs.to_epsg() else area_crs.to_wkt(),
            "work": str(work), "answers": ans, "started": time.strftime("%Y-%m-%d %H:%M:%S"),
            "status": "running", "progress": "starting", "pid": os.getpid(), "allow_non_utm": bool(allow_non_utm)}
    save_last_run(last)

    def progress(msg, status="running"):
        last.update(progress=msg, status=status, updated=time.strftime("%Y-%m-%d %H:%M:%S"))
        save_last_run(last)

    open_work(work)
    lock = pc.acquire_lock(work)
    try:
        result = _run_locked(final, l1_path, out, work, wcrs, area_crs, ans, workers, max_memory_gb,
                             memory_buffer_gb, clean, progress, t_start, max_chunk_polygons, summary_line)
    except KeyboardInterrupt:
        pc.release_lock(lock)
        progress("stopped by the user", "interrupted")
        raise
    except BaseException as e:
        pc.release_lock(lock)
        progress(f"{type(e).__name__}: {e}"[:300], "failed")
        raise
    pc.release_lock(lock)
    if clean and result.get("written") == str(out):
        clean_work(work, 1, remove_folder=True)
    progress("finished", "finished")
    return result


def _plan_machine(workers, max_memory_gb, memory_buffer_gb):
    cores, (avail, total) = pc.n_cpus(), pc.memory_status()
    buf = pc.memory_buffer_bytes(total, memory_buffer_gb)
    budget = int(max_memory_gb * 1024 ** 3) if max_memory_gb else max(avail - buf, 256 * 1024 ** 2)
    w = workers or cores
    w = max(1, min(w, 61 if os.name == "nt" else w, int(budget // (WORKER_BASE_BYTES + 64 * 1024 ** 2)) or 1))
    return cores, avail, buf, budget, w


def _run_locked(final, l1_path, out, work, wcrs, area_crs, ans, workers, max_memory_gb, memory_buffer_gb,
                clean, progress, t_start, max_chunk_polygons, summary_line=False):
    k1i, k2i, k3i = stage_keys(final, l1_path, out, area_crs, ans)
    k1, k2, k3 = pc._hash(k1i), pc._hash(k2i), pc._hash(k3i)
    man_path = work / "manifest.json"
    man = None
    if man_path.exists():
        try:
            man = json.loads(man_path.read_text())
        except Exception:
            man = None
    if man is None or man.get("k1") != k1:
        if man is not None or any(_OURS.match(f.name) or f.name in ("s1", "s2") for f in work.iterdir()):
            why = describe_changes(man.get("k1_items", {}), k1i) if man else "work folder unreadable"
            print(f"Inputs changed since the saved overlay ({why}): everything is recomputed.")
        clean_work(work, 1)
        man = {"k1": k1, "k1_items": k1i}
        pc._write_json(man_path, man)
    elif man.get("k2") != k2:
        if man.get("k2"):
            print(f"Classification settings changed ({describe_changes(man.get('k2_items', {}), k2i)}): "
                  f"the saved overlay is reused, classification and writing run again.")
        clean_work(work, 2)
    tol = tolerance_of(wcrs)
    cores, avail, buf, budget, w = _plan_machine(workers, max_memory_gb, memory_buffer_gb)
    stats = {}
    times, notes = {}, []

    def stage(key, msg):
        t0 = pc._T0[0]
        tick(msg)
        times[key] = times.get(key, 0.0) + pc._T0[0] - t0

    # ---- stage 1: prepare + overlay
    s1 = work / "s1"
    chunks_bin = work / "chunks.bin"
    if not man.get("overlay_done"):
        if not chunks_bin.exists():
            need = 4 * sum(q.stat().st_size for p in (final, l1_path) for q in pc._sig_paths(p, "l1") if q.exists())
            free = shutil.disk_usage(out.parent).free
            if need + 50 * 1024 ** 2 > free:
                e = DiskSpaceError(f"not enough disk space next to the output: about {need / 1024 ** 3:.2f} GB "
                                   f"needed, {free / 1024 ** 3:.2f} GB free in {out.parent}")
                e.needed, e.free = need, free
                raise e
            progress("reading inputs")
            l1 = pc.load_l1(l1_path, wcrs, workers=w)
            fg = pc.load_l1(final, wcrs, workers=w)
            stage("read", f"inputs read, reprojected to the working CRS and made valid: {len(l1):,} L1 polygons, "
                          f"{len(fg):,} FINAL polygons")
            chunks, boxes, per_worker = plan_chunks(l1, fg, w, budget, max_chunk_polygons)
            write_valid(work / "l1_valid.gpkg", l1, wcrs, lid=True)
            write_valid(work / "final_valid.gpkg", fg, wcrs)
            pc._save({"chunks": chunks, "boxes": boxes, "per_worker": per_worker, "n_l1": len(l1)}, chunks_bin)
            del l1, fg
        cinfo = pc._load(chunks_bin)
        chunks, boxes = cinfo["chunks"], cinfo["boxes"]
        w1 = max(1, min(w, int(budget // max(cinfo["per_worker"], 1)) or 1, len(chunks) or 1))
        print(f"machine: {cores} cores, {avail / 1024 ** 3:.1f} GB available, buffer {buf / 1024 ** 3:.1f} GB, "
              f"budget {budget / 1024 ** 3:.1f} GB -> {w1} workers, {len(chunks)} chunks of L1 polygons "
              f"(biggest chunk about {cinfo['per_worker'] / 1024 ** 3:.2f} GB)")
        s1.mkdir(exist_ok=True)
        tasks = [{"l1": str(work / "l1_valid.gpkg"), "final": str(work / "final_valid.gpkg"), "lids": c,
                  "bbox": bx, "tol": tol, "path": str(s1 / f"c{i:06d}.s1")} for i, (c, bx) in enumerate(zip(chunks, boxes))]
        progress("overlay")
        run_pool(chunk_overlay, tasks, w1, buf, "overlay", progress)
        stage("overlay", "stage 1 overlay (clip, erase, touches remaining)")
        pairs = []
        for t in tasks:
            pairs += pc._load(t["path"])["overlaps"]
        pc._save(l1_clusters(work / "l1_valid.gpkg", pairs, tol), work / "clusters.bin")
        man["overlay_done"] = True
        pc._write_json(man_path, man)
    else:
        print("reusing the saved overlay (stage 1)")
        notes.append("saved overlay")
    cinfo = pc._load(chunks_bin)
    chunks, boxes = cinfo["chunks"], cinfo["boxes"]

    # ---- stage 2: classes, dissolve, areas
    classified = work / "classified.bin"
    if not (man.get("k2") == k2 and classified.exists()):
        man["k2"], man["k2_items"] = k2, k2i
        man.pop("k3", None)
        pc._write_json(man_path, man)
        s2 = work / "s2"
        s2.mkdir(exist_ok=True)
        m = 4 * tol
        tasks = [{"s1": str(s1 / f"c{i:06d}.s1"), "l1": str(work / "l1_valid.gpkg"), "tol": tol,
                  "work_crs": wcrs.to_wkt(), "area_crs": area_crs.to_wkt(), "rule": ans["rule"],
                  "sliver_acres": ans["sliver_acres"], "bbox_pad": (bx[0] - m, bx[1] - m, bx[2] + m, bx[3] + m),
                  "path": str(s2 / f"c{i:06d}.s2")} for i, bx in enumerate(boxes)]
        w2 = max(1, min(w, int(budget // max(cinfo["per_worker"], 1)) or 1, len(tasks) or 1))
        progress("classes")
        run_pool(chunk_classes, tasks, w2, buf, "classes", progress)
        stage("rule", "stage 2 sliver rule and areas per chunk")
        progress("dissolve")
        stats = finish_classes(tasks, work, wcrs, area_crs, tol, w, budget)
        stage("dissolve", f"dissolve of pieces on L1 seams: {stats['seam_in']:,} pieces -> {stats['seam_out']:,}")
        man["stats"] = stats
        pc._write_json(man_path, man)
        _clean_dir(s2)
    else:
        print("reusing the saved classes (stage 2)")
        notes.append("saved classes")
        stats = man["stats"]

    # ---- stage 3: write
    if man.get("k3") == k3 and out.exists():
        print(f"output {out.name} is already up to date for these settings")
        notes.append("output")
        written, wst = out, man.get("written_stats", {})
    else:
        progress("writing output")
        wst, written, note = write_output(classified, out, wcrs, CLASS_CHOICES[ans["classes"]], w)
        stage("write", f"stage 3 write {written}")
        if note:
            print(f"NOTE: {note}")
        if written == out:
            man["k3"], man["written_stats"] = k3, wst
            pc._write_json(man_path, man)
    print_summary(stats, wst, ans, out=written, work=work, clean=clean, times=times, notes=notes,
                  total_s=time.time() - t_start, area_crs=area_crs, summary_line=summary_line)
    res = dict(stats)
    res.update(written=str(written), written_stats=wst)
    return res


def l1_clusters(l1_valid, pairs, tol):
    """Groups of overlapping L1 polygons and their unions (for the area of the union of L1)."""
    if not pairs:
        return {"members": [], "wkb": empty()}
    ids = sorted({x for p in pairs for x in p})
    parent = {x: x for x in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    groups = {}
    for x in ids:
        groups.setdefault(find(x), []).append(x)
    _, _, wkb, fd = pyogrio.raw.read(str(l1_valid), columns=["lid"], where=f"lid IN ({','.join(map(str, ids))})")
    geo = dict(zip(np.asarray(fd[0]).tolist(), shapely.from_wkb(wkb)))
    members = list(groups.values())
    unions = [_union(np.array([geo[x] for x in g], dtype=object), tol) for g in members]
    return {"members": members, "wkb": pc.to_wkb(np.array(unions, dtype=object)), "geoms_wkb":
            [pc.to_wkb(np.array([geo[x] for x in g], dtype=object)) for g in members]}


def finish_classes(tasks, work, wcrs, area_crs, tol, workers, budget):
    """Collect the chunk results, dissolve the seam pieces, drop noise and save classified.bin."""
    st = {"pieces": 0, "touching": 0, "touching_acres": 0.0, "absorbed": 0, "absorbed_acres": 0.0,
          "l1_m2_sum": 0.0, "noise_overlay": 0, "noise_overlay_m2": 0.0}
    seam_w, seam_c, seam_l = [], [], []
    nbytes = 0
    for t in tasks:
        d = pc._load(t["path"])
        for k in ("pieces", "touching", "touching_acres", "absorbed", "absorbed_acres", "noise_overlay",
                  "noise_overlay_m2"):
            st[k] += d["stats"][k]
        st["l1_m2_sum"] += d["l1_m2"]
        if len(d["seam_wkb"]):
            seam_w.append(d["seam_wkb"])
            seam_c.append(d["seam_cls"])
            seam_l.append(d["seam_lid"])
            nbytes += sum(len(x) for x in d["seam_wkb"])
    if budget and nbytes * 8 > budget:
        raise UserError(f"the pieces on L1 seams need about {nbytes * 8 / 1024 ** 3:.1f} GB to dissolve, more than "
                        f"the memory budget; raise --max-memory-gb")
    cl = pc._load(work / "clusters.bin")
    l1_m2 = st["l1_m2_sum"]
    for g_wkb, u_wkb in zip(cl.get("geoms_wkb", []), cl["wkb"]):   # union of overlapping L1 = sum - overlaps
        members = float(area_m2(shapely.from_wkb(g_wkb), wcrs, area_crs).sum())
        l1_m2 += float(area_m2(np.array([shapely.from_wkb(u_wkb)], dtype=object), wcrs, area_crs).sum()) - members
    st["l1_union_acres"] = l1_m2 / ACRE_M2
    st["l1_overlap_groups"] = len(cl["members"])
    sw = shapely.from_wkb(np.concatenate(seam_w)) if seam_w else empty()
    sc = np.concatenate(seam_c) if seam_c else np.zeros(0, dtype=np.int32)
    sl = np.concatenate(seam_l) if seam_l else np.zeros(0, dtype=np.int64)
    st["seam_in"] = int(len(sw))
    g81 = dissolve(sw[sc == REMAIN], sl[sc == REMAIN], tol, workers, "81", budget)
    g82, l82, st["trimmed_82_pieces"] = trim_harvested(g81, sw[sc == HARVEST], sl[sc == HARVEST], tol)
    g82 = dissolve(g82, l82, tol, workers, "82", budget)
    st["seam_out"] = int(len(g81) + len(g82))
    tmp = work / "classified.bin.tmp"
    acc = {REMAIN: [0, 0.0], HARVEST: [0, 0.0]}
    noise = [0]
    import pickle

    def dump(fh, wkb, cls, m2):
        ok = m2 >= NOISE_M2
        noise[0] += int((~ok).sum())
        wkb, cls, m2 = wkb[ok], cls[ok], m2[ok]
        for c in (REMAIN, HARVEST):
            acc[c][0] += int((cls == c).sum())
            acc[c][1] += float(m2[cls == c].sum()) / ACRE_M2
        if len(wkb):
            pickle.dump({"wkb": wkb, "cls": cls, "m2": m2}, fh, protocol=4)

    with open(tmp, "wb") as fh:
        for t in tasks:
            d = pc._load(t["path"])
            dump(fh, d["fin_wkb"], d["fin_cls"], d["fin_m2"])
        for g, c in ((g81, REMAIN), (g82, HARVEST)):
            if len(g):
                dump(fh, pc.to_wkb(g), np.full(len(g), c, dtype=np.int32), area_m2(g, wcrs, area_crs))
    os.replace(tmp, work / "classified.bin")
    st.update(n81=acc[REMAIN][0], acres81=acc[REMAIN][1], n82=acc[HARVEST][0], acres82=acc[HARVEST][1],
              noise=noise[0])
    return st


def write_output(classified, out, wcrs, classes, workers, batch=200_000):
    out = Path(out)
    tmp = out.with_name(out.stem + "__partial" + out.suffix)
    for x in pc.file_set(tmp, pc.out_exts(out)).values():
        if x.exists():
            x.unlink()
    driver = "GPKG" if out.suffix.lower() == ".gpkg" else "ESRI Shapefile"
    layer = out.stem if driver == "GPKG" else None
    st = {"count": 0, "first": True, REMAIN: [0, 0.0], HARVEST: [0, 0.0]}
    to4326 = pyproj.CRS(wcrs) != pyproj.CRS(4326)
    buf = []

    def flush():
        if not buf:
            return
        w = np.concatenate([b[0] for b in buf])
        c = np.concatenate([b[1] for b in buf])
        a = np.concatenate([b[2] for b in buf])
        pyogrio.raw.write(str(tmp), geometry=w, field_data=[c.astype("int32"), a], fields=["predicted", "area_ac"],
                          geometry_type="Polygon", crs="EPSG:4326", driver=driver, append=not st["first"], layer=layer)
        st["first"] = False
        buf.clear()

    n = 0
    for d in pc._iter_chunks(classified):
        m = np.isin(d["cls"], classes)
        if not m.any():
            continue
        wkb, cls, a = d["wkb"][m], d["cls"][m], np.round(d["m2"][m] / ACRE_M2, 3)
        if to4326:
            wkb = pc.to_wkb(pc.project_geoms(shapely.from_wkb(wkb), wcrs, 4326, workers))
        for c in (REMAIN, HARVEST):
            st[c][0] += int((cls == c).sum())
            st[c][1] += float(a[cls == c].sum())
        buf.append((wkb, cls, a))
        n += len(wkb)
        st["count"] += len(wkb)
        if n >= batch:
            flush()
            n = 0
    flush()
    if st["first"]:
        pyogrio.raw.write(str(tmp), geometry=empty(), field_data=[np.zeros(0, dtype="int32"), np.zeros(0)],
                          fields=["predicted", "area_ac"], geometry_type="Polygon", crs="EPSG:4326", driver=driver,
                          layer=layer)
    n_read = pyogrio.read_info(str(tmp))["features"]
    if n_read != st["count"]:
        raise RuntimeError(f"output check failed: wrote {st['count']} polygons but the file has {n_read}")
    written, note = pc.install_output(tmp, out)
    res = {"count": st["count"], "n81": st[REMAIN][0], "acres81": st[REMAIN][1], "n82": st[HARVEST][0],
           "acres82": st[HARVEST][1]}
    return res, written, note


def _rows(text, width=88):
    return [(x,) for x in _wrap(text, width)]


def _wrap(text, width=88):
    """One row per wrapped line: two spaces indent, hanging indent of two more on the next lines."""
    import textwrap
    return list(textwrap.wrap(text, width, initial_indent="  ", subsequent_indent="    "))


def print_result_block(title, sections):
    """Print a titled block. sections = [(heading, rows)]; a row is a tuple of strings.
    Rows with 2+ cells are aligned in columns (first cell left, the next two right-aligned, any further cell
    left as free text); a row with one cell is printed as it is. An empty heading prints no heading line."""
    wid = [0, 0, 0]
    for _, rows in sections:
        for r in rows:
            if len(r) > 1:
                for i in range(min(len(r), 3)):
                    wid[i] = max(wid[i], len(r[i]))
    lines = []
    for heading, rows in sections:
        if lines:
            lines.append("")
        if heading:
            lines.append(heading)
        for r in rows:
            if len(r) == 1:
                lines.append(r[0])
                continue
            cells = [r[0].ljust(wid[0])] + [c.rjust(wid[i]) for i, c in enumerate(r[1:3], 1)]
            line = "  ".join(cells) + "".join("  " + c for c in r[3:])
            lines.append(line.rstrip())
    width = max([48] + [len(x) for x in lines if len(x) <= 90])
    head = f" {title} "
    print("=" * ((width - len(head)) // 2) + head + "=" * (width - len(head) - (width - len(head)) // 2))
    print("\n".join(lines))
    print("=" * width)


def _folder_size(p):
    total = 0
    for f in Path(p).rglob("*"):
        try:
            if f.is_file():
                total += f.stat().st_size
        except OSError:
            pass
    return total


def _human_size(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:,.0f} {unit}" if unit == "B" else f"{n:,.1f} {unit}"
        n /= 1024


def _summary_lines(st, wst, ans):
    a81, a82, ul = st["acres81"], st["acres82"], st["l1_union_acres"]
    diff = a81 + a82 - ul
    pct = 100 * diff / ul if ul else 0.0
    rule = f"on, <= {ans['sliver_acres']} acres" if ans["rule"] else "off"
    print(f"SUMMARY sliver_rule={rule} harvested_pieces={st['pieces']:,} touching_remaining={st['touching']:,} "
          f"turned_to_81={st['absorbed']:,} ({st['absorbed_acres']:,.3f} acres) "
          f"noise_pieces_removed={st['noise_overlay'] + st['noise']:,} ({st['noise_overlay_m2'] / ACRE_M2:.4f} acres) "
          f"overlapping_L1_groups={st['l1_overlap_groups']:,} harvested_pieces_cut_by_81={st['trimmed_82_pieces']:,}")
    print(f"SUMMARY class 81: {st['n81']:,} polygons, {a81:,.3f} acres | class 82: {st['n82']:,} polygons, "
          f"{a82:,.3f} acres | 81 + 82 = {a81 + a82:,.3f} acres | union of L1 = {ul:,.3f} acres | "
          f"difference {diff:,.3f} acres ({pct:.4f} %)")
    if wst:
        print(f"SUMMARY written ({ans['classes']}): {wst['count']:,} polygons "
              f"(81: {wst['n81']:,}, {wst['acres81']:,.3f} acres; 82: {wst['n82']:,}, {wst['acres82']:,.3f} acres; "
              f"sums of the rounded area_ac)")


def print_summary(st, wst, ans, out=None, work=None, clean=False, times=None, notes=(), total_s=None, area_crs=None,
                  summary_line=False):
    a81, a82, ul = st["acres81"], st["acres82"], st["l1_union_acres"]
    diff = a81 + a82 - ul
    pct = 100 * diff / ul if ul else 0.0
    ok = abs(pct) <= 0.01
    n = lambda v: f"{v:,}"          # noqa: E731
    ac = lambda v: f"{v:,.3f}"      # noqa: E731
    if notes:
        parts = [x for x in notes if x != "output"]
        if parts:
            print("Reused from an earlier run: " + " and ".join(parts) + ".")
        if "output" in notes:
            print("The output file was already up to date for these settings; nothing was written again.")
    table = [("Class", "Polygons", "Acres"),
             ("  81  Remaining cane", n(st["n81"]), ac(a81)),
             ("  82  Harvested", n(st["n82"]), ac(a82)),
             ("  Total (81 + 82)", n(st["n81"] + st["n82"]), ac(a81 + a82))]
    classes = CLASS_CHOICES[ans["classes"]]
    if wst:
        w_sum = sum(wst[f"acres{c}"] for c in classes)
        exact = sum({REMAIN: a81, HARVEST: a82}[c] for c in classes)
        if abs(w_sum - exact) > 0.001:
            table.extend(_rows(f"Note: adding up the area_ac column of the file gives {ac(w_sum)} acres "
                               f"(each polygon is rounded to 3 decimals)."))
    check = [("  L1 area", "", ac(ul)),
             ("  Difference", "", ac(diff), f"({abs(pct):.4f} %)  {'OK' if ok else 'CHECK'}")]
    if not ok:
        check.extend(_rows(f"WARNING: 81 + 82 differs from the area of L1 by {pct:.4f} % (more than 0.01 %)"))
    if st["l1_overlap_groups"]:
        check.extend(_rows(f"L1 polygons overlap each other in {st['l1_overlap_groups']:,} places; the overlap is "
                           f"counted once, as remaining cane where both classes meet."))
    if st["trimmed_82_pieces"]:
        check.extend(_rows(f"{st['trimmed_82_pieces']:,} harvested pieces were cut back where they overlapped "
                           f"remaining cane."))
    noise = ("  Numerical noise removed", n(st["noise_overlay"] + st["noise"]), f"{st['noise_overlay_m2'] / ACRE_M2:.3f}")
    if ans["rule"]:
        small = [(f"Small harvested pieces (rule: {ans['sliver_acres']} acres or smaller AND touching remaining cane)",
                  [("  Harvested pieces found", n(st["pieces"])),
                   ("  Touching remaining cane", n(st["touching"])),
                   ("  Changed to remaining cane", n(st["absorbed"]), ac(st["absorbed_acres"])), noise])]
    else:
        small = [("Cleaning (sliver rule is off)", [noise])]
    crs_txt = area_crs.name if area_crs is not None else "?"
    if area_crs is not None and area_crs.to_epsg():
        crs_txt += f" (EPSG:{area_crs.to_epsg()})"
    settings = [("  Sliver rule : " + (f"on, pieces of {ans['sliver_acres']} acres or smaller" if ans["rule"] else "off"),),
                (f"  Projection  : {crs_txt}",),
                ("  Classes     : " + {"both": "both (81 and 82)", "81": "81 only", "82": "82 only"}[ans["classes"]],)]
    files = []
    if out is not None:
        cnt = f"{wst['count']:,} polygons" if wst else "polygons not counted"
        if ans["classes"] != "both":
            cnt += f", class {ans['classes']} only"
        files.append((f"  Output : {out}   ({cnt})",))
        if clean:
            files.append(("  Cache  : removed after the run (--clean)",))
        elif work is not None:
            files.append((f"  Cache  : {work} ({_human_size(_folder_size(work))})",))
            files.append(("           A rerun that only changes the sliver rule, the projection or the classes reuses it.",))
            big = " It is large; delete it when you no longer need quick reruns." if _folder_size(work) > 500 * 1024 ** 2 else ""
            files.append(("           Safe to delete." + big,))
    sections = [("", table), ("Area check (remaining + harvested must equal L1)", check)] + small
    sections += [("Settings used", settings), ("Files", files)]
    if total_s is not None:
        names = (("read", "read"), ("overlay", "overlay"), ("rule", "rule and areas"), ("dissolve", "dissolve"),
                 ("write", "write"))
        parts = " | ".join(f"{lab} {times[k]:.1f}" for k, lab in names if times and k in times)
        sections.append(("", [(f"Time     : {total_s:.1f} s" + (f"  ({parts})" if parts else ""),)]))
    print_result_block("RESULT", sections)
    if summary_line:
        _summary_lines(st, wst, ans)


# ---------------------------------------------------------------- interactive mode
def made_by_this_tool(p):
    p = Path(p)
    if (p.parent / (p.stem + "_work") / MARKER).exists():
        return True
    last = load_last_run()
    try:
        return bool(last) and Path(last["out"]).resolve() == p.resolve()
    except Exception:
        return False


_BAD_CHARS = set('<>:"/\\|?*')
_RESERVED = {"con", "prn", "aux", "nul"} | {f"com{i}" for i in range(1, 10)} | {f"lpt{i}" for i in range(1, 10)}


def _name_check(a):
    a = a.strip()
    if Path(a).suffix.lower() in (".shp", ".gpkg"):
        a = a[:-len(Path(a).suffix)]
    bad = sorted({c for c in a if c in _BAD_CHARS or ord(c) < 32})
    if not a:
        return None, "Type a name."
    if bad:
        return None, f"A file name cannot contain {' '.join(bad)} (Windows rule). Use letters, digits, - or _."
    if a[-1] in ". " or a.split(".")[0].lower() in _RESERVED:
        return None, "This name is not allowed on Windows (reserved word, or ends with a dot or space)."
    return a, None


def ask_output(default_dir, default_stem, defaults):
    """Three questions: output folder, name (no extension), format. Returns the output Path.
    A full file path pasted as the folder is split into folder, name and format."""
    prev = Path(defaults["out"]) if defaults and defaults.get("out") else None
    folder_def = str(prev.parent if prev else default_dir)
    stem_def = prev.stem if prev else default_stem
    fmt_def = "2" if (prev and prev.suffix.lower() == ".gpkg") else "1"
    while True:
        raw = pc.clean_path(pc.ask_text("   Output folder", folder_def))
        folder = Path(raw)
        if folder.suffix.lower() in (".shp", ".gpkg"):
            stem_def, fmt_def = folder.stem, "2" if folder.suffix.lower() == ".gpkg" else "1"
            folder = folder.parent
        if folder.exists() and not folder.is_dir():
            print(f"  {folder} is a file, not a folder.")
            continue
        if not folder.exists():
            if not pc.ask_yes_no(f"   The folder {folder} does not exist. Create it?", True):
                continue
            try:
                folder.mkdir(parents=True)
            except OSError as e:
                print(f"  Cannot create it: {e}")
                continue
        if not pc.folder_writable(folder):
            print(f"  Cannot write in {folder}. Choose another folder.")
            continue
        free = shutil.disk_usage(folder).free
        if free < 200 * 1024 ** 2:
            print(f"  Only {free / 1024 ** 2:.0f} MB free in {folder}. Choose a folder on a disk with more space.")
            continue
        break
    while True:
        stem = pc.ask_text("   Output name", stem_def, _name_check)
        print("   A shapefile cannot be larger than 2 GB; choose 2 for very large areas.")
        fmt = pc.ask_text("   Save as:  1) Shapefile (.shp)   2) GeoPackage (.gpkg) ", fmt_def,
                          lambda a: (a.strip(), None) if a.strip() in ("1", "2") else (None, "Type 1 or 2."))
        out = folder / (stem + (".shp" if fmt == "1" else ".gpkg"))
        if out.exists():
            if made_by_this_tool(out):
                if not pc.ask_yes_no(f"   {out.name} was made by an earlier run of this tool. Replace it?", True):
                    continue
            elif not pc.ask_yes_no(f"   {out.name} already exists. Overwrite?", False):
                continue
        return out


def _classes_check(a):
    a = a.strip().lower()
    return (a, None) if a in CLASS_CHOICES else (None, "Type both, 81 or 82.")


def interactive(defaults):
    print("Remaining cane (81) and harvested (82) from the checked layer and L1 - answer the questions, "
          "Enter keeps the [default].\n")
    exts = (".shp", ".gpkg", ".geojson")
    final = pc.ask_text("1) Checked remaining-cane layer (FINAL, .shp / .gpkg)", defaults.get("final"),
                        pc._file_check(exts, "FINAL layer"))
    l1 = pc.ask_text("2) L1 layer (.shp / .gpkg)", defaults.get("l1"), pc._file_check(exts, "L1 layer"))
    assume = False
    try:
        fc, lc = layer_crs(final), layer_crs(l1)
    except Exception as e:
        raise UserError(f"cannot read the layers: {e}")
    if not fc and not lc:
        raise SystemExit("Stopped: neither layer has a coordinate system (.prj missing).")
    if not fc or not lc:
        which, other = ("FINAL", "L1") if not fc else ("L1", "FINAL")
        print(f"   The {which} layer has no coordinate system (.prj missing).")
        if not pc.ask_yes_no(f"   Assume it uses the same coordinate system as the {other} layer?", False):
            raise SystemExit(f"Stopped: give the {which} layer a coordinate system and run again.")
        assume = True
    print("3) Output")
    out = ask_output(final.parent, final.stem + "_harvest_final", defaults)
    rule = pc.ask_yes_no("4) Turn small harvested pieces that touch remaining cane into remaining cane?",
                         defaults.get("rule", True))
    sliver = defaults.get("sliver_acres", 0.2)
    if rule:
        sliver = pc.ask_text("   Largest piece size in acres (pieces <= this size change)", sliver, pc.parse_acres)
    crs = pyproj.CRS(lc or fc)
    _, lb = pyogrio.read_bounds(str(l1))
    lb = np.asarray(lb).T
    lb = lb[np.isfinite(lb).all(axis=1)]
    if len(lb) == 0:
        raise UserError("the L1 layer has no polygons")
    bounds = (lb[:, 0].min(), lb[:, 1].min(), lb[:, 2].max(), lb[:, 3].max())
    opts = pc.utm_options(crs, bounds)
    north = pc.data_centre(crs, bounds)[1] >= 0
    if opts:
        print("5) Projection used to calculate acres (detected from the data extent):")
        for i, (c, label, note) in enumerate(opts, 1):
            print(f"   {i}) {label}  EPSG:{c.to_epsg()}  ({note})")
    else:
        print("5) The data cross the 180 degree meridian: no UTM zone can be suggested.")
    allow = False
    while True:
        utm = pc.ask_text("   Choose number, zone (e.g. 47N) or EPSG code" if opts else "   Type a zone (e.g. 1S) or EPSG code",
                          pc.default_area_crs(defaults.get("utm"), crs, bounds, opts),
                          lambda a: pc.parse_crs_msg(a, north, opts))
        is_utm, far = pc.area_crs_notes(utm, crs, bounds)
        if not is_utm:
            print(f"   {pc.NON_UTM_NOTE}")
            if not pc.ask_yes_no(f"   Use {utm.name} anyway?", False):
                continue
            allow = True
        if far:
            print(f"   {far}")
            if not pc.ask_yes_no("   Use it anyway?", False):
                continue
        break
    classes = pc.ask_text("6) Classes to write: both, 81 (remaining) or 82 (harvested)", defaults.get("classes", "both"),
                          _classes_check)
    print("\nYour answers:")
    print(f"  FINAL       : {final}\n  L1 layer    : {l1}\n  output      : {out}")
    print(f"  sliver rule : {'pieces <= ' + str(sliver) + ' acres touching remaining -> 81' if rule else 'off'}")
    print(f"  projection  : {utm.name} (EPSG:{utm.to_epsg()})")
    print(f"  classes     : {classes}")
    if not pc.ask_yes_no("Start?", True):
        print("Cancelled.")
        return None
    return dict(final=final, l1_path=l1, out=out, sliver_acres=sliver, rule=rule, utm=utm, classes=classes,
                allow_non_utm=allow, assume_crs=assume)


def chunks_progress(work):
    try:
        n = len(pc._load(Path(work) / "chunks.bin")["chunks"])
        man = json.loads((Path(work) / "manifest.json").read_text())
        if man.get("overlay_done"):
            return n, n
        return sum(1 for f in (Path(work) / "s1").iterdir() if f.name.endswith(".s1")), n
    except Exception:
        return 0, 0


def describe_last(last):
    ans = last.get("answers", {})
    d, n = chunks_progress(last.get("work", ""))
    try:
        epsg = pyproj.CRS(last["utm"]).to_epsg()
    except Exception:
        epsg = "?"
    state = last.get("status", "?")
    if state == "running" and pc._pid_alive(last.get("pid")) is False:
        state = "interrupted"
    print("A previous run was found:")
    print(f"  FINAL       : {last['final']}\n  L1 layer    : {last['l1']}\n  output      : {last['out']}")
    rule = f"pieces <= {ans.get('sliver_acres')} acres touching remaining -> 81" if ans.get("rule", True) else "off"
    print(f"  sliver rule : {rule}")
    print(f"  projection  : EPSG:{epsg}")
    print(f"  classes     : {ans.get('classes', 'both')}")
    extra = f" ({d} of {n} overlay chunks saved)" if n and state != "finished" else ""
    print(f"  status      : {state}, {last.get('progress', '')}{extra}, started {last.get('started', '?')}")


def _validate_cli(a, ap):
    for name in ("final", "l1"):
        p = Path(pc.clean_path(getattr(a, name)))
        if not p.is_file():
            ap.error(f"--{name}: file not found: {p}")
    out = Path(pc.clean_path(a.out))
    if out.suffix.lower() not in (".shp", ".gpkg"):
        ap.error("--out must end with .shp or .gpkg")
    if not out.parent.is_dir():
        ap.error(f"--out: folder does not exist: {out.parent}")
    if a.sliver_acres < 0 or not math.isfinite(a.sliver_acres):
        ap.error("--sliver-acres must be 0 or more")
    if a.workers is not None and a.workers < 1:
        ap.error("--workers must be 1 or more")
    if a.max_memory_gb is not None and a.max_memory_gb <= 0:
        ap.error("--max-memory-gb must be more than 0")
    if a.memory_buffer_gb is not None and a.memory_buffer_gb < 0:
        ap.error("--memory-buffer-gb must be 0 or more")
    try:
        l1p = pc.clean_path(a.l1)
        crs = layer_crs(l1p) or layer_crs(pc.clean_path(a.final))
        _, lb = pyogrio.read_bounds(l1p)
        lb = np.asarray(lb).T
        lb = lb[np.isfinite(lb).all(axis=1)]
    except Exception as e:
        ap.error(f"cannot read the layers ({e})")
    if crs is None:
        ap.error("neither layer has a coordinate system")
    if len(lb) == 0:
        ap.error("--l1 has no polygons")
    try:
        return pc.resolve_area_crs(a.utm, crs, (lb[:, 0].min(), lb[:, 1].min(), lb[:, 2].max(), lb[:, 3].max()),
                                   a.allow_non_utm, warn=False)
    except UserError as e:
        ap.error(f"--utm: {e}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--final", help="checked remaining-cane layer (.shp / .gpkg)")
    ap.add_argument("--l1", help="L1 layer: area that had cane before this report")
    ap.add_argument("--out", help="output .shp or .gpkg")
    ap.add_argument("--sliver-acres", type=float, default=0.2,
                    help="harvested pieces touching remaining cane with area <= this become 81 (default 0.2)")
    ap.add_argument("--no-sliver-rule", action="store_true", help="keep every harvested piece as 82")
    ap.add_argument("--classes", choices=sorted(CLASS_CHOICES), default="both", help="classes to write")
    ap.add_argument("--utm", default=None, help="projected CRS for acres: 47N, 32647 or EPSG:32647 (default: auto)")
    ap.add_argument("--allow-non-utm", action="store_true", help="accept a projected CRS that is not a UTM zone")
    ap.add_argument("--assume-crs", action="store_true",
                    help="a layer without coordinate system is taken to be in the other layer's system")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--max-memory-gb", type=float, default=None,
                    help="memory budget (default: available memory minus the buffer)")
    ap.add_argument("--memory-buffer-gb", type=float, default=None,
                    help="memory kept free (default 2, or 1.5 on machines with 8 GB RAM or less)")
    ap.add_argument("--clean", action="store_true", help="remove the work folder after success")
    ap.add_argument("--summary-line", action="store_true",
                    help="also print the machine-readable SUMMARY key=value lines after the result block")
    ap.add_argument("--ask", action="store_true", help="question-and-answer mode")
    a = ap.parse_args(argv)
    ask = a.ask or len(sys.argv if argv is None else argv) <= (1 if argv is None else 0)
    common = dict(workers=a.workers, max_memory_gb=a.max_memory_gb, memory_buffer_gb=a.memory_buffer_gb, clean=a.clean)
    if ask:
        try:
            last = load_last_run()
            ans = None
            defaults = dict(final=a.final, l1=a.l1, out=a.out)
            if last:
                describe_last(last)
                la = last.get("answers", {})
                if pc.ask_yes_no("Continue the last run with these settings?", True):
                    ans = dict(final=Path(last["final"]), l1_path=Path(last["l1"]), out=Path(last["out"]),
                               sliver_acres=la.get("sliver_acres", 0.2), rule=la.get("rule", True),
                               utm=pyproj.CRS(last["utm"]), classes=la.get("classes", "both"),
                               allow_non_utm=last.get("allow_non_utm", False), assume_crs=la.get("assume_crs", False))
                else:
                    epsg = pyproj.CRS(last["utm"]).to_epsg()
                    defaults = dict(final=last["final"], l1=last["l1"], out=last["out"], rule=la.get("rule", True),
                                    sliver_acres=la.get("sliver_acres", 0.2), classes=la.get("classes", "both"),
                                    utm=f"EPSG:{epsg}" if epsg else None)
                    print()
            if ans is None:
                ans = interactive(defaults)
            while ans:
                try:
                    run(**ans, **common)
                    break
                except DiskSpaceError as e:
                    print(f"\n{e}")
                    if not pc.ask_yes_no("Choose another output path?", True):
                        break
                    ans["out"] = ask_output(Path(ans["out"]).parent, Path(ans["out"]).stem, {})
        except KeyboardInterrupt:
            print("\nStopped.")
        except SystemExit as e:
            print(f"\n{e}" if e.code not in (None, 0) else "\nStopped.")
        except UserError as e:
            print(f"\nERROR: {e}")
        except Exception as e:  # keep the window readable, no traceback for the analyst
            print(f"\nERROR: {type(e).__name__}: {e}")
        try:
            input("\nPress Enter to close")
        except EOFError:
            pass
        return
    if not (a.final and a.l1 and a.out):
        ap.error("--final, --l1 and --out are required (or run without arguments for questions)")
    utm = _validate_cli(a, ap)
    try:
        run(Path(pc.clean_path(a.final)), Path(pc.clean_path(a.l1)), Path(pc.clean_path(a.out)), a.sliver_acres,
            not a.no_sliver_rule, utm, a.classes, allow_non_utm=a.allow_non_utm, assume_crs=a.assume_crs,
            summary_line=a.summary_line, **common)
    except UserError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)
    except KeyboardInterrupt:
        print("\nStopped.", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
