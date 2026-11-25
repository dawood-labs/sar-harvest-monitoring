"""Brute-force and layout tests for src/preprocess_classified.py (speed / memory changes).

Reference = slow cell-by-cell majority filter + polygonise the whole array + union(cane) intersected
with union(L1) + split into parts. The tool must give the same polygons (topologically equal, same
count, same acres) at several tile sizes, and vertex-identical output in every execution layout
(threads in this program, worker processes, processes x threads).
Run: python -m pytest -q tests/test_preprocess_layouts.py
"""
import contextlib
import io
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pyogrio.raw
import rasterio
import shapely
from rasterio import features
from rasterio.transform import from_origin

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src" if (_ROOT / "src").is_dir() else _ROOT))
import preprocess_classified as pc  # noqa: E402

PX = 0.0001
ORIGIN = (4.0, 1.0)


# ---------------------------------------------------------------- slow reference
def literal_filter(a, cane, nb, thr, nodata):
    """Cell-by-cell implementation of the documented majority rules (Esri)."""
    rows, cols = a.shape
    ring_off = pc._RING if nb == 8 else [pc._RING[j] for j in (0, 2, 4, 6)]
    out = np.zeros(a.shape, bool)
    for r in range(rows):
        for c in range(cols):
            cen = a[r, c]
            if nodata is not None and cen == nodata:
                continue
            ring = []
            for dr, dc in ring_off:
                rr, cc = r + dr, c + dc
                ok = 0 <= rr < rows and 0 <= cc < cols and not (nodata is not None and a[rr, cc] == nodata)
                ring.append(a[rr, cc] if ok else None)
            edge_r, edge_c = r in (0, rows - 1), c in (0, cols - 1)
            n = len(ring)
            k = (2 if nb == 4 else 3) if (edge_r or edge_c) else ((nb // 2 + 1) if thr == "majority" else nb // 2)
            qual = set()
            for v in set(x for x in ring if x is not None):
                for s in range(n):
                    idx = [(s + i) % n for i in range(k)]
                    if all(ring[i] == v for i in idx):
                        if (edge_r or edge_c) and nb == 8:
                            along = ({2, 6} if edge_r else set()) | ({0, 4} if edge_c else set())
                            if not (set(idx) & along):
                                continue
                        qual.add(v)
                        break
            out[r, c] = (list(qual)[0] if len(qual) == 1 else cen) == cane
    return out


def reference(arr, l1_geoms, do_filter, nodata=None, cane=1, nb=4):
    tf = from_origin(ORIGIN[0], ORIGIN[1], PX, PX)
    if do_filter:
        m = literal_filter(arr, cane, nb, "majority", nodata)
    else:
        m = arr == cane
        if nodata is not None:
            m &= arr != nodata
    if not m.any():
        return np.array([], dtype=object)
    shp = features.shapes(m.view(np.uint8), mask=m, transform=tf, connectivity=4)
    polys = [shapely.geometry.shape(g) for g, _ in shp]
    cu = shapely.union_all(np.array(polys, dtype=object))
    lu = shapely.union_all(shapely.make_valid(np.array(l1_geoms, dtype=object)))
    parts = pc.polys_only(np.array([shapely.intersection(cu, lu)], dtype=object))
    utm = pc.zone_crs(31, True)
    m2 = pc.area_m2(parts, "EPSG:4326", utm)
    return parts[m2 >= pc.NOISE_M2]


def write_case(d, arr, l1_geoms, nodata=None):
    tf = from_origin(ORIGIN[0], ORIGIN[1], PX, PX)
    with rasterio.open(d / "r.tif", "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1,
                       dtype="uint8", crs="EPSG:4326", transform=tf, nodata=nodata) as o:
        o.write(arr, 1)
    pyogrio.raw.write(str(d / "l1.gpkg"), geometry=shapely.to_wkb(np.array(l1_geoms, dtype=object)), field_data=[],
                      fields=[], geometry_type="Unknown", crs="EPSG:4326", driver="GPKG", layer="l1")
    return d / "r.tif", d / "l1.gpkg"


def box_px(r0, c0, r1, c1):
    """Box in world coordinates from (fractional) cell rows / columns."""
    return shapely.box(ORIGIN[0] + c0 * PX, ORIGIN[1] - r1 * PX, ORIGIN[0] + c1 * PX, ORIGIN[1] - r0 * PX)


def read_out(path):
    meta, _, g, f = pyogrio.raw.read(str(path))
    return shapely.from_wkb(g), np.asarray(f[list(meta["fields"]).index("area_ac")], float)


def topo_same(a, b):
    """Same number of polygons and each one equal to exactly one of the other set (topologically, or up to
    floating-point noise where an L1 edge crosses another edge: area of the difference below 1e-16 deg2)."""
    if len(a) != len(b):
        return False, f"count {len(a)} != {len(b)}"
    if len(a) == 0:
        return True, ""
    tree = shapely.STRtree(b)
    used = np.zeros(len(b), bool)
    for g in a:
        cand = [j for j in tree.query(g, predicate="intersects")
                if not used[j] and (shapely.equals(g, b[j]) or shapely.symmetric_difference(g, b[j]).area < 1e-16)]
        if not cand:
            return False, f"no match for a polygon of area {g.area:.3e}"
        used[cand[0]] = True
    return True, ""


def vertex_same(a, b):
    ka = sorted(zip(shapely.to_wkb(a[0]).tolist(), a[1].round(6).tolist()))
    kb = sorted(zip(shapely.to_wkb(b[0]).tolist(), b[1].round(6).tolist()))
    return ka == kb


def run_tool(d, r, l1, tag, do_filter, tile, procs=None, threads=None, workers=None):
    out = d / f"{tag}.gpkg"
    with contextlib.redirect_stdout(io.StringIO()):
        pc.run(r, l1, out, min_acres=0.0, keep_slivers=True, do_filter=do_filter, tile_size=tile, max_memory_gb=1,
               clean=True, processes=procs, threads=threads, workers=workers, utm="31N")
    return read_out(out)


# ---------------------------------------------------------------- cases
def case_ring(n=60):
    a = np.full((n, n), 4, np.uint8)
    yy, xx = np.mgrid[:n, :n]
    rr = np.hypot(yy - n / 2 + 0.3, xx - n / 2 + 0.1)
    a[(rr > n / 5) & (rr < n / 3)] = 1
    return a, [box_px(2.5, 2.5, n - 2.3, n - 2.7)]


def case_stair(n=48):
    a = np.full((n, n), 4, np.uint8)
    for i in range(n):
        a[i, : i + 1] = 1
    tri = shapely.Polygon([(ORIGIN[0], ORIGIN[1]), (ORIGIN[0] + n * PX, ORIGIN[1] - n * PX), (ORIGIN[0], ORIGIN[1] - n * PX)])
    return a, [box_px(1.4, 0.0, n - 0.5, n - 3.6), tri]


def case_checker(n=40):
    a = np.where((np.add.outer(np.arange(n), np.arange(n)) % 2) == 0, 1, 4).astype(np.uint8)
    a[10:30, 10:30] = 1
    return a, [box_px(0.5, 0.5, n - 0.5, n - 0.5)]


def case_l1_edges(n=50):
    rng = np.random.default_rng(7)
    a = np.where(rng.random((n, n)) < 0.75, 1, 4).astype(np.uint8)
    geoms = [box_px(0, 0, 25, 25),                       # edges on cell lines, sharing an edge with the next
             box_px(0, 25, 25, 50),
             box_px(25, 0, 50, 24.5),                    # edge through cells
             shapely.Polygon([(ORIGIN[0] + 26 * PX, ORIGIN[1] - 26 * PX), (ORIGIN[0] + 49 * PX, ORIGIN[1] - 27.3 * PX),
                              (ORIGIN[0] + 40 * PX, ORIGIN[1] - 49 * PX)])]
    return a, geoms


def case_bad_l1(n=50):
    rng = np.random.default_rng(11)
    z = rng.random((n, n))
    for _ in range(2):
        z = (z + np.roll(z, 1, 0) + np.roll(z, -1, 0) + np.roll(z, 1, 1) + np.roll(z, -1, 1)) / 5
    a = np.where(z > np.median(z), 1, 4).astype(np.uint8)
    x, y = ORIGIN
    bow = shapely.Polygon([(x + 2 * PX, y - 2 * PX), (x + 20 * PX, y - 20 * PX), (x + 20 * PX, y - 2 * PX),
                           (x + 2 * PX, y - 20 * PX)])                                     # self-crossing (invalid)
    multi = shapely.MultiPolygon([box_px(22, 22, 30, 30), box_px(35.5, 35.5, 48, 48)])     # multipart
    over = [box_px(5, 25, 20, 45), box_px(10, 30, 30, 49.2)]                             # overlapping
    return a, [bow, multi] + over


def case_nodata(n=40):
    rng = np.random.default_rng(3)
    a = rng.choice([0, 1, 1, 1, 4], size=(n, n)).astype(np.uint8)
    return a, [box_px(1, 1, n - 1, n - 1)]


def case_thin(rows=5, cols=3000):
    rng = np.random.default_rng(5)
    a = np.where(rng.random((rows, cols)) < 0.7, 1, 4).astype(np.uint8)
    return a, [box_px(0.5, 3.3, 4.6, 1500.2), box_px(0, 1500.2, 5, 2999.1)]


CASES = {"ring": case_ring, "stair": case_stair, "checker": case_checker, "l1_edges": case_l1_edges,
         "bad_l1": case_bad_l1, "nodata": case_nodata, "thin": case_thin}


def _check_case(name, tiles, layouts, filters=(True, False)):
    arr, geoms = CASES[name]()
    nodata = 0 if name == "nodata" else None
    d = Path(tempfile.mkdtemp())
    r, l1 = write_case(d, arr, geoms, nodata)
    for do_filter in filters:
        ref = reference(arr, geoms, do_filter, nodata)
        first = None
        for tile in tiles:
            for k, lay in enumerate(layouts):
                got = run_tool(d, r, l1, f"{name}_{do_filter}_{tile}_{k}", do_filter, tile, *lay)
                ok, why = topo_same(got[0], ref)
                assert ok, (name, do_filter, tile, lay, why)
                if first is None:
                    first = got
                if tile == tiles[0]:
                    assert vertex_same(got, first), (name, do_filter, tile, lay, "layouts differ")
                assert abs(got[1].sum() - first[1].sum()) < 0.002 + 1e-6 * len(got[1]), (name, tile)


IN_PROCESS = [(1, 1), (1, 3)]          # threads inside this program
WITH_PROCS = [(1, 1), (1, 4), (2, 1), (2, 2)]


def test_ring_across_seams():
    os.environ["PREPROCESS_TEST_BLOCK"] = "16"
    old, pc.BLOCK = pc.BLOCK, 16       # block seams inside tiles as well
    try:
        _check_case("ring", (7, 13, 32, 1000), IN_PROCESS)
    finally:
        pc.BLOCK = old
        os.environ.pop("PREPROCESS_TEST_BLOCK")


def test_staircase():
    _check_case("stair", (5, 16, 1000), IN_PROCESS)


def test_checkerboard():
    _check_case("checker", (6, 17, 1000), IN_PROCESS)


def test_cane_on_l1_edges():
    _check_case("l1_edges", (25, 10, 1000), IN_PROCESS)


def test_overlapping_invalid_multipart_l1():
    _check_case("bad_l1", (9, 20, 1000), IN_PROCESS)


def test_nodata_tag():
    _check_case("nodata", (8, 1000), IN_PROCESS)


def test_thin_raster():
    _check_case("thin", (2, 64, 5000), IN_PROCESS, filters=(True,))


def test_layouts_with_processes_identical():
    """Same tile size, every layout (threads here, processes, processes x threads): vertex-identical."""
    _check_case("bad_l1", (11,), WITH_PROCS, filters=(True,))


# ---------------------------------------------------------------- planner, memory, cache, prompts
def test_plan_layout_budgets():
    gb = 1024 ** 3
    bpc = pc.plan_bpc(17, "container")
    seen = {}
    for b in (0.5, 1, 2, 4, 8):
        p, t, tile, total = pc.plan_layout(6835, 6775, b * gb, 12, bpc)
        assert p * t >= total and total == 12, (b, p, t, total)
        assert p == 1 or p * pc.WORKER_BASE_BYTES + total * pc.thread_task_bytes(tile, bpc) <= b * gb
        seen[b] = (p, t)
    assert seen[0.5] == (1, 12) and seen[8] == (12, 1) and 1 < seen[2][0] < 12
    assert pc.plan_layout(100, 100, 8 * gb, 12, bpc, processes=3, threads=2)[:2] == (3, 2)
    assert pc.plan_layout(100, 100, 8 * gb, 12, bpc, workers=8, processes=2)[:2] == (2, 4)
    assert pc.plan_layout(100, 100, 8 * gb, 12, bpc, workers=8, threads=4)[:2] == (2, 4)
    assert pc.plan_bpc(17, "desktop") < pc.plan_bpc(17, "container")


def test_memory_buffer_and_mode():
    gb = 1024 ** 3
    for avail, budget in ((2.5, 2.0), (12, 11.0), (16, 15.0)):
        assert abs(avail - pc.memory_buffer_bytes(0, None, int(avail * gb)) / gb - budget) < 1e-6
    os.environ["PREPROCESS_MEMORY_MODE"] = "desktop"
    try:
        assert pc.memory_mode() == "desktop"
    finally:
        os.environ.pop("PREPROCESS_MEMORY_MODE")
    assert pc.memory_mode() in ("desktop", "container")


def test_valid_l1_wkb_equals_load_l1():
    _, geoms = case_bad_l1()
    geoms = geoms + [shapely.Polygon(), box_px(1, 1, 3, 3)]
    d = Path(tempfile.mkdtemp())
    pyogrio.raw.write(str(d / "l1.gpkg"), geometry=shapely.to_wkb(np.array(geoms, dtype=object)), field_data=[],
                      fields=[], geometry_type="Unknown", crs="EPSG:4326", driver="GPKG", layer="l1")
    for crs in ("EPSG:4326", "EPSG:32631"):
        a = pc.polys_only(pc.load_l1(d / "l1.gpkg", crs, workers=2))
        b = shapely.from_wkb(pc._valid_l1_wkb(d / "l1.gpkg", crs, workers=2))
        assert len(a) == len(b) and shapely.equals_exact(a, b, 0).all(), crs


def test_l1_cache_reuse_and_own_files_only():
    d = Path(tempfile.mkdtemp())
    old_dir = pc.L1_CACHE_DIR
    pc.L1_CACHE_DIR = d / "cache"
    os.environ["PREPROCESS_L1_CACHE_GB"] = "0"          # cap 0: evict everything not in use
    try:
        r, l1 = write_case(d, *case_ring())
        (d / "w").mkdir()
        p1 = pc.prepare_l1(l1, "EPSG:4326", d / "w", 2)
        assert p1.parent == pc.L1_CACHE_DIR and p1.exists()
        assert pc.prepare_l1(l1, "EPSG:4326", d / "w", 2) == p1          # cached
        (pc.L1_CACHE_DIR / "l1v_0123456789abcdef.gpkg").write_bytes(b"not ours")   # not in the index
        idx = json.loads((pc.L1_CACHE_DIR / pc.L1_CACHE_INDEX).read_text())["files"]
        idx[p1.name]["used"] = 0                                         # long unused: may be evicted
        pc._write_json(pc.L1_CACHE_DIR / pc.L1_CACHE_INDEX, {"files": idx})
        p2 = pc.prepare_l1(l1, "EPSG:32631", d / "w", 2)                 # another CRS -> another entry, evicts p1
        assert p2.exists() and not p1.exists()
        assert (pc.L1_CACHE_DIR / "l1v_0123456789abcdef.gpkg").exists()  # never deleted
    finally:
        pc.L1_CACHE_DIR = old_dir
        os.environ.pop("PREPROCESS_L1_CACHE_GB")


def _piped(fn, answers, *args):
    out, old = io.StringIO(), sys.stdin
    sys.stdin = io.StringIO("".join(a + "\n" for a in answers))
    try:
        with contextlib.redirect_stdout(out):
            res = fn(*args)
    finally:
        sys.stdin = old
    return res, out.getvalue()


def test_ask_output_three_questions():
    d = Path(tempfile.mkdtemp())
    res, txt = _piped(pc.ask_output, ["", "", ""], d, "r_remaining_cane", {})
    assert res == d / "r_remaining_cane.shp"
    assert f"3) Output folder [{d}]:" in txt and "   Output name [r_remaining_cane]:" in txt
    assert "   A shapefile cannot be larger than 2 GB; choose 2 for very large areas." in txt
    assert "   Save as:  1) Shapefile (.shp)   2) GeoPackage (.gpkg)  [1]:" in txt
    res, txt = _piped(pc.ask_output, [str(d), "bad:name", "CON", "dot.", "my out (v2).shp", "gpkg"], d, "x", {})
    assert res == d / "my out (v2).gpkg" and "cannot contain :" in txt and "reserved name" in txt and "dot" in txt
    res, _ = _piped(pc.ask_output, [str(d / "new sub" / "a.gpkg"), "", "", ""], d, "x", {})   # full path pasted
    assert res == d / "new sub" / "a.gpkg" and (d / "new sub").is_dir()
    (d / "f.shp").write_bytes(b"someone else's")
    res, txt = _piped(pc.ask_output, ["", "f", "1", "", "g", ".shp"], d, "x", {})                # N -> name again
    assert res == d / "g.shp" and "f.shp already exists. Overwrite? [y/N]:" in txt
    res, txt = _piped(pc.ask_output, ["", "", ""], d, "x", {"out": str(d / "prev.gpkg")})         # previous answers
    assert res == d / "prev.gpkg" and "[prev]" in txt and "[2]" in txt


def test_result_block_rows_add_up():
    s = {"count": 10, "acres": 12.5, "sn": 3, "sa": 0.25, "noise": 2, "na": 0.0}
    ans = {"keep_slivers": False, "min_acres": 0.2, "do_filter": True, "neighbors": 4, "threshold": "majority"}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        pc.print_run_result(s, ans, pc.zone_crs(31, True), Path("o.shp"), Path("o_work"), True, Path("x"),
                            ["low memory"], None, {"tiles": 1.0}, 2.0, summary_line=True)
    t = buf.getvalue()
    assert t.isascii() and "NOTE: low memory" in t and "RESULT" in t
    assert "Cane pieces inside L1" in t and "15" in t and "12.750" in t and "SUMMARY polygons=10" in t


def test_memory_error_in_threads_splits_tile():
    arr, geoms = case_bad_l1()
    d = Path(tempfile.mkdtemp())
    r, l1 = write_case(d, arr, geoms)
    ref = run_tool(d, r, l1, "ref", True, 1000, 1, 1)
    os.environ["PREPROCESS_TEST_MEMFAIL"] = "0_0_25_25"
    try:
        got = run_tool(d, r, l1, "split", True, 25, 1, 3)
    finally:
        os.environ.pop("PREPROCESS_TEST_MEMFAIL")
    ok, why = topo_same(got[0], ref[0])
    assert ok, why


def test_dead_process_retried_with_threads():
    arr, geoms = case_bad_l1()
    d = Path(tempfile.mkdtemp())
    r, l1 = write_case(d, arr, geoms)
    ref = run_tool(d, r, l1, "ref", True, 25, 1, 1)
    os.environ["PREPROCESS_TEST_MEMFAIL"] = "kill:0_25_25_50"
    try:
        got = run_tool(d, r, l1, "kill", True, 25, 2, 2)    # killed twice -> processed as 4 sub-tiles
    finally:
        os.environ.pop("PREPROCESS_TEST_MEMFAIL")
    ok, why = topo_same(got[0], ref[0])
    assert ok, why


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
