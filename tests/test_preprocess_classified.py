"""Tests for src/preprocess_classified.py (run: python tests/test_preprocess_classified.py or pytest)."""
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import preprocess_classified as pc  # noqa: E402

pc.LAST_RUN = Path(tempfile.mkdtemp()) / "last_run.json"   # never touch the user's own last-run record

PATHS = [
    ('r"C:\\a\\b.tif"', "C:/a/b.tif"),
    ("& 'C:\\a b\\c.tif'", "C:/a b/c.tif"),
    ('"C:\\Users\\x y\\data\\in.tif"', "C:/Users/x y/data/in.tif"),
    ("  'C:/mixed\\path/f.shp'   ", "C:/mixed/path/f.shp"),
    ("C:\\plain\\f.tif ", "C:/plain/f.tif"),
    ("file:///C:/data/my%20file.tif", "C:/data/my file.tif"),
    ("file:///home/u/a.tif", "/home/u/a.tif"),
    ("\u201cC:\\a\\b.tif\u201d", "C:/a/b.tif"),
    ("& r'D:\\x\\y.gpkg'", "D:/x/y.gpkg"),
    ("\\\\server\\share\\a.tif", "//server/share/a.tif"),
    ("/data/a b/c.tif", "/data/a b/c.tif"),
]


def test_clean_path_table():
    for raw, want in PATHS:
        assert pc.clean_path(raw) == want, (raw, pc.clean_path(raw))


def test_clean_path_env(monkeypatch=None):
    import os
    os.environ["PC_TEST_DIR"] = "/tmp/pcdir"
    assert pc.clean_path("%PC_TEST_DIR%\\a.tif") == "/tmp/pcdir/a.tif"
    assert pc.clean_path("$PC_TEST_DIR/a.tif") == "/tmp/pcdir/a.tif"
    assert pc.clean_path("~/a.tif").endswith("/a.tif") and not pc.clean_path("~/a.tif").startswith("~")


def test_parse_crs():
    opts = [(pc.zone_crs(31, True), "UTM 31N", ""), (pc.zone_crs(32, True), "UTM 32N", "")]
    assert pc.parse_crs("1", True, opts).to_epsg() == 32631
    assert pc.parse_crs("2", True, opts).to_epsg() == 32632
    for t, want in [("42N", 32642), ("utm 47 n", 32647), ("UTM47N", 32647), ("33s", 32733), ("1N", 32601),
                    ("32642", 32642), ("EPSG:32642", 32642), ("epsg 32642", 32642), ("42S", 32742)]:
        assert pc.parse_crs(t, True, opts).to_epsg() == want, t
    assert pc.parse_crs("42", False).to_epsg() == 32742          # no list shown: bare zone, hemisphere default
    crs, hint = pc.parse_crs_msg("42", True, opts)               # list shown: 42 is not a list number
    assert crs is None and "42N" in hint
    crs, hint = pc.parse_crs_msg("3", True, opts)
    assert crs is None and "3N" in hint
    for bad in ["4326", "abc", "", "99N", "EPSG:2277"]:             # geographic / feet / nonsense
        assert pc.parse_crs(bad, True, opts) is None, bad
    assert pc.parse_crs("3857", True, opts).to_epsg() == 3857      # metre CRS parses; flagged below
    assert pc.area_crs_notes(pc.parse_crs("3857"), "EPSG:4326", (10, 5, 11, 6))[0] is False
    is_utm, far = pc.area_crs_notes(pc.zone_crs(44, True), "EPSG:4326", (10, 5, 11, 6))
    assert is_utm and "zones away" in far
    assert pc.area_crs_notes(pc.zone_crs(33, True), "EPSG:4326", (10, 5, 11, 6)) == (True, None)


def test_utm_options_lists_zones():
    from pyproj import CRS
    o = pc.utm_options(CRS.from_epsg(4326), (3.0, 1.0, 3.5, 1.5))
    assert o[0][0].to_epsg() == 32631 and "suggested" in o[0][2]
    o = pc.utm_options(CRS.from_epsg(4326), (5.0, 1.0, 8.0, 1.5))
    assert {c.to_epsg() for c, _, _ in o} == {32631, 32632}


def filt(rows, nb=4, thr="majority", nodata=None, cane=1):
    a = np.array(rows, dtype=np.uint8)
    return pc.majority_cane(a, nodata, cane, 2, nb, thr)


def grid(n=5, fill=4):
    return np.full((n, n), fill, dtype=np.uint8)


def put(a, cells, v):
    for r, c in cells:
        a[r, c] = v
    return a


def test_four_majority():
    a = put(grid(), [(1, 2), (2, 3), (3, 2)], 1)          # N,E,S cane around (2,2)=4
    assert filt(a)[2, 2]
    a = put(grid(), [(1, 2), (2, 3)], 1)                   # only 2 of 4
    assert not filt(a)[2, 2]
    a = put(grid(), [(2, 2)], 1)                           # lone cane cell is replaced
    assert not filt(a)[2, 2]
    a = put(grid(), [(2, 2), (1, 2), (2, 3)], 1)           # cane centre, 2 cane + 2 other neighbours stays
    assert filt(a)[2, 2]


def test_four_edge_and_corner():
    a = put(grid(), [(1, 2), (0, 1)], 1)                   # top edge (0,2): W and S cane (adjacent)
    assert filt(a)[0, 2]
    a = put(grid(), [(0, 1), (0, 3)], 1)                   # W and E only (opposite)
    assert not filt(a)[0, 2]
    a = put(grid(), [(0, 1), (1, 0)], 1)                   # corner (0,0): E and S
    assert filt(a)[0, 0]
    a = put(grid(), [(0, 1)], 1)
    assert not filt(a)[0, 0]


def test_four_half():
    a = put(grid(), [(1, 2), (2, 3)], 1)                   # N,E cane but S,W = 4 also pair up: tie, centre kept
    assert not filt(a, 4, "half")[2, 2]
    a = put(put(grid(), [(1, 2), (2, 3)], 1), [(3, 2)], 3)  # S differs: only N,E pair -> replaced
    assert filt(a, 4, "half")[2, 2]
    a = put(grid(), [(1, 2), (3, 2)], 1)                   # N,S opposite
    assert not filt(a, 4, "half")[2, 2]


def test_half_tie_keeps_centre():
    a = grid(fill=3)
    put(a, [(1, 2), (2, 3)], 4)                            # N,E = 4 ; S,W = 3 ; centre 1 (cane)
    a[2, 2] = 1
    assert filt(a, 4, "half")[2, 2]                        # tie -> centre kept (cane)
    a[2, 2] = 3
    put(a, [(3, 2), (2, 1)], 1)                            # N,E=4, S,W=1, centre 3 -> tie, stays 3
    assert not filt(a, 4, "half")[2, 2]


def test_eight_majority_run():
    a = put(grid(7), [(2, 3), (2, 4), (3, 4), (4, 4), (4, 3)], 1)   # N,NE,E,SE,S around (3,3)
    assert filt(a, 8)[3, 3]
    a = put(grid(7), [(2, 3), (2, 4), (3, 4), (4, 4), (3, 2)], 1)   # N,NE,E,SE + W (not contiguous)
    assert not filt(a, 8)[3, 3]
    a = put(grid(7), [(2, 3), (2, 4), (3, 4), (4, 4)], 1)           # run of 4 only
    assert not filt(a, 8)[3, 3]
    assert not filt(a, 8, "half")[3, 3]                              # HALF tie with the other 4: kept
    a[4, 3] = 3                                                      # break the other run -> cane wins
    assert filt(a, 8, "half")[3, 3]
    a = put(grid(7), [(2, 3), (2, 4), (4, 4), (4, 3)], 1)           # 4 cells, broken run
    assert not filt(a, 8, "half")[3, 3]


def test_eight_edge_and_corner():
    a = put(grid(), [(0, 1), (1, 1), (1, 2)], 1)           # top edge (0,2): W,SW,S
    assert filt(a, 8)[0, 2]
    a = put(grid(), [(1, 1), (1, 2), (1, 3)], 1)           # SW,S,SE: none along the edge
    assert not filt(a, 8)[0, 2]
    a = put(grid(), [(0, 1), (1, 1), (1, 0)], 1)           # corner (0,0): E,SE,S
    assert filt(a, 8)[0, 0]
    a = put(grid(), [(0, 1), (1, 0)], 1)
    assert not filt(a, 8)[0, 0]


def test_nodata():
    a = np.array(grid(), dtype=np.uint8)
    a[2, 2] = 0
    assert not filt(a, nodata=0)[2, 2]                     # nodata stays nodata (never cane)
    a = put(grid(), [(1, 2), (2, 3), (3, 2)], 0)           # nodata neighbours do not vote
    assert not filt(a, nodata=0)[2, 2]
    b = put(grid(), [(1, 2), (2, 3), (3, 2)], 1)
    assert filt(b, nodata=0)[2, 2]


def _literal(a, cane, nb, thr, nodata):
    """Slow cell-by-cell implementation of the documented rules."""
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
            edge_r = r in (0, rows - 1)
            edge_c = c in (0, cols - 1)
            n = len(ring)
            if edge_r or edge_c:
                k = 2 if nb == 4 else 3
            else:
                k = (nb // 2 + 1) if thr == "majority" else nb // 2
            qual = set()
            for v in set(x for x in ring if x is not None):
                for s in range(n):
                    idx = [(s + i) % n for i in range(k)]
                    if all(ring[i] == v for i in idx):
                        if (edge_r or edge_c) and nb == 8:
                            along = set()
                            if edge_r:
                                along |= {2, 6}
                            if edge_c:
                                along |= {0, 4}
                            if not (set(idx) & along):
                                continue
                        qual.add(v)
                        break
            new = list(qual)[0] if len(qual) == 1 else cen
            out[r, c] = new == cane
    return out


def test_vectorised_equals_literal_random():
    rng = np.random.default_rng(1)
    for nb in (4, 8):
        for thr in ("majority", "half"):
            for nodata in (None, 0):
                a = rng.choice([1, 4, 4, 7, 0], size=(23, 19)).astype(np.uint8)
                got = pc.majority_cane(a, nodata, 1, 3, nb, thr)
                want = _literal(a, 1, nb, thr, nodata)
                assert (got == want).all(), (nb, thr, nodata, int((got != want).sum()))


def test_tile_halo_equals_whole():
    rng = np.random.default_rng(2)
    a = rng.choice([1, 4, 4, 7], size=(41, 37)).astype(np.uint8)
    for nb in (4, 8):
        for thr in ("majority", "half"):
            whole = pc.majority_cane(a, None, 1, 1, nb, thr)
            got = np.zeros_like(whole)
            for (r0, c0, r1, c1) in pc.tile_windows(*a.shape, 13):
                hr0, hr1, hc0, hc1 = max(r0 - 1, 0), min(r1 + 1, 41), max(c0 - 1, 0), min(c1 + 1, 37)
                t = pc.filter_array(a[hr0:hr1, hc0:hc1], None, 1, nb, thr,
                                    (r0 == 0, r1 == 41, c0 == 0, c1 == 37))
                got[r0:r1, c0:c1] = t[r0 - hr0:r1 - hr0, c0 - hc0:c1 - hc0]
            assert (got == whole).all(), (nb, thr)


def test_memory_buffer():
    gb = 1024 ** 3
    # owner rule: buffer = clamp(20 % of available, 0.5 GB, 1 GB); budget = available - buffer
    for avail, want_budget in ((2.5, 2.0), (12, 11.0), (16, 15.0), (1.0, 0.5), (5.0, 4.0)):
        buf = pc.memory_buffer_bytes(16 * gb, None, int(avail * gb))
        assert abs((avail * gb - buf) / gb - want_budget) < 1e-6, (avail, buf / gb)
    assert pc.memory_buffer_bytes(16 * gb, 3) == 3 * gb
    assert pc.memory_status()[0] >= 0


def test_stage_keys_depend_only_on_their_inputs(tmp=None):
    import tempfile
    d = Path(tempfile.mkdtemp())
    r, l1, out = d / "a.tif", d / "b.shp", d / "o.shp"
    r.write_bytes(b"x")
    l1.write_bytes(b"y")
    base = dict(cane_value=1, do_filter=True, neighbors=4, threshold="majority", keep_slivers=False, min_acres=0.2)
    u36, u37 = pc.zone_crs(31, True), pc.zone_crs(32, True)
    t0, f0 = pc.stage_keys(r, l1, out, u36, base)
    t1, f1 = pc.stage_keys(r, l1, out, u37, dict(base, min_acres=0.5))
    assert pc._hash(t0) == pc._hash(t1) and pc._hash(f0) != pc._hash(f1)       # late settings: tiles reused
    t2, _ = pc.stage_keys(r, l1, out, u36, dict(base, neighbors=8))
    assert pc._hash(t0) != pc._hash(t2)                                          # majority setting: tiles redone
    assert "neighbours" in pc.describe_changes(t0, t2)
    t3, _ = pc.stage_keys(r, l1, out, u36, dict(base, do_filter=False, neighbors=8))
    t4, _ = pc.stage_keys(r, l1, out, u36, dict(base, do_filter=False, neighbors=4))
    assert pc._hash(t3) == pc._hash(t4)                                          # neighbours irrelevant without filter


def test_tiled_run_equals_single_tile():
    import tempfile
    import rasterio
    from rasterio.transform import from_origin
    import pyogrio.raw
    import shapely
    d = Path(tempfile.mkdtemp())
    rng = np.random.default_rng(3)
    z = rng.random((150, 170))
    for _ in range(3):  # smooth into blobs
        z = (z + np.roll(z, 1, 0) + np.roll(z, -1, 0) + np.roll(z, 1, 1) + np.roll(z, -1, 1)) / 5
    arr = np.where(z > np.median(z), 1, 4).astype(np.uint8)
    tf = from_origin(4.0, 1.0, 0.0001, 0.0001)
    with rasterio.open(d / "r.tif", "w", driver="GTiff", height=150, width=170, count=1, dtype="uint8",
                       crs="EPSG:4326", transform=tf) as o:
        o.write(arr, 1)
    boxes = [shapely.box(4.0010, 0.9860, 4.0100, 0.9990), shapely.box(4.0090, 0.9850, 4.0165, 0.9950),
             shapely.box(4.0020, 0.9840, 4.0140, 0.9880)]   # overlapping and touching L1 polygons
    pyogrio.raw.write(str(d / "l1.gpkg"), geometry=shapely.to_wkb(np.array(boxes, dtype=object)),
                      field_data=[], fields=[], geometry_type="Polygon", crs="EPSG:4326", driver="GPKG", layer="l1")
    res = {}
    for tag, tile in (("tiles", 40), ("single", 1000)):
        pc.run(d / "r.tif", d / "l1.gpkg", d / f"{tag}.gpkg", min_acres=0.0, tile_size=tile, workers=2,
               max_memory_gb=1, clean=True)
        _, _, g, _ = pyogrio.raw.read(str(d / f"{tag}.gpkg"))
        res[tag] = shapely.from_wkb(g)
    a, b = res["tiles"], res["single"]
    assert len(a) == len(b) > 0
    assert shapely.symmetric_difference(shapely.union_all(a), shapely.union_all(b)).area < 1e-12


def test_filter_many_classes_equals_literal():
    rng = np.random.default_rng(5)
    for nb in (4, 8):
        for thr in ("majority", "half"):
            a = rng.integers(0, 200, size=(17, 21)).astype(np.uint8)
            a[rng.random(a.shape) < 0.5] = 1           # plenty of cane runs among 200 classes
            a[rng.random(a.shape) < 0.2] = 7
            for nodata in (None, 0):
                got = pc.majority_cane(a, nodata, 1, 1, nb, thr)
                assert (got == _literal(a, 1, nb, thr, nodata)).all(), (nb, thr, nodata)


def _small_case(d, seed=3):
    import rasterio
    from rasterio.transform import from_origin
    import pyogrio.raw
    import shapely
    rng = np.random.default_rng(seed)
    z = rng.random((150, 170))
    for _ in range(3):
        z = (z + np.roll(z, 1, 0) + np.roll(z, -1, 0) + np.roll(z, 1, 1) + np.roll(z, -1, 1)) / 5
    arr = np.where(z > np.median(z), 1, 4).astype(np.uint8)
    with rasterio.open(d / "r.tif", "w", driver="GTiff", height=150, width=170, count=1, dtype="uint8",
                       crs="EPSG:4326", transform=from_origin(4.0, 1.0, 0.0001, 0.0001)) as o:
        o.write(arr, 1)
    boxes = [shapely.box(4.0010, 0.9860, 4.0100, 0.9990), shapely.box(4.0090, 0.9850, 4.0165, 0.9950),
             shapely.box(4.0020, 0.9840, 4.0140, 0.9880)]
    pyogrio.raw.write(str(d / "l1.gpkg"), geometry=shapely.to_wkb(np.array(boxes, dtype=object)),
                      field_data=[], fields=[], geometry_type="Polygon", crs="EPSG:4326", driver="GPKG", layer="l1")
    return d / "r.tif", d / "l1.gpkg"


def _read(path):
    import pyogrio.raw
    import shapely
    _, _, g, _ = pyogrio.raw.read(str(path))
    return shapely.from_wkb(g)


def _same(a, b):
    import shapely
    return len(a) == len(b) > 0 and shapely.symmetric_difference(shapely.union_all(a), shapely.union_all(b)).area < 1e-14


def test_bracket_names_and_old_sidecars():
    import tempfile
    d = Path(tempfile.mkdtemp())
    r, l1 = _small_case(d)
    ref = None
    for name in ("cane [v2].shp", "cane[1].gpkg"):
        out = d / name
        pc.run(r, l1, out, min_acres=0.0, workers=2, max_memory_gb=1)
        assert out.exists(), name
        assert not [f.name for f in d.iterdir() if "__partial" in f.name or "__old" in f.name]
        g = _read(out)
        ref = g if ref is None else ref
        assert _same(g, ref)
    out = d / "cane [v2].shp"
    for ext in (".sbn", ".sbx", ".qix"):
        (d / ("cane [v2]" + ext)).write_bytes(b"old index")
    pc.run(r, l1, out, min_acres=0.01, workers=2, max_memory_gb=1)   # only the final stage reruns
    left = sorted(f.name for f in d.iterdir() if f.name.startswith("cane [v2]."))
    assert left == ["cane [v2].cpg", "cane [v2].dbf", "cane [v2].prj", "cane [v2].shp", "cane [v2].shx"], left


def test_foreign_work_folder_is_never_deleted():
    import tempfile
    d = Path(tempfile.mkdtemp())
    r, l1 = _small_case(d)
    (d / "o_work").mkdir()
    (d / "o_work" / "notes.txt").write_text("keep me")
    try:
        pc.run(r, l1, d / "o.gpkg", workers=2, max_memory_gb=1)
        raise AssertionError("expected a refusal")
    except pc.UserError as e:
        assert "not made by this script" in str(e)
    assert (d / "o_work" / "notes.txt").read_text() == "keep me"


def test_resume_after_partly_split_tile():
    import os
    import tempfile
    d = Path(tempfile.mkdtemp())
    r, l1 = _small_case(d)
    pc.run(r, l1, d / "ref.gpkg", min_acres=0.0, workers=2, max_memory_gb=1, tile_size=1000, clean=True)
    # tile (0,0,75,85) fails, then its 3rd sub-tile fails too: 2 of 4 sub-tiles are saved, the run stops
    os.environ["PREPROCESS_TEST_MEMFAIL"] = "0_0_75_85,37_0_75_42"
    try:
        pc.run(r, l1, d / "o.gpkg", min_acres=0.0, workers=2, max_memory_gb=1, tile_size=85)
        raise AssertionError("expected a failure")
    except pc.UserError:
        pass
    finally:
        os.environ.pop("PREPROCESS_TEST_MEMFAIL")
    tiles = d / "o_work" / "tiles"
    assert (tiles / "0_0_37_42.fin").exists() and not (tiles / "0_0_75_85.fin").exists()
    pc.run(r, l1, d / "o.gpkg", min_acres=0.0, workers=2, max_memory_gb=1)     # resume finishes the sub-tiles
    assert _same(_read(d / "o.gpkg"), _read(d / "ref.gpkg"))


def test_dead_worker_restarts_pool():
    import os
    import tempfile
    d = Path(tempfile.mkdtemp())
    r, l1 = _small_case(d)
    pc.run(r, l1, d / "ref.gpkg", min_acres=0.0, workers=2, max_memory_gb=1, tile_size=1000, clean=True)
    os.environ["PREPROCESS_TEST_MEMFAIL"] = "kill:0_85_75_170"
    try:
        pc.run(r, l1, d / "o.gpkg", min_acres=0.0, workers=2, max_memory_gb=1, tile_size=85, clean=True)
    finally:
        os.environ.pop("PREPROCESS_TEST_MEMFAIL")
    assert _same(_read(d / "o.gpkg"), _read(d / "ref.gpkg"))


def test_cli_rejects_bad_projection():
    import tempfile
    d = Path(tempfile.mkdtemp())
    r, l1 = _small_case(d)
    base = ["--raster", str(r), "--l1", str(l1), "--out", str(d / "o.gpkg"), "--max-memory-gb", "1", "--workers", "2"]
    for bad in ("EPSG:4326", "EPSG:2263", "abc", "3857"):
        try:
            pc.main(base + ["--utm", bad])
            raise AssertionError(bad)
        except SystemExit as e:
            assert e.code == 2, (bad, e.code)
    pc.main(base + ["--utm", "31N", "--clean"])
    assert (d / "o.gpkg").exists()


def test_lock_and_env_rule():
    import json
    import os
    import tempfile
    d = Path(tempfile.mkdtemp())
    r, l1 = _small_case(d)
    w = d / "o_work"
    pc.open_work(w)
    (w / pc.LOCK).write_text(json.dumps({"pid": os.getppid(), "started": "x"}))   # a live process
    try:
        pc.run(r, l1, d / "o.gpkg", workers=2, max_memory_gb=1)
        raise AssertionError("expected the lock to stop the run")
    except pc.UserError as e:
        assert "another run" in str(e)
    (w / pc.LOCK).write_text(json.dumps({"pid": 2 ** 22 + 12345, "started": "x"}))  # dead process: taken over
    pc.run(r, l1, d / "o.gpkg", workers=2, max_memory_gb=1, clean=True)
    assert not w.exists()
    (d / "%NOPE_VAR%").mkdir()
    assert pc.clean_path(str(d / "%NOPE_VAR%")) == str(d / "%NOPE_VAR%").replace("\\", "/")
    os.environ["PC_TEST_HOME"] = "/x"
    (d / "%PC_TEST_HOME%").mkdir()
    assert pc.clean_path(str(d / "%PC_TEST_HOME%")).endswith("%PC_TEST_HOME%")    # exists as typed: kept


def test_key_covers_sidecars_and_version():
    import os
    import tempfile
    d = Path(tempfile.mkdtemp())
    for ext in (".shp", ".shx", ".dbf", ".prj"):
        (d / ("l" + ext)).write_bytes(b"x")
    (d / "r.tif").write_bytes(b"x")
    base = dict(cane_value=1, do_filter=True, neighbors=4, threshold="majority", keep_slivers=False, min_acres=0.2)
    t0, _ = pc.stage_keys(d / "r.tif", d / "l.shp", d / "o.shp", pc.zone_crs(31, True), base)
    st = os.stat(d / "l.prj")
    os.utime(d / "l.prj", ns=(st.st_atime_ns, st.st_mtime_ns + 1))      # 1 ns later, same size
    t1, _ = pc.stage_keys(d / "r.tif", d / "l.shp", d / "o.shp", pc.zone_crs(31, True), base)
    assert pc._hash(t0) != pc._hash(t1)
    assert t0["version"] == pc.CACHE_VERSION


def test_disk_space_and_failed_swap(monkeypatch=None):
    import tempfile
    from collections import namedtuple
    d = Path(tempfile.mkdtemp())
    r, l1 = _small_case(d)
    real_usage, real_replace = pc.shutil.disk_usage, pc.os.replace
    pc.shutil.disk_usage = lambda p: namedtuple("u", "total used free")(10, 10, 1000)
    try:
        pc.run(r, l1, d / "o.shp", workers=2, max_memory_gb=1)
        raise AssertionError("expected a disk space stop")
    except pc.DiskSpaceError as e:
        assert "GB needed" in str(e)
    finally:
        pc.shutil.disk_usage = real_usage
    pc.run(r, l1, d / "o.shp", min_acres=0.0, workers=2, max_memory_gb=1)
    before = (d / "o.dbf").read_bytes()

    def locked(src, dst):                       # the .dbf of the old output is locked
        if str(dst).endswith("o__old.dbf"):
            raise PermissionError("locked")
        return real_replace(src, dst)
    pc.os.replace = locked
    try:
        res = pc.run(r, l1, d / "o.shp", min_acres=0.5, workers=2, max_memory_gb=1)
    finally:
        pc.os.replace = real_replace
    assert (d / "o.dbf").read_bytes() == before                          # old output untouched and complete
    assert sorted(f.suffix for f in d.iterdir() if f.name.startswith("o.")) == [".cpg", ".dbf", ".prj", ".shp", ".shx"]
    alt = Path(res["written"])
    assert alt != d / "o.shp" and alt.exists() and len(_read(alt)) == res["polygons"]


def _piped(fn, answers, *args):
    import contextlib
    import io
    import sys
    out, old = io.StringIO(), sys.stdin
    sys.stdin = io.StringIO("".join(a + "\n" for a in answers))
    try:
        with contextlib.redirect_stdout(out):
            res = fn(*args)
    finally:
        sys.stdin = old
    return res, out.getvalue()


def test_question_mode_replace_own_output():
    import tempfile
    d = Path(tempfile.mkdtemp())
    r, l1 = _small_case(d)
    pc.run(r, l1, d / "o.gpkg", workers=2, max_memory_gb=1)            # earlier run of this tool (work folder kept)
    (d / "x.gpkg").write_bytes(b"someone else's file")
    res, txt = _piped(pc.out_check, [""], str(d / "o.gpkg"))           # Enter alone = replace
    assert res[0] == d / "o.gpkg" and "o.gpkg was made by an earlier run of this tool. Replace it? [Y/n]:" in txt
    res, txt = _piped(pc.out_check, [""], str(d / "x.gpkg"))           # Enter alone = keep the foreign file
    assert res[0] is None and "x.gpkg already exists. Overwrite? [y/N]:" in txt


def test_question_mode_projection_default():
    import tempfile
    d = Path(tempfile.mkdtemp())
    r, l1 = _small_case(d)                                             # data centre in UTM zone 31N
    answers = [str(r), str(l1), str(d / "new.gpkg"), "", "", "", "", "", "", "", "n"]   # folder, name, format
    for prev, want in (("EPSG:32647", "[1]:"), ("EPSG:32632", "[EPSG:32632]:"), ("EPSG:32631", "[EPSG:32631]:")):
        res, txt = _piped(pc.interactive, answers, dict(utm=prev))
        line = [x for x in txt.split("\n") if "Choose number, zone" in x][0]
        assert want in line and "zones away" not in txt, (prev, line)
    assert "1) UTM 31N" in txt


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
