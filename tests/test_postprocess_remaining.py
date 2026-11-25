"""Tests for src/postprocess_remaining.py (run: python tests/test_postprocess_remaining.py or pytest).

Every case is compared with a brute-force reference written here independently: plain shapely,
one L1 polygon at a time, exactly as the model's steps read (clip, erase per L1 feature, explode,
select by intersection, area <= size, merge, dissolve by class, explode). Synthetic coordinates only.
"""
import contextlib
import io
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pyogrio.raw
import pyproj
import shapely
from shapely.geometry import MultiPolygon, Polygon, box

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import postprocess_remaining as ppr  # noqa: E402

ACRE = 4046.85642
CRS = "EPSG:32631"            # metric working CRS of the synthetic cases (neutral open-sea coordinates)
X0, Y0 = 500000.0, 100000.0
ppr.LAST_RUN = Path(tempfile.mkdtemp()) / "last_run.json"   # never touch the user's own last-run record


def B(x0, y0, x1, y1):
    return box(X0 + x0, Y0 + y0, X0 + x1, Y0 + y1)


# ---------------------------------------------------------------- brute-force reference
def parts(g):
    out = []
    for x in shapely.get_parts(np.array([g], dtype=object)):
        if x.geom_type == "Polygon" and not x.is_empty:
            out.append(x)
        elif x.geom_type in ("MultiPolygon", "GeometryCollection"):
            out += parts(x)
    return out


def reference(final, l1, size=0.2, rule=True, tol=0.001):
    """Literal model: per-feature clip, per-L1-feature erase, select by location, area rule, dissolve."""
    final = [shapely.make_valid(g) for g in final]
    l1 = [p for g in l1 for p in parts(shapely.make_valid(g))]
    ul = shapely.union_all(l1)
    remaining = [p for f in final for p in parts(f.intersection(ul))]          # Clip + Multipart To Singlepart
    rem = shapely.union_all(remaining) if remaining else Polygon()
    p81, p82 = [], []
    for L in l1:                                                                # Erase keeps L1 features apart
        for p in parts(L.difference(rem)):
            if p.area < 0.01:                                                   # numerical noise: dropped
                continue
            touches = (not rem.is_empty) and p.distance(rem) <= tol             # Select Layer By Location INTERSECT
            if rule and touches and round(p.area / ACRE, 9) <= size:            # "Selected" = 'Yes' AND "Area" <= size
                p81.append(p)
            else:
                p82.append(p)
    c81 = shapely.union_all(remaining + p81) if (remaining or p81) else Polygon()   # Merge + Dissolve
    c82 = shapely.union_all(p82).difference(c81) if p82 else Polygon()            # one class per point, 81 kept
    keep = lambda ps: [p for p in ps if p.area >= 0.01]                           # noqa: E731
    return keep(parts(c81)), keep(parts(c82)), ul.area


# ---------------------------------------------------------------- helpers
def write(path, geoms, crs=CRS):
    geoms = np.array(geoms, dtype=object)
    pyogrio.raw.write(str(path), geometry=shapely.to_wkb(geoms), field_data=[], fields=[], geometry_type="Unknown",
                      crs=crs, driver="GPKG", layer="x")
    return path


def read_out(path, crs=CRS):
    meta, _, w, fd = pyogrio.raw.read(str(path), columns=["predicted", "area_ac"])
    fd = [fd[list(meta["fields"]).index(k)] for k in ("predicted", "area_ac")]
    g = shapely.from_wkb(w) if len(w) else np.array([], dtype=object)
    tr = pyproj.Transformer.from_crs(4326, crs, always_xy=True)
    if len(g):
        g = shapely.transform(g, lambda xy: np.column_stack(tr.transform(xy[:, 0], xy[:, 1])))
    return g, (np.asarray(fd[0]) if len(w) else np.zeros(0, int)), (np.asarray(fd[1]) if len(w) else np.zeros(0))


def quiet(fn, *a, **k):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        res = fn(*a, **k)
    return res, out.getvalue()


def run_case(final, l1, d=None, final_crs=CRS, l1_crs=CRS, **kw):
    d = Path(d or tempfile.mkdtemp())
    same = kw.pop("same_inputs", False)                      # keep the files of the earlier run (resume tests)
    f = d / "final.gpkg" if same else write(d / "final.gpkg", final, final_crs)
    lp = d / "l1.gpkg" if same else write(d / "l1.gpkg", l1, l1_crs)
    kw.setdefault("workers", 2)
    kw.setdefault("max_memory_gb", 2)
    kw.setdefault("utm", "EPSG:32631")
    res, txt = quiet(ppr.run, f, lp, d / kw.pop("name", "out.gpkg"), **kw)
    return res, txt, d


def same_set(a, b, eps=1e-3):
    ua = shapely.union_all(a) if len(a) else Polygon()
    ub = shapely.union_all(b) if len(b) else Polygon()
    return ua.symmetric_difference(ub).area < eps


def check(final, l1, final_ref=None, l1_ref=None, size=0.2, rule=True, **kw):
    """Run the tool and compare class by class with the reference (count and shape)."""
    res, txt, d = run_case(final, l1, sliver_acres=size, rule=rule, **kw)
    g, cls, ac = read_out(d / kw.get("name", "out.gpkg"))
    r81, r82, ul = reference(final_ref or final, l1_ref or l1, size, rule)
    g81, g82 = g[cls == 81], g[cls == 82]
    assert len(g81) == len(r81) and len(g82) == len(r82), (len(g81), len(r81), len(g82), len(r82))
    assert same_set(g81, r81) and same_set(g82, r82)
    assert shapely.union_all(g81).intersection(shapely.union_all(g82)).area < 1e-3 if len(g81) and len(g82) else True
    ref_ac = sum(p.area for p in r81 + r82) / ACRE
    assert abs(res["acres81"] + res["acres82"] - ref_ac) < 1e-6 and abs(ref_ac - ul / ACRE) < 1e-5   # noise < 0.01 m2
    assert np.allclose(ac, np.round(shapely.area(g) / ACRE, 3), atol=1.5e-3)
    assert all(shapely.is_valid(g)) and set(np.unique(shapely.get_type_id(g))) <= {3}
    return g81, g82, txt, d


# ---------------------------------------------------------------- unit cases
def test_touching_and_not_touching():
    l1 = [B(0, 0, 200, 200),        # strip 200 x 4 m = 0.198 ac touching cane -> 81
          B(300, 0, 320, 20),       # small, no cane anywhere near -> 82
          B(500, 0, 540, 40),       # fully cane
          B(540, 40, 560, 60),      # small, touches the cane of the previous polygon at one corner -> 81
          B(600, 0, 700, 100)]      # 1 ha harvested next to cane -> 82 (large)
    final = [B(0, 0, 200, 196), B(500, 0, 540, 40), B(600, 0, 650, 100)]
    g81, g82, _, _ = check(final, l1)
    assert len(g82) == 2


def test_exactly_threshold():
    h = 809.371284 / 100.0          # strip 100 x h = exactly 0.2 acre
    l1 = [B(0, 0, 100, 50 + h), B(200, 0, 300, 50 + h + 0.001)]
    final = [B(0, 0, 100, 50), B(200, 0, 300, 50)]
    g81, g82, _, _ = check(final, l1)
    assert len(g81) == 2 and len(g82) == 1 and abs(g82[0].area - 100 * (h + 0.001)) < 1e-6


def test_sliver_split_across_two_l1_polygons():
    l1 = [B(0, 0, 100, 100), B(100, 0, 200, 100)]          # strip of 2 x 600 m2: each half <= 0.2, together 0.297
    final = [B(0, 0, 200, 94)]
    g81, g82, _, _ = check(final, l1)
    assert len(g81) == 1 and len(g82) == 0                  # per-L1-polygon rule, then dissolved into one polygon
    g81, g82, _, _ = check(final, [B(0, 0, 200, 100)])
    assert len(g82) == 1                                    # one L1 polygon: the 0.297 acre strip stays 82


def test_qc_outside_l1_and_overlapping_qc():
    l1 = [B(0, 0, 100, 100), B(150, 0, 250, 100)]
    final = [B(-50, -50, 60, 97), B(40, 10, 130, 60), B(200, 200, 300, 300), B(150, 0, 200, 100)]
    check(final, l1)


def test_overlapping_l1_polygons():
    l1 = [B(0, 0, 100, 100), B(50, 0, 150, 100)]
    final = [B(0, 0, 100, 97)]
    g81, g82, txt, _ = check(final, l1)
    assert "L1 polygons overlap each other in 1 place;" in txt


def test_invalid_holes_multipart():
    bow = Polygon([(X0, Y0), (X0 + 100, Y0 + 100), (X0 + 100, Y0), (X0, Y0 + 100)])            # self-crossing L1
    ring = Polygon(B(200, 0, 400, 200).exterior.coords, [B(280, 80, 320, 120).exterior.coords])   # L1 with a hole
    multi = MultiPolygon([B(500, 0, 560, 60), B(600, 0, 660, 60)])
    f_hole = Polygon(B(200, 0, 400, 200).exterior.coords,
                     [B(220, 20, 230, 30).exterior.coords, B(330, 130, 390, 190).exterior.coords])
    f_bad = Polygon([(X0 + 500, Y0), (X0 + 560, Y0 + 58), (X0 + 560, Y0), (X0 + 500, Y0 + 58)])
    final = [B(0, 0, 100, 40), f_hole, f_bad, MultiPolygon([B(600, 0, 660, 57), B(0, 60, 100, 100)])]
    check(final, [bow, ring, multi])


def test_different_crs():
    l1 = [B(0, 0, 200, 200), B(200, 0, 260, 60)]
    final_utm = [B(0, 0, 200, 196), B(200, 0, 260, 30)]
    tr = pyproj.Transformer.from_crs(CRS, 4326, always_xy=True)
    final_geo = [shapely.transform(g, lambda xy: np.column_stack(tr.transform(xy[:, 0], xy[:, 1]))) for g in final_utm]
    # the reference uses the exact UTM shapes: the round trip moves edges by ~1e-9 m and the tool, like the model,
    # still joins the two cane areas across the shared L1 edge (tolerance), so the counts must match the exact case
    check(final_geo, l1, final_ref=final_utm, final_crs="EPSG:4326")
    # L1 in EPSG:4326, FINAL in UTM: the working CRS is EPSG:4326, areas in UTM
    l1_geo = [shapely.transform(g, lambda xy: np.column_stack(tr.transform(xy[:, 0], xy[:, 1]))) for g in l1]
    res, _, d = run_case(final_utm, l1_geo, l1_crs="EPSG:4326")
    g, cls, _ = read_out(d / "out.gpkg")
    r81, r82, _ = reference(final_utm, l1)
    assert len(g[cls == 81]) == len(r81) and len(g[cls == 82]) == len(r82)
    assert same_set(g[cls == 81], r81, 0.05) and same_set(g[cls == 82], r82, 0.05)


def test_rule_off_and_classes():
    l1 = [B(0, 0, 200, 200), B(300, 0, 320, 20)]
    final = [B(0, 0, 200, 196)]
    check(final, l1, rule=False)
    _, txt, d = run_case(final, l1, classes="81", name="o81.gpkg")
    _, cls, _ = read_out(d / "o81.gpkg")
    assert set(cls) == {81}
    _, txt, d = run_case(final, l1, d=d, classes="82", name="o82.gpkg")
    _, cls, _ = read_out(d / "o82.gpkg")
    assert set(cls) == {82} and len(cls) == 1


def random_case(seed=1, n=6):
    """A field grid with shared edges, a few overlapping fields and cane blobs (some outside L1)."""
    rng = np.random.default_rng(seed)
    l1 = [B(i * 60, j * 60, i * 60 + 60, j * 60 + 60) for i in range(n) for j in range(n) if rng.random() < 0.85]
    l1 += [B(30, 30, 100, 80), B(200, 150, 260, 170)]
    final = []
    for _ in range(40):
        x, y = rng.uniform(-20, n * 60 + 20, 2)
        final.append(shapely.Point(X0 + x, Y0 + y).buffer(rng.uniform(5, 40), quad_segs=4))
    for _ in range(15):
        x, y = rng.uniform(0, n * 60, 2)
        final.append(B(x, y, x + rng.uniform(5, 50), y + rng.uniform(1, 8)))
    return final, l1


def test_random_cases_match_reference():
    for seed in (1, 2, 3):
        final, l1 = random_case(seed)
        check(final, l1, size=0.05)
        check(final, l1, size=0.2)


def test_chunked_equals_unchunked_and_resume():
    final, l1 = random_case(7, 8)
    _, _, d1 = run_case(final, l1, sliver_acres=0.05)
    g1, c1, a1 = read_out(d1 / "out.gpkg")
    d2 = Path(tempfile.mkdtemp())
    os.environ["POSTPROCESS_TEST_STOP_AFTER"] = "2"
    try:
        run_case(final, l1, d=d2, sliver_acres=0.05, max_chunk_polygons=4)
        raise AssertionError("the test stop did not happen")
    except KeyboardInterrupt:
        pass
    finally:
        del os.environ["POSTPROCESS_TEST_STOP_AFTER"]
    assert not (d2 / "out.gpkg").exists()
    res, txt, _ = run_case(final, l1, d=d2, sliver_acres=0.05, max_chunk_polygons=4, same_inputs=True)
    assert "resuming:" in txt and "overlay chunks already done" in txt
    nchunks = int(txt.split("resuming: ")[1].split(" of ")[1].split()[0])
    assert nchunks > 4
    g2, c2, a2 = read_out(d2 / "out.gpkg")
    for c in (81, 82):
        assert (c1 == c).sum() == (c2 == c).sum() and same_set(g1[c1 == c], g2[c2 == c])
    assert abs(a1.sum() - a2.sum()) < 1e-9


def test_threshold_only_rerun_uses_cache():
    final, l1 = random_case(4)
    _, txt, d = run_case(final, l1, sliver_acres=0.2)
    _, txt, d = run_case(final, l1, d=d, sliver_acres=0.05, same_inputs=True)
    assert "reusing the saved overlay" in txt and "stage 1 overlay" not in txt
    g, cls, _ = read_out(d / "out.gpkg")
    r81, r82, _ = reference(final, l1, 0.05)
    assert len(g[cls == 81]) == len(r81) and same_set(g[cls == 81], r81)
    _, txt, d = run_case(final, l1, d=d, sliver_acres=0.05, classes="82", same_inputs=True)
    assert "reusing the saved classes" in txt and "stage 3 write" in txt
    _, txt, d = run_case(final, l1, d=d, sliver_acres=0.05, classes="82", same_inputs=True)
    assert "already up to date" in txt
    final[0] = final[0].buffer(1)                                     # input changed: everything again
    _, txt, d = run_case(final, l1, d=d, sliver_acres=0.05, classes="82")
    assert "Inputs changed" in txt and "stage 1 overlay" in txt


def test_foreign_work_folder_is_never_deleted():
    d = Path(tempfile.mkdtemp())
    (d / "out_work").mkdir()
    (d / "out_work" / "mine.txt").write_text("x")
    try:
        run_case([B(0, 0, 10, 10)], [B(0, 0, 20, 20)], d=d)
        raise AssertionError("expected an error")
    except ppr.UserError as e:
        assert "not made by this script" in str(e)
    assert (d / "out_work" / "mine.txt").exists()


def test_cli_errors_exit_2():
    d = Path(tempfile.mkdtemp())
    f = write(d / "f.gpkg", [B(0, 0, 10, 10)])
    for args in (["--final", str(f), "--l1", str(f), "--out", str(d / "o.txt")],
                 ["--final", str(f), "--l1", str(d / "missing.shp"), "--out", str(d / "o.gpkg")],
                 ["--final", str(f), "--l1", str(f), "--out", str(d / "o.gpkg"), "--utm", "4326"]):
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                ppr.main(args)
            raise AssertionError("expected exit 2")
        except SystemExit as e:
            assert e.code == 2, args


def test_question_mode():
    d = Path(tempfile.mkdtemp())
    f = write(d / "f.gpkg", [B(0, 0, 200, 196)])
    l1 = write(d / "l1.gpkg", [B(0, 0, 200, 200)])
    answers = ["& '" + str(f).replace("/", "\\") + "'", '"' + str(l1) + '"', str(d / "new"), "", "bad:name", "q", "3", "2",
               "", "0.1", "", "both", "n"]
    old = sys.stdin
    sys.stdin = io.StringIO("".join(a + "\n" for a in answers))
    try:
        res, txt = quiet(ppr.interactive, {})
    finally:
        sys.stdin = old
    assert res is None and "Cancelled." in txt and "1) UTM 31N" in txt and "pieces <= 0.1 acres" in txt
    assert "Create it?" in txt and (d / "new").is_dir() and "Windows rule" in txt
    assert "Type 1 (shapefile) or 2 (GeoPackage)." in txt
    assert "[f_harvest_final]" in txt and str(d / "new" / "q.gpkg") in txt and "2 GB" in txt


def test_output_questions_split_pasted_path():
    d = Path(tempfile.mkdtemp())
    old = sys.stdin
    sys.stdin = io.StringIO(str(d / "x.gpkg") + "\n\n\n")      # pasted file path, then Enter twice
    try:
        out, txt = quiet(ppr.ask_output, d, "a_harvest_final", {})
    finally:
        sys.stdin = old
    assert out == d / "x.gpkg" and "[x]" in txt and "[2]" in txt


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name, flush=True)


def _num(x):
    return float(x.replace(",", ""))


def test_result_block_and_summary_line():
    final, l1 = random_case(2)
    _, txt, d = run_case(final, l1, sliver_acres=0.2)
    assert "==================== RESULT" in txt and "SUMMARY" not in txt
    rows = {}
    for line in txt.splitlines():
        for key in ("81  Remaining cane", "82  Harvested", "Total (81 + 82)"):
            if line.strip().startswith(key):
                rows[key] = line.split()[-2:]
    assert len(rows) == 3, txt
    n81, a81 = (_num(x) for x in rows["81  Remaining cane"])
    n82, a82 = (_num(x) for x in rows["82  Harvested"])
    nt, at = (_num(x) for x in rows["Total (81 + 82)"])
    assert n81 + n82 == nt and abs(a81 + a82 - at) < 0.0015
    area = [x for x in txt.splitlines() if x.strip().startswith("Difference")]
    assert area and area[0].rstrip().endswith("OK") and "Area check" in txt and "Time     :" in txt
    _, txt2, _ = run_case(final, l1, d=d, sliver_acres=0.2, same_inputs=True, summary_line=True)
    assert "SUMMARY class 81" in txt2 and "already up to date" in txt2 and "RESULT" in txt2
    _, txt3, _ = run_case(final, l1, d=d, sliver_acres=0.2, rule=False, classes="81", same_inputs=True)
    assert "Sliver rule : off" in txt3 and "Small harvested pieces" not in txt3 and "class 81 only" in txt3
