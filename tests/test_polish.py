"""Tests of the run messages, timing, work-folder reuse and layouts of preprocess_classified.py and
postprocess_remaining.py (synthetic data only). Run: python -m pytest -q tests/test_polish.py"""
import contextlib
import io
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pyogrio.raw
import rasterio
import shapely
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import postprocess_remaining as ppr  # noqa: E402
import preprocess_classified as pc  # noqa: E402

pc.LAST_RUN = Path(tempfile.mkdtemp()) / "last_run.json"     # never touch the user's own last-run records
ppr.LAST_RUN = Path(tempfile.mkdtemp()) / "last_run.json"
GB = 1024 ** 3


def quiet(fn, *a, **k):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        res = fn(*a, **k)
    return res, out.getvalue()


def piped(fn, answers, *a, **k):
    old = sys.stdin
    sys.stdin = io.StringIO("".join(x + "\n" for x in answers))
    try:
        return quiet(fn, *a, **k)
    finally:
        sys.stdin = old


def small_raster(d, seed=3):
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


def read_wkb(path):
    meta, _, g, f = pyogrio.raw.read(str(path))
    ac = np.asarray(f[list(meta["fields"]).index("area_ac")], float)
    return sorted(zip(shapely.to_wkb(shapely.normalize(shapely.from_wkb(g))).tolist(), ac.round(3).tolist()))


def time_parts(txt):
    lines = txt.splitlines()
    i = [k for k, x in enumerate(lines) if x.startswith("Time     :")][0]
    line = lines[i] + "".join(" " + x.strip() for x in lines[i + 1:i + 3] if x.startswith("     "))
    total = float(re.search(r"Time     : ([\d.]+) s", line).group(1))
    parts = {k.strip(): float(v) for k, v in re.findall(r"([a-zA-Z1 ]+?) ([\d.]+)(?: \||\))", line.split("(", 1)[1])}
    return total, parts


def folder_state(d):
    return sorted((f.name, f.stat().st_size, f.stat().st_mtime_ns) for f in Path(d).rglob("*") if f.is_file())


# ---------------------------------------------------------------- preprocess_classified
def test_pre_clock_leaves_out_questions_and_parts_add_up():
    d = Path(tempfile.mkdtemp())
    r, l1 = small_raster(d)
    quiet(pc.run, r, l1, d / "o.gpkg", workers=2, max_memory_gb=1)
    man = json.loads((d / "o_work" / "manifest.json").read_text())
    man["tiles_items"]["version"] = pc.CACHE_VERSION - 1                     # made by an older version
    man["tiles_key"] = "old"
    (d / "o_work" / "manifest.json").write_text(json.dumps(man))

    def slow_answer(work):
        time.sleep(1.5)                                                       # the user thinks about it
        return True
    t = time.time()
    _, txt = quiet(pc.run, r, l1, d / "o.gpkg", workers=2, max_memory_gb=1, confirm_clean=slow_answer)
    wall = time.time() - t
    assert "This work folder was made by an older version of the tool; its saved tiles cannot be reused." in txt
    assert "changed -> changed" not in txt
    plan = float(re.search(r"\[\s*([\d.]+)s\] plan", txt).group(1))
    total, parts = time_parts(txt)
    assert plan < 1.0 and total <= wall - 1.4, (plan, total, wall)
    assert set(parts) == {"plan", "prepare L1", "tiles", "merge", "write"}, parts
    assert abs(sum(parts.values()) - total) <= 0.05 * len(parts) + 0.1, (parts, total)


def test_pre_change_messages():
    d = Path(tempfile.mkdtemp())
    r, l1 = d / "r.tif", d / "l.gpkg"
    r.write_bytes(b"x" * 10)
    l1.write_bytes(b"y")
    base = dict(cane_value=1, do_filter=True, neighbors=4, threshold="majority", keep_slivers=False, min_acres=0.2)
    u = pc.zone_crs(31, True)
    t0, _ = pc.stage_keys(r, l1, d / "o.shp", u, base)
    time.sleep(0.01)
    r.write_bytes(b"x" * 2000)
    t1, _ = pc.stage_keys(r, l1, d / "o.shp", u, base)
    older, why = pc.explain_changes(t0, t1)
    assert not older and len(why) == 1 and why[0].startswith("The file r.tif (raster file) was modified since the last run")
    assert "(before: 10 B, saved 20" in why[0] and "; now: 2.0 KB, saved 20" in why[0]
    st = os.stat(r)
    os.utime(r, ns=(st.st_atime_ns, st.st_mtime_ns + 2 * 10 ** 6))            # 2 ms later, same size
    t3, _ = pc.stage_keys(r, l1, d / "o.shp", u, base)
    why = pc.explain_changes(t1, t3)[1][0]
    before, now = why.split("before: ")[1].split("; now: ")
    assert before != now.rstrip(").") and re.search(r"\d\d:\d\d:\d\d\.\d{3}", before), why
    t2, _ = pc.stage_keys(r, l1, d / "o.shp", u, dict(base, neighbors=8, cane_value=2))
    older, why = pc.explain_changes(t3, t2)
    assert why == ["Setting changed: cane value (1 -> 2), number of neighbours (4 -> 8)."], why
    assert pc.explain_changes(dict(t1, version=1), t1)[0] and pc.explain_changes(dict(t1, raster="sig"), t1)[0]
    assert pc.explain_changes(None, t1)[0]


def test_pre_low_memory_note_once_and_previous_run_screen():
    d = Path(tempfile.mkdtemp())
    r, l1 = small_raster(d)
    real = pc.memory_status
    pc.memory_status = lambda: (int(0.7 * GB), 8 * GB)
    try:
        _, txt = quiet(pc.run, r, l1, d / "o.gpkg", memory_mode="desktop")
    finally:
        pc.memory_status = real
    assert txt.count("NOTE: Free memory is low") == 1, txt
    assert txt.index("NOTE: Free memory is low") < txt.index("] plan (")
    last = json.loads(pc.LAST_RUN.read_text())
    _, scr = quiet(pc.describe_last, dict(last, utm="EPSG:32636"))
    assert "  status      : finished, started " in scr and "finished, finished" not in scr
    assert "  majority    : yes, 4 neighbours\n" in scr
    assert "  projection  : WGS 84 / UTM zone 36N (EPSG:32636)" in scr
    _, scr = quiet(pc.describe_last, dict(last, answers=dict(last["answers"], threshold="half"), status="failed",
                                          progress="UserError: x"))
    assert "yes, 4 neighbours, half rule" in scr and "status      : failed, UserError: x" in scr


def test_pre_result_rows_same_basis():
    d = Path(tempfile.mkdtemp())
    r, l1 = small_raster(d)
    rows = {}
    for size in (0.0001, 0.0002):
        _, txt = quiet(pc.run, r, l1, d / "o.gpkg", min_acres=size, workers=2, max_memory_gb=1)
        lines = txt.splitlines()
        get = lambda k: float([x for x in lines if x.strip().startswith(k)][0].split()[-1].replace(",", ""))  # noqa: E731
        first, sl, no, written = (get(k) for k in ("Cane pieces inside L1", "Removed: slivers", "Removed: numerical",
                                                   "Written to the output"))
        assert abs(first - sl - no - written) <= 0.0011, txt
        total = float(re.search(r"Total area\s+:\s+([\d,.]+) acres", txt).group(1).replace(",", ""))
        assert total == written
        file_sum = sum(a for _, a in read_wkb(d / "o.gpkg"))
        assert ("adding up the area_ac column" in txt) == (abs(file_sum - total) > 0.001)
        rows[size] = first
    assert rows[0.0001] == rows[0.0002]


def test_pre_reuses_previous_work_for_a_new_output_name():
    d = Path(tempfile.mkdtemp())
    r, l1 = small_raster(d)
    _, txt1 = quiet(pc.run, r, l1, d / "o.gpkg", min_acres=0.0, workers=2, max_memory_gb=1)
    assert "tiles (filter" in txt1
    before = folder_state(d / "o_work")
    _, txt2 = quiet(pc.run, r, l1, d / "new name.shp", min_acres=0.0, workers=2, max_memory_gb=1)
    assert f"Reusing saved work from {(d / 'o_work').resolve()}" in txt2
    assert "tiles (filter" not in txt2 and "L1 prepared" not in txt2 and "merge seams" not in txt2
    assert read_wkb(d / "new name.shp") == read_wkb(d / "o.gpkg")
    assert folder_state(d / "o_work") == before and not (d / "o_work" / pc.LOCK).exists()
    _, txt3 = quiet(pc.run, r, l1, d / "new name.shp", min_acres=0.0, workers=2, max_memory_gb=1)
    assert "already up to date" in txt3 and txt3.count("up to date") == 1
    (d / "o_work" / pc.LOCK).write_text(json.dumps({"pid": os.getppid(), "started": "x"}))   # a live run uses it
    _, txt4 = quiet(pc.run, r, l1, d / "third.gpkg", min_acres=0.0, workers=2, max_memory_gb=1)
    assert "Reusing" not in txt4 and "tiles (filter" in txt4
    assert json.loads((d / "o_work" / pc.LOCK).read_text())["pid"] == os.getppid()
    (d / "o_work" / pc.LOCK).unlink()
    pc.clean_work(d / "o_work", remove_folder=True)                          # the earlier folder is gone
    pc.save_last_run({"raster": str(r), "l1": str(l1), "out": str(d / "o.gpkg"), "work": str(d / "o_work")})
    _, txt5 = quiet(pc.run, r, l1, d / "fourth.gpkg", min_acres=0.0, workers=2, max_memory_gb=1)
    assert "Reusing" not in txt5 and "tiles (filter" in txt5
    assert read_wkb(d / "fourth.gpkg") == read_wkb(d / "o.gpkg")


def test_pre_question_defaults_after_no_and_skip_when_inputs_gone():
    d = Path(tempfile.mkdtemp())
    r, l1 = small_raster(d)
    pc.save_last_run({"raster": str(r), "l1": str(l1), "out": str(d / "prev out.gpkg"), "utm": "EPSG:32631",
                      "work": str(d / "prev out_work"), "answers": {"do_filter": True, "neighbors": 4},
                      "status": "finished", "progress": "finished", "started": "x"})
    answers = ["n", "", "", "", "", "", "", "", "", "", "", "n"]
    _, txt = piped(pc.main, answers, ["--ask"])
    assert "A previous run was found" in txt and "Cancelled." in txt
    assert f"3) Output folder [{d}]" in txt and "Output name [prev out]" in txt and "(.gpkg)  [2]:" in txt
    pc.save_last_run({"raster": str(d / "gone.tif"), "l1": str(l1), "out": str(d / "prev out.gpkg"),
                      "utm": "EPSG:32631", "answers": {}, "status": "finished", "progress": "finished"})
    _, txt = piped(pc.main, [str(r), "", "", "", "", "", "", "", "", "", "n"], ["--ask"])
    assert "A previous run was found" not in txt and "Continue the last run" not in txt and "Cancelled." in txt


# ---------------------------------------------------------------- postprocess_remaining
CRS = "EPSG:32631"
X0, Y0 = 500000.0, 100000.0


def B(x0, y0, x1, y1):
    return shapely.box(X0 + x0, Y0 + y0, X0 + x1, Y0 + y1)


def post_case(d):
    rng = np.random.default_rng(11)
    l1 = [B(i * 120, j * 120, i * 120 + 110 + 15 * (i == 1), j * 120 + 110) for i in range(3) for j in range(3)]
    final = [B(x, y, x + w, y + h) for x, y, w, h in rng.uniform([0, 0, 20, 20], [300, 300, 90, 90], (14, 4))]
    for name, g in (("final.gpkg", final), ("l1.gpkg", l1)):
        pyogrio.raw.write(str(d / name), geometry=shapely.to_wkb(np.array(g, dtype=object)), field_data=[], fields=[],
                          geometry_type="Polygon", crs=CRS, driver="GPKG", layer="x")
    return d / "final.gpkg", d / "l1.gpkg"


def post_run(f, l1, out, **kw):
    kw.setdefault("workers", 2)
    kw.setdefault("max_memory_gb", 2)
    kw.setdefault("utm", "EPSG:32631")
    return quiet(ppr.run, f, l1, out, **kw)


def test_post_reuse_lines_and_storage():
    d = Path(tempfile.mkdtemp())
    f, l1 = post_case(d)
    _, txt1 = post_run(f, l1, d / "a.gpkg", sliver_acres=0.5)
    assert "Machine: " in txt1 and "\nPlan: " in txt1 and "machine:" not in txt1
    assert not [x.name for x in (d / "a_work").iterdir() if "_valid" in x.name]   # removed after stage 1
    before = folder_state(d / "a_work")
    _, txt2 = post_run(f, l1, d / "b.shp", sliver_acres=0.5)
    assert f"Reusing saved work from {(d / 'a_work').resolve()}" in txt2
    assert "stage 1 overlay" not in txt2 and "stage 2" not in txt2 and "Reused from" not in txt2
    assert read_wkb(d / "b.shp") == read_wkb(d / "a.gpkg")
    assert folder_state(d / "a_work") == before and not (d / "a_work" / ppr.LOCK).exists()
    _, txt3 = post_run(f, l1, d / "b.shp", sliver_acres=0.1)                  # needs stage 2 again: reads a_work/s1
    assert "Reusing saved work from" in txt3
    assert "stage 1 overlay" not in txt3 and "stage 2 sliver rule" in txt3
    _, txt4 = post_run(f, l1, d / "b.shp", sliver_acres=0.1)
    _, txt5 = post_run(f, l1, d / "b.shp", sliver_acres=0.3)                  # own saved classes, new setting
    assert "Setting changed: sliver size (0.1 -> 0.3)." in txt5
    _, txt4 = post_run(f, l1, d / "b.shp", sliver_acres=0.3)
    assert txt4.count("already up to date") == 1 and "reusing the saved" not in txt4 and "Reused from" not in txt4
    total, parts = time_parts(txt1)
    assert abs(sum(parts.values()) - total) <= 0.05 * len(parts) + 0.1 and "plan" in parts


def test_post_valid_inputs_recreated_on_resume():
    d = Path(tempfile.mkdtemp())
    f, l1 = post_case(d)
    _, _ = post_run(f, l1, d / "ref.gpkg", clean=True)
    os.environ["POSTPROCESS_TEST_STOP_AFTER"] = "1"
    try:
        post_run(f, l1, d / "o.gpkg", max_chunk_polygons=2)
        raise AssertionError("the test stop did not happen")
    except KeyboardInterrupt:
        pass
    finally:
        del os.environ["POSTPROCESS_TEST_STOP_AFTER"]
    (d / "o_work" / "l1_valid.gpkg").unlink()                                 # lost while stopped
    _, txt = post_run(f, l1, d / "o.gpkg", max_chunk_polygons=2)
    assert "inputs read" in txt and "resuming:" in txt
    assert not [x.name for x in (d / "o_work").iterdir() if "_valid" in x.name]
    assert read_wkb(d / "o.gpkg") == read_wkb(d / "ref.gpkg")


def test_post_layouts_identical():
    res = []
    for procs, threads in ((None, None), (1, 3), (2, 2)):
        d = Path(tempfile.mkdtemp())
        f, l1 = post_case(d)
        _, txt = post_run(f, l1, d / "o.gpkg", max_chunk_polygons=2, workers=4, processes=procs, threads=threads)
        if procs:
            assert f"Plan: {procs} process{'es' if procs > 1 else ''} x {threads} threads" in txt
        res.append(read_wkb(d / "o.gpkg"))
    assert res[0] == res[1] == res[2]
    gb = 1024 ** 3
    assert ppr.plan_layout(97, 80 * 1024 ** 2, 12 * gb, 12, 15) == (12, 1)
    p, t = ppr.plan_layout(276, 30 * 1024 ** 2, 1 * gb, 12, 15)
    assert p * t == 12 and 1 <= p < 12


def test_post_wording():
    st = {"acres81": 10.0, "acres82": 5.0, "l1_union_acres": 15.0 + 1e-9, "n81": 2, "n82": 1, "pieces": 3,
          "touching": 1, "absorbed": 1, "absorbed_acres": 0.1, "noise_overlay": 0, "noise": 0, "noise_overlay_m2": 0.0,
          "l1_overlap_groups": 1, "trimmed_82_pieces": 1}
    ans = {"rule": True, "sliver_acres": 0.2, "classes": "both"}
    _, txt = quiet(ppr.print_summary, st, {}, ans, area_crs=pc.zone_crs(31, True))
    assert "-0.000" not in txt and "Difference" in txt
    assert "1 harvested piece was cut back where it overlapped" in txt and "overlap each other in 1 place;" in txt
    _, txt = quiet(ppr.print_summary, dict(st, l1_overlap_groups=3, trimmed_82_pieces=2), {}, ans)
    assert "2 harvested pieces were cut back where they overlapped" in txt and "in 3 places;" in txt
    last = {"final": "f", "l1": "l", "out": "o", "utm": "EPSG:32636", "answers": {"rule": True, "sliver_acres": 0.2},
            "status": "finished", "progress": "finished", "started": "x"}
    _, scr = quiet(ppr.describe_last, last)
    assert "status      : finished, started x" in scr and "WGS 84 / UTM zone 36N (EPSG:32636)" in scr
    d = Path(tempfile.mkdtemp())
    f, l1 = post_case(d)
    ppr.save_last_run(dict(last, final=str(d / "gone.gpkg"), l1=str(l1), out=str(d / "p.gpkg")))
    _, txt = piped(ppr.main, [str(f), "", "", "", "", "", "", "", "", "n"], ["--ask"])
    assert "A previous run was found" not in txt and "Cancelled." in txt and "[final_harvest_final]" in txt


def test_pre_chain_of_new_names_computes_once():
    d = Path(tempfile.mkdtemp())
    r, l1 = small_raster(d)
    outs, txts = [d / "a.gpkg", d / "b.shp", d / "c.gpkg", d / "d.shp"], []
    for o in outs:
        txts.append(quiet(pc.run, r, l1, o, min_acres=0.0, workers=2, max_memory_gb=1)[1])
    assert "tiles (filter" in txts[0]
    for t in txts[1:]:
        assert f"Reusing saved work from {(d / 'a_work').resolve()}" in t and "tiles (filter" not in t, t
    assert all(read_wkb(o) == read_wkb(outs[0]) for o in outs[1:])
    assert json.loads(pc.LAST_RUN.read_text())["data_work"] == str((d / "a_work").resolve())


def test_post_chain_of_new_names_computes_once():
    d = Path(tempfile.mkdtemp())
    f, l1 = post_case(d)
    outs, txts = [d / "a.gpkg", d / "b.shp", d / "c.gpkg", d / "d.shp"], []
    for o in outs:
        txts.append(post_run(f, l1, o)[1])
    assert "stage 1 overlay" in txts[0]
    for t in txts[1:]:
        assert f"Reusing saved work from {(d / 'a_work').resolve()}" in t, t
        assert "stage 1 overlay" not in t and "stage 2" not in t, t
    assert all(read_wkb(o) == read_wkb(outs[0]) for o in outs[1:])
    _, t5 = post_run(f, l1, d / "e.gpkg", sliver_acres=0.05)                 # new setting: only stage 2
    _, t6 = post_run(f, l1, d / "f.gpkg", sliver_acres=0.05)                 # its classes are reused too
    assert "stage 1 overlay" not in t5 and "stage 2 sliver rule" in t5
    assert "stage 1 overlay" not in t6 and "stage 2" not in t6 and "Reusing saved work" in t6
    assert read_wkb(d / "f.gpkg") == read_wkb(d / "e.gpkg")


# ---------------------------------------------------------------- same result for every tile size (seam vertices)
def _whole_reference(mask, tf, l1_geoms, utm):
    """No tiles, no blocks: polygonise the whole mask, union, intersect union(L1), explode, areas."""
    from rasterio import features
    polys = [shapely.geometry.shape(g) for g, _ in features.shapes(mask.view(np.uint8), mask=mask, transform=tf,
                                                                   connectivity=4)]
    cu = shapely.union_all(np.array(polys, dtype=object))
    lu = shapely.union_all(np.array(l1_geoms, dtype=object))
    parts = pc.polys_only(np.array([shapely.intersection(cu, lu)], dtype=object))
    ac = pc.area_m2(parts, "EPSG:4326", utm) / pc.ACRE_M2
    keep = ac * pc.ACRE_M2 >= pc.NOISE_M2
    return sorted(zip(shapely.to_wkb(shapely.normalize(parts[keep])).tolist(), np.round(ac[keep], 3).tolist()))


def test_pre_every_tile_size_gives_the_whole_array_result():
    """Long straight edges in degrees on a coarse grid: extra vertices left on block/tile seams would
    change the projected area (and did, by up to 0.1 acre). L1 edges on cell lines, a cane corner on
    an L1 edge exactly on a block seam (a real vertex: must stay), cane crossing seams everywhere."""
    d = Path(tempfile.mkdtemp())
    px = 0.003                                   # about 330 m cells: long edges, visible curvature in UTM
    tf = from_origin(4.0, 1.0, px, px)
    a = np.full((60, 70), 4, np.uint8)
    a[3:57, 5:66] = 1                            # big block crossing every seam
    a[20:40, 30:50] = 4                          # hole
    a[16:26, 0:12] = 1                           # touches the L1 edge (column 10) with corners on rows 16 / 26
    a[44:52, 60:70] = 4
    with rasterio.open(d / "r.tif", "w", driver="GTiff", height=60, width=70, count=1, dtype="uint8",
                       crs="EPSG:4326", transform=tf) as o:
        o.write(a, 1)
    cell = lambda r0, c0, r1, c1: shapely.box(4.0 + c0 * px, 1.0 - r1 * px, 4.0 + c1 * px, 1.0 - r0 * px)  # noqa: E731
    l1 = [cell(5, 10, 50, 64), cell(0, 40, 30, 70)]
    pyogrio.raw.write(str(d / "l1.gpkg"), geometry=shapely.to_wkb(np.array(l1, dtype=object)), field_data=[], fields=[],
                      geometry_type="Polygon", crs="EPSG:4326", driver="GPKG", layer="l1")
    utm = pc.zone_crs(31, True)
    for do_filter in (False, True):
        mask = pc.majority_cane(a, None, 1, 1) if do_filter else (a == 1)
        ref = _whole_reference(mask, tf, l1, utm)
        old_block, pc.BLOCK = pc.BLOCK, 16      # block seams on rows / columns 16, 32, 48
        os.environ["PREPROCESS_TEST_BLOCK"] = "16"
        try:
            for tile, lay in ((7, (1, 1)), (13, (1, 3)), (16, (2, 2)), (33, (1, 1)), (1000, (1, 1))):
                out = d / f"o_{do_filter}_{tile}.gpkg"
                quiet(pc.run, d / "r.tif", d / "l1.gpkg", out, min_acres=0.0, keep_slivers=True, do_filter=do_filter,
                      tile_size=tile, processes=lay[0], threads=lay[1], max_memory_gb=1, utm="31N", clean=True)
                got = read_wkb(out)
                assert got == ref, (do_filter, tile, len(got), len(ref))
        finally:
            pc.BLOCK = old_block
            os.environ.pop("PREPROCESS_TEST_BLOCK")


def test_pre_signature_notices_same_size_same_time_new_content():
    d = Path(tempfile.mkdtemp())
    r, l1 = small_raster(d)
    base = dict(cane_value=1, do_filter=True, neighbors=4, threshold="majority", keep_slivers=False, min_acres=0.2)
    t0, _ = pc.stage_keys(r, l1, d / "o.shp", pc.zone_crs(31, True), base)
    st = os.stat(r)
    with rasterio.open(r, "r+") as o:            # same size (uncompressed), other cells
        x = o.read(1)
        x[:5] = 4 if x[0, 0] == 1 else 1
        o.write(x, 1)
    os.utime(r, ns=(st.st_atime_ns, st.st_mtime_ns))
    assert os.stat(r).st_size == st.st_size and os.stat(r).st_mtime_ns == st.st_mtime_ns
    t1, _ = pc.stage_keys(r, l1, d / "o.shp", pc.zone_crs(31, True), base)
    assert pc._hash(t0) != pc._hash(t1)
    assert "has other content than at the last run" in pc.explain_changes(t0, t1)[1][0]


# ---------------------------------------------------------------- wording and safety questions
def test_post_alone_in_a_folder_says_what_is_missing():
    import subprocess
    d = Path(tempfile.mkdtemp())
    src = Path(__file__).resolve().parents[1] / "src" / "postprocess_remaining.py"
    (d / "postprocess_remaining.py").write_bytes(src.read_bytes())
    p = subprocess.run([sys.executable, "-I", str(d / "postprocess_remaining.py"), "--help"], capture_output=True, text=True)
    assert p.returncode == 1 and "Traceback" not in p.stderr
    assert "Put preprocess_classified.py in the same folder as this file." in p.stderr


def test_changed_shapefile_is_one_line_per_layer():
    d = Path(tempfile.mkdtemp())
    for ext in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
        (d / ("f" + ext)).write_bytes(b"x")
    t0 = pc._file_sig(d / "f.shp", "l1")
    time.sleep(0.01)
    for ext in (".shp", ".shx", ".dbf"):
        (d / ("f" + ext)).write_bytes(b"yy")
    t1 = pc._file_sig(d / "f.shp", "l1")
    older, why = pc.explain_changes({"version": 1, "final": t0}, {"version": 1, "final": t1}, ppr.KEY_NAMES)
    assert not older and len(why) == 1 and why[0].startswith("The FINAL file f.shp was modified since the last run")
    assert "changed .dbf .shp .shx" in why[0]


def test_question_details():
    d = Path(tempfile.mkdtemp())
    r, l1 = small_raster(d)
    answers = [str(r), str(l1), str(d), "o", "2", "", "", "", "", "9", "n", "7", "y", "n"]
    _, txt = piped(pc.interactive, answers, dict(utm="EPSG:32632"))
    assert "   Enter keeps your earlier choice: UTM 32N (EPSG:32632)" in txt
    assert "That is large: pieces smaller than 9 acres will be removed. Use it? [y/N]" in txt
    assert "   Type a smaller size." in txt and "slivers     : remove below 7.0 acres" in txt
    _, txt = piped(pc.ask_output, [str(d), "bad:name", "ok", "1"], d, "x", {})
    assert "[x]:    A file name cannot contain :" in txt          # hint indented like the prompts (3)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        pc.print_result_block("RESULT", [("Files", [("  Output : " + "/very/long" * 12 + "/out.gpkg",)])])
    lines = out.getvalue().splitlines()
    assert all(len(x) <= 80 for x in lines) and len(lines[0]) == len(lines[-1]) == 80
    assert lines[3].startswith(" " * 11 + "very") or lines[3].startswith(" " * 11)


def test_late_setting_change_lines():
    d = Path(tempfile.mkdtemp())
    r, l1 = small_raster(d)
    quiet(pc.run, r, l1, d / "o.gpkg", min_acres=0.2, workers=2, max_memory_gb=1)
    _, txt = quiet(pc.run, r, l1, d / "o.gpkg", min_acres=0.5, workers=2, max_memory_gb=1)
    assert "Setting changed: sliver size (0.2 -> 0.5). Only the last step runs again." in txt
    f, lp = post_case(d)
    post_run(f, lp, d / "p.gpkg")
    _, txt = post_run(f, lp, d / "p.gpkg", sliver_acres=0.5)
    assert "Setting changed: sliver size (0.2 -> 0.5). Only the last steps run again" in txt
    _, txt = post_run(f, lp, d / "p.gpkg", sliver_acres=0.5, classes="82")
    assert "Setting changed: classes (both -> 82)" in txt and "Only the last step runs again." in txt


def test_result_block_wraps_each_line_once():
    paths = [r"C:\Users\analyst.name\Documents\Harvest Monitoring\Area 1\2026\ImageClassifier_remaining_cane",
             r"C:\Users\a\x_work", "D:\\" + "\\".join(["very long folder name"] * 6) + "\\out"]
    for p in paths:
        rows = [("How the result was reached", "Polygons", "Acres"), ("  Written to the output", "28,246", "28,731.028")]
        rows += pc.wrap_rows("Note: adding up the area_ac column of the file gives 28,731.001 acres (each polygon is "
                             "rounded to 3 decimals).")
        files = [(f"  Output : {p}.shp   (56,703 polygons)",)]
        files += pc.cache_rows("Cache ", p + "_work", "72.5 MB", "A rerun that only changes the sliver size, the "
                               "projection or the output name reuses it and takes seconds. Safe to delete.")
        tl = pc.time_line({"plan": 1.0, "prepare L1": 4.6, "tiles": 11.3, "merge": 1.4, "write": 2.1}, 20.4,
                          (("plan", "plan"), ("prepare L1", "prepare L1"), ("tiles", "tiles"), ("merge", "merge"),
                           ("write", "write")))
        sections = [("", rows), ("Files", files), ("", [(tl,)])]
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            pc.print_result_block("RESULT", sections)
        lines = out.getvalue().splitlines()
        assert all(len(x) <= 80 for x in lines), [x for x in lines if len(x) > 80]
        assert len(lines[0]) == len(lines[-1]) == 80
        for _, rs in sections:                       # every logical line: wrapped once, no lonely words
            for r in rs:
                if len(r) != 1:
                    continue
                parts = pc.wrap_line(r[0])
                assert "".join(x.strip() for x in parts).replace(" ", "") == r[0].replace(" ", ""), parts
                for k, x in enumerate(parts[1:-1], 1):
                    assert len(x.split()) >= 3 or "\\" in x, (k, parts)
                if len(parts) > 1 and "|" in r[0]:
                    assert all(x.rstrip().endswith("|") for x in parts[:-1]), parts
        assert any("write 2.1)" in x for x in lines)
        assert "(72.5 MB)" in out.getvalue()


def test_output_defaults_follow_the_input():
    d = Path(tempfile.mkdtemp())
    (d / "old").mkdir()
    (d / "new").mkdir()
    r1, l1 = small_raster(d / "old")
    r2, _ = small_raster(d / "new")
    prev = {"raster": str(r1), "out": str(d / "old" / "prev_out.gpkg"), "utm": "EPSG:32631"}
    ans = [str(r2), str(l1), "", "", "", "", "", "", "", "", "", "n"]
    _, txt = piped(pc.interactive, ans, dict(prev, l1=str(l1)))
    assert f"3) Output folder [{d / 'new'}]" in txt and "Output name [r_remaining_cane]" in txt and "(.gpkg)  [2]:" in txt
    _, txt = piped(pc.interactive, [str(r1)] + ans[1:], dict(prev, l1=str(l1)))
    assert f"3) Output folder [{d / 'old'}]" in txt and "Output name [prev_out]" in txt
    f1, lp = post_case(d / "old")
    f2, _ = post_case(d / "new")
    prev = {"final": str(f1), "l1": str(lp), "out": str(d / "old" / "test_output_harvest_final_2.shp"), "utm": "EPSG:32631"}
    ans = [str(f2), str(lp), "", "", "", "", "", "", "", "", "n"]
    _, txt = piped(ppr.interactive, ans, prev)
    assert f"3) Output folder [{d / 'new'}]" in txt and "Output name [final_harvest_final]" in txt and "(.gpkg)  [1]:" in txt
    _, txt = piped(ppr.interactive, [str(f1)] + ans[1:], prev)
    assert f"3) Output folder [{d / 'old'}]" in txt and "Output name [test_output_harvest_final_2]" in txt
