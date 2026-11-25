"""Look closely at one or a few places of a large area: why does the SAR report say what it says there?

Prints, for the neighbourhood of the points, the crop mask, the model probability, the SAR class and the
original report as small pixel grids (the points are in brackets), and per point: probability on every report
date, the original report codes, raw VH of every pass of each track (also after the report) with the 5 x 5\nmean the model sees, and NDVI of every
optical image. Nothing is written.

Usage: python src/inspect_points.py <report config.yaml> <report date> <lat,lon> [<lat,lon> ...]
"""
from __future__ import annotations

import glob
import sys
import warnings

import numpy as np
import rasterio
import yaml
from pyproj import Transformer
from rasterio.windows import Window

warnings.filterwarnings("ignore")


def main(cfg_path: str, report: str, *points: str) -> None:
    cfg = yaml.safe_load(open(cfg_path))
    aoi, interim, area = cfg["aoi"], cfg["paths"]["interim"] + "/", "full"
    out = f"outputs/{aoi}_{area}{cfg.get('output_tag', '')}/{aoi}_{area}_"
    with rasterio.open(interim + f"{aoi}_cane_mask_{area}.tif") as m:
        tr = Transformer.from_crs("EPSG:4326", m.crs, always_xy=True)
        rc = [m.index(*tr.transform(float(p.split(",")[1]), float(p.split(",")[0]))) for p in points]
        r0, c0 = min(r for r, _ in rc) - 6, min(c for _, c in rc) - 6
        win = Window(c0, r0, max(c for _, c in rc) - c0 + 7, max(r for r, _ in rc) - r0 + 7)
        mask = m.read(1, window=win)
    if len(rc) == 2:
        print("distance between the points:", round(10 * float(np.hypot(rc[0][0] - rc[1][0], rc[0][1] - rc[1][1]))), "m")

    def grid(name: str, f: str, fmt: str = "%3d") -> None:
        with rasterio.open(f) as s:
            a = s.read(1, window=win)
        print(name)
        for i, row in enumerate(a):
            print("   " + " ".join(("[" if (i + r0, j + c0) in rc else " ") + fmt % v + ("]" if (i + r0, j + c0) in rc else " ") for j, v in enumerate(row)))

    grid(f"probability {report} (0-100, 255 = outside the crop mask)", out + f"sar_probability_{report}_v1.tif")
    grid(f"SAR class {report}", out + f"sar_harvest_{report}_v1.tif", "%2d")
    last = sorted(str(d) for d in cfg["original_reports"])[cfg["report_dates"].index(report)]
    grid(f"original report {last}", interim + f"{aoi}_original_{area}_{last}.tif")
    for k, (r, c) in enumerate(rc):
        w1 = Window(c, r, 1, 1)

        def one(f: str, b: int = 1):
            with rasterio.open(f) as s:
                return s.read(b, window=w1)[0, 0]

        print(f"\nPOINT {'ABCDEFGH'[k]} ({points[k]}): probability by report date", {d: int(one(out + f"sar_probability_{d}_v1.tif")) for d in cfg["report_dates"]})
        print("   original codes by report:", {f[-14:-4][5:]: int(one(f)) for f in sorted(glob.glob(interim + f"{aoi}_original_{area}_*.tif"))})
        for t, direction in cfg["s1_tracks"].items():
            fs = sorted(glob.glob(f"{cfg['paths']['raw']}/s1/{aoi}_s1_ro{int(t):03d}_{direction[:3].lower()}_*.tif"))
            print(f"   VH track {int(t):03d}:", " ".join(f"{f[-9:-4]}:{'  nan' if x == -32768 else '%5.1f' % (x / 100)}" for f, x in ((f, one(f, 2)) for f in fs)))
            w5 = Window(c - 2, r - 2, 5, 5)                       # what the model sees: mean of the 5 x 5 crop-mask pixels around
            with rasterio.open(interim + f"{aoi}_cane_mask_{area}.tif") as m5:
                in5 = m5.read(1, window=w5) == 1
            vals = []
            for f in fs:
                with rasterio.open(f) as s5:
                    v = s5.read(2, window=w5).astype("float32")
                ok = in5 & (v != -32768)
                vals.append(f"{f[-9:-4]}:{10 * np.log10(np.mean(10 ** (v[ok] / 1000))):5.1f}" if ok.any() else f"{f[-9:-4]}:  nan")
            print(f"   5x5 mean ({int(in5.sum())} of 25 pixels are cane):", " ".join(vals))
        nd = []
        for f in sorted(glob.glob(f"{cfg['paths']['raw']}/s2/*.tif")):
            with rasterio.open(f) as s:
                b = s.read([3, 5, 6], window=w1)[:, 0, 0].astype(float)
            nd.append(f"{f[-9:-4]}:" + (f"{(b[1] - b[0]) / (b[1] + b[0] + 1e-6):.2f}" if b[2] >= 60 and b[1] > 0 else "no image"))
        print("   NDVI:", " ".join(nd))
        with rasterio.open(interim + f"{aoi}_patch_acres_{area}.tif") as s:
            print("   cane patch size:", round(float(s.read(1, window=w1)[0, 0]), 2), "acres")


if __name__ == "__main__":
    main(*sys.argv[1:])
