"""Figure for non-specialists: which Sentinel-1 tracks pass over an area, where, and how often (dark theme).

Panel A  map: cane coloured by the main tracks that see it (both / only one), with the strip edges of the main tracks
Panel B  calendar: every pass of every track in the season, per satellite, with the report dates
Panel C  how often a window of N days holds images of at least two different tracks (both areas measured)
Reads image metadata and footprints from Earth Engine (no pixels, no cost) and the pass lists in data/interim.

Usage: python src/track_figure.py configs/config.yaml
Output: reports/figures/<aoi>_s1_tracks.png
"""
from __future__ import annotations

import glob
import sys
import warnings

import ee
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import yaml
from matplotlib.patches import Polygon as MplPolygon
from pyproj import Transformer

from s1_passes import init_ee

warnings.filterwarnings("ignore")
SURF, PAGE, INK, INK2, MUTED, GRID = "#1a1a19", "#0d0d0d", "#f0efec", "#c3c2b7", "#898781", "#2c2c2a"
COL = {164: "#3987e5", 26: "#d95926", 91: "#199e70", 99: "#c98500"}
NAME = {164: "track 164 (morning)", 26: "track 026 (evening)", 91: "track 091 (morning)", 99: "track 099 (evening)"}


def main(cfg_path: str) -> None:
    cfg = yaml.safe_load(open(cfg_path))
    aoi = cfg["aoi"]
    init_ee(cfg["gee"])
    passes = pd.read_csv(sorted(glob.glob(f"data/interim/{aoi}_s1_passes_*.csv"))[-1], parse_dates=["date"])
    with rasterio.open(f"data/interim/{aoi}_cane_mask_full.tif") as m:
        a = m.read(1, out_shape=(m.height // 20, m.width // 20)) == 1
        r, c = np.nonzero(a)
        x, y = rasterio.transform.xy(m.transform * m.transform.scale(20, 20), r, c)
        lon, lat = Transformer.from_crs(m.crs, "EPSG:4326", always_xy=True).transform(np.array(x), np.array(y))
    box = ee.Geometry.Rectangle([float(lon.min()) - 0.3, float(lat.min()) - 0.3, float(lon.max()) + 0.3, float(lat.max()) + 0.3])

    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.edgecolor": "#383835", "font.size": 10})
    fig = plt.figure(figsize=(16, 9.2), facecolor=PAGE)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.3], height_ratios=[1.1, 1], hspace=0.45, wspace=0.22)
    ax_map, ax_cal, ax_win = fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])
    for ax in (ax_map, ax_cal, ax_win):
        ax.set_facecolor(SURF)

    # A: which tracks see each piece of cane (from the downloaded images), plus the swath edges
    def covered(track: int, direction: str) -> np.ndarray:
        best = None                                     # the pass of this track that covers the most cane
        for f in sorted(glob.glob(f"{cfg['paths']['raw']}/s1/{aoi}_s1_ro{track:03d}_{direction}_*.tif")):
            with rasterio.open(f) as s1:
                v = s1.read(2, out_shape=(s1.height // 20, s1.width // 20))[r, c] != s1.nodata
            if best is None or v.sum() > best.sum():
                best = v
        return best
    c164, c26 = covered(164, "des"), covered(26, "asc")
    groups = [(c164 & c26, "#8a8983", "both main tracks (164 + 026)"),
              (~c164 & c26, COL[91], "no 164 here -> track 091 fills in"),
              (c164 & ~c26, COL[99], "no 026 here -> track 099 fills in")]
    for sel, col, lab in groups:
        ax_map.scatter(lon[sel], lat[sel], s=3, color=col, lw=0, zorder=2, label=f"{lab} ({sel.mean():.0%} of cane)")
    for t in (164, 26):
        day = passes[(passes.track == t) & (passes.satellite == "S1A")].date.max().strftime("%Y-%m-%d")
        g = (ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(box).filterDate(day, ee.Date(day).advance(1, "day"))
             .filter(ee.Filter.eq("relativeOrbitNumber_start", t)).geometry().dissolve(1000).intersection(box, 1000).getInfo())
        polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        for k, poly in enumerate(polys):
            ax_map.add_patch(MplPolygon(np.array(poly[0]), closed=True, fill=False, ec=COL[t], lw=2.2, zorder=3,
                                        label=f"edge of {NAME[t].split(' (')[0]}'s strip" if k == 0 else None))
    ax_map.set_xlim(lon.min() - 0.15, lon.max() + 0.15)
    ax_map.set_ylim(lat.min() - 0.15, lat.max() + 0.15)
    ax_map.set_aspect(1 / np.cos(np.radians(lat.mean())))
    ax_map.set_xlabel("longitude"); ax_map.set_ylabel("latitude")
    ax_map.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=9, facecolor=SURF, edgecolor="#383835", labelcolor=INK2,
                  markerscale=3, ncol=1)
    ax_map.set_title("A. Which tracks see each piece of cane", loc="left", fontsize=13, fontweight="bold", pad=10)
    ax_map.grid(color=GRID, lw=0.6)
    ax_map.text(0.01, 0.995, "Each track images a strip ~250 km wide.\nThe area is wider than the overlap of the two main strips,\nso its edges need a neighbouring track.",
                transform=ax_map.transAxes, va="top", fontsize=9, color=INK2)

    # B: calendar of passes
    order = [164, 26, 91, 99]
    for i, t in enumerate(order):
        g = passes[passes.track == t]
        ax_cal.axhline(i, color=GRID, lw=0.8, zorder=0)
        for sat, mk in (("S1A", "o"), ("S1C", "D")):
            s = g[g.satellite == sat]
            ax_cal.scatter(s.date, [i] * len(s), marker=mk, s=70, color=COL[t], edgecolor=SURF, lw=1.5, zorder=3,
                           alpha=1.0 if s.cane_share.median() >= 0.5 or t in (91, 99) else 0.35)
        share = g.cane_share.max()
        ax_cal.text(passes.date.max() + pd.Timedelta(days=1.5), i, f"covers {share:.0%} of the cane", va="center", fontsize=9, color=INK2)
    for j, r in enumerate(cfg["report_dates"]):
        ax_cal.axvline(pd.Timestamp(r), color=INK2, ls=":", lw=1.2, zorder=1)
        ax_cal.text(pd.Timestamp(r) + pd.Timedelta(days=0.6 if j == 2 else -0.6), len(order) - 0.35 + (0.25 if j == 2 else 0),
                    "report " + pd.Timestamp(r).strftime("%d %b"), ha="left" if j == 2 else "right", fontsize=8, color=INK2)
    ax_cal.set_yticks(range(len(order)), [NAME[t] for t in order], fontsize=9)
    ax_cal.set_ylim(-0.6, len(order) + 0.15)
    ax_cal.set_xlim(passes.date.min() - pd.Timedelta(days=2), passes.date.max() + pd.Timedelta(days=14))
    ax_cal.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%d %b"))
    ax_cal.invert_yaxis()
    ax_cal.scatter([], [], marker="o", s=50, color=INK2, label="satellite S1A")
    ax_cal.scatter([], [], marker="D", s=50, color=INK2, label="satellite S1C")
    ax_cal.legend(loc="lower left", fontsize=9, facecolor=SURF, edgecolor="#383835", labelcolor=INK2, ncol=2)
    ax_cal.set_title("B. When each track takes a picture (one dot = one radar image)", loc="left", fontsize=13, fontweight="bold", pad=10)
    ax_cal.text(0, -0.13, "Track 164: every 6 days (two satellites take turns).  Track 026: every 12 days on the full area (only S1A covers it;\n"
                "faded diamonds = S1C images that cover only 13 % of the cane).", transform=ax_cal.transAxes, fontsize=9, color=INK2, va="top")

    # C: windows
    rows = []
    for f in sorted(glob.glob("data/interim/*_s1_passes_*.csv")):
        d = pd.read_csv(f, parse_dates=["date"])
        big = d[d.cane_share >= 0.5].drop_duplicates(["date", "track"])
        for n in range(4, 16):
            days = pd.date_range(big.date.min() + pd.Timedelta(days=n - 1), big.date.max())
            k = [big[(big.date > e - pd.Timedelta(days=n)) & (big.date <= e)].track.nunique() for e in days]
            rows.append({"area": "area 2 (this area)" if aoi in f else "area 1", "days": n, "share": 100 * np.mean(np.array(k) >= 2)})
    w = pd.DataFrame(rows)
    for (area, g), col in zip(w.groupby("area", sort=False), ("#9085e9", "#3987e5")):
        ax_win.plot(g.days, g.share, color=col, lw=2.2, marker="o", ms=7, label=area, zorder=3)
    ax_win.axvline(12, color=INK2, ls=":", lw=1.2)
    ax_win.text(12.15, 50, "12 days:\nalways\n2 tracks", fontsize=9, color=INK2)
    ax_win.axvline(7, color=MUTED, ls=":", lw=1)
    ax_win.text(7.15, 8, "one week", fontsize=9, color=MUTED)
    ax_win.set_xlabel("length of the window (days)"); ax_win.set_ylabel("% of windows with images\nof 2 different tracks")
    ax_win.set_ylim(0, 105); ax_win.set_xticks(range(4, 16))
    ax_win.grid(color=GRID, lw=0.6)
    ax_win.legend(loc="upper left", fontsize=9, facecolor=SURF, edgecolor="#383835", labelcolor=INK2)
    ax_win.set_title("C. How long until we have images from 2 different tracks?", loc="left", fontsize=13, fontweight="bold", pad=10)
    fig.suptitle("Sentinel-1 radar over the cane area: tracks, coverage and timing", x=0.06, ha="left", fontsize=16, fontweight="bold", y=0.96)
    out = f"reports/figures/{aoi}_s1_tracks.png"
    fig.savefig(out, dpi=150, facecolor=PAGE, bbox_inches="tight")
    print("written", out)


if __name__ == "__main__":
    main(*sys.argv[1:2])
