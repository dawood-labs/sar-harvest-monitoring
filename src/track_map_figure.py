"""Map figure for non-specialists: every Sentinel-1 track over the area, how many tracks see each place, when.

Left   how many tracks see each place (1-4), with the area outline and the cane
Right  one small map per track: its strip, flight direction, local time, repeat, share of the cane it sees
Footprints and times come from Earth Engine image metadata (no pixels, no cost).

Usage: python src/track_map_figure.py configs/config.yaml
Output: reports/figures/<aoi>_s1_track_map.png
"""
from __future__ import annotations

import glob
import sys
import warnings
from datetime import datetime, timedelta

import ee
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import shapely
import yaml
from matplotlib.colors import ListedColormap
from matplotlib.patches import Polygon as MplPolygon
from pyproj import Transformer

from s1_passes import init_ee

warnings.filterwarnings("ignore")
SURF, PAGE, INK, INK2, MUTED, GRID = "#1a1a19", "#0d0d0d", "#f0efec", "#c3c2b7", "#898781", "#2c2c2a"
COL = {164: "#3987e5", 91: "#199e70", 26: "#d95926", 99: "#c98500"}


def main(cfg_path: str) -> None:
    cfg = yaml.safe_load(open(cfg_path))
    aoi = cfg["aoi"]
    init_ee(cfg["gee"])
    passes = pd.read_csv(sorted(glob.glob(f"data/interim/{aoi}_s1_passes_*.csv"))[-1], parse_dates=["date"])
    area = gpd.read_file(f"data/interim/{aoi.split('_')[0]}_full.gpkg").to_crs(4326)
    with rasterio.open(f"data/interim/{aoi}_cane_mask_full.tif") as m:
        a = m.read(1, out_shape=(m.height // 20, m.width // 20)) == 1
        r, c = np.nonzero(a)
        x, y = rasterio.transform.xy(m.transform * m.transform.scale(20, 20), r, c)
        lon, lat = Transformer.from_crs(m.crs, "EPSG:4326", always_xy=True).transform(np.array(x), np.array(y))
    x0, y0, x1, y1 = area.total_bounds
    pad = 0.6
    view = (x0 - pad, y0 - pad, x1 + pad, y1 + pad)
    box = ee.Geometry.Rectangle(list(view))

    tracks = {}
    for t in (164, 91, 26, 99):
        s = passes[(passes.track == t) & (passes.satellite == "S1A")]
        day = s.date.max().strftime("%Y-%m-%d")
        col = (ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(box).filterDate(day, ee.Date(day).advance(1, "day"))
               .filter(ee.Filter.eq("relativeOrbitNumber_start", t)))
        g = shapely.geometry.shape(col.geometry().dissolve(1000).getInfo())
        when = datetime.utcfromtimestamp(col.aggregate_min("system:time_start").getInfo() / 1000) + timedelta(hours=7)
        g_all = passes[passes.track == t]
        full = g_all[g_all.cane_share >= 0.5].date.sort_values().diff().dt.days.dropna()
        tracks[t] = {"geom": g, "dir": "descending" if s["pass"].iloc[0] == "DES" else "ascending", "time": when.strftime("%H:%M"),
                     "repeat": (lambda v: f"{int(v.min())}" if v.min() == v.max() else f"{int(v.min())}-{int(v.max())}")(
                         full if len(full) else g_all.date.sort_values().diff().dt.days.dropna()),
                     "share": g_all.cane_share.max(), "sats": "S1A + S1C" if (g_all[g_all.cane_share >= g_all.cane_share.max() - 0.01].satellite.nunique() > 1) else "S1A"}
        print(t, tracks[t]["dir"], tracks[t]["time"], tracks[t]["repeat"], tracks[t]["sats"], round(tracks[t]["share"], 2))

    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.edgecolor": "#383835", "font.size": 10})
    fig = plt.figure(figsize=(17, 10), facecolor=PAGE)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.45, 1, 1], hspace=0.3, wspace=0.18)
    aspect = 1 / np.cos(np.radians((y0 + y1) / 2))

    def base(ax):
        ax.set_facecolor(SURF)
        ax.set_xlim(view[0], view[2]); ax.set_ylim(view[1], view[3]); ax.set_aspect(aspect)
        ax.grid(color=GRID, lw=0.5)
        ax.scatter(lon, lat, s=1.5, color="#d8d6cf", lw=0, zorder=4)
        for geom in area.geometry:
            ax.plot(*geom.exterior.xy if geom.geom_type == "Polygon" else geom.convex_hull.exterior.xy, color=INK, lw=1.4, ls="--", zorder=5)

    # left: how many tracks see each place
    ax = fig.add_subplot(gs[:, 0]); base(ax)
    gx, gy = np.meshgrid(np.linspace(view[0], view[2], 360), np.linspace(view[1], view[3], 300))
    pts = shapely.points(gx.ravel(), gy.ravel())
    n = sum(shapely.contains(tr["geom"], pts).astype(int) for tr in tracks.values()).reshape(gx.shape)
    cmap = ListedColormap(["#1a1a19", "#1d3557", "#255f9c", "#3987e5", "#9cc3f5"])
    ax.imshow(n, origin="lower", extent=(view[0], view[2], view[1], view[3]), cmap=cmap, vmin=-0.5, vmax=4.5, zorder=1, alpha=0.95, interpolation="nearest")
    for k in (1, 2, 3, 4):
        ax.scatter([], [], marker="s", s=90, color=cmap(k), label=f"{k} track{'s' if k > 1 else ''}")
    ax.scatter([], [], s=12, color="#d8d6cf", label="sugarcane fields")
    ax.plot([], [], color=INK, ls="--", lw=1.4, label="area of interest")
    ax.legend(loc="lower left", fontsize=10, facecolor=SURF, edgecolor="#383835", labelcolor=INK2, title="tracks seeing the place",
              title_fontsize=10)
    ax.set_title("How many radar tracks see each place", loc="left", fontsize=14, fontweight="bold", pad=10)
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    covered = [shapely.contains(tr["geom"], shapely.points(lon, lat)) for tr in tracks.values()]
    k = np.sum(covered, axis=0)
    ax.text(0.01, 0.99, "Of the sugarcane:\n" + "\n".join(f"  seen by {i} tracks: {np.mean(k == i):.0%}" for i in (1, 2, 3, 4) if np.mean(k == i) > 0.005),
            transform=ax.transAxes, va="top", fontsize=10, color=INK, bbox=dict(fc=SURF, ec="#383835", alpha=0.9))

    # right: one map per track
    for i, t in enumerate((164, 26, 91, 99)):
        tr = tracks[t]
        ax = fig.add_subplot(gs[i // 2, 1 + i % 2]); base(ax)
        for poly in (tr["geom"].geoms if tr["geom"].geom_type == "MultiPolygon" else [tr["geom"]]):
            xy = np.array(poly.exterior.coords)
            ax.add_patch(MplPolygon(xy, closed=True, fc=COL[t], alpha=0.35, lw=0, zorder=2))
            ax.add_patch(MplPolygon(xy, closed=True, fill=False, ec=COL[t], lw=2, zorder=3))
        up = tr["dir"] == "ascending"
        cx = view[2] - 0.45 if not up else view[0] + 0.45
        ax.annotate("", xy=(cx + (-0.12 if up else 0.12) * 0, view[3] - 0.35 if up else view[1] + 0.35),
                    xytext=(cx, view[1] + 0.35 if up else view[3] - 0.35),
                    arrowprops=dict(arrowstyle="-|>", color=INK, lw=2.2, mutation_scale=22), zorder=6)
        ax.text(cx + (0.12 if up else -0.12), (view[1] + view[3]) / 2, "flies north" if up else "flies south", rotation=90,
                ha="left" if up else "right", va="center", fontsize=9, color=INK, zorder=6,
                bbox=dict(fc=SURF, ec="none", alpha=0.8, pad=1))
        ax.set_title(f"Track {t:03d}: {tr['dir']}, {tr['time']}\nevery {tr['repeat']} days, sees {tr['share']:.0%} of the cane",
                     loc="left", fontsize=11, fontweight="bold", color=INK, pad=6)
        ax.set_xticklabels([]); ax.set_yticklabels([])
    fig.suptitle("Sentinel-1 radar tracks over the area of interest", x=0.05, ha="left", fontsize=17, fontweight="bold", y=0.975)
    fig.text(0.05, 0.035, "Descending = the satellite flies from north to south in the early morning and looks to the west.   "
             "Ascending = it flies from south to north in the evening and looks to the east.\n"
             "Each track is a fixed strip about 250 km wide; the same track always comes back over the same strip at the same time of day (times = local time).   "
             "Coloured strip = what that track sees.", fontsize=10, color=INK2)
    out = f"reports/figures/{aoi}_s1_track_map.png"
    fig.savefig(out, dpi=150, facecolor=PAGE, bbox_inches="tight")
    print("written", out)


if __name__ == "__main__":
    main(*sys.argv[1:2])
