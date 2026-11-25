"""EDA test of two ideas on the pixel table (eda_extract.py), before any model:

  (a) What should a pixel be compared with?  its own season start (the old way)  /  its own previous passes
      /  the standing cane around it at the same pass (75th percentile of cane pixels in its 3 km block).
      Why: standing cane dries during the season, so "fell since the start" also flags standing cane.
  (b) Pixel or whole field?  the pixel's 5 x 5 mean  /  the mean of its whole cane patch  /  both together.
      Why: most pixels lie near a field edge, where a 5 x 5 mean mixes neighbours in.
Measure: AUC (1 = cut and standing fully apart, 0.5 = useless) and, with the limit set so that 80 % of the cut
cane is caught, the share of standing cane wrongly flagged. Latest VH pass of each track on or before the last
report; (a) on pure pixels, (b) on ALL labelled pixels because edges are the point there.

Usage: python src/eda_ideas_test.py <area> <config.yaml>
Output: data/interim/<aoi>_eda_ideas_<area>.csv
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import yaml
from scipy import ndimage
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")


def score(v: np.ndarray, cut: np.ndarray, stand: np.ndarray) -> tuple[float, float]:
    """AUC, and the share of standing flagged when the limit catches 80 % of the cut pixels (lower v = more cut)."""
    a, b = v[cut], v[stand]
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    y = np.r_[np.ones(len(a)), np.zeros(len(b))]
    return roc_auc_score(y, -np.r_[a, b]), float((b < np.percentile(a, 80)).mean())


def block_ref(s: np.ndarray, block: np.ndarray) -> np.ndarray:
    """Per pass: 75th percentile of the cane pixels in each 3 km block = level of the standing cane around."""
    order = np.argsort(block)
    starts = np.r_[0, np.flatnonzero(np.diff(block[order])) + 1, len(block)]
    out = np.full(s.shape, np.nan, "float32")
    for i, j in zip(starts[:-1], starts[1:]):
        idx = order[i:j]
        if len(idx) >= 50:
            out[:, idx] = np.nanpercentile(s[:, idx], 75, axis=1)[:, None]
    return out


def main(area: str, cfg_path: str) -> None:
    cfg = yaml.safe_load(open(cfg_path))
    interim = Path(cfg["paths"]["interim"])
    z = dict(np.load(interim / f"{cfg['aoi']}_eda_table_{area}.npz"))
    d1, d2, d3 = z["report_days"]
    c1, c2, c3 = (z[f"code_{d}"] for d in (d1, d2, d3))
    pure = z[f"inner_{d3}"]
    recent = (c1 == 81) & ((c2 == 82) | (c3 == 82))             # cut between report 1 and report 3
    old, stand = c1 == 82, c3 == 81
    block = (z["rows"] // 300).astype("int64") * 100000 + z["cols"] // 300
    with rasterio.open(interim / f"{cfg['aoi']}_cane_mask_{area}.tif") as m:
        lab, _ = ndimage.label(m.read(1) == 1)
    patch = lab[z["rows"], z["cols"]]
    del lab
    rows = []
    for t in cfg["s1_tracks"]:
        s = z[f"vh5_{t}"]
        ref = block_ref(s, block)
        rel = s - ref                                            # level against the standing cane around
        ways = {"a1 own season start": s[-1] - np.nanmean(s[:3], axis=0),
                "a2 own previous 3 passes": s[-1] - np.nanmean(s[-4:-1], axis=0),
                "a3 biggest fall between passes (last 3 steps)": np.nanmin(np.diff(s[-4:], axis=0), axis=0),
                "a4 standing cane around, now": rel[-1],
                "a5 around, now minus own start": rel[-1] - np.nanmean(rel[:3], axis=0),
                "a6 around, now minus own previous 3": rel[-1] - np.nanmean(rel[-4:-1], axis=0)}
        for name, v in ways.items():
            for group, cut in (("recent cut", recent), ("all cut", recent | old)):
                auc, false = score(v, cut & pure, stand & pure)
                rows.append({"track": t, "idea": name, "pixels": "pure", "cut group": group, "auc": auc, "standing_flagged_at_80pct_caught": false})
        # (b) pixel against whole patch, on all labelled pixels; the best simple measure of (a) is used: a5
        v = ways["a5 around, now minus own start"]
        ok = np.isfinite(v)
        n = np.bincount(patch[ok], minlength=patch.max() + 1)
        mean = np.bincount(patch[ok], weights=v[ok], minlength=patch.max() + 1) / np.maximum(n, 1)
        field = np.where(n[patch] >= 5, mean[patch], np.nan)
        for name, w in (("b1 pixel 5 x 5 mean", v), ("b2 whole patch mean", field), ("b3 pixel + patch together", (v + field) / 2)):
            for sel_name, sel in (("all pixels", np.ones(len(v), bool)), ("edge pixels (not pure)", ~pure), ("pure", pure)):
                auc, false = score(w, (recent | old) & sel, stand & sel)
                rows.append({"track": t, "idea": name, "pixels": sel_name, "cut group": "all cut", "auc": auc, "standing_flagged_at_80pct_caught": false})
    # how often is a whole cane patch one state? (by area, last report)
    cut3 = (recent | old | ((c1 != 81) & (c3 != 81))).astype(float)
    n = np.bincount(patch, minlength=patch.max() + 1)
    share = np.bincount(patch, weights=cut3, minlength=patch.max() + 1) / np.maximum(n, 1)
    one = ((share <= 0.1) | (share >= 0.9))[patch]
    print(f"cane area lying in patches that are one state (>= 90 % cut or >= 90 % standing) on the last report: {one.mean():.0%}")
    df = pd.DataFrame(rows)
    df.to_csv(interim / f"{cfg['aoi']}_eda_ideas_{area}.csv", index=False)
    pd.set_option("display.width", 200)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main(*sys.argv[1:3])
