"""Save the training sample of an area that has the in-memory pixel table, for use in another area's training.

Why: a model trained on two areas together may work in a new area without that area's own reports.
The sample is exactly the one run_area.py trains its final model on: per report date up to 300,000 random
crop-mask pixels at least 20 m from an edge, the 64 date-free features and harvested / standing from the
original reports. Features are computed for the report dates in parallel threads.

Usage: python src/export_training_sample.py <area> <config.yaml> [averaging window in pixels, default 5]
Output: data/interim/<aoi>_train_sample_<area>_export.npz (X, y)
"""
from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

import transfer_test as tt
from compare_detectors import smoothed_series
from retrain_test import rate_features
from sar_features import load_pixels, local_std_in_mask

SAMPLE = 300_000


def main(area: str, cfg_path: str, smooth: str = "5") -> None:
    cfg, d = load_pixels(area, cfg_path)
    reports, rng = cfg["report_dates"], np.random.default_rng(cfg["seed"])
    series = smoothed_series(cfg, d, int(smooth))
    texture = {t: local_std_in_mask(d[f"vh_{t}"], d["rows"], d["cols"], 5) for t in cfg["s1_tracks"]}

    def one(job: tuple[int, str]) -> tuple[np.ndarray, np.ndarray]:
        n, r = job
        y, known = tt.harvested_by(d, n)
        pool = np.flatnonzero(known & ~tt.near_edge(d, y, known))
        idx = np.sort(np.random.default_rng(cfg["seed"] + n).choice(pool, min(len(pool), SAMPLE), replace=False))
        x = np.column_stack([tt.date_free_features(cfg, d, series, r), rate_features(cfg, d, series, texture, r)])
        return x[idx], y[idx].astype("int8")

    with ThreadPoolExecutor(len(reports)) as ex:
        parts = list(ex.map(one, enumerate(reports, 1)))
    X, y = np.vstack([p[0] for p in parts]), np.concatenate([p[1] for p in parts])
    out = Path(cfg["paths"]["interim"]) / f"{cfg['aoi']}_train_sample_{area}_export{'' if smooth == '5' else '_smooth' + smooth}.npz"
    np.savez(out, X=X, y=y)
    print(f"{len(y):,} samples, harvested {y.mean():.1%}, {X.shape[1]} features -> {out}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
