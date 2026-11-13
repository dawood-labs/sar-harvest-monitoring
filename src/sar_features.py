"""Shared SAR helpers: load the cached pixel table and smooth inside the crop mask only."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import yaml
from scipy.ndimage import uniform_filter

from resources import workers


def load_pixels(area: str, cfg_path: str = "configs/config.yaml") -> tuple[dict, dict]:
    """Config and the cached table of crop-mask pixels (see eda_prepare.py); dB values as float32 with NaN."""
    cfg = yaml.safe_load(open(cfg_path))
    z = np.load(Path(cfg["paths"]["interim"]) / f"{cfg['aoi']}_eda_pixels_{area}.npz")
    d = {k: z[k] for k in z.files}
    for k in d:
        if k[:3] in ("vv_", "vh_"):
            d[k] = np.where(d[k] == -32768, np.nan, d[k] / 100).astype("float32")
    return cfg, d


def smooth_in_mask(db: np.ndarray, rows: np.ndarray, cols: np.ndarray, size: int) -> np.ndarray:
    """Mean of each pixel's size x size neighbourhood, using crop-mask pixels only.

    Why: one SAR pixel is noisy (speckle); averaging neighbours shows the field's real level. The average is taken on
    power (not dB) and pixels outside the mask never enter it. db = (dates, pixels); returns the same shape in dB.
    """
    if size <= 1:
        return db
    shape = (int(rows.max()) + 1, int(cols.max()) + 1)

    def one(line: np.ndarray) -> np.ndarray:
        power = np.zeros(shape, "float32")
        weight = np.zeros(shape, "float32")
        ok = np.isfinite(line)
        power[rows[ok], cols[ok]] = 10 ** (line[ok] / 10)
        weight[rows[ok], cols[ok]] = 1
        mean = uniform_filter(power, size) / np.maximum(uniform_filter(weight, size), 1e-6)
        res = 10 * np.log10(np.maximum(mean[rows, cols], 1e-6))
        res[~ok] = np.nan
        return res

    with ThreadPoolExecutor(workers()) as ex:      # one date per thread; numpy / scipy release the GIL
        return np.stack(list(ex.map(one, db)))
