"""Tests for CMS-HCC and actuarial actuarial features actuarial."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _make_triangle_frame(
    *,
    triangle_id: str = "A",
    origins: list[int] | None = None,
    periods: list[int] | None = None,
    values: np.ndarray | None = None,
) -> pd.DataFrame:
    if origins is None:
        origins = [2018, 2019, 2020]
    if periods is None:
        periods = [1, 2, 3]
    if values is None:
        values = np.array([
            [100, 180, 240],
            [110, 195, np.nan],
            [120, np.nan, np.nan],
        ])
    cols = [str(p) for p in periods]
    data: dict[str, object] = {
        "triangle_id": [triangle_id] * len(origins),
        "origin_year": origins,
    }
    for p, col in zip(periods, cols, strict=True):
        data[col] = values[:, periods.index(p)].tolist()
    df = pd.DataFrame(data)
    return df
