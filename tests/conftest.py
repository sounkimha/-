"""테스트용 합성 시세 (네트워크 없이 코드 경로를 검증하기 위한 것. 전략 성과 평가용 아님)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from trader.config import load_config  # noqa: E402


def make_bars(
    n_days: int = 60,
    *,
    tz: str = "Asia/Seoul",
    first_bar: str = "09:00",
    bars_per_day: int = 7,
    start: str = "2025-01-06",
    seed: int = 0,
    vol: float = 0.005,
    price: float = 100.0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    hh, mm = (int(x) for x in first_bar.split(":"))
    rows, idx = [], []
    for day in pd.bdate_range(start, periods=n_days):
        gap = rng.normal(0, vol * 2)
        for k in range(bars_per_day):
            ts = pd.Timestamp(day.date()).tz_localize(tz) + pd.Timedelta(hours=hh + k, minutes=mm)
            o = price * (1 + gap if k == 0 else 1)
            c = o * (1 + rng.normal(0, vol))
            h = max(o, c) * (1 + abs(rng.normal(0, vol / 2)))
            low = min(o, c) * (1 - abs(rng.normal(0, vol / 2)))
            rows.append((o, h, low, c, float(rng.integers(1_000, 5_000))))
            idx.append(ts)
            price = c
    return pd.DataFrame(rows, index=pd.DatetimeIndex(idx, name="time"), columns=["open", "high", "low", "close", "volume"])


def bars_from_rows(rows, tz="Asia/Seoul") -> pd.DataFrame:
    """[(시각 문자열, o, h, l, c), ...] → OHLCV."""
    idx = pd.DatetimeIndex([pd.Timestamp(r[0]).tz_localize(tz) for r in rows], name="time")
    data = [(r[1], r[2], r[3], r[4], 1000.0) for r in rows]
    return pd.DataFrame(data, index=idx, columns=["open", "high", "low", "close", "volume"])


@pytest.fixture
def cfg():
    return load_config(ROOT / "config.yaml")
