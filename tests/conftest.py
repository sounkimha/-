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
    bars_per_day: int | None = None,
    start: str = "2025-01-06",
    seed: int = 0,
    vol: float = 0.005,
    price: float = 100.0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    if bars_per_day is None:  # 야후 실제 구조: 국장 09:00~14:00 6봉, 미장 09:30~15:30 7봉
        bars_per_day = 7 if first_bar == "09:30" else 6
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


def daily_bars(closes, opens=None, start="2025-01-06", lows=None, tz="Asia/Seoul") -> pd.DataFrame:
    """영업일 00:00 인덱스의 일봉 (야후 일봉과 같은 모양). 시가를 안 주면 종가와 같게."""
    days = pd.bdate_range(start, periods=len(closes))
    idx = pd.DatetimeIndex([pd.Timestamp(d.date()).tz_localize(tz) for d in days], name="time")
    c = np.asarray(closes, float)
    o = c.copy() if opens is None else np.asarray(opens, float)
    lo = np.minimum(o, c) if lows is None else np.asarray(lows, float)
    return pd.DataFrame({"open": o, "high": np.maximum(o, c), "low": lo, "close": c, "volume": 1000.0}, index=idx)


def random_daily(n=160, seed=0, start="2024-01-01", drift=0.0008, vol=0.02, price=10_000.0) -> pd.DataFrame:
    """갭이 있는 랜덤워크 일봉 (규칙 전략이 진입·청산을 여러 번 하도록)."""
    rng = np.random.default_rng(seed)
    close = price * np.cumprod(1 + rng.normal(drift, vol, n))
    gap = rng.normal(0, vol * 0.4, n)
    opens = np.r_[close[0], close[:-1] * (1 + gap[1:])]
    df = daily_bars(close, opens, start=start)
    df["high"] = df[["open", "close"]].max(axis=1) * (1 + np.abs(rng.normal(0, vol / 4, n)))
    df["low"] = df[["open", "close"]].min(axis=1) * (1 - np.abs(rng.normal(0, vol / 4, n)))
    return df


def bars_from_rows(rows, tz="Asia/Seoul") -> pd.DataFrame:
    """[(시각 문자열, o, h, l, c), ...] → OHLCV."""
    idx = pd.DatetimeIndex([pd.Timestamp(r[0]).tz_localize(tz) for r in rows], name="time")
    data = [(r[1], r[2], r[3], r[4], 1000.0) for r in rows]
    return pd.DataFrame(data, index=idx, columns=["open", "high", "low", "close", "volume"])


@pytest.fixture
def cfg():
    return load_config(ROOT / "config.yaml")
