"""차트 특징값과 학습 라벨.

규칙: t 시점 특징값은 t 봉의 '종가까지' 정보만 쓴다 (rolling/ewm/shift(+) 만 사용).
미래를 보는 연산(shift(-k), center=True, 그룹 전체 통계)은 라벨에만 쓴다.
tests/test_features.py 가 '앞부분만 잘라서 계산해도 값이 같다'로 이를 검증한다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FeatureConfig

META_COLS = ("time", "symbol", "y", "fwd_ret", "label_time")


def rsi(close: pd.Series, period: int) -> pd.Series:
    """Wilder RSI (0~100)."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss
    out = 100 - 100 / (1 + rs)
    out = out.mask((avg_loss == 0) & (avg_gain > 0), 100.0)
    out = out.mask((avg_loss == 0) & (avg_gain == 0), 50.0)
    return out


def build_features(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    o, h, l, c, v = (df[k] for k in ("open", "high", "low", "close", "volume"))
    ret1 = c.pct_change()
    day = pd.Series(df.index.normalize(), index=df.index)
    f = pd.DataFrame(index=df.index)

    # 기간별 수익률
    for n in cfg.return_windows:
        f[f"ret_{n}"] = c.pct_change(n)
    # RSI
    f[f"rsi_{cfg.rsi_period}"] = rsi(c, cfg.rsi_period)
    # 이동평균 대비 위치 + 단기/장기 이평 비율
    mas = {n: c.rolling(n).mean() for n in cfg.ma_windows}
    for n, ma in mas.items():
        f[f"ma_ratio_{n}"] = c / ma - 1
    if len(cfg.ma_windows) >= 2:
        short, long_ = min(cfg.ma_windows), max(cfg.ma_windows)
        f[f"ma_{short}_{long_}"] = mas[short] / mas[long_] - 1
    # 변동성
    for n in cfg.volatility_windows:
        f[f"volat_{n}"] = ret1.rolling(n).std()
    # 거래량 비율 (0 거래량 평균은 NaN 처리)
    f["volume_ratio"] = v / v.rolling(cfg.volume_window).mean().replace(0, np.nan)
    # 봉 크기 / 꼬리
    f["body"] = (c - o) / o
    f["upper_wick"] = (h - np.maximum(o, c)) / o
    f["lower_wick"] = (np.minimum(o, c) - l) / o
    f["bar_range"] = (h - l) / o
    # 갭(직전 종가 대비 시가) — 하루 첫 봉이면 오버나이트 갭
    f["gap"] = o / c.shift(1) - 1
    # 장중 위치
    f["bar_of_day"] = df.groupby(day).cumcount().astype(float)
    f["day_return"] = c / o.groupby(day).transform("first") - 1
    f["hour"] = df.index.hour + df.index.minute / 60
    # 최근 N봉 고가/저가 대비 위치
    w = cfg.range_window
    f[f"dist_high_{w}"] = c / h.rolling(w).max() - 1
    f[f"dist_low_{w}"] = c / l.rolling(w).min() - 1
    return f.replace([np.inf, -np.inf], np.nan)


def build_label(df: pd.DataFrame, cost: float) -> tuple[pd.Series, pd.Series, pd.Series]:
    """y_t = 1  ⇔  다음 봉(t+1)의 시가→종가 수익률 > 왕복비용.

    t 봉 종가에서 신호를 보고 t+1 봉 시가에 진입하는 실제 매매 순서와 맞춘 정의다.
    반환: (y, 다음 봉 수익률, 라벨이 의존하는 봉의 시작 시각). 마지막 봉은 NaN.
    """
    fwd = df["close"].shift(-1) / df["open"].shift(-1) - 1
    y = (fwd > cost).astype(float).where(fwd.notna())
    label_time = pd.Series(df.index, index=df.index).shift(-1)
    return y, fwd, label_time


def make_dataset(bars: dict[str, pd.DataFrame], cfg: FeatureConfig, cost: float) -> pd.DataFrame:
    """종목별 특징값+라벨을 하나의 긴 표로 합친다 (index 는 0..n-1, 시간은 'time' 컬럼)."""
    frames = []
    for symbol, df in bars.items():
        f = build_features(df, cfg)
        y, fwd, label_time = build_label(df, cost)
        f = f.assign(y=y, fwd_ret=fwd, label_time=label_time)
        f.insert(0, "symbol", symbol)
        f.index.name = "time"
        frames.append(f.reset_index())
    if not frames:
        raise ValueError("데이터셋을 만들 종목 데이터가 없습니다")
    ds = pd.concat(frames, ignore_index=True)
    return ds.sort_values(["time", "symbol"], kind="stable").reset_index(drop=True)


def feature_columns(ds: pd.DataFrame) -> list[str]:
    return [c for c in ds.columns if c not in META_COLS]
