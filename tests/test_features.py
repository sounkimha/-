import numpy as np
import pandas as pd
import pandas.testing as pdt

from conftest import make_bars
from trader.features import build_features, build_label, feature_columns, make_dataset, rsi


def test_features_use_no_future_data(cfg):
    """앞부분만 잘라서 계산한 특징값 == 전체로 계산한 특징값의 같은 구간 (미래 정보가 섞이면 달라진다)."""
    df = make_bars(40, seed=1)
    full = build_features(df, cfg.features)
    for k in (30, 75, 150, len(df) - 1):
        part = build_features(df.iloc[:k], cfg.features)
        pdt.assert_frame_equal(part, full.iloc[:k], check_exact=False, rtol=1e-9, atol=1e-12)


def test_features_unchanged_when_future_bars_are_altered(cfg):
    """t 이후 봉을 바꿔도 t 까지의 특징값은 그대로여야 한다."""
    df = make_bars(30, seed=2)
    t = 120
    altered = df.copy()
    altered.iloc[t + 1 :, :4] *= 1.5
    a = build_features(df, cfg.features).iloc[: t + 1]
    b = build_features(altered, cfg.features).iloc[: t + 1]
    pdt.assert_frame_equal(a, b)


def test_label_is_next_bar_open_to_close_above_cost():
    df = make_bars(5, seed=3)
    cost = 0.0025
    y, fwd, label_time = build_label(df, cost)
    for i in range(len(df) - 1):
        r = df["close"].iloc[i + 1] / df["open"].iloc[i + 1] - 1
        assert np.isclose(fwd.iloc[i], r)
        assert y.iloc[i] == float(r > cost)
        assert label_time.iloc[i] == df.index[i + 1]
    assert np.isnan(y.iloc[-1]) and pd.isna(label_time.iloc[-1])


def test_rsi_bounds_and_constant_series():
    s = pd.Series(np.linspace(100, 120, 50))
    r = rsi(s, 14).dropna()
    assert (r == 100).all()  # 계속 오르기만 하면 100
    r2 = rsi(pd.Series([100.0] * 30), 14).dropna()
    assert (r2 == 50).all()
    noisy = rsi(pd.Series(100 + np.random.default_rng(0).normal(0, 1, 500).cumsum()), 14).dropna()
    assert noisy.between(0, 100).all()


def test_make_dataset_long_format(cfg):
    bars = {"A": make_bars(10, seed=1), "B": make_bars(10, seed=2)}
    ds = make_dataset(bars, cfg.features, 0.001)
    assert len(ds) == 140
    assert ds["time"].is_monotonic_increasing
    assert set(ds["symbol"]) == {"A", "B"}
    cols = feature_columns(ds)
    assert "y" not in cols and "fwd_ret" not in cols and "label_time" not in cols
    assert {"ret_1", "rsi_14", "ma_ratio_20", "volat_24", "volume_ratio", "body", "gap"} <= set(cols)
