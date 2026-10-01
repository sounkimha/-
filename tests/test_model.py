from dataclasses import replace

import numpy as np
import pandas as pd

from conftest import make_bars
from trader.features import make_dataset
from trader.model import evaluate_predictions, predict_latest, walk_forward


def _small_model(cfg, **kw):
    params = {"hgb": {"max_depth": 2, "max_iter": 30, "min_samples_leaf": 20, "early_stopping": False}}
    return replace(cfg.model, min_train_days=30, retrain_every_days=5, params=params, **kw)


def test_walk_forward_never_trains_on_future_labels(cfg):
    bars = {"A": make_bars(70, seed=1), "B": make_bars(70, seed=2)}
    ds = make_dataset(bars, cfg.features, 0.002)
    wf = walk_forward(ds, _small_model(cfg), 0.002)
    assert wf.folds, "폴드가 하나도 없음"
    for f in wf.folds:
        # 학습에 쓴 라벨(다음 봉)이 전부 테스트 구간 시작 전에 확정되어 있어야 한다
        assert f.train_last_label < f.test_start
    # 처음 min_train_days 거래일에는 예측이 없어야 한다
    first_test = wf.folds[0].test_start
    assert wf.prob[ds["time"] < first_test].isna().all()
    assert wf.prob[ds["time"] >= first_test].notna().any()


def test_walk_forward_rolling_window_and_per_symbol(cfg):
    bars = {"A": make_bars(70, seed=1), "B": make_bars(70, seed=2)}
    ds = make_dataset(bars, cfg.features, 0.002)
    wf = walk_forward(ds, _small_model(cfg, pooled=False, train_window_days=20), 0.002)
    assert {f.symbols for f in wf.folds} == {("A",), ("B",)}
    for f in wf.folds:
        assert f.train_last_label < f.test_start
        assert f.train_first >= f.test_start - pd.Timedelta(days=40)  # 최근 20거래일(+주말)만


def test_model_learns_a_planted_pattern(cfg):
    """다음 봉 방향이 직전 봉 몸통 방향을 따르도록 심어 두면 모델이 찾아야 한다 (AUC > 0.6)."""
    rng = np.random.default_rng(7)
    df = make_bars(120, seed=7)
    o = df["open"].to_numpy().copy()
    c = df["close"].to_numpy().copy()
    for i in range(1, len(df)):
        o[i] = c[i - 1]
        prev_body = c[i - 1] / o[i - 1] - 1
        c[i] = o[i] * (1 + 0.8 * prev_body + rng.normal(0, 0.003))
    df = df.assign(open=o, close=c)
    df["high"] = df[["open", "close"]].max(axis=1) * 1.001
    df["low"] = df[["open", "close"]].min(axis=1) * 0.999
    ds = make_dataset({"X": df}, cfg.features, 0.001)
    wf = walk_forward(ds, _small_model(cfg), 0.001)
    ev = evaluate_predictions(ds, wf.prob, 0.5)
    assert ev["auc"] > 0.6, ev


def test_predict_latest_one_row_per_symbol(cfg):
    bars = {"A": make_bars(50, seed=1), "B": make_bars(50, seed=2)}
    ds = make_dataset(bars, cfg.features, 0.002)
    latest = predict_latest(ds, _small_model(cfg), 0.002)
    assert list(latest.index) == ["A", "B"]
    assert latest["prob"].between(0, 1).all()
    assert (latest["time"] == bars["A"].index[-1]).all()
