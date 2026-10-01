"""분류 모델 + 워크포워드 학습/예측.

워크포워드: 거래일을 순서대로 나눠, 각 구간(retrain_every_days)을 예측할 때마다
'그 구간 시작 전에 라벨이 확정된 데이터'로만 새로 학습한다.
라벨이 다음 봉에 의존하므로, 라벨 봉(label_time)이 테스트 시작 이후인 샘플은 학습에서 뺀다.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import ModelConfig
from .features import feature_columns

log = logging.getLogger(__name__)


class InsufficientDataError(RuntimeError):
    pass


def make_classifier(cfg: ModelConfig):
    params = dict(cfg.params.get(cfg.type, {}) or {})
    if cfg.type == "hgb":
        return HistGradientBoostingClassifier(random_state=cfg.random_state, **params)
    if cfg.type == "logistic":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(**params))
    if cfg.type == "random_forest":
        return make_pipeline(
            SimpleImputer(strategy="median"), RandomForestClassifier(random_state=cfg.random_state, **params)
        )
    raise ValueError(f"지원하지 않는 model.type: {cfg.type}")


@dataclass
class FittedModel:
    """분류기 + 기대수익 계산용 통계(학습 구간에서 라벨 1/0 일 때 평균 다음 봉 수익률)."""

    clf: object | None
    up_mean: float
    down_mean: float
    base_rate: float
    n_train: int

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if len(X) == 0:
            return np.empty(0)
        if self.clf is None:  # 학습 데이터에 한 가지 라벨만 있을 때: 기저율로 예측
            return np.full(len(X), self.base_rate)
        col = list(self.clf.classes_).index(1)
        return self.clf.predict_proba(X)[:, col]

    def expected_return(self, prob: np.ndarray, cost: float) -> np.ndarray:
        """비용 차감 기대수익 ≈ p·E[r|상승] + (1-p)·E[r|비상승] - 왕복비용."""
        return prob * self.up_mean + (1 - prob) * self.down_mean - cost


def fit_model(X: pd.DataFrame, y: pd.Series, fwd: pd.Series, cfg: ModelConfig) -> FittedModel:
    y = y.astype(int)
    if len(y) == 0:
        raise InsufficientDataError("학습 샘플이 없습니다")
    up = fwd[y == 1]
    down = fwd[y == 0]
    base = float(y.mean())
    clf = None
    if y.nunique() == 2:
        clf = make_classifier(cfg).fit(X, y)
    else:
        log.warning("학습 라벨이 한 종류뿐이라 기저율(%.3f)로 예측합니다", base)
    return FittedModel(
        clf=clf,
        up_mean=float(up.mean()) if len(up) else 0.0,
        down_mean=float(down.mean()) if len(down) else 0.0,
        base_rate=base,
        n_train=len(y),
    )


def valid_rows(ds: pd.DataFrame, cols: list[str]) -> pd.Series:
    """특징값이 모두 계산된 행 (지표 워밍업 구간 제외)."""
    return ds[cols].notna().all(axis=1)


@dataclass
class Fold:
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_first: pd.Timestamp | None
    train_last_label: pd.Timestamp | None  # 학습에 쓴 라벨 봉 중 가장 늦은 시각 (< test_start 이어야 함)
    n_train: int
    n_test: int
    symbols: tuple[str, ...]


@dataclass
class WalkForwardResult:
    prob: pd.Series  # ds 와 같은 index. NaN = 예측 안 함(학습 구간·워밍업)
    exp_ret: pd.Series  # 비용 차감 기대수익
    folds: list[Fold]


def walk_forward(ds: pd.DataFrame, cfg: ModelConfig, cost: float) -> WalkForwardResult:
    cols = feature_columns(ds)
    prob = pd.Series(np.nan, index=ds.index)
    exp_ret = pd.Series(np.nan, index=ds.index)
    folds: list[Fold] = []
    groups = [ds] if cfg.pooled else [g for _, g in ds.groupby("symbol", sort=False)]
    for g in groups:
        _walk_forward_group(g, cols, cfg, cost, prob, exp_ret, folds)
    return WalkForwardResult(prob=prob, exp_ret=exp_ret, folds=folds)


def _walk_forward_group(g, cols, cfg, cost, prob, exp_ret, folds) -> None:
    day_code, days = pd.factorize(g["time"].dt.normalize(), sort=True)
    day_code = pd.Series(day_code, index=g.index)
    n_days = len(days)
    if n_days <= cfg.min_train_days:
        raise InsufficientDataError(
            f"거래일 {n_days}일 ≤ min_train_days {cfg.min_train_days}일: 워크포워드 테스트 구간이 없습니다"
        )
    valid = valid_rows(g, cols)
    labeled = valid & g["y"].notna()
    for k in range(cfg.min_train_days, n_days, cfg.retrain_every_days):
        test_mask = (day_code >= k) & (day_code < k + cfg.retrain_every_days)
        test_start = g.loc[test_mask, "time"].min()
        train_mask = labeled & (g["label_time"] < test_start)
        if cfg.train_window_days > 0:
            train_mask &= day_code >= k - cfg.train_window_days
        pred_mask = test_mask & valid
        if not train_mask.any() or not pred_mask.any():
            continue
        tr = g.loc[train_mask]
        model = fit_model(tr[cols], tr["y"], tr["fwd_ret"], cfg)
        p = model.predict_proba(g.loc[pred_mask, cols])
        prob.loc[pred_mask[pred_mask].index] = p
        exp_ret.loc[pred_mask[pred_mask].index] = model.expected_return(p, cost)
        folds.append(
            Fold(
                test_start=test_start,
                test_end=g.loc[test_mask, "time"].max(),
                train_first=tr["time"].min(),
                train_last_label=tr["label_time"].max(),
                n_train=len(tr),
                n_test=int(pred_mask.sum()),
                symbols=tuple(sorted(g["symbol"].unique())),
            )
        )


def fit_latest(ds: pd.DataFrame, cfg: ModelConfig) -> dict[str, FittedModel]:
    """실시간 신호용: 라벨이 확정된 모든 데이터로 학습. {종목: 모델} (pooled 면 모두 같은 모델)."""
    cols = feature_columns(ds)
    groups = {"*": ds} if cfg.pooled else {s: g for s, g in ds.groupby("symbol", sort=False)}
    models: dict[str, FittedModel] = {}
    for key, g in groups.items():
        train = g.loc[valid_rows(g, cols) & g["y"].notna()]
        if cfg.train_window_days > 0:
            day_code, _ = pd.factorize(train["time"].dt.normalize(), sort=True)
            train = train.loc[day_code >= day_code.max() - cfg.train_window_days + 1]
        model = fit_model(train[cols], train["y"], train["fwd_ret"], cfg)
        for sym in g["symbol"].unique():
            models[sym] = model
    return models


def predict_latest(ds: pd.DataFrame, cfg: ModelConfig, cost: float) -> pd.DataFrame:
    """종목별 가장 최근 완성 봉의 상승확률·기대수익."""
    cols = feature_columns(ds)
    models = fit_latest(ds, cfg)
    latest = ds.groupby("symbol", sort=False).tail(1)
    rows = []
    for _, row in latest.iterrows():
        X = row[cols].to_frame().T.astype(float)
        if X.isna().any(axis=None):
            p = np.array([np.nan])
        else:
            p = models[row["symbol"]].predict_proba(X)
        er = models[row["symbol"]].expected_return(p, cost)
        rows.append({"symbol": row["symbol"], "time": row["time"], "prob": float(p[0]), "exp_ret": float(er[0])})
    return pd.DataFrame(rows).set_index("symbol")


def evaluate_predictions(ds: pd.DataFrame, prob: pd.Series, threshold: float) -> dict[str, float]:
    """표본 외(워크포워드) 예측 품질: AUC, 기저율, 임계값 이상 신호의 적중률."""
    m = prob.notna() & ds["y"].notna()
    y = ds.loc[m, "y"].astype(int)
    p = prob[m]
    sig = p >= threshold
    return {
        "n": int(m.sum()),
        "base_rate": float(y.mean()) if len(y) else float("nan"),
        "auc": float(roc_auc_score(y, p)) if y.nunique() == 2 else float("nan"),
        "signals": int(sig.sum()),
        "precision": float(y[sig].mean()) if sig.any() else float("nan"),
    }
