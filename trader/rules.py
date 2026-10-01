"""일봉 규칙 기반 신호 — 학습용 추세 돌파 (수익성 미입증).

모든 값은 T일 종가까지의 정보만 쓴다 (rolling 과 shift(+1) 만 사용).
  entry : 종가 > 장기 이동평균  그리고  종가 > 직전 N일(T일 제외) 최고 종가
  hold  : 종가 ≥ 청산 이동평균  (False 가 되면 다음 날 시가에 청산)
  score : 최근 N일 수익률 — 진입 후보가 빈자리보다 많을 때 높은 순으로 고른다
"""
from __future__ import annotations

import pandas as pd

from .config import StrategyConfig


def trend_breakout(df: pd.DataFrame, st: StrategyConfig) -> pd.DataFrame:
    c = df["close"]
    sma_long = c.rolling(st.long_ma).mean()
    sma_exit = c.rolling(st.exit_ma).mean()
    prior_high = c.shift(1).rolling(st.breakout_lookback).max()
    valid = sma_long.notna() & sma_exit.notna() & prior_high.notna()
    return pd.DataFrame(
        {
            "entry": valid & (c > sma_long) & (c > prior_high),
            "hold": valid & (c >= sma_exit),
            "score": c / c.shift(st.score_lookback) - 1,
            "sma_long": sma_long,
            "sma_exit": sma_exit,
            "prior_high": prior_high,
        },
        index=df.index,
    )


def first_valid_time(sig: pd.DataFrame) -> pd.Timestamp | None:
    ok = sig["sma_long"].notna() & sig["prior_high"].notna() & sig["sma_exit"].notna()
    return sig.index[ok.argmax()] if ok.any() else None
