from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from conftest import make_bars
from trader.backtest import Account, buy_hold_curve, run_backtest, summarize
from trader.config import CostConfig


def flat_days(n_days=2, price=100.0, start="2025-03-03", tz="Asia/Seoul"):
    """KR 시간표(09:00~15:00, 하루 7봉)로 가격이 평평한 봉."""
    idx = []
    for day in pd.bdate_range(start, periods=n_days):
        for k in range(7):
            idx.append(pd.Timestamp(day.date()).tz_localize(tz) + pd.Timedelta(hours=9 + k))
    df = pd.DataFrame(price, index=pd.DatetimeIndex(idx, name="time"), columns=["open", "high", "low", "close"])
    df["volume"] = 1000.0
    return df


def run(cfg, df, prob, *, strategy=None, risk=None):
    market = cfg.market("kr")
    sig = pd.DataFrame({"prob": prob, "exp_ret": 1.0}, index=df.index)
    st = replace(cfg.strategy, **(strategy or {}))
    rk = replace(cfg.risk, **(risk or {}))
    return run_backtest({"X": df}, {"X": sig}, market, st, rk)


def expected_flat_return(c: CostConfig) -> float:
    s, fee, tax = c.slippage, c.commission, c.sell_tax
    return (1 - s) * (1 - fee - tax) / ((1 + s) * (1 + fee)) - 1


def test_round_trip_cost_is_charged(cfg):
    df = flat_days(1)
    prob = [0.9] + [0.0] * 6  # 0번 봉 신호 → 1번 봉 시가 진입 → 2번 봉 시가 청산
    res = run(cfg, df, prob, strategy={"flatten_at_session_end": False})
    assert len(res.trades) == 1
    t = res.trades[0]
    assert t.entry_time == df.index[1] and t.exit_time == df.index[2] and t.exit_reason == "signal"
    costs = cfg.market("kr").costs
    assert t.ret == pytest.approx(expected_flat_return(costs), abs=1e-12)
    assert t.ret == pytest.approx(-costs.round_trip, abs=2e-5)  # 국장 기본 왕복 0.25%


def test_position_size_is_20_percent_of_equity(cfg):
    df = flat_days(1)
    res = run(cfg, df, [0.9] + [0.0] * 6, strategy={"flatten_at_session_end": False})
    t = res.trades[0]
    init = cfg.market("kr").initial_capital
    assert t.cost_basis <= init * 0.20  # 체결금액 + 수수료가 계좌의 20% 이내
    assert t.cost_basis > init * 0.20 - t.entry_price * 2  # 정수 주식 반올림 오차 이내로 꽉 채움


def test_intrabar_stop_fills_at_stop_price(cfg):
    df = flat_days(1)
    df.iloc[1] = [100.0, 100.5, 98.0, 99.5, 1000.0]  # 진입 봉에서 저가가 손절가(-1%) 아래로
    res = run(cfg, df, [0.9] * 7, strategy={"flatten_at_session_end": False})
    t = res.trades[0]
    s = cfg.market("kr").costs.slippage
    assert t.exit_reason == "stop" and t.exit_time == df.index[1]
    assert t.stop_price == pytest.approx(t.entry_price * 0.99)
    assert t.exit_price == pytest.approx(t.stop_price * (1 - s))


def test_gap_down_through_stop_loses_more_than_stop(cfg):
    df = flat_days(2)
    df.iloc[7:, :4] = 95.0  # 다음날 시가가 -5% 갭하락
    prob = [0.0] * 5 + [0.9, 0.9] + [0.9] * 7  # 14:00 봉 신호 → 15:00 진입 → 오버나이트 보유
    res = run(cfg, df, prob, strategy={"flatten_at_session_end": False})
    t = res.trades[0]
    assert t.exit_reason == "gap_stop" and t.exit_time == df.index[7]
    assert t.exit_price < t.stop_price  # 손절가보다 불리하게 체결
    assert t.ret < -0.049  # 손절 -1% 가 아니라 갭만큼 손실


def test_flatten_at_session_end_never_holds_overnight(cfg):
    df = flat_days(3)
    res = run(cfg, df, [0.9] * len(df), strategy={"flatten_at_session_end": True})
    assert res.trades
    for t in res.trades[:-1]:
        assert t.exit_time.date() == t.entry_time.date()
        assert t.exit_reason == "session_end"
        assert t.exit_time.hour == 15  # 마지막 봉(15:00) 시가에 청산
    assert not any(t.entry_time.hour == 15 for t in res.trades)  # 마지막 봉 신규 진입 금지


def test_drawdown_limit_halts_all_trading(cfg):
    df = flat_days(3)
    px = 100.0 * 0.99 ** np.arange(len(df))  # 봉마다 1%씩 하락
    df["open"] = df["high"] = px
    df["close"] = df["low"] = px * 0.995
    res = run(
        cfg,
        df,
        [0.9] * len(df),
        strategy={"flatten_at_session_end": False},
        risk={"position_size_pct": 100.0, "max_drawdown_pct": 5.0, "stop_loss_pct": 50.0},
    )
    assert res.halted_at is not None
    assert all(t.entry_time <= res.halted_at for t in res.trades)  # 중단 후 신규 진입 없음
    assert res.trades[-1].exit_reason == "halt"
    assert len(res.trades) == 1


def test_daily_trade_limit(cfg):
    df = flat_days(3)
    prob = [0.9 if i % 2 == 0 else 0.0 for i in range(len(df))]  # 진입·청산 반복 신호
    res = run(cfg, df, prob, strategy={"flatten_at_session_end": False}, risk={"max_trades_per_day": 2})
    per_day = pd.Series([t.entry_time.date() for t in res.trades]).value_counts()
    assert per_day.max() == 2
    unlimited = run(cfg, df, prob, strategy={"flatten_at_session_end": False}, risk={"max_trades_per_day": 10})
    assert len(unlimited.trades) > len(res.trades)


def test_buy_and_hold_includes_round_trip_cost(cfg):
    costs = cfg.market("kr").costs
    bh = buy_hold_curve(flat_days(1), costs)
    assert bh.iloc[-1] - 1 == pytest.approx(expected_flat_return(costs), abs=1e-12)


def test_no_signal_no_trade_and_summary_shape(cfg):
    df = make_bars(20, seed=4)
    res = run(cfg, df, [np.nan] * len(df))
    assert res.trades == []
    assert res.equity.iloc[-1] == pytest.approx(cfg.market("kr").initial_capital)
    table, info = summarize(res)
    assert list(table.columns) == ["종목", "전략수익%", "단순보유%", "거래수", "승률%", "평균손익%", "최대낙폭%(전략)", "최대낙폭%(보유)"]
    assert table.iloc[-1]["종목"] == "계좌 합계"
    assert info["매매중단"] == "없음"


def test_account_never_spends_more_than_cash(cfg):
    acct = Account(1000.0, cfg.market("kr").costs)
    t = acct.buy("X", 100.0, budget=1e9, when=pd.Timestamp("2025-01-01", tz="UTC"), stop_price_fn=lambda p: p * 0.99)
    assert t.qty == 9 and acct.cash >= 0
