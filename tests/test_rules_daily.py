"""일봉 규칙 전략: 신호 계산 · 백테스트 엔진(지정가·종가 손절·재진입 대기·종목 수 제한) · 주문표 · 일봉 시간 처리."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import trader.daily as daily
from conftest import ROOT
from trader.backtest import comparison_table, run_backtest, with_cost_multiplier
from trader.config import ConfigError, StrategyConfig, load_config
from trader.daily import Held, buy_limit_price, next_weekday, parse_held, plan_orders, rule_signals
from trader.data import bar_end, drop_incomplete_last_bar, expected_latest_bar, filter_session
from trader.rules import first_valid_time, trend_breakout

KST = "Asia/Seoul"
SMALL = replace(StrategyConfig(), type="rule_breakout", long_ma=5, breakout_lookback=3, exit_ma=3, score_lookback=3)


@pytest.fixture
def dcfg():
    return load_config(ROOT / "config.daily.yaml")


def daily_bars(closes, opens=None, start="2025-01-06", lows=None) -> pd.DataFrame:
    """영업일 00:00(KST) 인덱스의 일봉 (야후 일봉과 같은 모양)."""
    days = pd.bdate_range(start, periods=len(closes))
    idx = pd.DatetimeIndex([pd.Timestamp(d.date()).tz_localize(KST) for d in days], name="time")
    c = np.asarray(closes, float)
    o = c.copy() if opens is None else np.asarray(opens, float)
    lo = np.minimum(o, c) if lows is None else np.asarray(lows, float)
    return pd.DataFrame({"open": o, "high": np.maximum(o, c), "low": lo, "close": c, "volume": 1000.0}, index=idx)


def signals(df, entry, hold=True, score=0.0) -> pd.DataFrame:
    n = len(df)
    as_list = lambda v: list(v) if isinstance(v, (list, tuple, np.ndarray)) else [v] * n  # noqa: E731
    return pd.DataFrame({"entry": as_list(entry), "hold": as_list(hold), "score": as_list(score)}, index=df.index)


def run(dcfg, bars, sigs, strategy=None, risk=None):
    st = replace(dcfg.strategy, **(strategy or {}))
    rk = replace(dcfg.risk, **(risk or {}))
    return run_backtest(bars, sigs, dcfg.market("kr"), st, rk)


# --------------------------------------------------------------------------
# 규칙 신호
# --------------------------------------------------------------------------
def test_trend_breakout_rules_on_known_prices():
    df = daily_bars([10, 11, 12, 13, 14, 15, 15, 16, 14])
    sig = trend_breakout(df, SMALL)
    assert list(sig["entry"]) == [False, False, False, False, True, True, False, True, False]
    # 6번: 종가 15 = 직전 3일 최고 15 → '넘어야' 진입 (같으면 아님)
    assert sig["prior_high"].iloc[6] == 15
    assert bool(sig["hold"].iloc[6]) and not bool(sig["hold"].iloc[8])  # 8번: 14 < 3일선 15
    assert sig["score"].iloc[8] == pytest.approx(14 / 15 - 1)  # 3일 전(5번) 종가 대비
    assert first_valid_time(sig) == df.index[4]


def test_breakout_below_long_ma_is_not_an_entry():
    df = daily_bars([20, 20, 20, 20, 20, 10, 11, 12, 13])
    sig = trend_breakout(df, SMALL)
    assert df["close"].iloc[8] > sig["prior_high"].iloc[8]  # 돌파는 했지만
    assert df["close"].iloc[8] < sig["sma_long"].iloc[8]  # 장기선 아래
    assert not sig["entry"].iloc[8]


def test_trend_breakout_uses_no_future_data():
    rng = np.random.default_rng(3)
    df = daily_bars(100 * np.cumprod(1 + rng.normal(0.001, 0.02, 120)))
    full = trend_breakout(df, SMALL)
    for k in (5, 17, 60, 119):
        part = trend_breakout(df.iloc[: k + 1], SMALL).iloc[-1]
        pd.testing.assert_series_equal(part, full.iloc[k], check_names=False)


# --------------------------------------------------------------------------
# 백테스트 엔진 (일봉 규칙 경로)
# --------------------------------------------------------------------------
def test_max_positions_picks_highest_scores(dcfg):
    bars = {s: daily_bars([100.0] * 6) for s in "ABC"}
    sigs = {
        "A": signals(bars["A"], [True] + [False] * 5, score=0.1),
        "B": signals(bars["B"], [True] + [False] * 5, score=0.3),
        "C": signals(bars["C"], [True] + [False] * 5, score=0.2),
    }
    res = run(dcfg, bars, sigs, strategy={"max_positions": 2})
    assert sorted(t.symbol for t in res.trades) == ["B", "C"]
    assert all(t.entry_time == bars["B"].index[1] for t in res.trades)  # 신호 다음 날 시가
    budget = dcfg.market("kr").initial_capital * dcfg.risk.position_size_pct / 100
    assert all(t.cost_basis <= budget for t in res.trades)


def test_slots_free_up_only_after_a_sell(dcfg):
    bars = {s: daily_bars([100.0] * 6) for s in "AB"}
    hold_a = [True, True, False, False, False, False]  # A 는 2번 봉 종가에서 청산 신호 → 3번 시가 매도
    sigs = {"A": signals(bars["A"], [True] + [False] * 5, hold=hold_a), "B": signals(bars["B"], True)}
    res = run(dcfg, bars, sigs, strategy={"max_positions": 1, "reentry_cooldown_bars": 0})
    b = [t for t in res.trades if t.symbol == "B"]
    # A 가 0번 신호로 먼저 자리를 차지(점수 동률이면 설정 순서), B 는 A 의 매도 주문이 나간 2번 봉 신호로 3번 시가 매수
    assert b and b[0].entry_time == bars["B"].index[3]


def test_limit_order_is_unfilled_when_open_gaps_above(dcfg):
    df = daily_bars([100, 100, 100], opens=[100, 103, 100])  # 지정가 102 (종가+2%) < 시가 103
    res = run(dcfg, {"X": df}, {"X": signals(df, [True, False, False])})
    assert res.trades == [] and res.unfilled_entries == 1

    df2 = daily_bars([100, 100, 100], opens=[100, 101.9, 100])
    res2 = run(dcfg, {"X": df2}, {"X": signals(df2, [True, False, False])})
    t = res2.trades[0]
    assert res2.unfilled_entries == 0
    assert t.entry_price == pytest.approx(101.9 * (1 + dcfg.market("kr").costs.slippage))


def test_close_based_stop_ignores_intraday_low_and_exits_next_open(dcfg):
    closes = [100, 100, 95, 91, 89, 89]
    opens = [100, 100, 100, 94, 88, 89]
    lows = [100, 100, 85, 90, 87, 89]  # 2번 봉 장중 저가 85 는 손절가 아래지만 종가 95 는 위
    df = daily_bars(closes, opens=opens, lows=lows)
    res = run(dcfg, {"X": df}, {"X": signals(df, [True] + [False] * 5)})
    t = res.trades[0]
    assert t.stop_price == pytest.approx(t.entry_price * 0.92)
    # 3번 종가 91 ≤ 손절가(≈92.0) → 4번 시가 88 에 매도 (손절가보다 불리)
    assert t.exit_reason == "stop" and t.exit_time == df.index[4]
    assert t.exit_price == pytest.approx(88 * (1 - dcfg.market("kr").costs.slippage))

    intrabar = run(dcfg, {"X": df}, {"X": signals(df, [True] + [False] * 5)}, risk={"stop_check": "intrabar"})
    assert intrabar.trades[0].exit_time == df.index[2]  # 비교: 장중 판단이면 2번 봉에서 손절


def test_reentry_cooldown_blocks_new_entries_for_n_bars(dcfg):
    df = daily_bars([100.0] * 12)
    hold = [True, True, False] + [True] * 9  # 2번 종가에서 청산 신호 → 3번 시가 매도
    sigs = {"X": signals(df, True, hold=hold)}
    res = run(dcfg, {"X": df}, sigs, strategy={"reentry_cooldown_bars": 5})
    assert [t.entry_time for t in res.trades] == [df.index[1], df.index[9]]  # 3번 매도 → 8번 신호 → 9번 매수
    none = run(dcfg, {"X": df}, sigs, strategy={"reentry_cooldown_bars": 0})
    assert none.trades[1].entry_time == df.index[4]


def test_comparison_table_and_cost_multiplier(dcfg):
    df = daily_bars(np.linspace(100, 130, 30))
    res = run(dcfg, {"069500.KS": df}, {"069500.KS": signals(df, True)})
    comp = comparison_table(res, "069500.KS")
    assert list(comp["구분"]) == ["전략", "KODEX 200(069500.KS) 보유", "동일가중 1종목 보유"]
    assert comp["거래수"].iloc[0] == len(res.trades) and pd.isna(comp["거래수"].iloc[1])
    assert comp["최대낙폭%"].iloc[1] == pytest.approx(0.0, abs=0.1)

    doubled = with_cost_multiplier(dcfg, "kr", 2.0).market("kr").costs
    base = dcfg.market("kr").costs
    assert doubled.commission_pct == pytest.approx(2 * base.commission_pct)
    assert doubled.slippage_pct == pytest.approx(2 * base.slippage_pct)
    assert doubled.sell_tax_pct == base.sell_tax_pct


# --------------------------------------------------------------------------
# 일봉 시간 처리 · 설정 검증
# --------------------------------------------------------------------------
def test_daily_bar_timing(dcfg):
    kr = dcfg.market("kr")
    day = pd.Timestamp("2025-03-04", tz=KST)
    assert bar_end(day, kr, "1d") == pd.Timestamp("2025-03-04 15:30", tz=KST)
    df = daily_bars([1.0, 2.0], start="2025-03-03")  # 03-03, 03-04
    # 지연 30분: 16:00 전에는 그날 일봉을 미완성으로 본다
    assert drop_incomplete_last_bar(df, kr, "1d", pd.Timestamp("2025-03-04 15:59", tz=KST)).index[-1].day == 3
    assert drop_incomplete_last_bar(df, kr, "1d", pd.Timestamp("2025-03-04 16:00", tz=KST)).index[-1].day == 4
    assert expected_latest_bar(pd.Timestamp("2025-03-05 10:00", tz=KST), kr, "1d").date().day == 4
    assert expected_latest_bar(pd.Timestamp("2025-03-10 08:00", tz=KST), kr, "1d").date().day == 7  # 월 아침 → 금
    saturday = df.iloc[[-1]].set_axis(pd.DatetimeIndex([pd.Timestamp("2025-03-08", tz=KST)], name="time"))
    assert len(filter_session(pd.concat([df, saturday]), kr, "1d")) == 2  # 토요일 봉 제거
    assert next_weekday(pd.Timestamp("2025-03-07").date()).day == 10


def test_daily_config_loads_and_guards(dcfg, tmp_path):
    assert dcfg.strategy.type == "rule_breakout" and dcfg.data.interval == "1d"
    assert dcfg.market("kr").costs.sell_tax_pct == 0.0
    text = (ROOT / "config.daily.yaml").read_text(encoding="utf-8")
    bad = tmp_path / "flatten.yaml"
    bad.write_text(text.replace("flatten_at_session_end: false", "flatten_at_session_end: true"), encoding="utf-8")
    with pytest.raises(ConfigError, match="flatten_at_session_end"):
        load_config(bad)
    hourly = tmp_path / "hourly.yaml"
    hourly.write_text(text.replace("interval: 1d", "interval: 1h"), encoding="utf-8")
    with pytest.raises(ConfigError, match="일봉 전용"):
        load_config(hourly)


# --------------------------------------------------------------------------
# 주문표
# --------------------------------------------------------------------------
def test_parse_held(dcfg):
    kr = dcfg.market("kr")
    held = parse_held(["069500", "091160.KS@105000,102970"], kr)
    assert held == [Held("069500.KS"), Held("091160.KS", 105000.0), Held("102970.KS")]
    with pytest.raises(ConfigError, match="쉼표"):
        parse_held(["069500@105,000"], kr)  # 쉼표는 종목 구분자
    for bad in (["005930"], ["069500,069500.KS"], ["069500@abc"], ["069500@0"]):
        with pytest.raises(ConfigError):
            parse_held(bad, kr)


def sig_frame(rows) -> pd.DataFrame:
    base = {"valid": True, "stale": False, "진입": False, "유지": True, "청산선": 0.0, "점수%": 0.0}
    return pd.DataFrame([{**base, "종목": s, **r} for s, r in rows.items()], index=list(rows))


def test_plan_orders_sells_ranks_and_sizes(dcfg):
    sig = sig_frame(
        {
            "A": {"종가": 100.0, "유지": False, "청산선": 105.0},  # 보유 · 20일선 이탈 → 매도
            "B": {"종가": 91.0},  # 보유 · 매수가 100 의 -8%(92) 이하 → 손절
            "C": {"종가": 50.0, "진입": True, "점수%": 30.0},  # 보유 중이면 진입 신호여도 추가 매수 안 함
            "D": {"종가": 10_000.0, "진입": True, "점수%": 5.0},
            "E": {"종가": 111_520.0, "진입": True, "점수%": 10.0},
            "F": {"종가": 3_000.0, "진입": True, "점수%": 1.0},
            "G": {"종가": 3_000.0, "진입": True, "점수%": 50.0, "stale": True},  # 최신 일봉 없음 → 제외
        }
    )
    held = [Held("A"), Held("B", 100.0), Held("C")]
    plan = plan_orders(sig, held, 1_000_000, dcfg.strategy, dcfg.risk, dcfg.market("kr").costs)
    assert [o.ticker for o in plan.sells] == ["A", "B"]
    assert "청산선" in plan.sells[0].reason and "손절" in plan.sells[1].reason
    assert plan.slots == 2  # 최대 3종목 - 남는 보유(C) 1
    assert [o.ticker for o in plan.buys] == ["E", "D"] and plan.waiting == ["F"]
    e, d = plan.buys
    assert e.limit == 113_700  # 111,520 × 1.02 = 113,750.4 → 호가단위 100원 내림
    assert d.limit == 10_200
    fee = dcfg.market("kr").costs.commission
    assert e.qty == int(330_000 // (113_700 * (1 + fee))) == 2
    assert d.qty == int(330_000 // (10_200 * (1 + fee)))


def test_plan_orders_keeps_holdings_it_cannot_judge(dcfg):
    sig = sig_frame({"A": {"종가": 100.0, "유지": False, "stale": True}})
    plan = plan_orders(sig, [Held("A"), Held("Z")], 1_000_000, dcfg.strategy, dcfg.risk, dcfg.market("kr").costs)
    assert plan.sells == [] and plan.slots == 1 and len(plan.notes) == 2


def test_buy_limit_price_is_on_tick_grid():
    assert buy_limit_price(149_700, 2.0) == 152_600  # 152,694 → 100원 단위
    assert buy_limit_price(6_740, 2.0) == 6_870  # 6,874.8 → 10원 단위
    assert buy_limit_price(1_500, 2.0) == 1_530


def test_rule_signals_flags_stale_daily_data(dcfg, monkeypatch):
    kr = dcfg.market("kr")
    rng = np.random.default_rng(1)
    bars = {s: daily_bars(100 * np.cumprod(1 + rng.normal(0.001, 0.01, 260)), start="2024-03-01") for s in kr.symbols}
    last = bars["069500.KS"].index[-1]
    monkeypatch.setattr(daily, "load_market", lambda *a, **k: (bars, {}))
    same_evening = last + pd.Timedelta(hours=17)  # 그날 17:00 → 그날 일봉이 최신이어야 함
    sig, errors = rule_signals(dcfg, "kr", now=same_evening)
    assert not sig["stale"].any() and errors == {}
    expect = trend_breakout(bars["069500.KS"], dcfg.strategy).iloc[-1]
    assert sig.loc["069500.KS", "진입"] == bool(expect["entry"]) and sig.loc["069500.KS", "유지"] == bool(expect["hold"])
    next_evening = same_evening + pd.Timedelta(days=1)
    while next_evening.weekday() >= 5:
        next_evening += pd.Timedelta(days=1)
    stale, _ = rule_signals(dcfg, "kr", now=next_evening)
    assert stale["stale"].all() and stale["비고"].str.contains("지연").all()
