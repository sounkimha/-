"""일봉 trade: 가상계좌가 백테스트와 같은 결과를 내는지, 모의투자 주문/확인 단계(가짜 증권사)가 맞게 도는지."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import requests

import trader.broker
import trader.daily_trade as dt
from conftest import ROOT, daily_bars, random_daily
from trader.backtest import run_backtest
from trader.broker import Balance, OrderRequest, OrderResult, Position
from trader.config import load_config
from trader.rules import first_valid_time, trend_breakout

KST = "Asia/Seoul"


def no_network(*a, **k):
    raise AssertionError("테스트에서 네트워크 호출이 일어나면 안 됩니다")


def small_cfg(tmp_path, max_positions=2, limit_pct=2.0, **risk):
    cfg = load_config(ROOT / "config.daily.yaml")
    return replace(
        cfg,
        paths=replace(cfg.paths, state_dir=str(tmp_path / "state")),
        strategy=replace(
            cfg.strategy, long_ma=20, breakout_lookback=5, exit_ma=5, score_lookback=5,
            max_positions=max_positions, reentry_cooldown_bars=2, entry_limit_pct=limit_pct,
        ),
        risk=replace(cfg.risk, **risk),
    )


# --------------------------------------------------------------------------
# DRY_RUN 가상계좌
# --------------------------------------------------------------------------
@pytest.fixture
def paper(tmp_path, monkeypatch):
    monkeypatch.delenv("DRY_RUN", raising=False)
    monkeypatch.setattr(dt, "load_env", lambda root: None)
    monkeypatch.setattr(requests.Session, "request", no_network)
    syms = list(load_config(ROOT / "config.daily.yaml").market("kr").symbols)[:4]
    full = {s: random_daily(170, seed=10 + i) for i, s in enumerate(syms)}
    ctx = {"upto": None}

    def fake_load(cfg, mk, offline=False, refresh=False, now=None):  # 그날 저녁에 받을 수 있는 데이터만
        return {s: df[df.index <= ctx["upto"]] for s, df in full.items()}, {}

    monkeypatch.setattr(dt, "load_market", fake_load)
    return full, ctx, tmp_path


def run_days(cfg, ctx, days):
    events, reps = [], []
    for d in days:
        ctx["upto"] = d
        rep = dt.run_daily_cycle(cfg, "kr", now=d + pd.Timedelta(hours=17))
        events += rep.events
        reps.append(rep)
    return events, reps


@pytest.mark.parametrize(
    "mdd, size, max_pos, stop, limit",
    [
        (40.0, 33.0, 2, 8.0, 2.0),
        (4.0, 33.0, 2, 8.0, 2.0),  # 중간에 계좌 낙폭 한도로 중단되는 경우
        (40.0, 45.0, 3, 8.0, 2.0),  # 현금이 모자라 매수 순서(점수 순)가 결과를 바꾸는 경우
        (40.0, 33.0, 2, 2.0, 2.0),  # 종가 손절이 자주 걸리는 경우
        (40.0, 33.0, 2, 8.0, 0.3),  # 지정가 미체결이 자주 생기는 경우
    ],
)
def test_paper_account_matches_backtest_day_by_day(paper, mdd, size, max_pos, stop, limit):
    full, ctx, tmp_path = paper
    cfg = small_cfg(
        tmp_path, max_positions=max_pos, limit_pct=limit, max_drawdown_pct=mdd, position_size_pct=size, stop_loss_pct=stop
    )
    market = cfg.market("kr")
    sigs = {s: trend_breakout(df, cfg.strategy) for s, df in full.items()}
    start = min(first_valid_time(x) for x in sigs.values())
    bt = run_backtest(full, sigs, market, cfg.strategy, cfg.risk, start=start)
    days = [d for d in next(iter(full.values())).index if d >= start]
    events, reps = run_days(cfg, ctx, days)

    def key(day, s, qty, price, reason=""):
        return (day, market.label(s) if "(" not in s else s, int(qty), round(float(price), 2), reason)

    buys = sorted(key(e["일자"], e["종목"], e["수량"], e["가격"]) for e in events if e["구분"] == "매수")
    sells = sorted(key(e["일자"], e["종목"], e["수량"], e["가격"], e["사유"]) for e in events if e["구분"] == "매도")
    bt_buys = sorted(key(f"{t.entry_time:%Y-%m-%d}", t.symbol, t.qty, t.entry_price) for t in bt.trades)
    bt_sells = sorted(
        key(f"{t.exit_time:%Y-%m-%d}", t.symbol, t.qty, t.exit_price, dt.REASONS.get(t.exit_reason, t.exit_reason))
        for t in bt.trades if t.exit_reason != "end_of_data"
    )
    assert len(bt.trades) >= 6  # 비교가 의미 있을 만큼 거래가 있어야 함
    assert buys == bt_buys
    assert sells == bt_sells
    assert sum(e["구분"] == "미체결" for e in events) == bt.unfilled_entries
    equity = pd.read_csv(dt.daily_files(cfg, "kr", True)["equity"], encoding="utf-8-sig")
    assert equity["평가금액"].iloc[-2] == pytest.approx(bt.equity.iloc[-2], abs=0.01)  # 마지막 날은 백테스트가 강제 청산
    assert (bt.halted_at is not None) == reps[-1].halted == (mdd == 4.0)
    if stop == 2.0:
        assert sum(t.exit_reason == "stop" for t in bt.trades) >= 3  # 손절 경로가 실제로 검증되도록
    if limit == 0.3:
        assert bt.unfilled_entries >= 3  # 미체결 경로가 실제로 검증되도록
    if bt.halted_at is not None:
        assert not any(e["구분"] == "매수" and e["일자"] > f"{bt.halted_at:%Y-%m-%d}" for e in events)


def test_paper_catches_up_missed_days_and_is_idempotent(paper):
    full, ctx, tmp_path = paper
    every = small_cfg(tmp_path / "every", max_drawdown_pct=40.0)
    gaps = small_cfg(tmp_path / "gaps", max_drawdown_pct=40.0)
    days = list(next(iter(full.values())).index[30:])
    run_days(every, ctx, days)
    events_gap, reps = run_days(gaps, ctx, [days[0], days[10], days[11], days[40], days[-1], days[-1]])
    assert "밀린" in " ".join(reps[1].notes)
    assert reps[-1].events == [] and "새 일봉이 없습니다" in " ".join(reps[-1].notes)  # 같은 날 두 번 실행해도 중복 처리 없음
    a = dt.DailyState.load(dt.daily_files(every, "kr", True)["state"])
    b = dt.DailyState.load(dt.daily_files(gaps, "kr", True)["state"])
    assert a.cash == pytest.approx(b.cash) and a.positions == b.positions and a.pending == b.pending
    assert a.last_date == b.last_date == f"{days[-1]:%Y-%m-%d}"
    journal = pd.read_csv(dt.daily_files(gaps, "kr", True)["journal"], encoding="utf-8-sig")
    assert list(journal.columns) == dt.JOURNAL_COLUMNS and len(journal) == len(events_gap)


def test_paper_first_run_only_plans_and_shows_order_sheet(paper):
    full, ctx, tmp_path = paper
    cfg = small_cfg(tmp_path, max_drawdown_pct=40.0)
    days = next(iter(full.values())).index
    # 마지막 날에 진입 신호가 있는 날을 찾는다
    sigs = {s: trend_breakout(df, cfg.strategy) for s, df in full.items()}
    d = next(day for day in days[40:] if any(bool(x.loc[day, "entry"]) for x in sigs.values()))
    _, (rep,) = run_days(cfg, ctx, [d])
    assert rep.events == [] and rep.holdings == []
    assert rep.planned and all(p["구분"] == "매수" and p["주문"].startswith("지정가") for p in rep.planned)
    assert rep.equity == pytest.approx(cfg.market("kr").initial_capital)


# --------------------------------------------------------------------------
# DRY_RUN=false: 모의투자 (가짜 증권사)
# --------------------------------------------------------------------------
class FakeBroker:
    def __init__(self):
        self.cash = 400_000.0
        self.positions: dict[str, Position] = {}
        self.placed, self.cancels, self.rows, self.quotes, self.reject = [], [], {}, {}, set()
        self.n = 0

    def balance(self):
        mv = sum(p.qty * p.last_price for p in self.positions.values())
        return Balance("KRW", self.cash, self.cash + mv, dict(self.positions))

    def place(self, side, s, qty, limit=None):
        self.n += 1
        req = OrderRequest("domestic", side, s, qty, limit, "/x", "VTTC0012U", {})
        if (side, s) in self.reject:
            return OrderResult(req, sent=True, dry_run=False, ok=False, message="가짜 거부")
        self.placed.append((side, s, qty, limit))
        return OrderResult(req, sent=True, dry_run=False, ok=True, order_no=f"{self.n:010d}", org_no="00950")

    def cancel(self, s, order_no, org_no, qty):
        self.cancels.append((s, order_no, org_no, qty))
        req = OrderRequest("domestic", "cancel", s, qty, None, "/x", "VTTC0013U", {})
        return OrderResult(req, sent=True, dry_run=False, ok=True, order_no="0000009999")

    def daily_orders(self, day):
        return list(self.rows.get(day, []))

    def quote(self, s):
        return self.quotes.get(s, {"price": 0.0, "open": 0.0, "base": 0.0})

    def fill(self, day, order_no, s, side, qty, filled, avg):
        self.rows.setdefault(day, []).append(
            {"odno": order_no, "pdno": s.split(".")[0], "sll_buy_dvsn_cd": "01" if side == "sell" else "02",
             "ord_qty": str(qty), "tot_ccld_qty": str(filled), "avg_prvs": str(avg), "rmn_qty": str(qty - filled),
             "cncl_yn": "N", "ord_orgno": "00950"}
        )


@pytest.fixture
def vts(tmp_path, monkeypatch):
    monkeypatch.setenv("DRY_RUN", "false")
    for k, v in {"KIS_APP_KEY": "k", "KIS_APP_SECRET": "s", "KIS_ACCOUNT_NO": "12345678"}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(dt, "load_env", lambda root: None)
    monkeypatch.setattr(requests.Session, "request", no_network)
    fake = FakeBroker()
    monkeypatch.setattr(trader.broker, "make_broker", lambda *a, **k: fake)
    cfg = small_cfg(tmp_path, max_drawdown_pct=40.0)
    cfg = replace(cfg, strategy=replace(cfg.strategy, max_positions=3))
    s = list(cfg.market("kr").symbols)
    A, B, C, D, E = s[:5]
    up = lambda top: np.linspace(100, top, 40)  # noqa: E731  매일 신고가 → 마지막 날 진입 신호
    bars = {
        A: daily_bars(list(np.linspace(100, 130, 35)) + [125, 120, 115, 110, 105], start="2025-01-13"),  # 20일선 이탈
        B: daily_bars(up(140), start="2025-01-13"),  # 점수 1위
        C: daily_bars(up(135), start="2025-01-13"),
        D: daily_bars(up(132), start="2025-01-13"),
        E: daily_bars(up(150), start="2025-01-13"),  # 직접 보유 중인 종목 → 사지 않음
    }
    assert all(df.index[-1] == pd.Timestamp("2025-03-07", tz=KST) for df in bars.values())  # 금요일
    monkeypatch.setattr(dt, "load_market", lambda *a, **k: (bars, {}))
    fake.positions = {A: Position(A, 2000, 110.0, 105.0, sellable_qty=2000), E: Position(E, 10, 140.0, 150.0, sellable_qty=10)}
    files = dt.daily_files(cfg, "kr", False)
    st = dt.DailyState(files["state"], positions={A: {"qty": 2000, "entry_price": 110.0, "stop_price": 101.2, "entry_date": "2025-02-20"}})
    st.save()
    return cfg, fake, bars, (A, B, C, D, E), files


def kst(s):
    return pd.Timestamp(s, tz=KST)


def test_vts_order_phase_sells_buys_within_cash_and_defers_the_rest(vts):
    cfg, fake, bars, (A, B, C, D, E), files = vts
    rep = dt.run_daily_cycle(cfg, "kr", now=kst("2025-03-10 08:35"))  # 월요일 동시호가
    assert rep.phase == "주문" and not rep.dry_run
    budget = 0.33 * (400_000 + 2000 * 105 + 10 * 150)
    assert fake.placed[0] == ("sell", A, 2000, None)  # 시장가 매도
    side, sym, qty, limit = fake.placed[1]
    assert (side, sym, limit) == ("buy", B, 142)  # 140 × 1.02 = 142.8 → 호가단위(2천원 미만 1원)로 내림
    assert qty == int(budget // (limit * (1 + cfg.market("kr").costs.commission)))
    assert len(fake.placed) == 2  # 현금 40만원 → 매수 1건만, 나머지는 대기
    st = dt.DailyState.load(files["state"])
    assert [d["symbol"] for d in st.deferred] == [C, D]  # 점수 순
    assert all(p[1] != E for p in fake.placed)  # 직접 보유 종목은 건드리지 않음
    assert any("건드리지 않습니다" in n for n in rep.notes)

    again = dt.run_daily_cycle(cfg, "kr", now=kst("2025-03-10 08:45"))
    assert len(fake.placed) == 2 and any("이미 실행" in n for n in again.notes)  # 중복 주문 없음


def test_vts_check_phase_records_fills_cancels_rest_and_places_deferred(vts):
    cfg, fake, bars, (A, B, C, D, E), files = vts
    dt.run_daily_cycle(cfg, "kr", now=kst("2025-03-10 08:35"))
    (_, _, qa, _), (_, _, qb, lb) = fake.placed
    st = dt.DailyState.load(files["state"])
    no = {o["symbol"]: o["order_no"] for o in st.orders}
    fake.fill("20250310", no[A], A, "sell", qa, qa, 104.0)
    fake.fill("20250310", no[B], B, "buy", qb, qb - 100, 141.0)  # 일부만 체결 → 나머지 취소
    fake.positions = {B: Position(B, qb - 100, 141.0, 141.0, sellable_qty=qb - 100), E: fake.positions[E]}
    fake.cash = 400_000 + 2000 * 104 - (qb - 100) * 141.0
    lc = st.deferred[0]["limit"]
    fake.quotes = {C: {"price": lc, "open": lc - 1, "base": 0.0}, D: {"price": 999.0, "open": 999.0, "base": 0.0}}

    rep = dt.run_daily_cycle(cfg, "kr", now=kst("2025-03-10 09:05"))
    assert rep.phase == "확인"
    st = dt.DailyState.load(files["state"])
    assert A not in st.positions and st.last_exit[A] == "2025-03-10"
    memo = st.positions[B]
    assert memo["qty"] == qb - 100 and memo["entry_price"] == 141.0
    assert memo["stop_price"] == pytest.approx(141.0 * 0.92)
    assert fake.cancels == [(B, no[B], "00950", 100)]
    assert ("buy", C, st.orders[-1]["qty"], lc) == fake.placed[-1]  # 시가 ≤ 지정가 → 대기 매수 주문
    assert any(e["구분"] == "미체결" and D in e["종목"] for e in rep.events)  # 시가 > 지정가 → 주문 안 함
    assert st.deferred == []
    kinds = [e["구분"] for e in rep.events]
    assert {"매도 체결", "매수 체결", "매수 잔량 취소", "매수 주문", "미체결"} <= set(kinds)


def test_vts_rejected_auction_sell_is_retried_once_after_open(vts):
    cfg, fake, bars, (A, B, C, D, E), files = vts
    fake.reject = {("sell", A)}
    dt.run_daily_cycle(cfg, "kr", now=kst("2025-03-10 08:35"))
    assert not any(p[:2] == ("sell", A) for p in fake.placed)
    fake.reject = set()
    dt.run_daily_cycle(cfg, "kr", now=kst("2025-03-10 09:05"))
    assert [p for p in fake.placed if p[:2] == ("sell", A)] == [("sell", A, 2000, None)]
    dt.run_daily_cycle(cfg, "kr", now=kst("2025-03-10 09:20"))
    assert len([p for p in fake.placed if p[:2] == ("sell", A)]) == 1  # 한 번만


def test_vts_stale_data_uses_broker_base_price_for_holidays(vts, monkeypatch):
    cfg, fake, bars, (A, B, C, D, E), files = vts
    cut = {s: df.iloc[:-1] for s, df in bars.items()}  # 금요일 봉이 없음 (휴장? 지연?)
    monkeypatch.setattr(dt, "load_market", lambda *a, **k: (cut, {}))
    fake.quotes = {
        B: {"price": 0.0, "open": 0.0, "base": float(cut[B]["close"].iloc[-1])},  # 기준가 = 마지막 종가 → 금요일 휴장
        C: {"price": 0.0, "open": 0.0, "base": 1.0},  # 다름 → 데이터 지연
    }
    rep = dt.run_daily_cycle(cfg, "kr", now=kst("2025-03-10 08:35"))
    bought = {p[1] for p in fake.placed if p[0] == "buy"}
    assert B in bought and C not in bought
    assert any("최신 일봉이 아닙니다" in n and C in n for n in rep.notes)


def test_vts_previous_day_orders_are_reconciled_then_expired(vts):
    cfg, fake, bars, (A, B, C, D, E), files = vts
    dt.run_daily_cycle(cfg, "kr", now=kst("2025-03-10 08:35"))
    st = dt.DailyState.load(files["state"])
    no = {o["symbol"]: o["order_no"] for o in st.orders}
    (_, _, qb, _) = fake.placed[1]
    # 확인 단계를 못 돌렸는데 매수는 장중에 체결됨 → 다음 날 아침에 반영되고 나머지 주문은 만료
    fake.fill("20250310", no[B], B, "buy", qb, qb, 142.0)
    fake.positions = {B: Position(B, qb, 142.0, 142.0, sellable_qty=qb), E: fake.positions[E]}
    rep = dt.run_daily_cycle(cfg, "kr", now=kst("2025-03-11 08:35"))
    st = dt.DailyState.load(files["state"])
    assert st.positions[B]["qty"] == qb
    statuses = {o["symbol"]: o["status"] for o in st.orders if o["date"] == "2025-03-10"}
    assert statuses[B] == "filled" and statuses[A] == "expired"
    assert any("만료" in n for n in rep.notes)
    assert any("관리 종료" in n and A in n for n in rep.notes)  # 매도 체결 기록 없이 잔고에서 사라진 A


def test_vts_halt_during_check_phase_drops_deferred_buys(vts):
    cfg, fake, bars, (A, B, C, D, E), files = vts
    cfg = replace(cfg, risk=replace(cfg.risk, max_drawdown_pct=10.0))
    dt.run_daily_cycle(cfg, "kr", now=kst("2025-03-10 08:35"))
    st = dt.DailyState.load(files["state"])
    no = {o["symbol"]: o["order_no"] for o in st.orders}
    (_, _, qa, _), (_, _, qb, _) = fake.placed
    fake.fill("20250310", no[A], A, "sell", qa, qa, 104.0)
    fake.fill("20250310", no[B], B, "buy", qb, qb, 142.0)
    fake.positions = {B: Position(B, qb, 142.0, 60.0, sellable_qty=qb), E: fake.positions[E]}  # 장중 폭락
    fake.cash = 400_000 + 2000 * 104 - qb * 142.0
    fake.quotes = {C: {"price": 100.0, "open": 100.0, "base": 0.0}, D: {"price": 100.0, "open": 100.0, "base": 0.0}}
    rep = dt.run_daily_cycle(cfg, "kr", now=kst("2025-03-10 09:05"))
    assert rep.halted and dt.DailyState.load(files["state"]).deferred == []
    assert not any(p[0] == "buy" and p[1] in (C, D) for p in fake.placed)


def test_vts_outside_windows_only_reads(vts):
    cfg, fake, bars, *_ = vts
    rep = dt.run_daily_cycle(cfg, "kr", now=kst("2025-03-10 16:00"))
    assert rep.phase == "조회" and fake.placed == [] and fake.cancels == []
    assert dt.vts_phase(kst("2025-03-08 08:35")) == "조회"  # 토요일
    assert dt.vts_phase(kst("2025-03-10 08:29")) == "조회" and dt.vts_phase(kst("2025-03-10 15:20")) == "조회"
    assert dt.vts_phase(kst("2025-03-10 09:00")) == "조회" and dt.vts_phase(kst("2025-03-10 09:02")) == "확인"  # 체결 반영 대기
