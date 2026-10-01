"""매매 사이클 테스트: 네트워크 없이(가짜 데이터·가짜 증권사) 신호 → 리스크 → 주문/가상계좌 → 상태 저장까지."""
from dataclasses import replace

import pandas as pd
import pytest
import requests

import trader.broker
import trader.live as live
from conftest import make_bars
from trader.broker import Balance, KisApiError, OrderRequest, OrderResult, Position

KST, ET = "Asia/Seoul", "America/New_York"


@pytest.fixture
def setup(cfg, tmp_path, monkeypatch):
    monkeypatch.delenv("DRY_RUN", raising=False)  # 기본값(=DRY_RUN 켜짐)
    monkeypatch.setattr(live, "load_env", lambda root: None)  # 개발 PC 의 .env 를 읽지 않게

    def no_network(*a, **k):
        raise AssertionError("테스트에서 네트워크 호출이 일어나면 안 됩니다")

    monkeypatch.setattr(requests.Session, "request", no_network)
    c = replace(cfg, paths=replace(cfg.paths, state_dir=str(tmp_path / "state")))
    tickers = list(c.market("kr").symbols)
    ctx = {
        "bars": {t: make_bars(30, seed=i, start="2025-02-03") for i, t in enumerate(tickers)},  # 마지막 봉 03-14 15:00
        "errors": {},
        "prob": 0.9,
    }
    monkeypatch.setattr(
        live, "load_market", lambda cfg, mk, offline=False, refresh=False, now=None: (ctx["bars"], ctx["errors"])
    )

    def fake_predict(ds, model_cfg, cost):
        last = ds.groupby("symbol", sort=False).tail(1)
        return pd.DataFrame(
            {"time": last["time"].to_numpy(), "prob": ctx["prob"], "exp_ret": 0.01}, index=last["symbol"].to_numpy()
        )

    monkeypatch.setattr(live, "predict_latest", fake_predict)
    return c, ctx, tmp_path, tickers


def kst(s):
    return pd.Timestamp(s, tz=KST)


def load_state(cfg, tmp_path, mode="paper", market="kr"):
    return live.TradeState.load(tmp_path / "state" / f"trade_state_{market}_{mode}.json", cfg.market(market))


# --------------------------------------------------------------------------
# DRY_RUN (가상계좌)
# --------------------------------------------------------------------------
def test_dry_run_cycle_buys_then_sells_in_paper_account(setup):
    cfg, ctx, tmp_path, tickers = setup
    now = kst("2025-03-14 10:05")  # 금요일 장중
    rep = live.run_trade_cycle(cfg, "kr", now=now)
    assert rep.dry_run and rep.market_open
    buys = [e for e in rep.executed if e["side"] == "buy"]
    assert len(buys) == 3 and all(not e["sent"] for e in buys)
    assert (tmp_path / "state" / "trade_state_kr_paper.json").exists()

    ctx["prob"] = 0.1  # 신호 약화 → 전량 청산
    rep2 = live.run_trade_cycle(cfg, "kr", now=now + pd.Timedelta(hours=1))
    sells = [e for e in rep2.executed if e["side"] == "sell"]
    assert len(sells) == 3 and {e["reason"] for e in sells} == {"signal"}
    st = load_state(cfg, tmp_path)
    assert st.paper_positions == {}
    loss = cfg.market("kr").initial_capital - st.paper_cash
    assert 0 < loss < cfg.market("kr").initial_capital * 0.01  # 비용만큼만 줄었음


def test_daily_entry_limit_applies_in_live_cycle(setup):
    cfg, _, _, _ = setup
    cfg = replace(cfg, risk=replace(cfg.risk, max_trades_per_day=2))
    rep = live.run_trade_cycle(cfg, "kr", now=kst("2025-03-14 10:05"))
    assert len([e for e in rep.executed if e["side"] == "buy"]) == 2
    assert any("한도" in n for n in rep.notes)


def test_no_orders_outside_market_hours(setup):
    cfg, _, _, _ = setup
    night = kst("2025-03-14 20:00")
    rep = live.run_trade_cycle(cfg, "kr", now=night)
    assert rep.executed == [] and not rep.market_open
    rep2 = live.run_trade_cycle(cfg, "kr", now=night, ignore_hours=True)  # DRY_RUN 에서만 허용
    assert len(rep2.executed) == 3


def test_missing_price_for_held_symbol_does_not_abort_cycle(setup):
    cfg, ctx, tmp_path, tickers = setup
    live.run_trade_cycle(cfg, "kr", now=kst("2025-03-14 10:05"))
    ctx["bars"] = {t: b for t, b in ctx["bars"].items() if t != tickers[2]}  # 세 번째 종목 시세 수집 실패
    ctx["errors"] = {tickers[2]: "download failed"}
    rep = live.run_trade_cycle(cfg, "kr", now=kst("2025-03-14 11:05"))
    sold = {e["symbol"]: e["reason"] for e in rep.executed if e["side"] == "sell"}
    assert sold == {tickers[2]: "signal"}  # 신호를 못 구한 보유 종목만 보수적으로 청산, 나머지는 계속 보유
    assert any("시세 없음" in n for n in rep.notes)
    assert set(load_state(cfg, tmp_path).paper_positions) == {tickers[0], tickers[1]}


def test_stale_data_blocks_entries_and_clock_flattens_at_last_bar(setup):
    """미장 15:31(마지막 봉 구간)인데 최신 봉이 13:30 이면: 신규 매수 금지 + 보유분은 시계 기준으로 장마감 청산."""
    cfg, ctx, tmp_path, _ = setup
    us = list(cfg.market("us").symbols)
    full = {t: make_bars(30, seed=i, start="2025-02-03", tz=ET, first_bar="09:30") for i, t in enumerate(us)}
    day = pd.Timestamp("2025-03-14", tz=ET)

    ctx["bars"] = {t: b[b.index <= day + pd.Timedelta(hours=9, minutes=30)] for t, b in full.items()}
    rep = live.run_trade_cycle(cfg, "us", now=pd.Timestamp("2025-03-14 10:31", tz=ET))
    assert len([e for e in rep.executed if e["side"] == "buy"]) == 3

    ctx["bars"] = {t: b[b.index <= day + pd.Timedelta(hours=13, minutes=30)] for t, b in full.items()}  # 14:30 봉 누락
    rep = live.run_trade_cycle(cfg, "us", now=pd.Timestamp("2025-03-14 15:31", tz=ET))
    assert rep.signals["stale"].all() and (rep.signals["신호"] == live.WAIT).all()
    assert [e["side"] for e in rep.executed] == ["sell"] * 3
    assert {e["reason"] for e in rep.executed} == {"session_end"}
    assert load_state(cfg, tmp_path, market="us").paper_positions == {}


# --------------------------------------------------------------------------
# DRY_RUN=false (모의투자 서버) — 가짜 증권사로 대조 로직만 검증
# --------------------------------------------------------------------------
class FakeBroker:
    def __init__(self, equity=10_000_000.0):
        self.orders, self.positions, self.fail, self.equity = [], {}, {}, equity
        self.uncertain = False

    def price(self, s):
        return 100.0

    def balance(self):
        mv = sum(p.qty * p.last_price for p in self.positions.values())
        return Balance("KRW", self.equity - mv, self.equity, dict(self.positions))

    def _order(self, side, s, qty, ref):
        if s in self.fail:
            raise self.fail[s]
        self.orders.append((side, s, qty))
        req = OrderRequest("domestic", side, s, qty, None, "/x", "VTTC0012U", {})
        if self.uncertain:
            return OrderResult(req, sent=True, dry_run=False, ok=False, uncertain=True, message="ReadTimeout")
        return OrderResult(req, sent=True, dry_run=False, ok=True, order_no=f"{len(self.orders):04d}")

    def buy(self, s, qty, ref=None):
        return self._order("buy", s, qty, ref)

    def sell(self, s, qty, ref=None):
        return self._order("sell", s, qty, ref)


@pytest.fixture
def vts(setup, monkeypatch):
    cfg, ctx, tmp_path, tickers = setup
    monkeypatch.setenv("DRY_RUN", "false")
    for k, v in {"KIS_APP_KEY": "k", "KIS_APP_SECRET": "s", "KIS_ACCOUNT_NO": "12345678"}.items():
        monkeypatch.setenv(k, v)
    fake = FakeBroker()
    monkeypatch.setattr(trader.broker, "make_broker", lambda *a, **k: fake)
    return cfg, ctx, tmp_path, tickers, fake


def test_vts_cycle_reconciles_orders_with_broker_balance(vts):
    cfg, ctx, tmp_path, tickers, fake = vts
    now = kst("2025-03-14 10:05")
    fake.fail = {tickers[2]: KisApiError("가짜 거부")}  # 세 번째 주문 거부 → 기록만 하고 계속
    rep = live.run_trade_cycle(cfg, "kr", now=now)
    assert not rep.dry_run
    assert [o[:2] for o in fake.orders] == [("buy", tickers[0]), ("buy", tickers[1])]
    failed = [e for e in rep.executed if e["symbol"] == tickers[2]]
    assert failed and failed[0]["ok"] is False and "가짜 거부" in failed[0]["message"]
    st = load_state(cfg, tmp_path, "vts")
    assert set(st.live_positions) == {tickers[0], tickers[1]}
    assert st.live_positions[tickers[0]]["stop_price"] == pytest.approx(99.0)

    # 다음 사이클: 첫 종목만 체결(두 번째는 미체결) → 두 번째는 추적 종료, 신호 약화로 첫 종목 매도 '접수'
    q0 = fake.orders[0][2]
    fake.positions = {tickers[0]: Position(tickers[0], q0, 100.0, 100.0, sellable_qty=q0)}
    fake.orders, fake.fail, ctx["prob"] = [], {}, 0.1
    rep = live.run_trade_cycle(cfg, "kr", now=now + pd.Timedelta(hours=1))
    assert fake.orders == [("sell", tickers[0], q0)]
    assert any("추적 종료" in n for n in rep.notes)
    st = load_state(cfg, tmp_path, "vts")
    assert st.live_positions[tickers[0]]["exit_pending"] is True  # 접수 ≠ 체결 → 추적 유지

    # 매도가 아직 미체결(주문가능수량 0) → 중복 매도 없이 대기, 손절가 아래로 떨어져도 마찬가지
    fake.positions = {tickers[0]: Position(tickers[0], q0, 100.0, 50.0, sellable_qty=0)}
    fake.orders = []
    rep = live.run_trade_cycle(cfg, "kr", now=now + pd.Timedelta(hours=2))
    assert fake.orders == [] and any("체결 대기" in n for n in rep.notes)

    # 잔고에서 사라짐 → 청산 체결 확인 후 추적 종료
    fake.positions = {}
    rep = live.run_trade_cycle(cfg, "kr", now=now + pd.Timedelta(hours=3))
    assert any("청산 체결 확인" in n for n in rep.notes)
    assert load_state(cfg, tmp_path, "vts").live_positions == {}


def test_uncertain_buy_is_tracked_once_broker_shows_it(vts):
    cfg, ctx, tmp_path, tickers, fake = vts
    now = kst("2025-03-14 10:05")
    fake.uncertain = True  # 응답 없음(타임아웃) → 접수 여부 불명
    rep = live.run_trade_cycle(cfg, "kr", now=now)
    assert all(e.get("uncertain") for e in rep.executed)
    assert all(m["unconfirmed"] for m in load_state(cfg, tmp_path, "vts").live_positions.values())

    # 실제로는 첫 종목이 체결돼 있었고, 가격이 손절가 아래 → 추적이 이어져 손절 매도
    q0 = fake.orders[0][2]
    fake.positions = {tickers[0]: Position(tickers[0], q0, 100.0, 50.0, sellable_qty=q0)}
    fake.orders, fake.uncertain = [], False
    fake.price = lambda s: 50.0
    rep = live.run_trade_cycle(cfg, "kr", now=now + pd.Timedelta(hours=1))
    assert ("sell", tickers[0], q0) in fake.orders
    assert {e["reason"] for e in rep.executed if e["side"] == "sell"} == {"stop"}
    assert not any(e["side"] == "buy" and e["symbol"] == tickers[0] for e in rep.executed)  # 중복 매수 없음


def test_state_is_saved_even_if_an_order_raises(vts):
    cfg, ctx, tmp_path, tickers, fake = vts
    fake.fail = {tickers[1]: RuntimeError("예상 못 한 오류")}
    with pytest.raises(RuntimeError):
        live.run_trade_cycle(cfg, "kr", now=kst("2025-03-14 10:05"))
    st = load_state(cfg, tmp_path, "vts")
    assert set(st.live_positions) == {tickers[0]}  # 이미 접수된 첫 주문은 추적됨
    assert [h["symbol"] for h in st.history] == [tickers[0], tickers[1]]


def test_vts_drawdown_baseline_is_first_broker_balance(vts):
    cfg, ctx, tmp_path, tickers, fake = vts
    fake.equity = 8_500_000.0  # 설정의 initial_capital(1천만)보다 작은 모의계좌
    rep = live.run_trade_cycle(cfg, "kr", now=kst("2025-03-14 10:05"))
    assert not rep.halted and rep.drawdown == 0
    assert load_state(cfg, tmp_path, "vts").risk["peak_equity"] == 8_500_000.0
    assert not (tmp_path / "state" / "trade_state_kr_paper.json").exists()  # 가상계좌 상태와 분리
