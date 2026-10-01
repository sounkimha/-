"""DRY_RUN 매매 사이클: 네트워크 없이 신호 → 리스크 → 가상계좌 기록까지."""
from dataclasses import replace

import pandas as pd
import pytest
import requests

import trader.live as live
from conftest import make_bars


@pytest.fixture
def setup(cfg, tmp_path, monkeypatch):
    monkeypatch.delenv("DRY_RUN", raising=False)  # 기본값(=DRY_RUN 켜짐) 확인
    monkeypatch.setattr(live, "load_env", lambda root: None)  # 개발 PC 의 .env 를 읽지 않게

    def no_network(*a, **k):
        raise AssertionError("DRY_RUN 사이클에서 네트워크 호출이 일어나면 안 됩니다")

    monkeypatch.setattr(requests.Session, "request", no_network)
    c = replace(cfg, paths=replace(cfg.paths, state_dir=str(tmp_path / "state")))
    tickers = list(c.market("kr").symbols)
    bars = {t: make_bars(30, seed=i, start="2025-02-03") for i, t in enumerate(tickers)}
    monkeypatch.setattr(live, "load_market", lambda cfg, mk, offline=False, refresh=False, now=None: (bars, {}))
    probs = {"value": 0.9}

    def fake_predict(ds, model_cfg, cost):
        last = ds.groupby("symbol", sort=False).tail(1)
        return pd.DataFrame({"time": last["time"].to_numpy(), "prob": probs["value"], "exp_ret": 0.01}, index=last["symbol"].to_numpy())

    monkeypatch.setattr(live, "predict_latest", fake_predict)
    return c, probs, tmp_path


def test_dry_run_cycle_buys_then_sells_in_paper_account(setup):
    cfg, probs, tmp_path = setup
    now = pd.Timestamp("2025-03-14 10:05", tz="Asia/Seoul")  # 금요일 장중
    rep = live.run_trade_cycle(cfg, "kr", now=now)
    assert rep.dry_run and rep.market_open
    buys = [e for e in rep.executed if e["side"] == "buy"]
    assert len(buys) == 3 and all(not e["sent"] for e in buys)
    state_file = tmp_path / "state" / "trade_state_kr.json"
    assert state_file.exists()

    probs["value"] = 0.1  # 신호 약화 → 전량 청산
    rep2 = live.run_trade_cycle(cfg, "kr", now=now + pd.Timedelta(hours=1))
    sells = [e for e in rep2.executed if e["side"] == "sell"]
    assert len(sells) == 3 and {e["reason"] for e in sells} == {"signal"}
    st = live.TradeState.load(state_file, cfg.market("kr"))
    assert st.paper_positions == {}
    loss = cfg.market("kr").initial_capital - st.paper_cash
    assert 0 < loss < cfg.market("kr").initial_capital * 0.01  # 비용만큼만 줄었음


def test_daily_entry_limit_applies_in_live_cycle(setup):
    cfg, probs, _ = setup
    cfg = replace(cfg, risk=replace(cfg.risk, max_trades_per_day=2))
    rep = live.run_trade_cycle(cfg, "kr", now=pd.Timestamp("2025-03-14 10:05", tz="Asia/Seoul"))
    assert len([e for e in rep.executed if e["side"] == "buy"]) == 2
    assert any("한도" in n for n in rep.notes)


def test_no_orders_outside_market_hours(setup):
    cfg, _, _ = setup
    night = pd.Timestamp("2025-03-14 20:00", tz="Asia/Seoul")
    rep = live.run_trade_cycle(cfg, "kr", now=night)
    assert rep.executed == [] and not rep.market_open
    rep2 = live.run_trade_cycle(cfg, "kr", now=night, ignore_hours=True)  # DRY_RUN 에서만 허용
    assert len(rep2.executed) == 3


class FakeBroker:
    """DRY_RUN=false 경로 검증용 가짜 증권사 (네트워크 없음)."""

    def __init__(self):
        from trader.broker import Balance

        self.Balance = Balance
        self.orders, self.positions, self.fail = [], {}, set()

    def price(self, s):
        return 100.0

    def balance(self):
        mv = sum(p.qty * p.last_price for p in self.positions.values())
        return self.Balance("KRW", 10_000_000 - mv, 10_000_000, dict(self.positions))

    def _order(self, side, s, qty, ref):
        from trader.broker import KisApiError, OrderRequest, OrderResult

        if s in self.fail:
            raise KisApiError("가짜 오류")
        self.orders.append((side, s, qty))
        req = OrderRequest("domestic", side, s, qty, None, "/x", "VTTC0012U", {})
        return OrderResult(req, sent=True, dry_run=False, ok=True, order_no=f"{len(self.orders):04d}")

    def buy(self, s, qty, ref=None):
        return self._order("buy", s, qty, ref)

    def sell(self, s, qty, ref=None):
        return self._order("sell", s, qty, ref)


def test_live_cycle_with_dry_run_off_reconciles_with_broker(setup, monkeypatch):
    import trader.broker
    from trader.broker import Position

    cfg, probs, tmp_path = setup
    monkeypatch.setenv("DRY_RUN", "false")
    for k, v in {"KIS_APP_KEY": "k", "KIS_APP_SECRET": "s", "KIS_ACCOUNT_NO": "12345678"}.items():
        monkeypatch.setenv(k, v)
    fake = FakeBroker()
    monkeypatch.setattr(trader.broker, "make_broker", lambda *a, **k: fake)
    now = pd.Timestamp("2025-03-14 10:05", tz="Asia/Seoul")
    tickers = list(cfg.market("kr").symbols)

    fake.fail = {tickers[2]}  # 세 번째 주문은 API 오류 → 기록만 하고 계속
    rep = live.run_trade_cycle(cfg, "kr", now=now)
    assert not rep.dry_run
    assert [o[:2] for o in fake.orders] == [("buy", tickers[0]), ("buy", tickers[1])]
    failed = [e for e in rep.executed if e["symbol"] == tickers[2]]
    assert failed and failed[0]["ok"] is False and "가짜 오류" in failed[0]["message"]
    st = live.TradeState.load(tmp_path / "state" / "trade_state_kr.json", cfg.market("kr"))
    assert set(st.live_positions) == {tickers[0], tickers[1]}
    assert st.live_positions[tickers[0]]["stop_price"] == pytest.approx(99.0)

    # 다음 사이클: 첫 종목만 체결됨(두 번째는 미체결) → 상태에서 정리, 신호 약화로 첫 종목만 매도
    q0 = fake.orders[0][2]
    fake.positions = {tickers[0]: Position(tickers[0], q0, 100.0, 100.0)}
    fake.orders, fake.fail, probs["value"] = [], set(), 0.1
    live.run_trade_cycle(cfg, "kr", now=now + pd.Timedelta(hours=1))
    assert fake.orders == [("sell", tickers[0], q0)]
    st = live.TradeState.load(tmp_path / "state" / "trade_state_kr.json", cfg.market("kr"))
    assert st.live_positions == {}
