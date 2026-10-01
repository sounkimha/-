from trader.config import RiskConfig
from trader.risk import RiskManager


def rm(**kw):
    return RiskManager(RiskConfig(**kw), initial_equity=1_000_000)


def test_stop_and_size():
    r = rm(stop_loss_pct=1.0, position_size_pct=20.0)
    assert r.stop_price(10_000) == 9_900
    assert r.position_budget(1_000_000) == 200_000
    assert r.order_quantity(200_000, 10_000, cash=1_000_000) == 20
    assert r.order_quantity(200_000, 10_000, cash=55_000) == 5  # 현금이 모자라면 현금 한도
    assert r.order_quantity(200_000, 0, cash=1_000_000) == 0


def test_drawdown_halt_is_sticky():
    r = rm(max_drawdown_pct=10.0)
    assert not r.update_equity(1_100_000)  # 고점 갱신
    assert not r.update_equity(1_000_000)  # -9.1%
    assert r.update_equity(990_000, when="t1")  # -10% 도달 → 중단
    assert r.halted and r.halted_at == "t1"
    assert not r.update_equity(2_000_000)  # 회복해도 자동 재개하지 않음
    assert r.halted
    ok, why = r.can_enter("2025-01-01")
    assert not ok and "매매중단" in why


def test_daily_limit_and_state_roundtrip():
    r = rm(max_trades_per_day=2)
    for _ in range(2):
        assert r.can_enter("2025-01-02")[0]
        r.record_entry("2025-01-02")
    assert not r.can_enter("2025-01-02")[0]
    assert r.can_enter("2025-01-03")[0]
    r2 = RiskManager.from_state(r.cfg, r.to_state())
    assert r2.daily_entries == {"2025-01-02": 2}
    assert r2.peak_equity == r.peak_equity and r2.halted == r.halted
