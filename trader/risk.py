"""손실 제한 장치. 백테스트와 실매매(trade 명령)가 같은 규칙을 쓴다.

손실을 0으로 만들 수는 없다. 여기서 하는 일은 '한 번에, 하루에, 계좌 전체로' 잃을 수 있는 양을 제한하는 것이다.
  - 1회 손절: 진입가 대비 -stop_loss_pct (갭으로 더 크게 잃을 수 있음 → 백테스트에 반영)
  - 1회 투입: 계좌 평가금액의 position_size_pct
  - 계좌 최대낙폭: 고점 대비 -max_drawdown_pct 도달 시 매매 중단 (재개는 사람이 직접)
  - 하루 최대 신규 진입 횟수: max_trades_per_day
"""
from __future__ import annotations

from typing import Any

from .config import RiskConfig

_EPS = 1e-12


class RiskManager:
    def __init__(self, cfg: RiskConfig, initial_equity: float | None = None):
        self.cfg = cfg
        self.peak_equity: float | None = initial_equity
        self.halted = False
        self.halt_reason: str | None = None
        self.halted_at: str | None = None
        self.daily_entries: dict[str, int] = {}

    # --- 주문 크기 / 손절가 -------------------------------------------------
    def stop_price(self, entry_price: float) -> float:
        return entry_price * (1 - self.cfg.stop_loss_pct / 100)

    def position_budget(self, equity: float) -> float:
        return max(equity, 0.0) * self.cfg.position_size_pct / 100

    def order_quantity(self, budget: float, price: float, cash: float, cost_rate: float = 0.0) -> int:
        """예산(계좌의 N%)과 가용 현금 안에서 살 수 있는 정수 주식 수. cost_rate = 슬리피지+수수료 비율."""
        if price <= 0:
            return 0
        spend = min(budget, cash)
        return max(int(spend // (price * (1 + cost_rate))), 0)

    # --- 계좌 낙폭 -----------------------------------------------------------
    def drawdown(self, equity: float) -> float:
        if not self.peak_equity:
            return 0.0
        return equity / self.peak_equity - 1

    def update_equity(self, equity: float, when: Any = None) -> bool:
        """평가금액 갱신. 이번 호출로 최대낙폭 한도에 닿아 매매가 중단되면 True."""
        if self.peak_equity is None or equity > self.peak_equity:
            self.peak_equity = equity
        if self.halted:
            return False
        dd = self.drawdown(equity)
        if dd <= -self.cfg.max_drawdown_pct / 100 + _EPS:
            self.halted = True
            self.halted_at = str(when) if when is not None else None
            self.halt_reason = f"계좌 고점 대비 {dd:.2%} (한도 -{self.cfg.max_drawdown_pct:g}%)"
            return True
        return False

    # --- 진입 허용 여부 --------------------------------------------------------
    def can_enter(self, day: str) -> tuple[bool, str]:
        if self.halted:
            return False, f"매매중단: {self.halt_reason}"
        if self.daily_entries.get(day, 0) >= self.cfg.max_trades_per_day:
            return False, f"일일 신규진입 한도 {self.cfg.max_trades_per_day}회 도달"
        return True, ""

    def record_entry(self, day: str) -> None:
        self.daily_entries[day] = self.daily_entries.get(day, 0) + 1

    # --- 상태 저장 (실매매용) --------------------------------------------------
    def to_state(self, keep_days: int = 10) -> dict[str, Any]:
        recent = dict(sorted(self.daily_entries.items())[-keep_days:])
        return {
            "peak_equity": self.peak_equity,
            "halted": self.halted,
            "halt_reason": self.halt_reason,
            "halted_at": self.halted_at,
            "daily_entries": recent,
        }

    @classmethod
    def from_state(cls, cfg: RiskConfig, state: dict[str, Any] | None, initial_equity: float | None = None):
        rm = cls(cfg, initial_equity)
        if state:
            rm.peak_equity = state.get("peak_equity", initial_equity)
            rm.halted = bool(state.get("halted", False))
            rm.halt_reason = state.get("halt_reason")
            rm.halted_at = state.get("halted_at")
            rm.daily_entries = {str(k): int(v) for k, v in (state.get("daily_entries") or {}).items()}
        return rm


def resume_state(risk_state: dict[str, Any], resume: bool) -> list[str]:
    """trade --resume: 매매 중단을 풀고 고점 기준을 비운다 (다음 평가금액이 새 고점).
    halted 만 false 로 바꾸면 옛 고점 대비 낙폭이 그대로라 다음 실행에서 곧바로 다시 멈춘다."""
    if not resume:
        return []
    if not risk_state.get("halted"):
        return ["--resume: 매매 중단 상태가 아니라서 바꾼 것이 없습니다"]
    risk_state.update(halted=False, halt_reason=None, halted_at=None, peak_equity=None)
    return ["매매 재개: 중단을 풀고 고점 기준을 이번 평가금액으로 다시 잡습니다"]
