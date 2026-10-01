"""실시간 신호 + 1회 매매 사이클.

trade 사이클은 매 봉이 끝날 때마다(스케줄러로) 한 번씩 실행하는 것을 전제로 한다.
  1) 최신 완성 봉까지 데이터 → 모델 학습 → 종목별 상승확률
  2) 손실 제한 확인 (계좌 최대낙폭 중단, 손절가, 일일 진입 횟수)
  3) 청산 → 진입 순서로 주문
     - DRY_RUN=true(기본): 증권사로 아무것도 보내지 않고 state/ 의 가상계좌에만 기록 (현재 종가로 체결 가정)
     - DRY_RUN=false: 한국투자증권 '모의투자' 서버로 주문 전송 (실전 서버는 코드에서 거부)
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from .backtest import Account, Trade
from .broker.kis_client import KisError
from .config import AppConfig, ConfigError, MarketConfig, is_dry_run, load_credentials, load_env
from .data import DataError, is_last_bar_start, is_market_open, load_market, next_bar_start
from .features import make_dataset
from .model import predict_latest
from .risk import RiskManager

log = logging.getLogger(__name__)

BUY, WAIT = "매수", "관망"


# --------------------------------------------------------------------------
# 실시간 신호
# --------------------------------------------------------------------------
def compute_signals(
    cfg: AppConfig, market_key: str, *, offline: bool = False, now: pd.Timestamp | None = None
) -> tuple[pd.DataFrame, dict[str, str]]:
    """종목별 최신 완성 봉 기준 상승확률과 매수/관망 신호."""
    market = cfg.market(market_key)
    bars, errors = load_market(cfg, market_key, offline=offline, refresh=not offline, now=now)
    if not bars:
        raise DataError(f"[{market.name}] 모든 종목의 데이터 수집에 실패했습니다: {errors}")
    cost = market.costs.round_trip
    latest = predict_latest(make_dataset(bars, cfg.features, cost), cfg.model, cost)
    latest = latest.reindex([s for s in market.symbols if s in latest.index])  # 설정 파일 순서로
    st = cfg.strategy
    rows = []
    for sym, r in latest.iterrows():
        nxt = next_bar_start(r["time"], market, cfg.data.interval)
        last_blocked = st.flatten_at_session_end and is_last_bar_start(nxt, market, cfg.data.interval)
        ok = r["prob"] >= st.entry_threshold and r["exp_ret"] >= st.min_expected_return
        if ok and not last_blocked:
            signal, note = BUY, ""
        elif ok:
            signal, note = WAIT, "다음 봉이 장 마지막 봉 → 진입 금지"
        else:
            signal, note = WAIT, ""
        rows.append(
            {
                "symbol": sym,
                "종목": market.label(sym),
                "최신봉": r["time"],
                "종가": float(bars[sym]["close"].iloc[-1]),
                "상승확률": r["prob"],
                "기대수익%": r["exp_ret"] * 100,
                "신호": signal,
                "다음봉": nxt,
                "next_is_last": is_last_bar_start(nxt, market, cfg.data.interval),
                "비고": note,
            }
        )
    return pd.DataFrame(rows).set_index("symbol"), errors


# --------------------------------------------------------------------------
# 상태 파일 (리스크 상태 · DRY_RUN 가상계좌 · 모의투자 포지션 메모)
# --------------------------------------------------------------------------
@dataclass
class TradeState:
    path: Path
    market: MarketConfig
    risk: dict[str, Any] = field(default_factory=dict)
    paper_cash: float | None = None
    paper_positions: dict[str, dict] = field(default_factory=dict)
    live_positions: dict[str, dict] = field(default_factory=dict)  # 이 시스템이 모의투자로 연 포지션
    history: list[dict] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path, market: MarketConfig) -> "TradeState":
        if not path.exists():
            return cls(path=path, market=market, paper_cash=market.initial_capital)
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            path=path,
            market=market,
            risk=raw.get("risk") or {},
            paper_cash=raw.get("paper_cash", market.initial_capital),
            paper_positions=raw.get("paper_positions") or {},
            live_positions=raw.get("live_positions") or {},
            history=raw.get("history") or [],
        )

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "market": self.market.key,
            "risk": self.risk,
            "paper_cash": self.paper_cash,
            "paper_positions": self.paper_positions,
            "live_positions": self.live_positions,
            "history": self.history[-500:],
        }
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        tmp.replace(self.path)


@dataclass
class OrderIntent:
    side: str  # buy | sell
    symbol: str
    qty: int
    ref_price: float
    reason: str


@dataclass
class CycleReport:
    market: MarketConfig
    now: pd.Timestamp
    dry_run: bool
    market_open: bool
    signals: pd.DataFrame
    equity: float
    drawdown: float
    halted: bool
    halt_reason: str | None
    intents: list[OrderIntent]
    executed: list[dict]
    notes: list[str]
    data_errors: dict[str, str]


def _decide_exit(sig: pd.Series | None, price: float, stop: float, rm: RiskManager, cfg: AppConfig, bars_held: int):
    st = cfg.strategy
    if rm.halted and cfg.risk.liquidate_on_halt:
        return "halt"
    if price <= stop:
        return "stop"
    if sig is None:
        return "signal"
    if st.flatten_at_session_end and bool(sig["next_is_last"]):
        return "session_end"
    if not sig["상승확률"] >= st.exit_threshold:
        return "signal"
    if st.max_hold_bars and bars_held >= st.max_hold_bars:
        return "max_hold"
    return None


def run_trade_cycle(
    cfg: AppConfig,
    market_key: str,
    *,
    offline: bool = False,
    ignore_hours: bool = False,
    now: pd.Timestamp | None = None,
) -> CycleReport:
    market = cfg.market(market_key)
    load_env(cfg.root)
    dry_run = is_dry_run()
    now = pd.Timestamp.now(tz=market.timezone) if now is None else now.tz_convert(market.timezone)
    notes: list[str] = []

    signals, errors = compute_signals(cfg, market_key, offline=offline, now=now)
    state = TradeState.load(cfg.path(cfg.paths.state_dir) / f"trade_state_{market_key}.json", market)
    rm = RiskManager.from_state(cfg.risk, state.risk, market.initial_capital)
    open_now = is_market_open(now, market)
    may_trade = open_now or (dry_run and ignore_hours)
    if not open_now:
        notes.append("장 운영시간이 아닙니다" + (" (DRY_RUN --ignore-hours: 가상계좌로만 진행)" if may_trade else " → 주문 생략"))

    prices = {s: float(signals.loc[s, "종가"]) for s in signals.index}
    broker = None
    if dry_run:
        acct = Account(state.paper_cash, market.costs)
        acct.positions = {s: Trade.from_dict(d) for s, d in state.paper_positions.items()}
        if may_trade:
            for t in acct.positions.values():
                t.bars_held += 1  # 지난 사이클 이후 봉 하나를 더 보유
        held = {s: (t.qty, t.stop_price, t.bars_held) for s, t in acct.positions.items()}
        missing = [s for s in acct.positions if s not in prices]
        if missing:
            raise DataError(f"보유 종목 시세가 없어 평가할 수 없습니다: {missing}")
        equity, cash = acct.equity(prices), acct.cash
    else:
        from .broker import make_broker  # 실제 전송 모드에서만 증권사 모듈 사용

        creds = load_credentials()
        if creds is None:
            raise ConfigError("DRY_RUN=false 인데 .env 에 KIS 모의투자 자격정보가 없습니다")
        broker = make_broker(cfg, market_key, creds, dry_run=False)
        for s in signals.index:  # 주문 기준가는 증권사 현재가 (yfinance 는 지연될 수 있음)
            live_price = broker.price(s)
            if live_price > 0:
                prices[s] = live_price
        bal = broker.balance()
        equity, cash = bal.total_equity, bal.cash
        held = {}
        for s, memo in list(state.live_positions.items()):
            pos = bal.positions.get(s)
            if pos is None or pos.qty <= 0:  # 체결 안 됐거나 이미 정리된 포지션
                state.live_positions.pop(s)
                continue
            memo["bars_held"] = int(memo.get("bars_held", 0)) + (1 if may_trade else 0)
            held[s] = (min(int(memo["qty"]), pos.qty), float(memo["stop_price"]), memo["bars_held"])
            if s not in signals.index and pos.last_price > 0:
                prices[s] = pos.last_price

    rm.update_equity(equity, now)
    intents: list[OrderIntent] = []
    if may_trade:
        for s, (qty, stop, bars_held) in held.items():
            sig = signals.loc[s] if s in signals.index else None
            reason = _decide_exit(sig, prices[s], stop, rm, cfg, bars_held)
            if reason:
                intents.append(OrderIntent("sell", s, qty, prices[s], reason))
        if not rm.halted:
            day = now.date().isoformat()
            budget = rm.position_budget(equity)
            for s, sig in signals.iterrows():
                if s in held or sig["신호"] != BUY:
                    continue
                ok, why = rm.can_enter(day)
                if not ok:
                    notes.append(f"{market.label(s)} 매수 보류: {why}")
                    continue
                qty = rm.order_quantity(budget, prices[s], cash, market.costs.slippage + market.costs.commission)
                if qty <= 0:
                    notes.append(f"{market.label(s)} 매수 보류: 예산/현금 부족")
                    continue
                intents.append(OrderIntent("buy", s, qty, prices[s], "signal"))
                cash -= qty * prices[s] * (1 + market.costs.slippage + market.costs.commission)
                rm.record_entry(day)
        else:
            notes.append(f"매매 중단 상태: {rm.halt_reason} (state 파일을 확인하고 사람이 직접 해제)")

    executed: list[dict] = []
    for it in intents:
        rec = {"time": now.isoformat(), "side": it.side, "symbol": it.symbol, "qty": it.qty, "ref_price": it.ref_price, "reason": it.reason, "dry_run": dry_run}
        if dry_run:
            if it.side == "sell":
                tr = acct.sell(it.symbol, it.ref_price, now, it.reason)
                rec.update(sent=False, fill_price=tr.exit_price, pnl=tr.pnl, note="가상계좌 체결(현재 종가 가정)")
            else:
                tr = acct.buy(it.symbol, it.ref_price, float("inf"), now, rm.stop_price, max_qty=it.qty)
                rec.update(sent=False, fill_price=tr.entry_price if tr else None, note="가상계좌 체결(현재 종가 가정)")
        else:
            try:
                res = broker.sell(it.symbol, it.qty, it.ref_price) if it.side == "sell" else broker.buy(it.symbol, it.qty, it.ref_price)
            except KisError as e:  # 토큰/네트워크 오류: 이 주문만 실패로 기록하고 다음 주문 진행
                rec.update(sent=False, ok=False, message=f"{type(e).__name__}: {e}")
                executed.append(rec)
                continue
            rec.update(sent=res.sent, ok=res.ok, order_no=res.order_no, message=res.message)
            if res.ok and it.side == "buy":
                state.live_positions[it.symbol] = {
                    "qty": it.qty,
                    "entry_price": it.ref_price,
                    "stop_price": rm.stop_price(it.ref_price),
                    "entry_time": now.isoformat(),
                    "bars_held": 0,
                    "order_no": res.order_no,
                }
            elif res.ok and it.side == "sell":
                state.live_positions.pop(it.symbol, None)
        executed.append(rec)

    if dry_run:
        state.paper_cash = acct.cash
        state.paper_positions = {s: t.to_dict() for s, t in acct.positions.items()}
        equity = acct.equity(prices)
    state.risk = rm.to_state()
    state.history.extend(executed)
    state.save()

    return CycleReport(
        market=market,
        now=now,
        dry_run=dry_run,
        market_open=open_now,
        signals=signals,
        equity=equity,
        drawdown=rm.drawdown(equity),
        halted=rm.halted,
        halt_reason=rm.halt_reason,
        intents=intents,
        executed=executed,
        notes=notes,
        data_errors=errors,
    )
