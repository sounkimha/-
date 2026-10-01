"""실시간 신호 + 1회 매매 사이클.

trade 사이클은 매 봉이 끝날 때마다(스케줄러로) 한 번씩 실행하는 것을 전제로 한다.
  1) 최신 완성 봉까지 데이터 → 모델 학습 → 종목별 상승확률
  2) 손실 제한 확인 (계좌 최대낙폭 중단, 손절가, 장 마지막 봉 구간, 일일 진입 횟수, 데이터 지연)
  3) 청산 → 진입 순서로 주문
     - DRY_RUN=true(기본): 증권사로 아무것도 보내지 않고 state/ 의 가상계좌에만 기록 (최근 가격 체결 가정)
     - DRY_RUN=false: 한국투자증권 '모의투자' 서버로 주문 전송 (실전 서버는 코드에서 거부)
  상태 파일은 모드별로 따로 둔다: state/trade_state_<시장>_paper.json / _vts.json
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
from .data import (
    DataError,
    expected_latest_bar,
    in_last_bar,
    is_last_bar_start,
    is_market_open,
    load_market,
    next_bar_start,
)
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
    interval = cfg.data.interval
    clock = pd.Timestamp.now(tz=market.timezone) if now is None else now.tz_convert(market.timezone)
    bars, errors = load_market(cfg, market_key, offline=offline, refresh=not offline, now=now)
    if not bars:
        raise DataError(f"[{market.name}] 모든 종목의 데이터 수집에 실패했습니다: {errors}")
    cost = market.costs.round_trip
    latest = predict_latest(make_dataset(bars, cfg.features, cost), cfg.model, cost)
    latest = latest.reindex([s for s in market.symbols if s in latest.index])  # 설정 파일 순서로
    expected = None if offline else expected_latest_bar(clock, market, interval)
    st = cfg.strategy
    rows = []
    for sym, r in latest.iterrows():
        nxt = next_bar_start(r["time"], market, interval)
        next_is_last = is_last_bar_start(nxt, market, interval)
        stale = expected is not None and r["time"] < expected
        ok = r["prob"] >= st.entry_threshold and r["exp_ret"] >= st.min_expected_return
        if ok and stale:
            signal, note = WAIT, f"최신 봉 지연/누락(기대 {expected:%m-%d %H:%M}) → 진입 보류"
        elif ok and st.flatten_at_session_end and next_is_last:
            signal, note = WAIT, "다음 봉이 장 마지막 봉 → 진입 금지"
        elif ok:
            signal, note = BUY, ""
        else:
            signal, note = WAIT, ""
        df = bars[sym]
        rows.append(
            {
                "symbol": sym,
                "종목": market.label(sym),
                "최신봉": r["time"],
                "종가": float(df["close"].iloc[-1]),
                "현재가": float(df.attrs.get("last_price", df["close"].iloc[-1])),
                "상승확률": r["prob"],
                "기대수익%": r["exp_ret"] * 100,
                "신호": signal,
                "다음봉": nxt,
                "next_is_last": next_is_last,
                "stale": stale,
                "비고": note,
            }
        )
    return pd.DataFrame(rows).set_index("symbol"), errors


# --------------------------------------------------------------------------
# 상태 파일 (리스크 상태 · DRY_RUN 가상계좌 · 모의투자 포지션 메모) — 모드별로 분리
# --------------------------------------------------------------------------
def state_path(cfg: AppConfig, market_key: str, dry_run: bool) -> Path:
    mode = "paper" if dry_run else "vts"
    return cfg.path(cfg.paths.state_dir) / f"trade_state_{market_key}_{mode}.json"


@dataclass
class TradeState:
    path: Path
    market: MarketConfig
    risk: dict[str, Any] = field(default_factory=dict)
    paper_cash: float | None = None
    paper_positions: dict[str, dict] = field(default_factory=dict)
    live_positions: dict[str, dict] = field(default_factory=dict)  # 이 시스템이 모의투자로 연 포지션
    last_prices: dict[str, float] = field(default_factory=dict)  # 시세를 못 받았을 때 쓸 마지막 가격
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
            last_prices={k: float(v) for k, v in (raw.get("last_prices") or {}).items()},
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
            "last_prices": self.last_prices,
            "history": self.history[-500:],
        }
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        tmp.replace(self.path)


@dataclass
class Holding:
    qty: int
    stop_price: float
    bars_held: int
    sellable: int  # 지금 매도 주문을 낼 수 있는 수량 (기존 매도 주문이 걸려 있으면 0)


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


def _decide_exit(
    sig: pd.Series | None, price: float, h: Holding, rm: RiskManager, cfg: AppConfig, last_bar_now: bool
) -> str | None:
    st = cfg.strategy
    if rm.halted and cfg.risk.liquidate_on_halt:
        return "halt"
    if price <= h.stop_price:
        return "stop"
    if st.flatten_at_session_end and (last_bar_now or (sig is not None and bool(sig["next_is_last"]))):
        return "session_end"
    if sig is None:  # 신호를 계산하지 못한 보유 종목은 보수적으로 청산
        return "signal"
    if not sig["상승확률"] >= st.exit_threshold:
        return "signal"
    if st.max_hold_bars and h.bars_held >= st.max_hold_bars:
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
    interval = cfg.data.interval
    load_env(cfg.root)
    dry_run = is_dry_run()
    now = pd.Timestamp.now(tz=market.timezone) if now is None else now.tz_convert(market.timezone)
    notes: list[str] = []

    signals, errors = compute_signals(cfg, market_key, offline=offline, now=now)
    state = TradeState.load(state_path(cfg, market_key, dry_run), market)
    # 낙폭 기준: 가상계좌는 설정의 시작금액, 모의투자는 첫 잔고 조회값에서 시작
    rm = RiskManager.from_state(cfg.risk, state.risk, market.initial_capital if dry_run else None)
    open_now = is_market_open(now, market)
    may_trade = open_now or (dry_run and ignore_hours)
    last_bar_now = open_now and in_last_bar(now, market, interval)
    if not open_now:
        notes.append("장 운영시간이 아닙니다" + (" (DRY_RUN --ignore-hours: 가상계좌로만 진행)" if may_trade else " → 주문 생략"))

    prices = dict(state.last_prices)
    prices.update({s: float(signals.loc[s, "현재가"]) for s in signals.index})
    acct: Account | None = None
    broker = None
    held: dict[str, Holding] = {}
    intents: list[OrderIntent] = []
    executed: list[dict] = []
    equity = float("nan")
    try:
        if dry_run:
            acct = Account(state.paper_cash, market.costs)
            acct.positions = {s: Trade.from_dict(d) for s, d in state.paper_positions.items()}
            for s, t in acct.positions.items():
                if may_trade:
                    t.bars_held += 1  # 지난 사이클 이후 봉 하나를 더 보유
                if s not in signals.index:
                    prices.setdefault(s, t.entry_price)
                    notes.append(f"{market.label(s)} 시세 없음 → 마지막으로 알려진 가격 {prices[s]:,.2f} 사용")
                held[s] = Holding(t.qty, t.stop_price, t.bars_held, sellable=t.qty)
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
            for s, memo in list(state.live_positions.items()):
                pos = bal.positions.get(s)
                if pos is None or pos.qty <= 0:  # 청산 체결됐거나, 매수가 미체결·미접수
                    state.live_positions.pop(s)
                    what = "청산 체결 확인" if memo.get("exit_pending") else "잔고 없음(미체결·미접수) → 추적 종료"
                    notes.append(f"{market.label(s)}: {what}")
                    continue
                memo.pop("unconfirmed", None)  # 잔고로 확인됨
                memo["qty"] = min(int(memo["qty"]), pos.qty)
                if may_trade:
                    memo["bars_held"] = int(memo.get("bars_held", 0)) + 1
                sellable = pos.sellable_qty if pos.sellable_qty is not None else pos.qty
                if sellable > 0:
                    memo.pop("exit_pending", None)  # 걸려 있는 매도 주문이 없음
                held[s] = Holding(memo["qty"], float(memo["stop_price"]), int(memo.get("bars_held", 0)), min(sellable, memo["qty"]))
                if s not in signals.index and pos.last_price > 0:
                    prices[s] = pos.last_price
        state.last_prices.update({s: prices[s] for s in signals.index})

        rm.update_equity(equity, now)
        if may_trade:
            for s, h in held.items():
                sig = signals.loc[s] if s in signals.index else None
                reason = _decide_exit(sig, prices[s], h, rm, cfg, last_bar_now)
                if not reason:
                    continue
                if h.sellable <= 0:
                    notes.append(f"{market.label(s)} 청산({reason}) 보류: 이전 매도 주문이 아직 체결 대기 중")
                    continue
                intents.append(OrderIntent("sell", s, h.sellable, prices[s], reason))
            if rm.halted:
                notes.append(f"매매 중단 상태: {rm.halt_reason} (state 파일을 확인하고 사람이 직접 해제)")
            elif cfg.strategy.flatten_at_session_end and last_bar_now:
                notes.append("장 마지막 봉 구간 → 신규 진입 없음")
            else:
                day = now.date().isoformat()
                budget = rm.position_budget(equity)
                cost_rate = market.costs.slippage + market.costs.commission
                for s, sig in signals.iterrows():
                    if s in held:
                        continue
                    if sig["stale"] and sig["상승확률"] >= cfg.strategy.entry_threshold:
                        notes.append(f"{market.label(s)} 매수 보류: {sig['비고']}")
                    if sig["신호"] != BUY:
                        continue
                    ok, why = rm.can_enter(day)
                    if not ok:
                        notes.append(f"{market.label(s)} 매수 보류: {why}")
                        continue
                    qty = rm.order_quantity(budget, prices[s], cash, cost_rate)
                    if qty <= 0:
                        notes.append(f"{market.label(s)} 매수 보류: 예산/현금 부족")
                        continue
                    intents.append(OrderIntent("buy", s, qty, prices[s], "signal"))
                    cash -= qty * prices[s] * (1 + cost_rate)
                    rm.record_entry(day)

        for it in intents:
            rec = {"time": now.isoformat(), "side": it.side, "symbol": it.symbol, "qty": it.qty, "ref_price": it.ref_price, "reason": it.reason, "dry_run": dry_run}
            executed.append(rec)  # 중간에 예외가 나도 기록이 남도록 먼저 넣는다
            if dry_run:
                if it.side == "sell":
                    tr = acct.sell(it.symbol, it.ref_price, now, it.reason)
                    rec.update(sent=False, fill_price=tr.exit_price, pnl=tr.pnl, note="가상계좌 체결(최근 가격 가정)")
                else:
                    tr = acct.buy(it.symbol, it.ref_price, float("inf"), now, rm.stop_price, max_qty=it.qty)
                    rec.update(sent=False, fill_price=tr.entry_price if tr else None, note="가상계좌 체결(최근 가격 가정)")
                continue
            try:
                res = broker.sell(it.symbol, it.qty, it.ref_price) if it.side == "sell" else broker.buy(it.symbol, it.qty, it.ref_price)
            except KisError as e:  # 전송 전 실패(토큰 등): 이 주문만 실패로 기록하고 계속
                rec.update(sent=False, ok=False, message=f"{type(e).__name__}: {e}")
                continue
            rec.update(sent=res.sent, ok=res.ok, uncertain=res.uncertain, order_no=res.order_no, message=res.message)
            if not (res.ok or res.uncertain):
                continue
            if it.side == "buy":  # 접수(또는 접수 여부 불명) → 다음 사이클에 잔고로 확인
                state.live_positions[it.symbol] = {
                    "qty": it.qty,
                    "entry_price": it.ref_price,
                    "stop_price": rm.stop_price(it.ref_price),
                    "entry_time": now.isoformat(),
                    "bars_held": 0,
                    "order_no": res.order_no,
                    "unconfirmed": True,
                }
            elif it.symbol in state.live_positions:  # 매도 '접수' ≠ 체결 → 잔고에서 사라질 때까지 추적 유지
                state.live_positions[it.symbol].update(exit_pending=True, exit_reason=it.reason, exit_order_no=res.order_no)
    finally:
        if acct is not None:
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
