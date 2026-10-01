"""일봉 규칙(rule_breakout)의 매매 사이클 — trade 명령.

DRY_RUN=true (기본): 가상계좌. 증권사로 아무것도 보내지 않는다. 장 마감 후(16:00~) 하루 1회 실행.
  1) 어제 계획한 주문을 오늘 '시가'로 정산 — 백테스트와 같은 규칙 (매도 먼저, 지정가보다 높게 시작한 매수는 미체결)
  2) 오늘 종가로 평가 → 계좌 낙폭 확인 → 손절·청산·진입을 계획해 다음 거래일 시가 주문으로 저장
  실행을 며칠 빼먹으면 밀린 거래일을 순서대로 처리한다. 체결 기록·일별 평가금액은 CSV 로 남긴다.

DRY_RUN=false (사람이 .env 에서 직접 바꾼 경우만): 한국투자증권 모의투자 계좌로 주문. 하루 두 단계.
  - 주문 단계 (08:30~09:00 동시호가): 전 거래일 종가 기준으로 매도(시장가)·매수(지정가 = 종가+N%) 주문.
    현금이 모자란 매수는 '매도 대금 대기'로 미룬다 (동시호가 중에는 아직 안 팔린 대금이 주문가능금액에 안 잡힘).
  - 확인 단계 (09:02~15:20, 여러 번 실행 가능): 체결 조회 → 매수 체결분 기록, 남은 매수 잔량 취소,
    동시호가에서 거부된 매도는 시장가로 한 번 재시도, 미뤄 둔 매수는 오늘 시가가 지정가 이하일 때만 같은 지정가로 주문.
  이 시스템이 연 포지션만 관리한다 (계좌의 다른 보유 종목은 건드리지 않음).
"""
from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field, fields
from datetime import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .backtest import Account, Trade, buy_hold_curve, max_drawdown
from .broker.domestic import to_krx_code
from .broker.kis_client import KisError, to_float
from .config import AppConfig, ConfigError, MarketConfig, is_dry_run, load_credentials, load_env
from .daily import buy_limit_price
from .data import DataError, expected_latest_bar, load_market
from .risk import RiskManager, resume_state
from .rules import trend_breakout

log = logging.getLogger(__name__)

# 확인 단계는 09:02 부터: 09:00 동시호가 체결이 조회에 잡히기 전에 '미체결'로 보고 취소하지 않도록
ORDER_START, ORDER_END, CHECK_START, CHECK_END = time(8, 30), time(9, 0), time(9, 2), time(15, 20)
REASONS = {"signal": "청산선 이탈", "stop": "손절", "halt": "계좌 낙폭 한도", "entry": "추세 돌파"}
MAX_LAG_DAYS = 3  # 가상계좌: 한 종목의 최신 일봉이 이 거래일 수까지 늦으면 기다리고, 더 늦으면 거래정지로 본다
JOURNAL_COLUMNS = ["일자", "구분", "종목", "수량", "가격", "금액", "비용", "손익", "사유", "비고"]
EQUITY_COLUMNS = ["일자", "현금", "평가금액", "고점대비%", "보유종목수"]
_LIVE = ("submitted", "uncertain")  # 결과를 아직 모르는 주문
_OPEN_STATUSES = _LIVE + ("cancel_requested",)  # 체결 조회로 다시 확인할 주문 (취소 요청 뒤 체결된 수량까지 반영)


# --------------------------------------------------------------------------
# 상태 · 기록 파일
# --------------------------------------------------------------------------
def daily_files(cfg: AppConfig, market_key: str, dry_run: bool) -> dict[str, Path]:
    stem = f"daily_{market_key}_{'paper' if dry_run else 'vts'}"
    d = cfg.path(cfg.paths.state_dir)
    return {"state": d / f"{stem}_state.json", "journal": d / f"{stem}_journal.csv", "equity": d / f"{stem}_equity.csv"}


@dataclass
class DailyState:
    path: Path
    risk: dict[str, Any] = field(default_factory=dict)
    last_date: str | None = None  # 가상계좌: 마지막으로 처리한 일봉 날짜
    cash: float | None = None  # 가상계좌 현금
    positions: dict[str, dict] = field(default_factory=dict)  # 가상계좌: Trade / 모의투자: 이 시스템이 연 포지션 메모
    last_close: dict[str, float] = field(default_factory=dict)
    pending: list[dict] = field(default_factory=list)  # 가상계좌: 다음 시가에 체결할 주문
    last_exit: dict[str, str] = field(default_factory=dict)  # 종목 → 마지막 매도 체결일 (재진입 대기)
    orders: list[dict] = field(default_factory=list)  # 모의투자: 낸 주문과 상태
    deferred: list[dict] = field(default_factory=list)  # 모의투자: 매도 대금을 기다리는 매수
    order_phase_date: str | None = None  # 모의투자: 주문 단계를 마친 날

    @classmethod
    def load(cls, path: Path) -> "DailyState":
        if not path.exists():
            return cls(path)
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls(path, **{f.name: raw[f.name] for f in fields(cls) if f.name != "path" and f.name in raw})

    def save(self) -> None:
        self.orders = self.orders[-300:]
        payload = {f.name: getattr(self, f.name) for f in fields(self) if f.name != "path"}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        tmp.replace(self.path)


def append_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8-sig" if new else "utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerows(rows)


def _event(day, kind, market, s, qty=None, price=None, amount=None, cost=None, pnl=None, reason="", note="") -> dict:
    r2 = lambda v: None if v is None else round(float(v), 2)  # noqa: E731
    return {
        "일자": day, "구분": kind, "종목": market.label(s) if s else "", "수량": qty, "가격": r2(price),
        "금액": r2(amount), "비용": r2(cost), "손익": r2(pnl), "사유": REASONS.get(reason, reason), "비고": note,
    }


@dataclass
class DailyReport:
    market: MarketConfig
    now: pd.Timestamp
    dry_run: bool
    phase: str
    base_date: str | None  # 판단에 쓴 최신 일봉 날짜
    equity: float
    cash: float
    drawdown: float
    halted: bool
    halt_reason: str | None
    holdings: list[dict]
    events: list[dict]
    planned: list[dict]
    notes: list[str]
    data_errors: dict[str, str]
    files: dict[str, Path]


def _cooling(bars: pd.DataFrame, last_exit: str | None, at: pd.Timestamp, n: int) -> bool:
    """마지막 매도일로부터 n 봉이 지나지 않았는지 (백테스트와 같은 봉 수 기준)."""
    if not n or not last_exit:
        return False
    idx = bars.index
    e = idx.searchsorted(pd.Timestamp(last_exit).tz_localize(idx.tz))
    return idx.get_loc(at) - e < n


# --------------------------------------------------------------------------
# 진입점
# --------------------------------------------------------------------------
def run_daily_cycle(
    cfg: AppConfig, market_key: str, *, offline: bool = False, now: pd.Timestamp | None = None, resume: bool = False
) -> DailyReport:
    if cfg.strategy.type != "rule_breakout":
        raise ConfigError("run_daily_cycle 은 strategy.type=rule_breakout 전용입니다")
    if cfg.risk.stop_check != "close":
        raise ConfigError("일봉 trade 는 risk.stop_check: close 만 지원합니다 (장중 감시를 하지 않음)")
    market = cfg.market(market_key)
    load_env(cfg.root)
    dry_run = is_dry_run()
    now = pd.Timestamp.now(tz=market.timezone) if now is None else now.tz_convert(market.timezone)
    bars, errors = load_market(cfg, market_key, offline=offline, refresh=not offline, now=now)
    if not bars:
        raise DataError(f"[{market.name}] 모든 종목의 데이터 수집에 실패했습니다: {errors}")
    sigs = {s: trend_breakout(df, cfg.strategy) for s, df in bars.items()}
    files = daily_files(cfg, market_key, dry_run)
    state = DailyState.load(files["state"])
    notes = resume_state(state.risk, resume)
    run = _paper_cycle if dry_run else _vts_cycle
    return run(cfg, market, bars, sigs, state, now, errors, files, notes)


def _horizon(market, bars, syms, dates, notes) -> pd.Timestamp:
    """모든 종목에 일봉이 있는 마지막 날. 늦게 들어오는 종목을 기다려야 그날을 빠뜨리지 않는다."""
    pos = {d: i for i, d in enumerate(dates)}
    last = horizon = dates[-1]
    for s in syms:
        s_last = bars[s].index[-1]
        lag = pos[last] - pos[s_last]
        if lag == 0:
            continue
        if lag <= MAX_LAG_DAYS:
            horizon = min(horizon, s_last)
            notes.append(f"{market.label(s)}: 최신 일봉이 아직 없습니다 (마지막 {s_last:%Y-%m-%d}) → 모든 종목을 그날까지만 처리하고 기다립니다")
        else:
            notes.append(f"{market.label(s)}: {lag}거래일째 새 일봉이 없습니다 → 거래정지 등으로 보고 기다리지 않습니다")
    return horizon


def _paper_cycle(cfg, market, bars, sigs, state, now, errors, files, notes) -> DailyReport:
    st, rk, costs = cfg.strategy, cfg.risk, market.costs
    acct = Account(market.initial_capital if state.cash is None else state.cash, costs)
    acct.positions = {s: Trade.from_dict(d) for s, d in state.positions.items()}
    rm = RiskManager.from_state(rk, state.risk, market.initial_capital)
    syms = [s for s in market.symbols if s in bars]
    dates = sorted(set().union(*(set(bars[s].index) for s in syms)))
    horizon = _horizon(market, bars, syms, dates, notes)
    if errors:  # 한 종목이라도 못 받으면 그날을 건너뛰지 않도록 전체를 미룬다 (다음 실행 때 밀린 날로 처리)
        todo = []
        notes.append("데이터 수집에 실패한 종목이 있어 오늘은 처리하지 않습니다 (가상계좌 그대로, 다음 실행 때 밀린 날을 처리)")
    elif state.last_date is None:
        todo = [horizon]
        notes.append(
            f"가상계좌를 {market.initial_capital:,.0f}{market.currency}로 시작합니다. "
            f"{horizon:%Y-%m-%d} 종가로 계획한 주문이 다음 거래일 시가에 처음 체결됩니다."
        )
    else:
        todo = [d for d in dates if f"{d:%Y-%m-%d}" > state.last_date and d <= horizon]
        if not todo:
            notes.append(f"새로 처리할 일봉이 없습니다 (마지막 처리 {state.last_date}). 장 마감 후 데이터가 들어오면 다시 실행하세요.")
        elif len(todo) > 1:
            notes.append(f"밀린 {len(todo)}거래일을 순서대로 처리했습니다.")
    pending = {o["symbol"]: o for o in state.pending}
    last_close = dict(state.last_close)
    events: list[dict] = []
    equity_rows: list[dict] = []
    equity = acct.equity({s: last_close.get(s, t.entry_price) for s, t in acct.positions.items()})
    try:
        for d in todo:
            day = f"{d:%Y-%m-%d}"
            active = [s for s in syms if d in bars[s].index]
            row = {s: bars[s].loc[d] for s in active}
            # 1) 시가: 매도 → 매수
            for s in active:
                o = pending.get(s)
                if o is not None and o["side"] == "sell":
                    del pending[s]
                    if s in acct.positions:
                        t = acct.sell(s, float(row[s]["open"]), d, o["reason"])
                        state.last_exit[s] = day
                        events.append(
                            _event(day, "매도", market, s, t.qty, t.exit_price, t.exit_price * t.qty, t.exit_fee, t.pnl, o["reason"])
                        )
            active_set = set(active)
            for s in [s for s, o in list(pending.items()) if o["side"] == "buy" and s in active_set]:  # 계획 순서 = 점수 순
                o = pending.pop(s)
                if s in acct.positions:
                    continue
                op = float(row[s]["open"])
                if o["limit"] is not None and op > o["limit"]:
                    events.append(_event(day, "미체결", market, s, price=op, reason="entry", note=f"시가 {op:,.0f} > 지정가 {o['limit']:,.0f} → 취소"))
                    continue
                ok, why = rm.can_enter(day)
                t = acct.buy(s, op, o["budget"], d, rm.stop_price) if ok else None
                if t is None:
                    events.append(_event(day, "매수 못함", market, s, price=op, reason="entry", note=why or "예산·현금 부족"))
                    continue
                rm.record_entry(day)
                events.append(_event(day, "매수", market, s, t.qty, t.entry_price, t.entry_price * t.qty, t.entry_fee, reason="entry"))
            for s in active:
                if s in acct.positions:
                    acct.positions[s].bars_held += 1
                last_close[s] = float(row[s]["close"])

            # 2) 종가 평가 → 계좌 낙폭
            equity = acct.equity({s: last_close.get(s, t.entry_price) for s, t in acct.positions.items()})
            if rm.update_equity(equity, day):
                notes.append(f"{day}: 계좌 낙폭 한도 도달 → 매매 중단 ({rm.halt_reason}). 원인을 확인한 뒤 trade --resume 으로 재개")
            if rm.halted:
                for s in [s for s, o in pending.items() if o["side"] == "buy"]:
                    del pending[s]
                if rk.liquidate_on_halt:
                    for s in acct.positions:
                        pending[s] = {"side": "sell", "symbol": s, "reason": "halt", "plan_date": day, "budget": 0.0, "limit": None}

            # 3) 다음 거래일 시가 주문 계획 (오늘 종가까지의 정보만 사용)
            candidates = []
            for s in active:
                if s in pending:
                    continue
                sig, c = sigs[s].loc[d], last_close[s]
                pos = acct.positions.get(s)
                if pos is not None:
                    reason = "stop" if c <= pos.stop_price else (None if bool(sig["hold"]) else "signal")
                    if reason is None and st.max_hold_bars and pos.bars_held >= st.max_hold_bars:
                        reason = "max_hold"
                    if reason:
                        pending[s] = {"side": "sell", "symbol": s, "reason": reason, "plan_date": day, "budget": 0.0, "limit": None}
                elif not rm.halted and bool(sig["entry"]) and not _cooling(bars[s], state.last_exit.get(s), d, st.reentry_cooldown_bars):
                    sc = float(sig["score"])
                    candidates.append((sc if sc == sc else -np.inf, s))
            candidates.sort(key=lambda x: -x[0])  # 안정 정렬: 점수가 같으면 설정 파일 순서
            if st.max_positions:
                n_after = (
                    len(acct.positions)
                    - sum(1 for o in pending.values() if o["side"] == "sell")
                    + sum(1 for o in pending.values() if o["side"] == "buy")
                )
                candidates = candidates[: max(st.max_positions - n_after, 0)]
            for _, s in candidates:
                # 백테스트와 같게 호가단위로 내리지 않은 지정가로 판정한다 (표에 보여 주는 실주문 가격과 최대 1호가 차이)
                limit = last_close[s] * (1 + st.entry_limit_pct / 100) if st.entry_limit_pct > 0 else None
                pending[s] = {"side": "buy", "symbol": s, "reason": "entry", "plan_date": day, "budget": rm.position_budget(equity), "limit": limit}
            equity_rows.append(
                {"일자": day, "현금": round(acct.cash, 2), "평가금액": round(equity, 2),
                 "고점대비%": round(rm.drawdown(equity) * 100, 2), "보유종목수": len(acct.positions)}
            )
            state.last_date = day
    finally:
        state.cash = acct.cash
        state.positions = {s: t.to_dict() for s, t in acct.positions.items()}
        state.pending = list(pending.values())
        state.last_close = last_close
        state.risk = rm.to_state()
        state.save()
        append_csv(files["journal"], events, JOURNAL_COLUMNS)
        append_csv(files["equity"], equity_rows, EQUITY_COLUMNS)

    holdings = [
        {
            "종목": market.label(s), "수량": t.qty, "매수일": f"{t.entry_time:%Y-%m-%d}", "매수가": t.entry_price,
            "최근가": last_close.get(s), "평가손익%": (last_close.get(s, t.entry_price) / t.entry_price - 1) * 100,
            "손절가": t.stop_price,
        }
        for s, t in acct.positions.items()
    ]
    planned = []
    for o in pending.values():
        s = o["symbol"]
        if o["side"] == "sell":
            qty = acct.positions[s].qty if s in acct.positions else None
            planned.append({"구분": "매도", "종목": market.label(s), "주문": "시장가", "수량": qty, "예산": None, "사유": REASONS.get(o["reason"], o["reason"])})
        else:
            price = buy_limit_price(last_close[s], st.entry_limit_pct) if o["limit"] is not None else None
            ref = price if price is not None else last_close[s]
            qty = int(o["budget"] // (ref * (1 + costs.commission)))
            planned.append({"구분": "매수", "종목": market.label(s), "주문": f"지정가 {price:,.0f}" if price else "시장가", "수량": qty, "예산": o["budget"], "사유": REASONS["entry"]})
    return DailyReport(
        market=market, now=now, dry_run=True, phase="정산·계획", base_date=state.last_date, equity=equity,
        cash=acct.cash, drawdown=rm.drawdown(equity), halted=rm.halted, halt_reason=rm.halt_reason,
        holdings=holdings, events=events, planned=planned, notes=notes, data_errors=errors, files=files,
    )


def paper_summary(cfg: AppConfig, market_key: str, bars: dict[str, pd.DataFrame]) -> dict[str, Any] | None:
    """가상계좌 기록으로 운영 기간 성과를 요약하고, 같은 기간 기준 ETF(설정의 첫 종목) 단순보유와 비교."""
    market = cfg.market(market_key)
    files = daily_files(cfg, market_key, True)
    if not files["equity"].exists():
        return None
    eq = pd.read_csv(files["equity"], encoding="utf-8-sig")
    journal = (
        pd.read_csv(files["journal"], encoding="utf-8-sig") if files["journal"].exists() else pd.DataFrame(columns=JOURNAL_COLUMNS)
    )
    state = DailyState.load(files["state"])
    init = market.initial_capital
    start, last = str(eq["일자"].iloc[0]), str(eq["일자"].iloc[-1])
    final = float(eq["평가금액"].iloc[-1])
    bench = next(iter(market.symbols))
    bench_ret = None
    b = bars.get(bench)
    if b is not None:  # 첫 체결이 가능한 날(시작 다음 거래일) 시가에 사서 마지막 처리일 종가까지
        tz = b.index.tz
        part = b[(b.index > pd.Timestamp(start).tz_localize(tz)) & (b.index <= pd.Timestamp(last).tz_localize(tz))]
        if len(part):
            bench_ret = float(buy_hold_curve(part, market.costs).iloc[-1] - 1)
    sells = journal[journal["구분"] == "매도"]
    pnl = pd.to_numeric(sells["손익"], errors="coerce")
    return {
        "market": market,
        "start": start,
        "last": last,
        "days": len(eq) - 1,  # 첫 행은 시작일(계획만 함)
        "equity": final,
        "ret": final / init - 1,
        "mdd": max_drawdown(pd.concat([pd.Series([init]), eq["평가금액"]], ignore_index=True)),
        "bench": market.label(bench),
        "bench_ret": bench_ret,
        "buys": int((journal["구분"] == "매수").sum()),
        "closed": len(sells),
        "wins": int((pnl > 0).sum()),
        "realized": float(pnl.sum()),
        "unfilled": int((journal["구분"] == "미체결").sum()),
        "halted": bool(state.risk.get("halted")),
        "halt_reason": state.risk.get("halt_reason"),
        "holdings": {s: int(t["qty"]) for s, t in state.positions.items()},
        "pending": [(o["side"], o["symbol"]) for o in state.pending],
    }


# --------------------------------------------------------------------------
# DRY_RUN=false: 한국투자증권 모의투자
# --------------------------------------------------------------------------
def vts_phase(now: pd.Timestamp) -> str:
    if now.weekday() >= 5:
        return "조회"
    t = now.time()
    if ORDER_START <= t < ORDER_END:
        return "주문"
    if CHECK_START <= t < CHECK_END:
        return "확인"
    return "조회"


def _submit(broker, state, events, market, day, side, s, qty, limit, reason, **extra) -> dict:
    rec = {"date": day, "side": side, "symbol": s, "qty": int(qty), "limit": limit, "reason": reason,
           "status": "submitted", "filled": 0, "order_no": None, "org_no": None, **extra}
    state.orders.append(rec)  # 예외가 나도 기록이 남도록 먼저 넣는다
    try:
        res = broker.place(side, s, int(qty), limit)
    except (KisError, ValueError) as e:  # 전송 전 실패(토큰·주문 형식 등)
        rec.update(status="rejected", message=f"{type(e).__name__}: {e}")
    else:
        rec.update(order_no=res.order_no, org_no=getattr(res, "org_no", None), message=res.message)
        if res.uncertain:
            rec["status"] = "uncertain"
        elif not res.ok:
            rec["status"] = "rejected"
    label = {"submitted": "주문", "uncertain": "주문(응답 없음 → 체결 조회로 확인)", "rejected": "주문 거부"}[rec["status"]]
    kind = f"{'매도' if side == 'sell' else '매수'} {label}"
    events.append(_event(day, kind, market, s, qty, limit, reason=reason, note="시장가" if limit is None else rec.get("message", "")))
    return rec


def _apply_fills(rows, orders, state, broker, market, rk, events, notes, *, cancel_rest: bool) -> None:
    """체결 조회 결과를 주문 기록·포지션 메모에 반영. cancel_rest 면 남은 매수 잔량을 취소한다."""
    by_no = {str(r.get("odno", "")).lstrip("0"): r for r in rows}
    used = {str(o.get("order_no") or "").lstrip("0") for o in state.orders if o.get("order_no")}
    for o in orders:
        s, day = o["symbol"], o["date"]
        r = by_no.get(str(o.get("order_no") or "").lstrip("0")) if o.get("order_no") else None
        if r is None and o["status"] == "uncertain":  # 응답을 못 받은 주문: 같은 종목·방향의 아직 안 쓴 주문
            side_cd = "01" if o["side"] == "sell" else "02"
            r = next(
                (x for x in rows if x.get("pdno") == to_krx_code(s) and x.get("sll_buy_dvsn_cd") == side_cd
                 and str(x.get("odno", "")).lstrip("0") not in used),
                None,
            )
            if r is not None:
                o["order_no"] = r.get("odno")
                used.add(str(r.get("odno", "")).lstrip("0"))
        if r is None:
            notes.append(f"{market.label(s)}: 체결 조회에 주문이 아직 없습니다 ({day}) → 다음 확인 때 다시 봄")
            continue
        o["org_no"] = o.get("org_no") or r.get("ord_orgno") or r.get("ord_gno_brno")
        filled, avg = int(to_float(r.get("tot_ccld_qty"))), to_float(r.get("avg_prvs"))
        rest, cancelled = int(to_float(r.get("rmn_qty"))), str(r.get("cncl_yn", "")).upper() == "Y"
        new = filled - int(o.get("filled", 0))
        if new > 0:
            o.update(filled=filled, avg_price=avg)
            if o["side"] == "buy":
                memo = state.positions.setdefault(s, {"qty": 0, "entry_date": day})
                memo.update(qty=int(memo["qty"]) + new, entry_price=avg, stop_price=avg * (1 - rk.stop_loss_pct / 100))
                fee = avg * new * market.costs.commission
                events.append(_event(day, "매수 체결", market, s, new, avg, avg * new, fee, reason=o["reason"]))
            else:
                memo = state.positions.get(s)
                gross = avg * new
                fee = gross * (market.costs.commission + market.costs.sell_tax)
                pnl = None if memo is None else (avg - float(memo["entry_price"])) * new - fee
                events.append(_event(day, "매도 체결", market, s, new, avg, gross, fee, pnl, o["reason"], "손익은 수수료 추정치 반영"))
                if memo is not None:
                    memo["qty"] = int(memo["qty"]) - new
                    if memo["qty"] <= 0:
                        state.positions.pop(s)
                        state.last_exit[s] = day
        if rest > 0 and not cancelled:
            if o["status"] == "cancel_requested":  # 취소가 닿기 전 체결분은 다음 조회에서 반영된다
                notes.append(f"{market.label(s)}: 매수 잔량 {rest}주 취소 확인 대기")
            elif o["side"] == "buy" and cancel_rest:
                try:
                    res = broker.cancel(s, str(o["order_no"]), str(o.get("org_no") or ""), rest)
                except KisError as e:
                    notes.append(f"{market.label(s)}: 미체결 매수 취소 실패 ({e}) → 다음 확인 때 다시 시도")
                    continue
                if res.ok:
                    o["status"] = "cancel_requested"
                    events.append(_event(day, "매수 잔량 취소 요청", market, s, rest, o.get("limit"), reason=o["reason"], note="시가에 체결되지 않은 지정가"))
                else:
                    notes.append(f"{market.label(s)}: 미체결 매수 취소 거부 ({res.message}) → 다음 확인 때 다시 시도")
            else:
                notes.append(f"{market.label(s)}: {'매도' if o['side'] == 'sell' else '매수'} 잔량 {rest}주 미체결")
        else:
            o["status"] = "filled" if filled >= o["qty"] else ("cancelled" if filled == 0 else "partial")


def _vts_cycle(cfg, market, bars, sigs, state, now, errors, files, notes) -> DailyReport:
    from .broker import make_broker  # 실제 전송 모드에서만 증권사 모듈 사용

    creds = load_credentials()
    if creds is None:
        raise ConfigError("DRY_RUN=false 인데 .env 에 KIS 모의투자 자격정보가 없습니다")
    broker = make_broker(cfg, market.key, creds, dry_run=False)
    rk = cfg.risk
    today = f"{now:%Y-%m-%d}"
    phase = vts_phase(now)
    events: list[dict] = []
    rm = RiskManager.from_state(rk, state.risk, None)  # 낙폭 기준은 첫 잔고 조회값에서 시작
    bal = None
    try:
        # 지난 거래일에 끝나지 않은 주문: 체결분을 반영하고 나머지는 만료(당일 주문은 장 마감에 소멸)
        for day in sorted({o["date"] for o in state.orders if o["date"] < today and o["status"] in _OPEN_STATUSES}):
            olds = [o for o in state.orders if o["date"] == day and o["status"] in _OPEN_STATUSES]
            _apply_fills(broker.daily_orders(day.replace("-", "")), olds, state, broker, market, rk, events, notes, cancel_rest=False)
            for o in olds:
                if o["status"] in _OPEN_STATUSES:
                    o["status"] = "expired"
        expired = [d for d in state.deferred if d["date"] < today]
        if expired:
            notes.append(f"지난 거래일의 대기 매수 {len(expired)}건은 만료했습니다")
            state.deferred = [d for d in state.deferred if d["date"] >= today]

        bal = broker.balance()
        for s, memo in list(state.positions.items()):  # 잔고와 메모 맞추기
            pos = bal.positions.get(s)
            if pos is None or pos.qty <= 0:
                state.positions.pop(s)
                state.last_exit[s] = today
                notes.append(f"{market.label(s)}: 잔고에 없음 → 관리 종료 (직접 매도했거나 체결 기록을 놓침)")
                continue
            if pos.qty < int(memo["qty"]):
                notes.append(f"{market.label(s)}: 잔고 {pos.qty}주 < 기록 {memo['qty']}주 → 잔고 기준으로 맞춤")
                memo["qty"] = pos.qty
            memo["sellable"] = min(int(memo["qty"]), pos.qty if pos.sellable_qty is None else pos.sellable_qty)
            memo["last_price"] = pos.last_price
        unmanaged = sorted(s for s in bal.positions if s not in state.positions)
        if unmanaged:
            notes.append("이 시스템이 연 포지션이 아닌 보유 종목은 건드리지 않습니다: " + ", ".join(market.label(s) for s in unmanaged))
        rm.update_equity(bal.total_equity, now)

        if phase == "주문":
            if state.order_phase_date == today:
                notes.append("오늘 주문 단계는 이미 실행했습니다 → 다시 주문하지 않음")
            else:
                _vts_order_phase(cfg, market, bars, sigs, state, bal, set(unmanaged), broker, rm, now, events, notes)
        elif phase == "확인":
            _vts_check_phase(cfg, market, state, broker, rm, now, events, notes)
        else:
            notes.append("주문(평일 08:30~09:00)·확인(09:02~15:20) 시간이 아닙니다 → 조회만 했습니다")
    finally:
        state.risk = rm.to_state()
        state.save()
        append_csv(files["journal"], events, JOURNAL_COLUMNS)

    equity = bal.total_equity if bal is not None else float("nan")
    holdings = [
        {
            "종목": market.label(s), "수량": m["qty"], "매수일": m.get("entry_date"), "매수가": m.get("entry_price"),
            "최근가": m.get("last_price"), "평가손익%": (m["last_price"] / m["entry_price"] - 1) * 100 if m.get("last_price") and m.get("entry_price") else None,
            "손절가": m.get("stop_price"),
        }
        for s, m in state.positions.items()
    ]
    planned = [
        {"구분": "매도" if o["side"] == "sell" else "매수", "종목": market.label(o["symbol"]), "주문": "시장가" if o["limit"] is None else f"지정가 {o['limit']:,.0f}",
         "수량": o["qty"], "예산": None, "사유": f"{REASONS.get(o['reason'], o['reason'])} · {o['status']}"}
        for o in state.orders if o["date"] == today
    ] + [
        {"구분": "매수 대기", "종목": market.label(d["symbol"]), "주문": "시장가" if d["limit"] is None else f"지정가 {d['limit']:,.0f}",
         "수량": d["qty"], "예산": None, "사유": "매도 체결 후 주문"}
        for d in state.deferred if d["date"] == today
    ]
    base = max(f"{df.index[-1]:%Y-%m-%d}" for df in bars.values())
    return DailyReport(
        market=market, now=now, dry_run=False, phase=phase, base_date=base, equity=equity,
        cash=bal.cash if bal is not None else float("nan"), drawdown=rm.drawdown(equity) if bal is not None else 0.0,
        halted=rm.halted, halt_reason=rm.halt_reason, holdings=holdings, events=events, planned=planned,
        notes=notes, data_errors=errors, files=files,
    )


def _vts_order_phase(cfg, market, bars, sigs, state, bal, unmanaged, broker, rm, now, events, notes) -> None:
    """동시호가 주문. 종목 단위로 한 번만 주문하고, 일시 오류로 판단 못 한 종목이 있으면 다시 실행할 수 있게 남긴다."""
    st, rk = cfg.strategy, cfg.risk
    today = f"{now:%Y-%m-%d}"
    fee = market.costs.commission
    done = {o["symbol"] for o in state.orders if o["date"] == today} | {d["symbol"] for d in state.deferred if d["date"] == today}
    expected = expected_latest_bar(now, market, cfg.data.interval)
    fresh: dict[str, bool | None] = {}  # None = 확인하지 못함 (수집 실패·조회 오류) → 다시 실행할 때 판단
    for s in market.symbols:
        if s in done or s not in bars:
            fresh[s] = None if s not in bars else True
            continue
        df = bars[s]
        last_day = df.index[-1].date()
        if expected is not None and last_day == expected.date():
            fresh[s] = True
            continue
        # 어제가 공휴일이면 그 전 거래일 데이터가 최신이다: 증권사 기준가(보통 전일 종가)와 마지막 종가가 같으면 최신으로 본다
        try:
            base = broker.quote(s)["base"]
        except KisError as e:
            fresh[s] = None
            notes.append(f"{market.label(s)}: 기준가 조회 실패 ({e}) → 이 종목은 다시 실행할 때 판단")
            continue
        fresh[s] = base > 0 and abs(float(df["close"].iloc[-1]) / base - 1) < 0.001
        if not fresh[s]:
            notes.append(f"{market.label(s)}: 최신 일봉이 아닙니다 (마지막 {last_day}) → 오늘은 이 종목을 판단하지 않음")
    complete = True

    for s, memo in list(state.positions.items()):
        if s in done:
            continue
        if fresh.get(s) is None:
            complete = False
            continue
        if not fresh[s]:
            continue
        sig, close = sigs[s].iloc[-1], float(bars[s]["close"].iloc[-1])
        if rm.halted and rk.liquidate_on_halt:
            reason = "halt"
        elif close <= float(memo["stop_price"]):
            reason = "stop"
        elif not bool(sig["hold"]):
            reason = "signal"
        else:
            continue
        qty = int(memo.get("sellable", memo["qty"]))
        if qty <= 0:
            notes.append(f"{market.label(s)}: 매도({REASONS.get(reason, reason)}) 보류 — 이전 매도 주문이 아직 걸려 있음")
            continue
        _submit(broker, state, events, market, today, "sell", s, qty, None, reason)

    if rm.halted:
        notes.append(f"매매 중단 상태: {rm.halt_reason} → 신규 매수 없음 (원인을 확인한 뒤 trade --resume 으로 재개)")
    else:
        # 빈자리·현금은 오늘 이미 낸 주문까지 반영해서 센다. 거부된 매도는 자리를 비우지 않는다.
        todays = [o for o in state.orders if o["date"] == today]
        selling = {o["symbol"] for o in todays if o["side"] == "sell" and o["status"] in _LIVE} & set(state.positions)
        buying = [o for o in todays if o["side"] == "buy" and o["status"] in _LIVE]
        waiting = [d for d in state.deferred if d["date"] == today]
        held_after = len(state.positions) - len(selling) + len(buying) + len(waiting)
        slots = (st.max_positions - held_after) if st.max_positions else len(bars)
        available = bal.cash - sum(o["qty"] * o.get("ref", o["limit"] or 0) * (1 + fee) for o in buying)
        cands = []
        for s in market.symbols:
            if s in done or s in state.positions or s in unmanaged:
                continue
            if fresh.get(s) is None:
                complete = False
                continue
            if not fresh[s]:
                continue
            sig = sigs[s].iloc[-1]
            if not bool(sig["entry"]) or _cooling(bars[s], state.last_exit.get(s), bars[s].index[-1], st.reentry_cooldown_bars):
                continue
            sc = float(sig["score"])
            cands.append((sc if sc == sc else -np.inf, s))
        cands.sort(key=lambda x: -x[0])
        budget = rm.position_budget(bal.total_equity)
        for rank, (_, s) in enumerate(cands, start=1):
            if rank > max(slots, 0):
                notes.append(f"{market.label(s)}: 진입 신호지만 빈자리가 없음 (최대 {st.max_positions}종목)")
                continue
            ok, why = rm.can_enter(today)
            if not ok:
                notes.append(f"{market.label(s)}: 매수 보류 — {why}")
                continue
            close = float(bars[s]["close"].iloc[-1])
            limit = buy_limit_price(close, st.entry_limit_pct) if st.entry_limit_pct > 0 else None
            ref = limit if limit is not None else close
            qty = int(budget // (ref * (1 + fee)))
            if qty <= 0:
                notes.append(f"{market.label(s)}: 1종목 예산 {budget:,.0f}원으로 1주도 살 수 없음")
                continue
            need = qty * ref * (1 + fee)
            rm.record_entry(today)  # 주문 기준으로 하루 진입 횟수를 센다
            if need <= available:
                _submit(broker, state, events, market, today, "buy", s, qty, limit, "entry", ref=ref)
                available -= need
            else:
                state.deferred.append({"date": today, "symbol": s, "qty": qty, "limit": limit, "reason": "entry"})
                events.append(_event(today, "매수 대기", market, s, qty, limit, need, reason="entry", note="현금 부족 → 매도 체결 후 주문"))

    if complete:
        state.order_phase_date = today
    else:
        notes.append("일부 종목을 확인하지 못했습니다 → 09:00 전에 다시 실행하면 그 종목만 이어서 판단합니다 (이미 낸 주문은 다시 내지 않음)")


def _vts_check_phase(cfg, market, state, broker, rm, now, events, notes) -> None:
    st, rk = cfg.strategy, cfg.risk
    today = f"{now:%Y-%m-%d}"
    todays = [o for o in state.orders if o["date"] == today]
    if state.order_phase_date != today and not todays and not any(d["date"] == today for d in state.deferred):
        notes.append("오늘 주문 단계 기록이 없습니다 → 08:30~09:00 에 trade 를 먼저 실행해야 합니다")
        return
    open_orders = [o for o in todays if o["status"] in _OPEN_STATUSES]
    if open_orders:
        _apply_fills(broker.daily_orders(now.strftime("%Y%m%d")), open_orders, state, broker, market, rk, events, notes, cancel_rest=True)
    # 동시호가에서 거부된 주문: 매도는 시장가로 한 번만 재시도, 매수는 대기 목록으로 (대기 후 낸 매수가 또 거부되면 끝)
    for o in [o for o in todays if o["status"] == "rejected"]:
        o["status"] = "retried"
        if o["side"] == "sell":
            if not o.get("retry"):
                _submit(broker, state, events, market, today, "sell", o["symbol"], o["qty"], None, o["reason"], retry=True)
        elif not o.get("deferred"):
            state.deferred.append({"date": today, "symbol": o["symbol"], "qty": o["qty"], "limit": o["limit"], "reason": o["reason"]})

    waiting = [d for d in state.deferred if d["date"] == today]
    if not waiting:
        return
    if rm.halted:
        notes.append(f"매매 중단 상태 ({rm.halt_reason}) → 대기 매수 {len(waiting)}건을 내지 않고 버립니다")
        state.deferred = [d for d in state.deferred if d["date"] != today]
        return
    todays = [o for o in state.orders if o["date"] == today]
    if any(o["side"] == "sell" and o["status"] in _LIVE for o in todays):
        notes.append("매도 체결을 기다리는 중 → 대기 매수는 다음 확인 때 주문합니다")
        return
    # 매도가 끝내 체결되지 않았으면 그 자리는 비지 않았다 → 남은 빈자리만큼만 산다
    buying = sum(1 for o in todays if o["side"] == "buy" and o["status"] in _LIVE)
    free = (st.max_positions - len(state.positions) - buying) if st.max_positions else len(waiting)
    available = broker.balance().cash
    fee = market.costs.commission
    keep = []
    for d in waiting:
        s = d["symbol"]
        try:
            q = broker.quote(s)
        except KisError as e:
            notes.append(f"{market.label(s)}: 시가 조회 실패 ({e}) → 다음 확인 때 다시")
            keep.append(d)
            continue
        if q["open"] <= 0:
            notes.append(f"{market.label(s)}: 아직 시가가 없습니다 → 다음 확인 때 다시")
            keep.append(d)
            continue
        if d["limit"] is not None and q["open"] > d["limit"]:
            events.append(_event(today, "미체결", market, s, d["qty"], q["open"], reason=d["reason"], note=f"시가 {q['open']:,.0f} > 지정가 {d['limit']:,.0f} → 주문 안 함"))
            continue
        if free <= 0:
            events.append(_event(today, "매수 못함", market, s, d["qty"], d["limit"], reason=d["reason"], note="빈자리 없음 (매도가 체결되지 않음)"))
            continue
        ref = d["limit"] if d["limit"] is not None else q["price"]
        qty = min(int(d["qty"]), int(available // (ref * (1 + fee))))
        if qty <= 0:
            events.append(_event(today, "매수 못함", market, s, d["qty"], ref, reason=d["reason"], note="매도 후에도 현금 부족"))
            continue
        _submit(broker, state, events, market, today, "buy", s, qty, d["limit"], d["reason"], deferred=True, ref=ref)
        available -= qty * ref * (1 + fee)
        free -= 1
    state.deferred = [d for d in state.deferred if d["date"] != today] + keep
