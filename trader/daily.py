"""일봉 규칙 전략(rule_breakout)의 최신 신호와 '다음 거래일 시가' 주문표.

증권사 주문 API 없이(예: 미래에셋 MTS 에 직접 입력) 쓰는 것을 전제로 한다. 주문을 보내는 코드는 없다.
  1) 장 마감 후 최신 완성 일봉까지 받아 종목별 진입·유지 상태 계산 (백테스트와 같은 rules.trend_breakout)
  2) 보유 종목(--held)과 평가금액(--equity)으로 다음 거래일 동시호가(08:30~09:00)에 낼 주문을 만든다
     - 매도: 유지 조건 이탈(종가 < 청산선) 또는 손절(종가 ≤ 매수가 × (1-손절%), 매수가를 준 경우만)
     - 매수: 진입 신호 종목을 점수 순으로 빈자리만큼. 지정가 = 종가 × (1+entry_limit_pct%) 를 호가단위로 내림
백테스트에는 있지만 여기서 자동으로 못 하는 것: 재진입 대기(매도 이력 필요), 계좌 낙폭 중단(고점 기록 필요).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, timedelta

import pandas as pd

from .broker.domestic import krx_tick_size
from .config import AppConfig, ConfigError, CostConfig, MarketConfig, RiskConfig, StrategyConfig
from .data import DataError, expected_latest_bar, load_market
from .rules import trend_breakout


@dataclass(frozen=True)
class Held:
    ticker: str
    entry_price: float | None = None  # 모르면 None → 손절은 사람이 판단


def parse_held(items: list[str], market: MarketConfig) -> list[Held]:
    """'069500', '069500.KS', '069500@105000'(매수가), 쉼표로 여러 개."""
    by_code = {t.split(".")[0]: t for t in market.symbols}
    out: list[Held] = []
    for raw in items:
        for item in (x.strip() for x in raw.split(",")):
            if not item:
                continue
            code, _, price = item.partition("@")
            if code.isdigit() and len(code) < 6:
                raise ConfigError("--held: 매수가에 쉼표(,)를 쓰지 마세요 — 쉼표는 종목 구분자입니다 (예: 069500@105000)")
            ticker = code if code in market.symbols else by_code.get(code.split(".")[0])
            if ticker is None:
                raise ConfigError(f"--held {item}: 설정 파일의 종목이 아닙니다 ({', '.join(by_code)})")
            if any(h.ticker == ticker for h in out):
                raise ConfigError(f"--held {item}: 같은 종목이 두 번 들어왔습니다")
            entry = None
            if price:
                try:
                    entry = float(price)
                except ValueError:
                    entry = -1.0
                if not entry > 0:
                    raise ConfigError(f"--held {item}: 매수가는 0보다 큰 숫자여야 합니다")
            out.append(Held(ticker, entry))
    return out


def next_weekday(d: date) -> date:
    d += timedelta(days=1)
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


def rule_signals(
    cfg: AppConfig, market_key: str, *, offline: bool = False, now: pd.Timestamp | None = None
) -> tuple[pd.DataFrame, dict[str, str]]:
    """종목별 최신 완성 일봉 기준 상태. 백테스트와 같은 계산식을 쓴다."""
    market = cfg.market(market_key)
    bars, errors = load_market(cfg, market_key, offline=offline, refresh=not offline, now=now)
    if not bars:
        raise DataError(f"[{market.name}] 모든 종목의 데이터 수집에 실패했습니다: {errors}")
    clock = pd.Timestamp.now(tz=market.timezone) if now is None else now.tz_convert(market.timezone)
    expected = None if offline else expected_latest_bar(clock, market, cfg.data.interval)
    rows = []
    for sym in market.symbols:
        if sym not in bars:
            continue
        df = bars[sym]
        s = trend_breakout(df, cfg.strategy).iloc[-1]
        day, close = df.index[-1].date(), float(df["close"].iloc[-1])
        valid = not (pd.isna(s["sma_long"]) or pd.isna(s["sma_exit"]) or pd.isna(s["prior_high"]))
        stale = expected is not None and day < expected.date()
        note = ""
        if not valid:
            note = f"데이터 부족 ({len(df)}일) — 판단 안 함"
        elif stale:
            note = f"최신 일봉 지연/누락 (기대 {expected:%m-%d}) → 신규 매수 보류"
        rows.append(
            {
                "symbol": sym,
                "종목": market.label(sym),
                "기준일": day,
                "종가": close,
                "장기선": s["sma_long"],
                "직전고점": s["prior_high"],
                "청산선": s["sma_exit"],
                "점수%": s["score"] * 100,
                "추세": bool(valid and close > s["sma_long"]),
                "진입": bool(s["entry"]),
                "유지": bool(s["hold"]),
                "valid": valid,
                "stale": stale,
                "비고": note,
            }
        )
    return pd.DataFrame(rows).set_index("symbol"), errors


@dataclass
class OrderLine:
    side: str  # 매도 | 매수
    ticker: str
    label: str
    reason: str
    limit: float | None = None  # 매수 지정가. 매도는 시장가(시가 단일가) → None
    qty: int | None = None  # 매도는 보유 전량 → None
    amount: float | None = None


@dataclass
class OrderPlan:
    budget: float  # 1종목 예산
    slots: int  # 매도 후 빈자리
    sells: list[OrderLine] = field(default_factory=list)
    buys: list[OrderLine] = field(default_factory=list)
    waiting: list[str] = field(default_factory=list)  # 진입 신호지만 빈자리가 없어 대기
    notes: list[str] = field(default_factory=list)


def buy_limit_price(close: float, limit_pct: float) -> float:
    """종가 × (1+N%) 를 주식 호가단위로 내림. 주식 호가단위는 ETF 호가단위(5원)의 배수라 ETF 에도 유효한 가격이다."""
    raw = close * (1 + limit_pct / 100)
    tick = krx_tick_size(raw)
    return float(math.floor(raw / tick) * tick)


def plan_orders(
    sig: pd.DataFrame, held: list[Held], equity: float, st: StrategyConfig, rk: RiskConfig, costs: CostConfig
) -> OrderPlan:
    kept: list[Held] = []
    sells: list[OrderLine] = []
    notes: list[str] = []
    for h in held:
        if h.ticker not in sig.index:
            notes.append(f"{h.ticker}: 데이터 수집 실패로 판단하지 못했습니다 — 직접 확인하세요")
            kept.append(h)
            continue
        r = sig.loc[h.ticker]
        if not r["valid"] or r["stale"]:
            notes.append(f"{r['종목']}: 최신 일봉이 없거나 데이터가 부족해 매도 판단을 보류합니다")
            kept.append(h)
            continue
        stop = None if h.entry_price is None else h.entry_price * (1 - rk.stop_loss_pct / 100)
        if stop is not None and r["종가"] <= stop:
            reason = f"손절: 종가 {r['종가']:,.0f} ≤ 매수가 {h.entry_price:,.0f}의 -{rk.stop_loss_pct:g}% ({stop:,.0f})"
        elif not r["유지"]:
            reason = f"종가 {r['종가']:,.0f} < 청산선 {r['청산선']:,.0f}"
        else:
            kept.append(h)
            continue
        sells.append(OrderLine("매도", h.ticker, r["종목"], reason))

    held_set = {h.ticker for h in held}  # 오늘 파는 종목도 같은 날 다시 사지 않는다 (백테스트와 동일)
    slots = (st.max_positions - len(kept)) if st.max_positions else len(sig)
    slots = max(min(slots, rk.max_trades_per_day), 0)
    budget = max(equity, 0.0) * rk.position_size_pct / 100
    cands = sig[sig["진입"] & sig["valid"] & ~sig["stale"] & ~sig.index.isin(list(held_set))]
    cands = cands.sort_values("점수%", ascending=False, kind="stable", na_position="last")
    plan = OrderPlan(budget=budget, slots=slots, sells=sells, notes=notes)
    for rank, (sym, r) in enumerate(cands.iterrows(), start=1):
        if rank > slots:
            plan.waiting.append(r["종목"])
            continue
        limit = buy_limit_price(r["종가"], st.entry_limit_pct) if st.entry_limit_pct > 0 else None
        ref = limit if limit is not None else r["종가"]
        qty = int(budget // (ref * (1 + costs.commission)))
        if qty <= 0:
            notes.append(f"{r['종목']}: 1종목 예산 {budget:,.0f}원으로 1주({ref:,.0f}원)도 살 수 없습니다")
            continue
        plan.buys.append(
            OrderLine("매수", sym, r["종목"], f"점수 {r['점수%']:+.1f}% · 순위 {rank}", limit, qty, qty * ref)
        )
    return plan
