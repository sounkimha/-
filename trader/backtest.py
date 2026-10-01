"""이벤트 방식 백테스트 (시장별 계좌 1개 · 여러 종목).

체결 규칙 (미래 정보 없이, 보수적으로):
- t 봉 종가에서 계산한 신호 → t+1 봉 '시가'에 체결. 매수가는 시가×(1+슬리피지), 매도가는 ×(1-슬리피지).
- 손절: 보유 중인 봉의 시가가 이미 손절가 아래(갭하락)면 그 시가에 청산 → 손절가보다 더 크게 손실.
        그렇지 않고 저가가 손절가에 닿으면 손절가에 청산. 진입한 봉에서도 손절될 수 있다.
- 장 마감 청산(flatten_at_session_end): 다음 봉이 그날 마지막 봉이면 그 시가에 청산하고, 마지막 봉 신규 진입은 막는다.
- 비용: 편도 수수료(매수·매도), 매도세, 편도 슬리피지.
- 손실 제한(1회 투입 비율·손절·계좌 최대낙폭 중단·일일 진입 횟수)은 risk.RiskManager 가 판단한다.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from functools import reduce
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np
import pandas as pd

from .config import AppConfig, CostConfig, MarketConfig, RiskConfig, StrategyConfig
from .features import make_dataset
from .model import WalkForwardResult, evaluate_predictions, walk_forward
from .risk import RiskManager

EXIT_REASON_KO = {
    "signal": "신호약화",
    "stop": "손절",
    "gap_stop": "갭손절(손절가보다 불리)",
    "session_end": "장마감 청산",
    "max_hold": "보유기간 만료",
    "halt": "매매중단 청산",
    "end_of_data": "기간 종료",
}


# --------------------------------------------------------------------------
# 체결 / 계좌
# --------------------------------------------------------------------------
@dataclass
class Trade:
    symbol: str
    entry_time: pd.Timestamp
    entry_price: float  # 슬리피지 포함 체결가
    qty: int
    stop_price: float
    entry_fee: float
    exit_time: pd.Timestamp | None = None
    exit_price: float | None = None
    exit_fee: float = 0.0  # 수수료 + 매도세
    exit_reason: str | None = None
    bars_held: int = 0

    @property
    def cost_basis(self) -> float:
        return self.entry_price * self.qty + self.entry_fee

    @property
    def pnl(self) -> float:
        if self.exit_price is None:
            raise ValueError("아직 청산되지 않은 거래입니다")
        return (self.exit_price - self.entry_price) * self.qty - self.entry_fee - self.exit_fee

    @property
    def ret(self) -> float:
        return self.pnl / self.cost_basis

    def unrealized(self, price: float) -> float:
        return (price - self.entry_price) * self.qty - self.entry_fee

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for k in ("entry_time", "exit_time"):
            d[k] = None if d[k] is None else pd.Timestamp(d[k]).isoformat()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Trade":
        d = dict(d)
        for k in ("entry_time", "exit_time"):
            d[k] = None if d.get(k) is None else pd.Timestamp(d[k])
        return cls(**d)


class Account:
    """현금 + 보유 포지션. 백테스트와 DRY_RUN 가상계좌가 같이 쓴다."""

    def __init__(self, cash: float, costs: CostConfig):
        self.cash = float(cash)
        self.costs = costs
        self.positions: dict[str, Trade] = {}
        self.closed: list[Trade] = []

    def buy(
        self,
        symbol: str,
        ref_price: float,
        budget: float,
        when: pd.Timestamp,
        stop_price_fn: Callable[[float], float],
        max_qty: int | None = None,
    ) -> Trade | None:
        price = ref_price * (1 + self.costs.slippage)
        qty = int(min(budget, self.cash) // (price * (1 + self.costs.commission)))
        if max_qty is not None:
            qty = min(qty, max_qty)
        if qty <= 0:
            return None
        fee = price * qty * self.costs.commission
        self.cash -= price * qty + fee
        trade = Trade(symbol, when, price, qty, stop_price=stop_price_fn(price), entry_fee=fee)
        self.positions[symbol] = trade
        return trade

    def sell(self, symbol: str, ref_price: float, when: pd.Timestamp, reason: str) -> Trade:
        trade = self.positions.pop(symbol)
        price = ref_price * (1 - self.costs.slippage)
        gross = price * trade.qty
        fee = gross * (self.costs.commission + self.costs.sell_tax)
        self.cash += gross - fee
        trade.exit_time, trade.exit_price, trade.exit_fee, trade.exit_reason = when, price, fee, reason
        self.closed.append(trade)
        return trade

    def market_value(self, prices: dict[str, float]) -> float:
        return sum(t.qty * prices[s] for s, t in self.positions.items())

    def equity(self, prices: dict[str, float]) -> float:
        return self.cash + self.market_value(prices)


# --------------------------------------------------------------------------
# 백테스트 본체
# --------------------------------------------------------------------------
@dataclass
class BacktestResult:
    market: MarketConfig
    initial_capital: float
    start: pd.Timestamp
    end: pd.Timestamp
    equity: pd.Series
    exposure: pd.Series
    trades: list[Trade]
    symbol_pnl: pd.DataFrame  # 종목별 누적 손익(실현+평가)
    buy_hold: dict[str, pd.Series]  # 종목별 단순보유 가치(시작=1, 비용 반영)
    halted_at: pd.Timestamp | None = None
    halt_reason: str | None = None
    symbols: list[str] = field(default_factory=list)


def _prepare(bars, signals, start) -> dict[str, SimpleNamespace]:
    data = {}
    for s, df in bars.items():
        if s not in signals:
            continue
        if start is not None:
            df = df[df.index >= start]
        if df.empty:
            continue
        sig = signals[s].reindex(df.index)
        day = df.index.normalize()
        data[s] = SimpleNamespace(
            df=df,
            time=df.index,
            o=df["open"].to_numpy(float),
            h=df["high"].to_numpy(float),
            l=df["low"].to_numpy(float),
            c=df["close"].to_numpy(float),
            prob=sig["prob"].to_numpy(float),
            er=sig["exp_ret"].to_numpy(float),
            is_last=np.r_[np.asarray(day[1:] != day[:-1]), True],
            n=len(df),
        )
    return data


def buy_hold_curve(df: pd.DataFrame, costs: CostConfig) -> pd.Series:
    """첫 봉 시가에 사서 들고 있다가 각 시점 종가에 판다고 했을 때의 가치(시작=1, 왕복비용 반영)."""
    entry = df["open"].iloc[0] * (1 + costs.slippage) * (1 + costs.commission)
    return df["close"] * (1 - costs.slippage) * (1 - costs.commission - costs.sell_tax) / entry


def run_backtest(
    bars: dict[str, pd.DataFrame],
    signals: dict[str, pd.DataFrame],
    market: MarketConfig,
    strategy: StrategyConfig,
    risk_cfg: RiskConfig,
    start: pd.Timestamp | None = None,
) -> BacktestResult:
    """signals[종목] = DataFrame(index=봉 시작시각, columns=[prob, exp_ret]) — 각 봉 종가 시점에 계산된 값."""
    data = _prepare(bars, signals, start)
    if not data:
        raise ValueError("백테스트할 데이터가 없습니다")
    syms = list(data)
    timeline = reduce(lambda a, b: a.union(b), [d.time for d in data.values()]).tz_convert(market.timezone)

    acct = Account(market.initial_capital, market.costs)
    rm = RiskManager(risk_cfg, market.initial_capital)
    pending: dict[str, tuple[str, str, float]] = {}  # 종목 -> (buy|sell, 사유, 매수예산)
    ptr = dict.fromkeys(syms, 0)
    last_close: dict[str, float] = {}
    realized = dict.fromkeys(syms, 0.0)
    eq, expo, sym_rows = [], [], []
    halted_at = None

    for ts in timeline:
        day_key = ts.date().isoformat()
        active = [s for s in syms if ptr[s] < data[s].n and data[s].time[ptr[s]] == ts]

        # 1) 이번 봉: 대기 주문 시가 체결 → 손절 확인
        for s in active:
            d, i = data[s], ptr[s]
            o, l, c = d.o[i], d.l[i], d.c[i]
            order = pending.pop(s, None)
            if order is not None:
                side, reason, budget = order
                if side == "sell" and s in acct.positions:
                    realized[s] += acct.sell(s, o, ts, reason).pnl
                elif side == "buy" and s not in acct.positions:
                    ok, _ = rm.can_enter(day_key)
                    if ok and acct.buy(s, o, budget, ts, rm.stop_price) is not None:
                        rm.record_entry(day_key)
            pos = acct.positions.get(s)
            if pos is not None:
                if pos.entry_time < ts and o <= pos.stop_price:
                    realized[s] += acct.sell(s, o, ts, "gap_stop").pnl
                elif l <= pos.stop_price:
                    realized[s] += acct.sell(s, pos.stop_price, ts, "stop").pnl
                else:
                    pos.bars_held += 1
            last_close[s] = c

        # 2) 종가 기준 평가 → 계좌 최대낙폭 확인
        equity = acct.equity(last_close)
        if rm.update_equity(equity, ts):
            halted_at = ts
        eq.append(equity)
        expo.append(acct.market_value(last_close) / equity if equity > 0 else 0.0)
        sym_rows.append(
            [realized[s] + (acct.positions[s].unrealized(last_close[s]) if s in acct.positions else 0.0) for s in syms]
        )
        if rm.halted:
            for s in [s for s, o in pending.items() if o[0] == "buy"]:
                del pending[s]
            if risk_cfg.liquidate_on_halt:
                for s in acct.positions:
                    pending[s] = ("sell", "halt", 0.0)

        # 3) 다음 봉 주문 결정 (이번 봉 종가까지의 정보만 사용)
        for s in active:
            d, i = data[s], ptr[s]
            ptr[s] = i + 1
            if s in pending or i + 1 >= d.n:
                continue
            next_is_last = bool(d.is_last[i + 1])
            p, er = d.prob[i], d.er[i]
            pos = acct.positions.get(s)
            if pos is not None:
                reason = None
                if strategy.flatten_at_session_end and next_is_last:
                    reason = "session_end"
                elif not p >= strategy.exit_threshold:  # NaN 이어도 청산
                    reason = "signal"
                elif strategy.max_hold_bars and pos.bars_held >= strategy.max_hold_bars:
                    reason = "max_hold"
                if reason:
                    pending[s] = ("sell", reason, 0.0)
            elif not rm.halted:
                blocked_last = strategy.flatten_at_session_end and next_is_last
                if p >= strategy.entry_threshold and er >= strategy.min_expected_return and not blocked_last:
                    pending[s] = ("buy", "", rm.position_budget(equity))

    # 기간 끝: 남은 포지션은 마지막 종가에 청산 (비용 반영)
    for s in list(acct.positions):
        d = data[s]
        realized[s] += acct.sell(s, d.c[-1], d.time[-1], "end_of_data").pnl
    eq[-1] = acct.cash
    sym_rows[-1] = [realized[s] for s in syms]

    return BacktestResult(
        market=market,
        initial_capital=market.initial_capital,
        start=timeline[0],
        end=timeline[-1],
        equity=pd.Series(eq, index=timeline, name="equity"),
        exposure=pd.Series(expo, index=timeline, name="exposure"),
        trades=sorted(acct.closed, key=lambda t: t.entry_time),
        symbol_pnl=pd.DataFrame(sym_rows, index=timeline, columns=syms),
        buy_hold={s: buy_hold_curve(data[s].df, market.costs) for s in syms},
        halted_at=halted_at,
        halt_reason=rm.halt_reason,
        symbols=syms,
    )


# --------------------------------------------------------------------------
# 요약
# --------------------------------------------------------------------------
def max_drawdown(curve: pd.Series) -> float:
    if curve.empty:
        return float("nan")
    return float((curve / curve.cummax() - 1).min())


def _trade_stats(trades: list[Trade]) -> tuple[int, float, float]:
    if not trades:
        return 0, float("nan"), float("nan")
    rets = np.array([t.ret for t in trades])
    return len(trades), float((rets > 0).mean() * 100), float(rets.mean() * 100)


def compounded_return(trades: list[Trade]) -> float:
    """투입금 기준 수익률: 매 거래에 같은 돈(배정액 전부)을 넣었다고 보고 거래 수익률을 복리로 곱한 값."""
    return float(np.prod([1 + t.ret for t in trades]) - 1) if trades else 0.0


def equal_weight_buy_hold(result: BacktestResult) -> pd.Series:
    curves = pd.concat(result.buy_hold, axis=1).sort_index()
    return curves.ffill().fillna(1.0).mean(axis=1)


def summarize(result: BacktestResult) -> tuple[pd.DataFrame, dict[str, Any]]:
    init = result.initial_capital
    rows = []
    for s in result.symbols:
        trades = [t for t in result.trades if t.symbol == s]
        n, win, avg = _trade_stats(trades)
        bh = result.buy_hold[s]
        rows.append(
            {
                "종목": result.market.label(s),
                "전략수익%(계좌)": result.symbol_pnl[s].iloc[-1] / init * 100,
                "전략수익%(투입금)": compounded_return(trades) * 100,
                "단순보유%": (bh.iloc[-1] - 1) * 100,
                "거래수": n,
                "승률%": win,
                "평균손익%": avg,
                "최대낙폭%(전략)": max_drawdown(init + result.symbol_pnl[s]) * 100,
                "최대낙폭%(보유)": max_drawdown(bh) * 100,
            }
        )
    ew = equal_weight_buy_hold(result)
    n, win, avg = _trade_stats(result.trades)
    rows.append(
        {
            "종목": "계좌 합계",
            "전략수익%(계좌)": (result.equity.iloc[-1] / init - 1) * 100,
            "전략수익%(투입금)": float("nan"),
            "단순보유%": (ew.iloc[-1] - 1) * 100,
            "거래수": n,
            "승률%": win,
            "평균손익%": avg,
            "최대낙폭%(전략)": max_drawdown(result.equity) * 100,
            "최대낙폭%(보유)": max_drawdown(ew) * 100,
        }
    )
    reasons = Counter(EXIT_REASON_KO.get(t.exit_reason, t.exit_reason) for t in result.trades)
    info = {
        "기간": f"{result.start:%Y-%m-%d %H:%M} ~ {result.end:%Y-%m-%d %H:%M} ({result.market.timezone})",
        "거래일": int(pd.Index(result.equity.index.normalize()).nunique()),
        "왕복비용%": result.market.costs.round_trip * 100,
        "평균투입비중%": float(result.exposure.mean() * 100),
        "청산사유": dict(reasons),
        "매매중단": f"{result.halted_at} — {result.halt_reason}" if result.halted_at is not None else "없음",
    }
    return pd.DataFrame(rows), info


def trades_frame(result: BacktestResult) -> pd.DataFrame:
    rows = []
    for t in result.trades:
        d = t.to_dict()
        d.update(pnl=t.pnl, ret_pct=t.ret * 100, exit_reason_ko=EXIT_REASON_KO.get(t.exit_reason, t.exit_reason))
        rows.append(d)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 시장 단위 파이프라인: 특징값 → 워크포워드 예측 → 백테스트 → 요약
# --------------------------------------------------------------------------
@dataclass
class MarketBacktest:
    result: BacktestResult
    walk_forward: WalkForwardResult
    model_eval: dict[str, float]
    table: pd.DataFrame
    info: dict[str, Any]


def signals_from_walk_forward(ds: pd.DataFrame, wf: WalkForwardResult) -> dict[str, pd.DataFrame]:
    frame = ds[["time", "symbol"]].assign(prob=wf.prob, exp_ret=wf.exp_ret)
    return {s: g.set_index("time")[["prob", "exp_ret"]] for s, g in frame.groupby("symbol", sort=False)}


def backtest_market(cfg: AppConfig, market_key: str, bars: dict[str, pd.DataFrame]) -> MarketBacktest:
    market = cfg.market(market_key)
    cost = market.costs.round_trip
    ds = make_dataset(bars, cfg.features, cost)
    wf = walk_forward(ds, cfg.model, cost)
    predicted = wf.prob.notna()
    if not predicted.any():
        raise ValueError("워크포워드 예측이 하나도 없습니다 (데이터가 너무 짧음)")
    start = ds.loc[predicted, "time"].min()
    result = run_backtest(bars, signals_from_walk_forward(ds, wf), market, cfg.strategy, cfg.risk, start=start)
    table, info = summarize(result)
    model_eval = evaluate_predictions(ds, wf.prob, cfg.strategy.entry_threshold)
    return MarketBacktest(result=result, walk_forward=wf, model_eval=model_eval, table=table, info=info)
