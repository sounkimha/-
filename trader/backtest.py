"""이벤트 방식 백테스트 (시장별 계좌 1개 · 여러 종목).

체결 규칙 (미래 정보 없이, 보수적으로):
- t 봉 종가에서 계산한 신호 → t+1 봉 '시가'에 체결. 매수가는 시가×(1+슬리피지), 매도가는 ×(1-슬리피지).
- 손절: 보유 중인 봉의 시가가 이미 손절가 아래(갭하락)면 그 시가에 청산 → 손절가보다 더 크게 손실.
        그렇지 않고 저가가 손절가에 닿으면 손절가에 청산. 진입한 봉에서도 손절될 수 있다.
        stop_check=close 면 봉 중에는 보지 않고, 종가가 손절가 이하일 때 다음 봉 시가에 청산(갭이면 더 손실).
- 지정가 매수(entry_limit_pct>0): 지정가 = 신호 봉 종가×(1+N%). 다음 봉 시가가 그보다 높으면 미체결(당일 취소).
- 같은 시가에서는 매도 → 매수 순서로 체결해 매도 대금을 바로 쓴다. 실제 시가 동시호가에서는 매도가 체결되기 전이라
  그 대금이 주문가능금액에 안 잡히므로, 실전에서는 '매도 체결 직후 매수'로 근사된다.
- 장 마감 청산(flatten_at_session_end): 다음 봉이 그날 마지막 봉이면 그 시가에 청산하고, 마지막 봉 신규 진입은 막는다.
- 비용: 편도 수수료(매수·매도), 매도세, 편도 슬리피지.
- 손실 제한(1회 투입 비율·손절·계좌 최대낙폭 중단·일일 진입 횟수)은 risk.RiskManager 가 판단한다.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field, replace
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
    unfilled_entries: int = 0  # 지정가보다 높게 시작해 체결되지 않은 매수 주문 수


def _column(sig: pd.DataFrame, name: str, dtype=float):
    return sig[name].to_numpy(dtype) if name in sig.columns else None


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
        nan = np.full(len(df), np.nan)
        data[s] = SimpleNamespace(
            df=df,
            time=df.index,
            o=df["open"].to_numpy(float),
            h=df["high"].to_numpy(float),
            l=df["low"].to_numpy(float),
            c=df["close"].to_numpy(float),
            prob=_column(sig, "prob") if "prob" in sig.columns else nan,
            er=_column(sig, "exp_ret") if "exp_ret" in sig.columns else nan,
            # 규칙 전략용(있으면 prob 대신 사용): entry/hold 는 NaN 이면 False
            entry=None if "entry" not in sig.columns else sig["entry"].fillna(False).to_numpy(bool),
            hold=None if "hold" not in sig.columns else sig["hold"].fillna(False).to_numpy(bool),
            score=_column(sig, "score"),
            is_last=np.r_[np.asarray(day[1:] != day[:-1]), True],
            n=len(df),
        )
    return data


def _wants_entry(d, i, st: StrategyConfig) -> bool:
    if d.entry is not None:
        return bool(d.entry[i])
    return bool(d.prob[i] >= st.entry_threshold and d.er[i] >= st.min_expected_return)


def _wants_hold(d, i, st: StrategyConfig) -> bool:
    if d.hold is not None:
        return bool(d.hold[i])
    return bool(d.prob[i] >= st.exit_threshold)  # NaN 이면 False → 청산


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
    """signals[종목] = 봉 종가 시점에 계산된 값.

    - ML: columns=[prob, exp_ret] → strategy 임계값으로 진입/청산
    - 규칙: columns=[entry, hold, score] → entry 면 진입 후보, hold 가 False 면 청산, score 높은 순으로 빈자리 채움
    """
    data = _prepare(bars, signals, start)
    if not data:
        raise ValueError("백테스트할 데이터가 없습니다")
    syms = list(data)
    timeline = reduce(lambda a, b: a.union(b), [d.time for d in data.values()]).tz_convert(market.timezone)

    acct = Account(market.initial_capital, market.costs)
    rm = RiskManager(risk_cfg, market.initial_capital)
    pending: dict[str, tuple[str, str, float, float | None]] = {}  # 종목 -> (buy|sell, 사유, 매수예산, 지정가)
    ptr = dict.fromkeys(syms, 0)
    last_close: dict[str, float] = {}
    last_exit: dict[str, int] = {}  # 종목 -> 마지막 청산 봉 번호 (재진입 대기용)
    realized = dict.fromkeys(syms, 0.0)
    eq, expo, sym_rows = [], [], []
    halted_at = None
    unfilled = 0
    intrabar_stop = risk_cfg.stop_check == "intrabar"

    def close_position(s: str, price: float, ts, reason: str) -> None:
        realized[s] += acct.sell(s, price, ts, reason).pnl
        last_exit[s] = ptr[s]

    for ts in timeline:
        day_key = ts.date().isoformat()
        active = [s for s in syms if ptr[s] < data[s].n and data[s].time[ptr[s]] == ts]
        active_set = set(active)

        # 1) 이번 봉 시가: 대기 매도 → 대기 매수 순서로 체결 (매도 대금을 같은 시가 매수에 쓸 수 있게)
        for s in active:
            order = pending.get(s)
            if order is not None and order[0] == "sell":
                del pending[s]
                if s in acct.positions:
                    close_position(s, data[s].o[ptr[s]], ts, order[1])
        for s in [s for s, o in list(pending.items()) if s in active_set and o[0] == "buy"]:
            _, _, budget, limit = pending.pop(s)
            d, i = data[s], ptr[s]
            if s in acct.positions:
                continue
            if limit is not None and d.o[i] > limit:  # 지정가보다 높게 시작 → 미체결(당일 취소)
                unfilled += 1
                continue
            ok, _ = rm.can_enter(day_key)
            if ok and acct.buy(s, d.o[i], budget, ts, rm.stop_price) is not None:
                rm.record_entry(day_key)
        # 손절(봉 중 판단 모드): 갭으로 시가가 이미 손절가 아래면 시가, 아니면 저가가 닿을 때 손절가
        for s in active:
            d, i = data[s], ptr[s]
            pos = acct.positions.get(s)
            if pos is not None and intrabar_stop:
                if pos.entry_time < ts and d.o[i] <= pos.stop_price:
                    close_position(s, d.o[i], ts, "gap_stop")
                elif d.l[i] <= pos.stop_price:
                    close_position(s, pos.stop_price, ts, "stop")
            pos = acct.positions.get(s)
            if pos is not None:
                pos.bars_held += 1
            last_close[s] = d.c[i]

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
                    pending[s] = ("sell", "halt", 0.0, None)

        # 3) 다음 봉 주문 결정 (이번 봉 종가까지의 정보만 사용): 청산 먼저, 그다음 진입 후보를 점수 순으로
        candidates: list[tuple[float, str]] = []
        for s in active:
            d, i = data[s], ptr[s]
            if s in pending or i + 1 >= d.n:
                continue
            next_is_last = bool(d.is_last[i + 1])
            pos = acct.positions.get(s)
            if pos is not None:
                reason = None
                if not intrabar_stop and d.c[i] <= pos.stop_price:  # 종가 손절 → 다음 시가 청산(갭이면 더 손실)
                    reason = "stop"
                elif strategy.flatten_at_session_end and next_is_last:
                    reason = "session_end"
                elif not _wants_hold(d, i, strategy):
                    reason = "signal"
                elif strategy.max_hold_bars and pos.bars_held >= strategy.max_hold_bars:
                    reason = "max_hold"
                if reason:
                    pending[s] = ("sell", reason, 0.0, None)
            elif not rm.halted:
                blocked_last = strategy.flatten_at_session_end and next_is_last
                cooling = strategy.reentry_cooldown_bars and s in last_exit and i - last_exit[s] < strategy.reentry_cooldown_bars
                if not blocked_last and not cooling and _wants_entry(d, i, strategy):
                    sc = d.score[i] if d.score is not None else 0.0
                    candidates.append((sc if sc == sc else -np.inf, s))  # NaN 점수는 맨 뒤
        if candidates:
            candidates.sort(key=lambda x: -x[0])  # 안정 정렬: 점수가 같으면 기존 순서 유지
            if strategy.max_positions:
                n_after = (
                    len(acct.positions)
                    - sum(1 for o in pending.values() if o[0] == "sell")
                    + sum(1 for o in pending.values() if o[0] == "buy")
                )
                candidates = candidates[: max(strategy.max_positions - n_after, 0)]
            for _, s in candidates:
                d, i = data[s], ptr[s]
                limit = d.c[i] * (1 + strategy.entry_limit_pct / 100) if strategy.entry_limit_pct > 0 else None
                pending[s] = ("buy", "", rm.position_budget(equity), limit)
        for s in active:
            ptr[s] += 1

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
        unfilled_entries=unfilled,
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


def _is_daily_index(idx: pd.DatetimeIndex) -> bool:
    return bool(len(idx)) and bool(((idx.hour == 0) & (idx.minute == 0)).all())


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
    fmt = "%Y-%m-%d" if _is_daily_index(result.equity.index) else "%Y-%m-%d %H:%M"
    info = {
        "기간": f"{result.start:{fmt}} ~ {result.end:{fmt}} ({result.market.timezone})",
        "거래일": int(pd.Index(result.equity.index.normalize()).nunique()),
        "왕복비용%": result.market.costs.round_trip * 100,
        "평균투입비중%": float(result.exposure.mean() * 100),
        "청산사유": dict(reasons),
        "매매중단": f"{result.halted_at:{fmt}} — {result.halt_reason}" if result.halted_at is not None else "없음",
        "지정가미체결": result.unfilled_entries,
    }
    return pd.DataFrame(rows), info


def _period_return(curve: pd.Series, mask: pd.Series, base: float) -> tuple[float, float]:
    part = curve[mask]
    start = curve[curve.index < part.index[0]]
    b = start.iloc[-1] if len(start) else base
    path = pd.concat([pd.Series([b]), part.reset_index(drop=True)])
    return float(part.iloc[-1] / b - 1), max_drawdown(path)


def _curve_stats(curve: pd.Series, base: float, years: float) -> dict[str, float]:
    total = curve.iloc[-1] / base
    daily = pd.concat([pd.Series([base]), curve.reset_index(drop=True)]).pct_change().dropna()
    sd = daily.std()
    return {
        "총수익%": (total - 1) * 100,
        "연환산%": (total ** (1 / years) - 1) * 100 if years > 0 and total > 0 else float("nan"),
        "최대낙폭%": max_drawdown(pd.concat([pd.Series([base]), curve])) * 100,
        "샤프(rf=0)": float(daily.mean() / sd * np.sqrt(252)) if sd > 0 else float("nan"),
    }


def comparison_table(
    result: BacktestResult, benchmark: str | None = None, extras: dict[str, BacktestResult] | None = None
) -> pd.DataFrame:
    """같은 기간의 전략 / (비용 가정을 바꾼) 전략 / 기준 종목 보유 / 동일가중 보유를 한 표로. 일봉 기준 지표."""
    years = (result.end - result.start).days / 365.25
    idx = result.equity.index
    rows = []
    for name, r in {"전략": result, **(extras or {})}.items():
        rows.append({"구분": name, **_curve_stats(r.equity, r.initial_capital, years),
                     "평균투입%": r.exposure.mean() * 100, "거래수": len(r.trades)})
    if benchmark:
        bench = result.buy_hold[benchmark].reindex(idx).ffill().fillna(1.0)
        rows.append({"구분": f"{result.market.label(benchmark)} 보유", **_curve_stats(bench, 1.0, years),
                     "평균투입%": 100.0, "거래수": float("nan")})
    ew = equal_weight_buy_hold(result).reindex(idx).ffill().fillna(1.0)
    rows.append({"구분": f"동일가중 {len(result.symbols)}종목 보유", **_curve_stats(ew, 1.0, years),
                 "평균투입%": 100.0, "거래수": float("nan")})
    out = pd.DataFrame(rows)
    out["거래수"] = out["거래수"].astype("Int64")
    return out


def with_cost_multiplier(cfg: AppConfig, market_key: str, mult: float) -> AppConfig:
    """수수료·슬리피지만 mult 배로 (세금은 정해진 값이라 그대로). 비용 민감도 점검용."""
    m = cfg.market(market_key)
    costs = replace(m.costs, commission_pct=m.costs.commission_pct * mult, slippage_pct=m.costs.slippage_pct * mult)
    return replace(cfg, markets={**cfg.markets, market_key: replace(m, costs=costs)})


def yearly_table(result: BacktestResult, benchmark: str | None = None) -> pd.DataFrame:
    """연도별: 전략 / 동일가중 단순보유 / 기준 종목 단순보유 수익률과 전략 MDD, 진입 수."""
    eq = result.equity
    ew = equal_weight_buy_hold(result).reindex(eq.index).ffill().fillna(1.0)
    bench = result.buy_hold.get(benchmark) if benchmark else None
    bench = None if bench is None else bench.reindex(eq.index).ffill().fillna(1.0)
    rows = []
    for year in sorted(set(eq.index.year)):
        m = pd.Series(eq.index.year == year, index=eq.index)
        r, mdd = _period_return(eq, m, result.initial_capital)
        row = {"연도": year, "전략%": r * 100, "전략MDD%": mdd * 100, "동일가중보유%": _period_return(ew, m, 1.0)[0] * 100}
        if bench is not None:
            row[f"{result.market.label(benchmark)} 보유%"] = _period_return(bench, m, 1.0)[0] * 100
        row["진입수"] = sum(1 for t in result.trades if t.entry_time.year == year)
        rows.append(row)
    return pd.DataFrame(rows)


def regime_table(result: BacktestResult, bench_bars: pd.DataFrame, benchmark: str, ma: int = 200, slope: int = 20) -> pd.DataFrame:
    """기준 종목의 200일선과 그 기울기로 국면을 나눠 국면별 연환산 수익률을 비교 (보고용, 매매에는 쓰지 않음)."""
    c = bench_bars["close"]
    sma = c.rolling(ma).mean()
    up = (c > sma) & (sma > sma.shift(slope))
    down = (c < sma) & (sma < sma.shift(slope))
    regime = pd.Series("횡보장", index=c.index).mask(up, "상승장").mask(down, "하락장").where(sma.notna())
    eq = result.equity
    reg = regime.reindex(eq.index).ffill()
    strat = eq.pct_change()
    bh = result.buy_hold[benchmark].reindex(eq.index).ffill().pct_change()
    rows = []
    for name in ("상승장", "횡보장", "하락장"):
        m = (reg == name) & strat.notna()
        n = int(m.sum())
        if n == 0:
            continue
        ann = lambda r: float((1 + r[m].fillna(0)).prod() ** (252 / n) - 1) * 100
        rows.append({"국면": name, "거래일": n, "전략 연환산%": ann(strat), f"{result.market.label(benchmark)} 연환산%": ann(bh)})
    return pd.DataFrame(rows)


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
    walk_forward: WalkForwardResult | None
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


def backtest_rules(cfg: AppConfig, market_key: str, bars: dict[str, pd.DataFrame]) -> MarketBacktest:
    """규칙 전략: 학습이 없으므로 워크포워드 없이, 파라미터는 사전에 고정(결과를 보고 고치지 않음)."""
    from .rules import first_valid_time, trend_breakout

    market = cfg.market(market_key)
    signals = {s: trend_breakout(df, cfg.strategy) for s, df in bars.items()}
    starts = [t for t in (first_valid_time(sig) for sig in signals.values()) if t is not None]
    if not starts:
        raise ValueError("규칙 신호를 계산할 만큼 데이터가 길지 않습니다 (장기 이동평균 기간 부족)")
    result = run_backtest(bars, signals, market, cfg.strategy, cfg.risk, start=min(starts))
    table, info = summarize(result)
    return MarketBacktest(result=result, walk_forward=None, model_eval={}, table=table, info=info)
