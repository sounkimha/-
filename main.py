#!/usr/bin/env python3
"""국장·미장 자동매매 학습 시스템 — 실행 진입점.

  python main.py fetch                       데이터 수집(캐시 저장)
  python main.py backtest [--market kr|us]   워크포워드 학습 + 백테스트 결과 표
  python main.py signal                      최신 봉 기준 종목별 상승확률 / 매수·관망
  python main.py trade --market kr           1회 매매 사이클 (DRY_RUN 기본: 주문 전송 안 함)
  python main.py broker-check                모의투자 토큰·잔고·시세 조회 확인 (주문 없음)

  일봉 ETF 프로필 (config.daily.yaml, strategy.type=rule_breakout):
  python main.py --config config.daily.yaml backtest
  python main.py --config config.daily.yaml signal --held 069500@105000 --equity 1000000
  python main.py --config config.daily.yaml trade --market kr   가상계좌 정산·계획 (DRY_RUN) / 모의투자 주문·확인
"""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace
from datetime import datetime

import pandas as pd
from tabulate import tabulate

from trader.backtest import (
    backtest_market,
    backtest_rules,
    comparison_table,
    regime_table,
    trades_frame,
    with_cost_multiplier,
    yearly_table,
)
from trader.broker.kis_client import KisError
from trader.config import ConfigError, is_dry_run, load_config, load_credentials, load_env
from trader.data import DataError, load_market
from trader.model import InsufficientDataError

log = logging.getLogger("main")


def table(df: pd.DataFrame, floatfmt: str = ".2f") -> str:
    shown = df.astype(object).where(df.notna(), None)  # NaN → '-'
    return tabulate(shown, headers="keys", tablefmt="github", showindex=False, floatfmt=floatfmt, missingval="-")


def market_keys(cfg, arg: str) -> list[str]:
    return list(cfg.markets) if arg == "all" else [cfg.market(arg).key]


def print_data_errors(errors: dict[str, str]) -> None:
    if errors:
        print("\n[데이터 수집 실패 — 다른 소스로 우회하지 않음]")
        for msg in errors.values():
            print(f"  - {msg}")


# --------------------------------------------------------------------------
def cmd_fetch(cfg, args) -> int:
    code = 0
    for mk in market_keys(cfg, args.market):
        market = cfg.market(mk)
        bars, errors = load_market(cfg, mk, refresh=True)
        rows = []
        for t, df in bars.items():
            days = df.index.normalize().nunique()
            rows.append(
                {
                    "종목": market.label(t),
                    "봉 수": len(df),
                    "거래일": days,
                    "하루 평균 봉": len(df) / max(days, 1),
                    "시작": f"{df.index[0]:%Y-%m-%d %H:%M}",
                    "끝": f"{df.index[-1]:%Y-%m-%d %H:%M}",
                }
            )
        print(f"\n## {market.name} ({market.timezone})")
        if rows:
            print(table(pd.DataFrame(rows), ".1f"))
        print_data_errors(errors)
        code = code or (2 if errors else 0)
    return code


def report_ml(cfg, mk, bars, bt) -> None:
    info = bt.info
    st, rk = cfg.strategy, cfg.risk
    print(f"기간: {info['기간']} · 거래일 {info['거래일']}일 (워크포워드 표본 외 구간만)")
    print(
        f"비용: 왕복 {info['왕복비용%']:.2f}% · 진입: 상승확률 ≥ {st.entry_threshold:.2f} & 기대수익 ≥ {st.min_expected_return_pct:.2f}%"
        f" · 청산: 상승확률 < {st.exit_threshold:.2f}"
    )
    print(
        f"손실 제한: 손절 -{rk.stop_loss_pct:g}% · 1회 투입 {rk.position_size_pct:g}% · 계좌 MDD -{rk.max_drawdown_pct:g}% 중단"
        f" · 일일 진입 {rk.max_trades_per_day}회 · 장마감 청산 {'ON' if st.flatten_at_session_end else 'OFF'}"
    )
    print()
    print(table(bt.table))
    print(f"\n평균 투입비중: {info['평균투입비중%']:.1f}% · 청산 사유: {info['청산사유'] or '-'} · 매매중단: {info['매매중단']}")
    ev = bt.model_eval
    print(
        f"모델(표본 외): AUC {ev['auc']:.3f} · 기저율 {ev['base_rate']:.1%} · "
        f"임계값 이상 신호 {ev['signals']}건, 그중 실제 상승 {ev['precision']:.1%} (n={ev['n']})"
    )
    total = bt.table.iloc[-1]
    verdict = "높습니다" if total["전략수익%(계좌)"] > total["단순보유%"] else "낮습니다"
    per = bt.table.iloc[:-1]
    worse = int((per["전략수익%(투입금)"] < per["단순보유%"]).sum())
    print(
        f"판정: 계좌 전략수익 {total['전략수익%(계좌)']:+.2f}% vs 동일가중 단순보유 {total['단순보유%']:+.2f}% → 전략이 단순보유보다 {verdict}"
        f" (평균 투입비중 {info['평균투입비중%']:.1f}%)."
        f" 종목별 투입금 기준으로는 {len(per)}개 중 {worse}개에서 전략 < 단순보유."
    )


def report_rules(cfg, mk, bars, bt) -> None:
    """일봉 규칙: 설정대로 결과 + (중단됐다면) 중단 없이 계속한 결과 + 비용 2배 + 연도·국면별 비교."""
    info, st, rk, market = bt.info, cfg.strategy, cfg.risk, cfg.market(mk)
    bench = next(s for s in market.symbols if s in bars)  # 설정 파일의 첫 종목 = 기준(시장 대표). 수집 실패 시 다음 종목
    print(f"기간: {info['기간']} · 거래일 {info['거래일']}일 ({st.long_ma}일선 준비 기간 이후)")
    print(
        f"규칙: 종가 > {st.long_ma}일선 & 종가 > 직전 {st.breakout_lookback}일 최고 종가 → 다음 날 시가 매수"
        f"(지정가 = 종가 +{st.entry_limit_pct:g}%, 넘게 시작하면 미체결) · 종가 < {st.exit_ma}일선 → 다음 날 시가 매도"
    )
    print(
        f"      최대 {st.max_positions or '무제한'}종목 (넘치면 {st.score_lookback}일 수익률 순) · 매도 후 {st.reentry_cooldown_bars}거래일 재진입 금지"
    )
    print(
        f"손실 제한: 종가 손절 -{rk.stop_loss_pct:g}% (다음 날 시가 청산) · 1종목 {rk.position_size_pct:g}% · "
        f"계좌 MDD -{rk.max_drawdown_pct:g}% 중단 · 비용 왕복 {info['왕복비용%']:.2f}%"
    )
    print()
    print(table(bt.table))
    print(
        f"\n평균 투입비중 {info['평균투입비중%']:.1f}% · 청산 사유: {info['청산사유'] or '-'}"
        f" · 지정가 미체결 {info['지정가미체결']}건 · 매매중단: {info['매매중단']}"
    )

    full, extras = bt.result, {}
    if bt.result.halted_at is not None:  # 중단 뒤로는 현금뿐이라 규칙의 성격을 볼 수 없다 → 중단 없이 계속한 결과를 함께
        no_halt = replace(cfg, risk=replace(cfg.risk, max_drawdown_pct=99.0))
        full = backtest_rules(no_halt, mk, bars).result
        extras["전략(MDD 중단 없이 계속)"] = full
        extras["전략(중단 없이·수수료·슬리피지 2배)"] = backtest_rules(with_cost_multiplier(no_halt, mk, 2.0), mk, bars).result
    else:
        extras["전략(수수료·슬리피지 2배)"] = backtest_rules(with_cost_multiplier(cfg, mk, 2.0), mk, bars).result
    comp = comparison_table(bt.result, bench, extras)
    print("\n### 같은 기간 비교 (일봉 기준 · 샤프는 무위험수익률 0 가정)")
    print(table(comp))
    label = " — MDD 중단 없이 계속했을 때" if full is not bt.result else ""
    print(f"\n### 연도별{label}")
    print(table(yearly_table(full, bench), ".1f"))
    print(f"\n### 시장 국면별{label} ({market.label(bench)} 종가와 200일선·20일 기울기로 구분, 연환산 · 보고용)")
    print(table(regime_table(full, bars[bench], bench), ".1f"))

    s = comp.iloc[0] if full is bt.result else comp.iloc[1]
    b = comp.iloc[-2]
    higher = "높고" if s["연환산%"] > b["연환산%"] else "낮고"
    smaller = "작습니다" if s["최대낙폭%"] > b["최대낙폭%"] else "큽니다"
    if bt.result.halted_at is not None:
        print(
            f"\n판정: 설정한 계좌 MDD -{rk.max_drawdown_pct:g}% 중단이 {bt.result.halted_at:%Y-%m-%d} 에 걸려 이후는 현금 "
            f"(설정대로 총 {comp.iloc[0]['총수익%']:+.1f}%). 중단 없이 계속했다면 최대낙폭 {s['최대낙폭%']:.1f}% → 이 규칙의 정상 낙폭이 한도보다 큽니다."
        )
    print(
        f"판정: {s['구분']} 연환산 {s['연환산%']:+.1f}% / 최대낙폭 {s['최대낙폭%']:.1f}% vs {b['구분']} {b['연환산%']:+.1f}% / {b['최대낙폭%']:.1f}%"
        f" → 수익은 보유보다 {higher}, 낙폭은 {smaller}."
    )


def cmd_backtest(cfg, args) -> int:
    code = 0
    reports = cfg.path(cfg.paths.reports_dir)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rules = cfg.strategy.type == "rule_breakout"
    for mk in market_keys(cfg, args.market):
        market = cfg.market(mk)
        print(f"\n## {market.name} 백테스트")
        bars, errors = load_market(cfg, mk, offline=args.offline)
        print_data_errors(errors)
        if errors:
            code = 2
        if not bars:
            print("  → 사용할 데이터가 없어 이 시장은 건너뜁니다.")
            continue
        try:
            bt = backtest_rules(cfg, mk, bars) if rules else backtest_market(cfg, mk, bars)
        except (InsufficientDataError, ValueError) as e:
            print(f"  → 백테스트 불가: {e}")
            code = 2
            continue
        (report_rules if rules else report_ml)(cfg, mk, bars, bt)

        reports.mkdir(parents=True, exist_ok=True)
        tag = f"{mk}_{cfg.strategy.type}_{stamp}"
        bt.table.to_csv(reports / f"backtest_{tag}_summary.csv", index=False, encoding="utf-8-sig")
        trades_frame(bt.result).to_csv(reports / f"backtest_{tag}_trades.csv", index=False, encoding="utf-8-sig")
        bt.result.equity.to_csv(reports / f"backtest_{tag}_equity.csv", encoding="utf-8-sig")
        print(f"저장: {reports}/backtest_{tag}_*.csv")
    print("\n※ 과거 백테스트 결과는 미래 수익을 보장하지 않습니다. 같은 기간으로 설정값을 반복 조정하면 과최적화됩니다.")
    return code


def cmd_signal(cfg, args) -> int:
    if cfg.strategy.type == "rule_breakout":
        return signal_rules(cfg, args)
    from trader.live import compute_signals

    if args.held or args.equity is not None:
        print("참고: --held / --equity 는 일봉 규칙 전략(rule_breakout)에서만 씁니다. 무시합니다.")
    code = 0
    for mk in market_keys(cfg, args.market):
        market = cfg.market(mk)
        print(f"\n## {market.name} 실시간 신호 (최신 완성 봉 기준)")
        try:
            sig, errors = compute_signals(cfg, mk, offline=args.offline)
        except (DataError, InsufficientDataError, ValueError) as e:
            print(f"  → 신호 계산 불가: {e}")
            code = 2
            continue
        out = sig[["종목", "최신봉", "종가", "상승확률", "기대수익%", "신호", "비고"]].copy()
        out["최신봉"] = out["최신봉"].map(lambda t: f"{t:%m-%d %H:%M}")
        print(table(out, ".3f"))
        st = cfg.strategy
        print(f"기준: 상승확률 ≥ {st.entry_threshold:.2f} 그리고 비용 차감 기대수익 ≥ {st.min_expected_return_pct:.2f}% 이면 매수")
        print_data_errors(errors)
        code = code or (2 if errors else 0)
    return code


def _won(x) -> str:
    return "-" if x is None or pd.isna(x) else f"{x:,.0f}"


def _ox(flag: bool) -> str:
    return "O" if flag else "-"


def signal_rules(cfg, args) -> int:
    """일봉 규칙: 종목별 상태 + 다음 거래일 시가 주문표 (주문은 보내지 않음)."""
    from trader.daily import next_weekday, parse_held, plan_orders, rule_signals

    st, rk = cfg.strategy, cfg.risk
    code = 0
    for mk in market_keys(cfg, args.market):
        market = cfg.market(mk)
        held = parse_held(args.held, market)
        equity = args.equity if args.equity is not None else market.initial_capital
        try:
            sig, errors = rule_signals(cfg, mk, offline=args.offline)
        except (DataError, ValueError) as e:
            print(f"  → 신호 계산 불가: {e}")
            code = 2
            continue
        base = max(sig["기준일"])
        order_day = next_weekday(base)
        print(f"\n## {market.name} — {base} 종가 기준 → {order_day} 시가 주문 (주말만 건너뜀 · 공휴일 미반영)")
        now = pd.Timestamp.now(tz=market.timezone)
        if (now.date(), now.time()) >= (order_day, market.session.open):
            print(f"  ! 지금({now:%m-%d %H:%M})은 이 주문표의 시가({order_day} 09:00)가 이미 지났습니다. 장 마감 후 다시 실행하세요.")
        out = pd.DataFrame(
            {
                "종목": sig["종목"],
                "기준일": sig["기준일"].map(lambda d: f"{d:%m-%d}"),
                "종가": sig["종가"].map(_won),
                f"{st.long_ma}일선": sig["장기선"].map(_won),
                f"직전{st.breakout_lookback}일고점": sig["직전고점"].map(_won),
                f"{st.exit_ma}일선": sig["청산선"].map(_won),
                f"{st.score_lookback}일수익%": sig["점수%"],
                "추세": sig["추세"].map(_ox),
                "진입": sig["진입"].map(_ox),
                "유지": sig["유지"].map(_ox),
                "비고": sig["비고"],
            }
        )
        print(table(out, ".1f"))
        print(
            f"추세 = 종가 > {st.long_ma}일선 · 진입 = 추세 & 종가 > 직전 {st.breakout_lookback}일 최고 종가"
            f" · 유지 = 종가 ≥ {st.exit_ma}일선 (보유 중인데 '-' 면 매도)"
        )

        plan = plan_orders(sig, held, equity, st, rk, market.costs)
        print(
            f"\n[주문표] 평가금액 {equity:,.0f}원 기준 · 1종목 예산 {plan.budget:,.0f}원 ({rk.position_size_pct:g}%)"
            f" · 보유 {len(held)}종목 → 매도 후 빈자리 {plan.slots}"
        )
        if plan.sells:
            print("매도 — 시가 동시호가, 시장가, 보유 전량")
            for o in plan.sells:
                print(f"  - {o.label}: {o.reason}")
        else:
            print("매도: 없음")
        if plan.buys:
            print("매수 — 지정가 (시가가 지정가보다 높으면 미체결 → 09:00 이후 취소, 따라 사지 않음)")
            rows = [
                {
                    "종목": o.label,
                    "지정가": _won(o.limit) if o.limit is not None else "시장가",
                    "수량": o.qty,
                    "금액": _won(o.amount),
                    "예산 사용%": f"{o.amount / plan.budget * 100:.0f}",
                    "근거": o.reason,
                }
                for o in plan.buys
            ]
            print(table(pd.DataFrame(rows)))
        else:
            print("매수: 없음")
        if plan.waiting:
            print(f"대기(빈자리 없음): {', '.join(plan.waiting)}")
        for note in plan.notes:
            print(f"  ! {note}")
        print(
            f"\n주문 방법 (MTS/HTS 에 직접 입력 — 이 프로그램은 주문을 보내지 않음)\n"
            f"  1. {order_day} 08:30~08:59 시가 동시호가에 주문. 거래소는 KRX 로 지정 (통합·SOR 이면 넥스트레이드로 갈 수 있음)\n"
            f"  2. 매도는 09:00 시가에 체결된다. 매수 자금이 그 매도 대금이면 매도 체결 직후 같은 지정가로 매수\n"
            f"     (동시호가 중에는 아직 안 팔린 대금이 주문가능금액에 잡히지 않음)\n"
            f"  3. 사람이 챙길 규칙: 판 종목은 {st.reentry_cooldown_bars}거래일 재매수 금지 · 계좌 고점 대비 -{rk.max_drawdown_pct:g}% 면 전부 팔고 중단\n"
            f"     · 보유 종목을 --held 코드@매수가 로 넣으면 종가 손절(-{rk.stop_loss_pct:g}%)도 판단"
        )
        print_data_errors(errors)
        code = code or (2 if errors else 0)
    return code


def trade_rules(cfg, args) -> int:
    """일봉 규칙: DRY_RUN 이면 가상계좌 정산·계획, 아니면 모의투자 주문/확인 단계."""
    from trader.daily_trade import run_daily_cycle

    rep = run_daily_cycle(cfg, args.market, offline=args.offline, resume=args.resume)
    m = rep.market
    mode = "DRY_RUN — 가상계좌 (주문 전송 안 함)" if rep.dry_run else "모의투자 서버로 주문 전송"
    print(f"\n## {m.name} 매매 사이클 — {rep.now:%Y-%m-%d %H:%M %Z} · 모드: {mode} · 단계: {rep.phase}")
    print(
        f"기준 일봉 {rep.base_date or '-'} · 평가금액 {_won(rep.equity)}{m.currency} · 현금 {_won(rep.cash)}"
        f" · 고점 대비 {rep.drawdown:+.2%} · 매매중단 {'예 — ' + str(rep.halt_reason) if rep.halted else '아니오'}"
    )
    for note in rep.notes:
        print(f"  - {note}")
    print("\n[이번 실행에서 처리한 것]")
    print(table(pd.DataFrame(rep.events), ",.0f") if rep.events else "없음")
    print("\n[보유]")
    if rep.holdings:
        h = pd.DataFrame(rep.holdings)
        for col in ("매수가", "최근가", "손절가"):
            h[col] = h[col].map(_won)
        h["평가손익%"] = h["평가손익%"].map(lambda v: "-" if v is None or pd.isna(v) else f"{v:+.1f}")
        print(table(h))
    else:
        print("없음")
    if rep.dry_run:
        print("\n[다음 거래일 시가 주문] (실계좌를 가상계좌와 똑같이 운용하려면 MTS 에 같은 주문을 넣으세요)")
    else:
        print("\n[오늘 주문]")
    print(table(pd.DataFrame(rep.planned), ",.0f") if rep.planned else "없음")
    extra = f" · 일별 평가: {rep.files['equity']}" if rep.dry_run else ""
    print(f"\n상태: {rep.files['state']} · 체결 기록: {rep.files['journal']}{extra}")
    if args.ignore_hours:
        print("참고: --ignore-hours 는 일봉 모드에서 쓰지 않습니다 (가상계좌는 완성된 일봉 기준, 모의투자는 시계 기준)")
    print_data_errors(rep.data_errors)
    return 0


def cmd_trade(cfg, args) -> int:
    if cfg.strategy.type == "rule_breakout":
        return trade_rules(cfg, args)
    from trader.live import run_trade_cycle

    rep = run_trade_cycle(cfg, args.market, offline=args.offline, ignore_hours=args.ignore_hours, resume=args.resume)
    m = rep.market
    mode = "DRY_RUN (주문 전송 안 함 · 가상계좌)" if rep.dry_run else "모의투자 서버로 주문 전송"
    print(f"\n## {m.name} 매매 사이클 — {rep.now:%Y-%m-%d %H:%M %Z} · 모드: {mode}")
    sig = rep.signals[["종목", "종가", "상승확률", "기대수익%", "신호", "비고"]]
    print(table(sig, ".3f"))
    print(
        f"\n평가금액 {rep.equity:,.2f} {m.currency} · 고점 대비 {rep.drawdown:+.2%}"
        f" · 매매중단 {'예 — ' + str(rep.halt_reason) if rep.halted else '아니오'}"
    )
    for note in rep.notes:
        print(f"  - {note}")
    if rep.executed:
        print("\n[주문]")
        print(table(pd.DataFrame(rep.executed), ".4f"))
    else:
        print("\n[주문] 없음")
    print_data_errors(rep.data_errors)
    return 0


def cmd_broker_check(cfg, args) -> int:
    """자격정보로 토큰 발급 + 잔고/시세 조회만 한다. 주문은 하지 않는다."""
    from trader.broker import make_broker

    load_env(cfg.root)
    creds = load_credentials()
    print(f"DRY_RUN = {is_dry_run()} (주문 전송 {'안 함' if is_dry_run() else '함 — 모의투자 서버'})")
    if creds is None:
        print(".env 에 KIS_APP_KEY / KIS_APP_SECRET / KIS_ACCOUNT_NO 가 없습니다. .env.example 을 복사해 채우세요.")
        return 1
    print(f"계좌: {creds.masked_account} · 서버: 모의투자")
    for mk in market_keys(cfg, args.market):
        broker = make_broker(cfg, mk, creds, dry_run=True)  # 조회만 하므로 주문 경로는 잠가 둔다
        first = next(iter(cfg.market(mk).symbols))
        print(f"\n## {cfg.market(mk).name}")
        print(f"현재가 {first}: {broker.price(first):,.2f}")
        bal = broker.balance()
        print(f"현금(근사) {bal.cash:,.2f} {bal.currency} · 평가금액 {bal.total_equity:,.2f} {bal.currency}")
        for p in bal.positions.values():
            print(f"  - {p.symbol}: {p.qty}주 @ {p.avg_price:,.2f} (현재 {p.last_price:,.2f})")
    return 0


# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="국장·미장 자동매매 학습 시스템 (1시간봉 ML · 일봉 ETF 규칙)")
    p.add_argument("--config", default=None, help="설정 파일 경로 (기본: config.yaml)")
    p.add_argument("-v", "--verbose", action="store_true", help="디버그 로그")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("fetch", help="데이터 수집")
    s.add_argument("--market", default="all", help="kr | us | all")
    s.set_defaults(func=cmd_fetch)

    s = sub.add_parser("backtest", help="백테스트 (ml: 워크포워드 학습 / rule_breakout: 일봉 규칙)")
    s.add_argument("--market", default="all", help="kr | us | all")
    s.add_argument("--offline", action="store_true", help="네트워크 없이 캐시만 사용")
    s.set_defaults(func=cmd_backtest)

    s = sub.add_parser("signal", help="최신 봉 기준 신호 (rule_breakout: 다음 거래일 시가 주문표)")
    s.add_argument("--market", default="all", help="kr | us | all")
    s.add_argument("--offline", action="store_true", help="네트워크 없이 캐시만 사용")
    s.add_argument(
        "--held", action="append", default=[], help="rule_breakout: 보유 종목. 069500 또는 069500@105000(매수가), 쉼표로 여러 개"
    )
    s.add_argument("--equity", type=float, default=None, help="rule_breakout: 계좌 평가금액(원). 기본 = initial_capital")
    s.set_defaults(func=cmd_signal)

    s = sub.add_parser("trade", help="1회 매매 사이클 (DRY_RUN 기본)")
    s.add_argument("--market", required=True, help="kr | us")
    s.add_argument("--offline", action="store_true", help="네트워크 없이 캐시만 사용")
    s.add_argument("--ignore-hours", action="store_true", help="DRY_RUN 일 때만: 장 시간이 아니어도 가상계좌로 진행")
    s.add_argument("--resume", action="store_true", help="계좌 낙폭 한도로 멈춘 매매를 재개 (고점 기준을 지금 평가금액으로 다시 잡음)")
    s.set_defaults(func=cmd_trade)

    s = sub.add_parser("broker-check", help="모의투자 토큰·잔고·시세 조회 (주문 없음)")
    s.add_argument("--market", default="all", help="kr | us | all")
    s.set_defaults(func=cmd_broker_check)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    for noisy in ("yfinance", "urllib3", "peewee"):
        logging.getLogger(noisy).setLevel(logging.WARNING if args.verbose else logging.CRITICAL)
    try:
        cfg = load_config(args.config)
        return args.func(cfg, args)
    except (ConfigError, DataError, InsufficientDataError, KisError) as e:
        print(f"오류: {type(e).__name__}: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
