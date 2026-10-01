#!/usr/bin/env python3
"""국장·미장 1시간봉 단타 자동매매 — 실행 진입점.

  python main.py fetch                       데이터 수집(캐시 저장)
  python main.py backtest [--market kr|us]   워크포워드 학습 + 백테스트 결과 표
  python main.py signal                      최신 봉 기준 종목별 상승확률 / 매수·관망
  python main.py trade --market kr           1회 매매 사이클 (DRY_RUN 기본: 주문 전송 안 함)
  python main.py broker-check                모의투자 토큰·잔고·시세 조회 확인 (주문 없음)
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime

import pandas as pd
from tabulate import tabulate

from trader.backtest import backtest_market, trades_frame
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


def cmd_backtest(cfg, args) -> int:
    code = 0
    reports = cfg.path(cfg.paths.reports_dir)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
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
            bt = backtest_market(cfg, mk, bars)
        except (InsufficientDataError, ValueError) as e:
            print(f"  → 백테스트 불가: {e}")
            code = 2
            continue

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

        reports.mkdir(parents=True, exist_ok=True)
        bt.table.to_csv(reports / f"backtest_{mk}_{stamp}_summary.csv", index=False, encoding="utf-8-sig")
        trades_frame(bt.result).to_csv(reports / f"backtest_{mk}_{stamp}_trades.csv", index=False, encoding="utf-8-sig")
        bt.result.equity.to_csv(reports / f"backtest_{mk}_{stamp}_equity.csv", encoding="utf-8-sig")
        print(f"저장: {reports}/backtest_{mk}_{stamp}_*.csv")
    print("\n※ 과거 백테스트 결과는 미래 수익을 보장하지 않습니다. 같은 기간으로 설정값을 반복 조정하면 과최적화됩니다.")
    return code


def cmd_signal(cfg, args) -> int:
    from trader.live import compute_signals

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


def cmd_trade(cfg, args) -> int:
    from trader.live import run_trade_cycle

    rep = run_trade_cycle(cfg, args.market, offline=args.offline, ignore_hours=args.ignore_hours)
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
    p = argparse.ArgumentParser(description="국장·미장 1시간봉 단타 자동매매")
    p.add_argument("--config", default=None, help="설정 파일 경로 (기본: config.yaml)")
    p.add_argument("-v", "--verbose", action="store_true", help="디버그 로그")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("fetch", help="데이터 수집")
    s.add_argument("--market", default="all", help="kr | us | all")
    s.set_defaults(func=cmd_fetch)

    s = sub.add_parser("backtest", help="워크포워드 학습 + 백테스트")
    s.add_argument("--market", default="all", help="kr | us | all")
    s.add_argument("--offline", action="store_true", help="네트워크 없이 캐시만 사용")
    s.set_defaults(func=cmd_backtest)

    s = sub.add_parser("signal", help="최신 봉 기준 신호")
    s.add_argument("--market", default="all", help="kr | us | all")
    s.add_argument("--offline", action="store_true", help="네트워크 없이 캐시만 사용")
    s.set_defaults(func=cmd_signal)

    s = sub.add_parser("trade", help="1회 매매 사이클 (DRY_RUN 기본)")
    s.add_argument("--market", required=True, help="kr | us")
    s.add_argument("--offline", action="store_true", help="네트워크 없이 캐시만 사용")
    s.add_argument("--ignore-hours", action="store_true", help="DRY_RUN 일 때만: 장 시간이 아니어도 가상계좌로 진행")
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
