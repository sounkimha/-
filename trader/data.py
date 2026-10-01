"""시세 수집(yfinance 1시간봉) + CSV 캐시 + 장 시간 헬퍼."""
from __future__ import annotations

import logging
import time as _time
from datetime import date, datetime, time, timedelta
from pathlib import Path

import pandas as pd

from .config import AppConfig, DataConfig, MarketConfig

log = logging.getLogger(__name__)

OHLCV = ["open", "high", "low", "close", "volume"]
PRICE_COLS = ["open", "high", "low", "close"]


class DataError(RuntimeError):
    """데이터 수집/로딩 실패. 다른 소스로 우회하지 않고 그대로 올려보낸다."""


def _utcnow() -> pd.Timestamp:  # 테스트에서 시각을 바꿔 끼우기 위한 한 곳
    return pd.Timestamp.now(tz="UTC")


# --------------------------------------------------------------------------
# 장 시간 (공휴일·조기폐장은 반영하지 않음: 주말만 건너뜀)
# 날짜+현지 시각으로 직접 만들어 서머타임 전환일에도 어긋나지 않게 한다.
# --------------------------------------------------------------------------
def _local_date(ts: pd.Timestamp | date, market: MarketConfig) -> date:
    if isinstance(ts, pd.Timestamp):
        ts = ts.tz_localize(market.timezone) if ts.tzinfo is None else ts.tz_convert(market.timezone)
        return ts.date()
    return ts


def _at(day: pd.Timestamp | date, hhmm: time, market: MarketConfig) -> pd.Timestamp:
    return pd.Timestamp(datetime.combine(_local_date(day, market), hhmm)).tz_localize(market.timezone)


def session_close(ts: pd.Timestamp, market: MarketConfig) -> pd.Timestamp:
    return _at(ts, market.session.close, market)


def session_bar_starts(day: pd.Timestamp | date, market: MarketConfig, interval: str = "1h") -> list[pd.Timestamp]:
    """그날 정규장 봉들의 시작 시각 (시가 시각부터 interval 간격)."""
    t, close, step = _at(day, market.session.open, market), _at(day, market.session.close, market), pd.Timedelta(interval)
    starts = []
    while t < close:
        starts.append(t)
        t += step
    return starts


def is_daily(interval: str) -> bool:
    return interval.endswith("d")


def bar_end(ts: pd.Timestamp, market: MarketConfig, interval: str = "1h") -> pd.Timestamp:
    """봉이 끝나는 시각. 장 마감에 걸친 봉(예: 미장 15:30 봉)은 장 마감 시각에 끝난다. 일봉은 그날 장 마감."""
    if is_daily(interval):
        return session_close(ts, market)
    end = ts + pd.Timedelta(interval)
    close = session_close(ts, market)
    return min(end, close) if ts < close else end


def is_last_bar_start(ts: pd.Timestamp, market: MarketConfig, interval: str = "1h") -> bool:
    """ts 에 시작하는 봉이 그날 마지막 봉인지 (장 시간표 기준, 가격 정보는 쓰지 않음)."""
    return ts + pd.Timedelta(interval) >= session_close(ts, market)


def next_bar_start(ts: pd.Timestamp, market: MarketConfig, interval: str = "1h") -> pd.Timestamp:
    nxt = ts + pd.Timedelta(interval)
    if nxt < session_close(ts, market):
        return nxt
    day = _local_date(ts, market) + timedelta(days=1)
    while day.weekday() >= 5:
        day += timedelta(days=1)
    return _at(day, market.session.open, market)


def is_market_open(now: pd.Timestamp, market: MarketConfig) -> bool:
    now = now.tz_convert(market.timezone)
    if now.weekday() >= 5:
        return False
    return _at(now, market.session.open, market) <= now < _at(now, market.session.close, market)


def in_last_bar(now: pd.Timestamp, market: MarketConfig, interval: str = "1h") -> bool:
    """시계 기준: 지금이 그날 마지막 봉 구간(마지막 봉 시작 ~ 장 마감)인지. 데이터가 늦어도 판단 가능."""
    now = now.tz_convert(market.timezone)
    starts = session_bar_starts(now, market, interval) if now.weekday() < 5 else []
    return bool(starts) and starts[-1] <= now < session_close(now, market)


def expected_latest_bar(now: pd.Timestamp, market: MarketConfig, interval: str = "1h") -> pd.Timestamp | None:
    """지금 시각(데이터 지연 포함)이면 이미 완성돼 있어야 할 가장 최근 봉의 시작 시각."""
    now = now.tz_convert(market.timezone)
    delay = pd.Timedelta(minutes=market.data_delay_minutes)
    day = now.date()
    for _ in range(14):
        if day.weekday() < 5:
            for start in reversed(session_bar_starts(day, market, interval)):
                if bar_end(start, market, interval) + delay <= now:
                    return start
        day -= timedelta(days=1)
    return None


# --------------------------------------------------------------------------
# 수집 / 정리 / 캐시
# --------------------------------------------------------------------------
def download_ohlcv(ticker: str, interval: str, period_days: int) -> pd.DataFrame:
    """yfinance 에서 OHLCV 를 받는다. 실패하면 DataError (대체 데이터로 바꾸지 않음)."""
    import yfinance as yf  # 실제로 받을 때만 import

    end = _utcnow()
    start = end - pd.Timedelta(days=period_days)
    try:
        raw = yf.Ticker(ticker).history(
            start=start.to_pydatetime(),
            end=end.to_pydatetime(),
            interval=interval,
            prepost=False,
            actions=False,
            auto_adjust=True,
            raise_errors=True,
        )
    except Exception as e:  # yfinance 는 네트워크 차단·심볼 오류·레이트리밋을 여러 예외로 던진다
        hint = ""
        if type(e).__name__ == "YFTzMissingError":
            hint = " — 네트워크가 막혀도 yfinance 가 이 메시지를 냅니다. 다른 종목의 연결 오류(403 등)도 함께 확인하세요"
        raise DataError(f"{ticker}: yfinance 다운로드 실패 ({type(e).__name__}: {e}){hint}") from e
    if raw is None or raw.empty:
        raise DataError(f"{ticker}: yfinance 가 빈 데이터를 반환했습니다")
    df = raw.rename(columns=str.lower)
    missing = [c for c in OHLCV if c not in df.columns]
    if missing:
        raise DataError(f"{ticker}: 컬럼 누락 {missing}")
    return df[OHLCV]


def clean_ohlcv(df: pd.DataFrame, tz: str) -> pd.DataFrame:
    """시간대 통일, 중복·결측·0 이하 가격 제거, 고가/저가가 시가/종가를 감싸도록 보정."""
    if df.empty:
        return df
    idx = pd.DatetimeIndex(df.index)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx
    out = df[OHLCV].astype(float).copy()
    out.index = idx.tz_convert(tz)
    out.index.name = "time"
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out = out.dropna(subset=PRICE_COLS)
    out = out[(out[PRICE_COLS] > 0).all(axis=1)]
    out["volume"] = out["volume"].fillna(0.0)
    out["high"] = out[PRICE_COLS].max(axis=1)
    out["low"] = out[PRICE_COLS].min(axis=1)
    return out


def filter_session(df: pd.DataFrame, market: MarketConfig, interval: str = "1h") -> pd.DataFrame:
    """정규장 안에서 시작한 봉만 남긴다 (시간외·장 마감 시각에 찍힌 봉 제거, 주말 제거). 일봉은 주말만 제거."""
    if df.empty:
        return df
    if is_daily(interval):
        return df[df.index.weekday < 5]
    minutes = df.index.hour * 60 + df.index.minute
    o, c = market.session.open, market.session.close
    keep = (minutes >= o.hour * 60 + o.minute) & (minutes < c.hour * 60 + c.minute) & (df.index.weekday < 5)
    return df[keep]


def drop_incomplete_last_bar(
    df: pd.DataFrame, market: MarketConfig, interval: str = "1h", now: pd.Timestamp | None = None
) -> pd.DataFrame:
    """아직 끝나지 않은(또는 지연 때문에 덜 들어온) 마지막 봉을 버린다."""
    if df.empty:
        return df
    now = _utcnow() if now is None else now
    complete_at = bar_end(df.index[-1], market, interval) + pd.Timedelta(minutes=market.data_delay_minutes)
    return df.iloc[:-1] if now < complete_at else df


def _cache_path(cache_dir: Path, ticker: str, interval: str) -> Path:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in ticker)
    return cache_dir / f"{safe}_{interval}.csv"


def _read_cache(path: Path, tz: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True)
    return clean_ohlcv(df, tz)


def _write_cache(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out.index = out.index.tz_convert("UTC")
    out.to_csv(path)


def load_symbol(
    ticker: str,
    market: MarketConfig,
    data_cfg: DataConfig,
    cache_dir: Path,
    *,
    offline: bool = False,
    refresh: bool = False,
    now: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """캐시가 신선하면 캐시, 아니면 다운로드. refresh=True 면 항상 다운로드 (실시간 신호용).

    캐시에는 '받은 시점에 이미 끝난 봉'만 저장한다. 덜 끝난 봉이 캐시에 남았다가
    나중에 완성 봉으로 취급되는 일을 막기 위해서다.
    반환 DataFrame 의 attrs["last_price"] 는 진행 중인 봉까지 포함한 가장 최근 가격(DRY_RUN 체결가용).
    """
    path = _cache_path(cache_dir, ticker, data_cfg.interval)
    fresh = path.exists() and (_time.time() - path.stat().st_mtime) < data_cfg.cache_max_age_minutes * 60
    latest_raw = None
    if offline or (fresh and not refresh):
        if not path.exists():
            raise DataError(f"{ticker}: 오프라인 모드인데 캐시가 없습니다 ({path}). 먼저 fetch 를 실행하세요")
        df = _read_cache(path, market.timezone)
    else:
        raw = filter_session(
            clean_ohlcv(download_ohlcv(ticker, data_cfg.interval, data_cfg.period_days), market.timezone),
            market,
            data_cfg.interval,
        )
        if not raw.empty:
            latest_raw = (float(raw["close"].iloc[-1]), raw.index[-1])
        df = drop_incomplete_last_bar(raw, market, data_cfg.interval)  # 받은 시점 기준
        _write_cache(df, path)
    df = drop_incomplete_last_bar(filter_session(df, market, data_cfg.interval), market, data_cfg.interval, now)
    if df.empty:
        raise DataError(f"{ticker}: 사용할 수 있는 봉이 없습니다")
    df.attrs["last_price"], df.attrs["last_price_time"] = latest_raw or (float(df["close"].iloc[-1]), df.index[-1])
    return df


def load_market(
    cfg: AppConfig,
    market_key: str,
    *,
    offline: bool = False,
    refresh: bool = False,
    now: pd.Timestamp | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    """시장의 모든 종목을 읽는다. (성공한 종목 데이터, 실패한 종목의 에러 메시지)."""
    market = cfg.market(market_key)
    cache_dir = cfg.path(cfg.data.cache_dir)
    bars: dict[str, pd.DataFrame] = {}
    errors: dict[str, str] = {}
    for ticker in market.symbols:
        try:
            bars[ticker] = load_symbol(
                ticker, market, cfg.data, cache_dir, offline=offline, refresh=refresh, now=now
            )
        except DataError as e:
            log.debug("%s", e)  # 호출한 쪽에서 모아서 출력
            errors[ticker] = str(e)
    return bars, errors
