import pandas as pd
import pytest

import trader.data as data
from conftest import make_bars
from trader.data import (
    DataError,
    bar_end,
    drop_incomplete_last_bar,
    expected_latest_bar,
    filter_session,
    in_last_bar,
    is_last_bar_start,
    is_market_open,
    load_symbol,
    next_bar_start,
)

KST, ET = "Asia/Seoul", "America/New_York"


def ts(s, tz):
    return pd.Timestamp(s, tz=tz)


def test_bar_timing_kr(cfg):
    kr = cfg.market("kr")
    # 야후 국장 1시간봉: 09:00~14:00 6봉 (15:00~15:30 없음) → 설정상 마감 15:00
    assert bar_end(ts("2025-03-04 13:00", KST), kr) == ts("2025-03-04 14:00", KST)
    assert bar_end(ts("2025-03-04 14:00", KST), kr) == ts("2025-03-04 15:00", KST)
    assert is_last_bar_start(ts("2025-03-04 14:00", KST), kr)
    assert not is_last_bar_start(ts("2025-03-04 13:00", KST), kr)
    assert next_bar_start(ts("2025-03-04 13:00", KST), kr) == ts("2025-03-04 14:00", KST)
    assert next_bar_start(ts("2025-03-07 14:00", KST), kr) == ts("2025-03-10 09:00", KST)  # 금 → 월
    assert in_last_bar(ts("2025-03-04 14:21", KST), kr) and not in_last_bar(ts("2025-03-04 13:59", KST), kr)
    # 데이터 지연 20분 가정: 09:21 에는 전날 14:00 봉까지가 와 있어야 한다
    assert expected_latest_bar(ts("2025-03-05 09:21", KST), kr) == ts("2025-03-04 14:00", KST)


def test_bar_timing_us_dst(cfg):
    us = cfg.market("us")
    assert is_last_bar_start(ts("2026-10-30 15:30", ET), us)
    # 서머타임 종료 주말(2026-11-01)을 넘어가도 월요일 09:30 현지시각
    assert next_bar_start(ts("2026-10-30 15:30", ET), us) == ts("2026-11-02 09:30", ET)
    assert is_market_open(ts("2026-10-01 10:00", ET), us)
    assert not is_market_open(ts("2026-10-01 16:00", ET), us)
    assert not is_market_open(ts("2026-10-03 11:00", ET), us)  # 토요일


def test_filter_session_drops_out_of_hours_bars(cfg):
    us = cfg.market("us")
    idx = pd.DatetimeIndex([ts(f"2025-03-04 {h}", ET) for h in ("08:30", "09:30", "15:30", "16:00")])
    df = pd.DataFrame({c: 1.0 for c in ("open", "high", "low", "close", "volume")}, index=idx)
    kept = filter_session(df, us)
    assert [t.strftime("%H:%M") for t in kept.index] == ["09:30", "15:30"]


def test_drop_incomplete_last_bar_respects_delay(cfg):
    kr = cfg.market("kr")  # 데이터 지연 20분 가정
    df = make_bars(2, start="2025-03-03")
    last = df.index[-1]  # 14:00 봉 → 15:00 종료 + 지연 20분 = 15:20 에 완성
    assert len(drop_incomplete_last_bar(df, kr, now=last + pd.Timedelta(minutes=79))) == len(df) - 1
    assert len(drop_incomplete_last_bar(df, kr, now=last + pd.Timedelta(minutes=81))) == len(df)


def test_download_failure_raises_without_fallback(cfg, tmp_path, monkeypatch):
    class Boom:
        def __init__(self, *a, **k):
            pass

        def history(self, **k):
            raise ConnectionError("CONNECT tunnel failed, response 403")

    import yfinance

    monkeypatch.setattr(yfinance, "Ticker", Boom)
    with pytest.raises(DataError, match="403"):
        load_symbol("AAPL", cfg.market("us"), cfg.data, tmp_path)
    assert not list(tmp_path.iterdir())  # 실패하면 캐시도 만들지 않음


def test_cache_roundtrip_and_offline(cfg, tmp_path, monkeypatch):
    df = make_bars(5, tz=ET, first_bar="09:30", start="2025-03-03")
    monkeypatch.setattr(data, "download_ohlcv", lambda t, i, p: df)
    a = load_symbol("AAPL", cfg.market("us"), cfg.data, tmp_path, now=ts("2025-04-01 12:00", ET))
    b = load_symbol("AAPL", cfg.market("us"), cfg.data, tmp_path, offline=True, now=ts("2025-04-01 12:00", ET))
    pd.testing.assert_frame_equal(a, b, check_freq=False)
    assert str(a.index.tz) == ET and len(a) == len(df)
    with pytest.raises(DataError, match="캐시가 없습니다"):
        load_symbol("MSFT", cfg.market("us"), cfg.data, tmp_path, offline=True)


def test_cache_never_stores_a_bar_that_was_unfinished_at_download(cfg, tmp_path, monkeypatch):
    us = cfg.market("us")
    df = make_bars(3, tz=ET, first_bar="09:30", start="2025-03-03")  # 마지막 봉: 03-05 15:30
    download_time = ts("2025-03-05 15:45", ET)  # 15:30 봉은 아직 진행 중(16:00 종료)
    monkeypatch.setattr(data, "_utcnow", lambda: download_time.tz_convert("UTC"))
    monkeypatch.setattr(data, "download_ohlcv", lambda t, i, p: df)
    first = load_symbol("AAPL", us, cfg.data, tmp_path)
    assert first.index[-1] == ts("2025-03-05 14:30", ET)
    # 10분 뒤(캐시는 아직 신선) 장이 끝났어도, 받을 때 미완성이던 15:30 봉이 캐시에서 나오면 안 된다
    later = ts("2025-03-05 16:20", ET)
    monkeypatch.setattr(data, "_utcnow", lambda: later.tz_convert("UTC"))
    cached = load_symbol("AAPL", us, cfg.data, tmp_path, now=later)
    assert cached.index[-1] == ts("2025-03-05 14:30", ET)
    fresh = load_symbol("AAPL", us, cfg.data, tmp_path, refresh=True, now=later)  # 실시간 신호는 새로 받음
    assert fresh.index[-1] == ts("2025-03-05 15:30", ET)
