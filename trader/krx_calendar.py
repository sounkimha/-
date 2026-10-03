"""한국거래소(KRX) 휴장일 — 주말이 아닌데 증시가 쉬는 날.

출처·검증 (2026-10-03 기준)
- 2025: 야후 KODEX 200 일봉에서 거래가 없는 평일 19일과 대조 (임시공휴일 1/27·대통령선거 6/3 포함)
- 2026: 증권사·언론의 2026년 휴장일 공지(평일 17일, 제헌절 7/17 포함)와 대조. 10/2 까지는 야후 데이터와도 일치
- 2027: 잠정 — 공휴일 달력 기준. 노동절(5/1 토)·제헌절(7/17 토)의 대체휴일 여부는 출처마다 달라 넣지 않았다.
  거래소 공식 휴장일은 매년 12월에 발표되므로 그때 확인해 고친다.
이 목록 밖의 날짜(2024 이전, 2028 이후)는 주말만 쉬는 것으로 본다.
"""
from __future__ import annotations

import logging
from datetime import date, timedelta

log = logging.getLogger(__name__)

_DATES = {
    2025: "01-01 01-27 01-28 01-29 01-30 03-03 05-01 05-05 05-06 06-03 06-06 08-15 10-03 10-06 10-07 10-08 10-09 12-25 12-31",
    2026: "01-01 02-16 02-17 02-18 03-02 05-01 05-05 05-25 06-03 07-17 08-17 09-24 09-25 10-05 10-09 12-25 12-31",
    2027: "01-01 02-08 02-09 03-01 05-05 05-13 08-16 09-14 09-15 09-16 10-04 10-11 12-27 12-31",  # 잠정
}
KRX_HOLIDAYS = frozenset(date.fromisoformat(f"{y}-{md}") for y, mds in _DATES.items() for md in mds.split())
COVERED_YEARS = frozenset(_DATES)
_warned: set[int] = set()


def is_krx_trading_day(d: date) -> bool:
    if d.weekday() >= 5:
        return False
    if d.year not in COVERED_YEARS and d.year not in _warned:
        _warned.add(d.year)
        log.warning("%d년 KRX 휴장일 목록이 없습니다 → 주말만 쉬는 것으로 봅니다 (trader/krx_calendar.py 갱신 필요)", d.year)
    return d not in KRX_HOLIDAYS


def next_krx_trading_day(d: date) -> date:
    d += timedelta(days=1)
    while not is_krx_trading_day(d):
        d += timedelta(days=1)
    return d
