"""한국투자증권 Open API 공통 클라이언트 — 모의투자 전용.

안전장치 (코드로 강제):
1. 접속 도메인은 모의투자 서버(openapivts.koreainvestment.com)만 허용. 다른 주소면 생성 단계에서 예외.
   리다이렉트는 따라가지 않는다 (주문 본문·앱키가 다른 호스트로 재전송되지 않게).
2. 주문 tr_id 는 모의투자용(V로 시작)만 허용.
3. dry_run=True 이면 주문 요청을 만들어 로그만 남기고 네트워크로 보내지 않는다 (토큰 발급도 하지 않음).
   dry_run 값은 .env 의 DRY_RUN 에서 오며, 명시적으로 false 를 넣기 전에는 항상 True.
4. 네트워크 오류 메시지에는 URL(쿼리스트링의 계좌번호)을 남기지 않는다.

엔드포인트·tr_id 출처: 한국투자증권 공식 GitHub koreainvestment/open-trading-api (examples_llm, 2026-09-28 커밋 기준).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

import requests

from ..config import KisCredentials

log = logging.getLogger(__name__)

VTS_BASE_URL = "https://openapivts.koreainvestment.com:29443"  # 모의투자 서버
_ALLOWED_HOST = "openapivts.koreainvestment.com"
_TOKEN_PATH = "/oauth2/tokenP"
_KST = ZoneInfo("Asia/Seoul")


class KisError(RuntimeError):
    pass


class KisApiError(KisError):
    """KIS 가 오류(rt_cd != 0, HTTP 오류, 리다이렉트)를 돌려준 경우. 주문은 접수되지 않은 것."""


class KisNetworkError(KisError):
    """타임아웃·연결 끊김 등. 주문이라면 서버가 접수했는지 알 수 없다."""


class OrderBlockedError(KisError):
    """안전장치에 의해 주문이 차단된 경우."""


def assert_paper_url(base_url: str) -> None:
    host = urlparse(base_url).hostname
    if host != _ALLOWED_HOST:
        raise OrderBlockedError(f"모의투자 서버({_ALLOWED_HOST}) 이외의 주소는 사용할 수 없습니다: {host}")


def mask(value: str, keep: int = 2) -> str:
    return value[:keep] + "*" * max(len(value) - keep, 0)


@dataclass(frozen=True)
class OrderRequest:
    market: str  # domestic | overseas
    side: str  # buy | sell
    symbol: str
    qty: int
    price: float | None  # 지정가(시장가면 None)
    path: str
    tr_id: str
    body: dict[str, str] = field(repr=False)

    def describe(self) -> str:
        body = dict(self.body)
        if "CANO" in body:
            body["CANO"] = mask(body["CANO"])
        price = "시장가" if self.price is None else f"{self.price:g}"
        return f"{self.market} {self.side} {self.symbol} x{self.qty} @ {price} tr_id={self.tr_id} body={body}"


@dataclass
class Position:
    symbol: str  # yfinance 티커 (매핑 안 되는 종목은 증권사 코드 그대로)
    qty: int
    avg_price: float
    last_price: float
    sellable_qty: int | None = None  # 주문가능수량 (매도 주문이 걸려 있으면 줄어듦)


@dataclass
class Balance:
    currency: str
    cash: float  # 주문에 쓸 수 있다고 보는 현금 (근사치, README 참고)
    total_equity: float
    positions: dict[str, Position]


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return default


@dataclass
class OrderResult:
    request: OrderRequest
    sent: bool  # 네트워크로 실제 전송했는지
    dry_run: bool
    ok: bool
    order_no: str | None = None
    message: str = ""
    uncertain: bool = False  # 전송했지만 응답을 못 받음 → 접수 여부 불명 (다음 사이클에 잔고로 확인)
    raw: dict[str, Any] | None = field(default=None, repr=False)


class KisClient:
    def __init__(
        self,
        credentials: KisCredentials | None,
        *,
        dry_run: bool = True,
        base_url: str = VTS_BASE_URL,
        token_cache: Path | None = None,
        min_interval_sec: float = 0.5,
        timeout_sec: float = 10.0,
        session: requests.Session | None = None,
    ):
        assert_paper_url(base_url)
        self.credentials = credentials
        self.dry_run = dry_run
        self.base_url = base_url.rstrip("/")
        self.token_cache = token_cache
        self.min_interval_sec = min_interval_sec
        self.timeout_sec = timeout_sec
        self._session = session or requests.Session()
        self._last_call = 0.0
        self._token: str | None = None
        self._token_expiry: datetime | None = None

    # --- 내부 ----------------------------------------------------------------
    def _creds(self) -> KisCredentials:
        if self.credentials is None:
            raise KisError(".env 에 KIS_APP_KEY / KIS_APP_SECRET / KIS_ACCOUNT_NO 가 없습니다 (.env.example 참고)")
        return self.credentials

    def _throttle(self) -> None:
        wait = self.min_interval_sec - (time.monotonic() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.monotonic()

    def _request(self, method: str, path: str, **kwargs) -> requests.Response:
        """모든 HTTP 호출의 단일 통로: 호출 간격, 리다이렉트 차단, 예외 메시지 정리."""
        self._throttle()
        try:
            resp = self._session.request(
                method, self.base_url + path, allow_redirects=False, timeout=self.timeout_sec, **kwargs
            )
        except requests.RequestException as e:
            # 원래 메시지에는 쿼리스트링(계좌번호)이 포함되므로 경로만 남기고 체인도 끊는다
            raise KisNetworkError(f"{type(e).__name__} ({method} {path})") from None
        if 300 <= resp.status_code < 400:
            raise KisApiError(f"HTTP {resp.status_code} 리다이렉트는 따라가지 않습니다 ({method} {path})")
        return resp

    def _key_fingerprint(self) -> str:
        return hashlib.sha256(self._creds().app_key.encode()).hexdigest()[:16]

    def _load_cached_token(self) -> bool:
        if not self.token_cache or not self.token_cache.exists():
            return False
        try:
            data = json.loads(self.token_cache.read_text(encoding="utf-8"))
            expiry = datetime.fromisoformat(data["expires_at"])
        except (ValueError, KeyError, OSError):
            return False
        if data.get("key") != self._key_fingerprint() or expiry - timedelta(minutes=10) <= datetime.now(_KST):
            return False
        self._token, self._token_expiry = data["token"], expiry
        return True

    def _save_token(self) -> None:
        if not self.token_cache:
            return
        self.token_cache.parent.mkdir(parents=True, exist_ok=True)
        payload = {"key": self._key_fingerprint(), "token": self._token, "expires_at": self._token_expiry.isoformat()}
        # 처음부터 소유자만 읽을 수 있게 만든 뒤 쓴다
        fd = os.open(self.token_cache, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(json.dumps(payload))
        os.chmod(self.token_cache, 0o600)

    def access_token(self) -> str:
        """접근토큰 (유효 1일). 메모리 → 캐시 파일 → 신규 발급 순. 발급 시 KIS 가 알림톡을 보낸다."""
        if self._token and self._token_expiry and self._token_expiry - timedelta(minutes=10) > datetime.now(_KST):
            return self._token
        if self._load_cached_token():
            return self._token  # type: ignore[return-value]
        creds = self._creds()
        resp = self._request(
            "POST",
            _TOKEN_PATH,
            data=json.dumps(
                {"grant_type": "client_credentials", "appkey": creds.app_key, "appsecret": creds.app_secret}
            ),
            headers={"content-type": "application/json; charset=utf-8"},
        )
        try:
            data = resp.json()
        except ValueError as e:
            raise KisApiError(f"토큰 발급 실패: HTTP {resp.status_code}") from e
        if resp.status_code != 200 or "access_token" not in data:
            raise KisApiError(
                f"토큰 발급 실패: HTTP {resp.status_code} {data.get('error_code', '')} {data.get('error_description', '')}"
            )
        self._token = data["access_token"]
        expired = data.get("access_token_token_expired")
        if expired:
            self._token_expiry = datetime.strptime(expired, "%Y-%m-%d %H:%M:%S").replace(tzinfo=_KST)
        else:
            self._token_expiry = datetime.now(_KST) + timedelta(seconds=int(data.get("expires_in", 86400)))
        self._save_token()
        return self._token

    def _headers(self, tr_id: str) -> dict[str, str]:
        creds = self._creds()
        return {
            "content-type": "application/json; charset=utf-8",
            "authorization": f"Bearer {self.access_token()}",
            "appkey": creds.app_key,
            "appsecret": creds.app_secret,
            "tr_id": tr_id,
            "custtype": "P",
            "tr_cont": "",
        }

    @staticmethod
    def _check(resp: requests.Response) -> dict[str, Any]:
        try:
            data = resp.json()
        except ValueError as e:
            raise KisApiError(f"HTTP {resp.status_code}: JSON 응답이 아닙니다") from e
        if resp.status_code != 200 or str(data.get("rt_cd")) != "0":
            raise KisApiError(
                f"HTTP {resp.status_code} rt_cd={data.get('rt_cd')} {data.get('msg_cd', '')}: {data.get('msg1', '')}".strip()
            )
        return data

    # --- 공개 API ------------------------------------------------------------
    @property
    def account(self) -> tuple[str, str]:
        creds = self._creds()
        return creds.account_no, creds.product_code

    def get(self, path: str, tr_id: str, params: dict[str, str]) -> dict[str, Any]:
        """조회(시세·잔고) 전용 GET. 주문이 아니므로 DRY_RUN 과 무관하지만, 자격정보가 있어야 한다."""
        return self._check(self._request("GET", path, headers=self._headers(tr_id), params=params))

    def submit_order(self, req: OrderRequest) -> OrderResult:
        """주문 전송의 유일한 통로. 여기서 DRY_RUN·tr_id 를 검사한다."""
        if not req.tr_id.startswith("V"):
            raise OrderBlockedError(f"모의투자 tr_id(V로 시작)만 허용합니다: {req.tr_id}")
        if self.dry_run:
            log.warning("[DRY_RUN] 주문 미전송: %s", req.describe())
            return OrderResult(req, sent=False, dry_run=True, ok=True, message="DRY_RUN: 전송하지 않음")

        # ---- 여기부터만 실제 네트워크 전송 (모의투자 서버) ----
        headers = self._headers(req.tr_id)  # 토큰 실패는 전송 전이므로 예외로 올려보낸다
        log.info("모의투자 주문 전송: %s", req.describe())
        try:
            resp = self._request("POST", req.path, headers=headers, data=json.dumps(req.body))
        except KisNetworkError as e:
            log.error("주문 응답을 받지 못함 — 접수 여부 불명, 다음 사이클에 잔고로 확인: %s", e)
            return OrderResult(req, sent=True, dry_run=False, ok=False, uncertain=True, message=str(e))
        except KisApiError as e:
            log.error("주문 거부: %s", e)
            return OrderResult(req, sent=True, dry_run=False, ok=False, message=str(e))
        try:
            data = self._check(resp)
        except KisApiError as e:
            log.error("주문 거부: %s", e)
            return OrderResult(req, sent=True, dry_run=False, ok=False, message=str(e))
        out = data.get("output") or {}
        order_no = out.get("ODNO") or out.get("odno")
        return OrderResult(req, sent=True, dry_run=False, ok=True, order_no=order_no, message=data.get("msg1", ""), raw=data)
