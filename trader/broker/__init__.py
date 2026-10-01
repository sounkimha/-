"""한국투자증권 Open API (모의투자 전용) 주문 모듈."""
from __future__ import annotations

from ..config import AppConfig, KisCredentials
from .domestic import DomesticBroker
from .kis_client import (
    VTS_BASE_URL,
    Balance,
    KisApiError,
    KisClient,
    KisError,
    KisNetworkError,
    OrderBlockedError,
    OrderRequest,
    OrderResult,
    Position,
)
from .overseas import OverseasBroker

# config.yaml 의 시장 키 → 브로커 종류
MARKET_BROKERS = {"kr": DomesticBroker, "us": OverseasBroker}


def make_client(cfg: AppConfig, credentials: KisCredentials | None, *, dry_run: bool) -> KisClient:
    return KisClient(
        credentials,
        dry_run=dry_run,
        base_url=VTS_BASE_URL,
        token_cache=cfg.path(cfg.paths.state_dir) / "kis_token.json",
        min_interval_sec=cfg.broker.min_request_interval_sec,
        timeout_sec=cfg.broker.request_timeout_sec,
    )


def make_broker(
    cfg: AppConfig, market_key: str, credentials: KisCredentials | None, *, dry_run: bool, client: KisClient | None = None
) -> DomesticBroker | OverseasBroker:
    if market_key not in MARKET_BROKERS:
        raise ValueError(f"시장 '{market_key}' 에 연결된 브로커가 없습니다 (kr | us)")
    client = client or make_client(cfg, credentials, dry_run=dry_run)
    return MARKET_BROKERS[market_key](client, cfg.broker, list(cfg.market(market_key).symbols))


__all__ = [
    "Balance",
    "DomesticBroker",
    "KisApiError",
    "KisClient",
    "KisError",
    "KisNetworkError",
    "OrderBlockedError",
    "OrderRequest",
    "OrderResult",
    "OverseasBroker",
    "Position",
    "VTS_BASE_URL",
    "make_broker",
    "make_client",
]
