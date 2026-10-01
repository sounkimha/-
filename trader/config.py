"""설정 로딩.

- 수치(종목, 비용, 손실 제한, 모델, 전략 등)는 config.yaml 에서 읽는다.
- 비밀값(앱키·시크릿·계좌번호)과 DRY_RUN 은 .env(환경변수)에서만 읽는다.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from datetime import time
from pathlib import Path
from typing import Any, Mapping

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = ROOT / "config.yaml"

# DRY_RUN 을 끄는 값은 이것들뿐이다. 오타·빈 값·미설정은 전부 '켜짐(True)'으로 본다.
_DRY_RUN_OFF_VALUES = {"false", "0", "no", "off"}


class ConfigError(ValueError):
    """설정 파일/환경변수 오류."""


def _build(cls, data: Mapping[str, Any] | None, where: str):
    """dict -> dataclass. 오타 난 키를 조용히 무시하지 않고 에러로 알린다."""
    data = dict(data or {})
    unknown = set(data) - {f.name for f in fields(cls)}
    if unknown:
        raise ConfigError(f"{where}: 알 수 없는 설정 키 {sorted(unknown)}")
    try:
        return cls(**data)
    except TypeError as e:
        raise ConfigError(f"{where}: {e}") from e


def _parse_hhmm(value: Any, where: str) -> time:
    if not isinstance(value, str):
        raise ConfigError(f'{where}: 시각은 "09:00" 처럼 따옴표로 감싼 문자열이어야 합니다 (받은 값: {value!r})')
    try:
        hh, mm = value.split(":")
        return time(int(hh), int(mm))
    except ValueError as e:
        raise ConfigError(f"{where}: 시각 형식 오류 {value!r}") from e


@dataclass(frozen=True)
class CostConfig:
    commission_pct: float  # 편도 수수료 %
    sell_tax_pct: float  # 매도 시 세금 %
    slippage_pct: float  # 편도 슬리피지 %

    @property
    def commission(self) -> float:
        return self.commission_pct / 100

    @property
    def sell_tax(self) -> float:
        return self.sell_tax_pct / 100

    @property
    def slippage(self) -> float:
        return self.slippage_pct / 100

    @property
    def round_trip(self) -> float:
        """왕복 총비용(비율). 라벨 기준선이자 백테스트 비용."""
        return 2 * self.commission + self.sell_tax + 2 * self.slippage


@dataclass(frozen=True)
class SessionConfig:
    open: time
    close: time


@dataclass(frozen=True)
class MarketConfig:
    key: str
    name: str
    currency: str
    timezone: str
    initial_capital: float
    data_delay_minutes: int
    session: SessionConfig
    symbols: dict[str, str]  # yfinance 티커 -> 표시 이름
    costs: CostConfig

    def label(self, ticker: str) -> str:
        return f"{self.symbols.get(ticker, ticker)}({ticker})"


@dataclass(frozen=True)
class DataConfig:
    interval: str = "1h"
    period_days: int = 729
    cache_dir: str = "data_cache"
    cache_max_age_minutes: int = 30


@dataclass(frozen=True)
class FeatureConfig:
    return_windows: tuple[int, ...] = (1, 2, 4, 8, 24)
    rsi_period: int = 14
    ma_windows: tuple[int, ...] = (5, 20, 60)
    volatility_windows: tuple[int, ...] = (8, 24)
    volume_window: int = 20
    range_window: int = 24


@dataclass(frozen=True)
class ModelConfig:
    type: str = "hgb"
    pooled: bool = True
    min_train_days: int = 180
    retrain_every_days: int = 5
    train_window_days: int = 0
    random_state: int = 42
    params: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class StrategyConfig:
    entry_threshold: float = 0.50
    min_expected_return_pct: float = 0.0
    exit_threshold: float = 0.45
    max_hold_bars: int = 0
    flatten_at_session_end: bool = True

    @property
    def min_expected_return(self) -> float:
        return self.min_expected_return_pct / 100


@dataclass(frozen=True)
class RiskConfig:
    stop_loss_pct: float = 1.0
    position_size_pct: float = 20.0
    max_drawdown_pct: float = 10.0
    liquidate_on_halt: bool = True
    max_trades_per_day: int = 4


@dataclass(frozen=True)
class DomesticBrokerConfig:
    exchange_id: str = "KRX"
    order_type: str = "market"
    limit_buffer_pct: float = 0.3


@dataclass(frozen=True)
class OverseasBrokerConfig:
    order_type: str = "limit"
    limit_buffer_pct: float = 0.3
    exchanges: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class TrIdConfig:
    domestic_buy: str = "VTTC0012U"
    domestic_sell: str = "VTTC0011U"
    domestic_balance: str = "VTTC8434R"
    domestic_price: str = "FHKST01010100"
    overseas_buy: str = "VTTT1002U"
    overseas_sell: str = "VTTT1001U"
    overseas_balance: str = "VTTS3012R"
    overseas_buying_power: str = "VTTS3007R"
    overseas_price: str = "HHDFS00000300"


@dataclass(frozen=True)
class BrokerConfig:
    min_request_interval_sec: float = 0.5
    request_timeout_sec: float = 10.0
    domestic: DomesticBrokerConfig = field(default_factory=DomesticBrokerConfig)
    overseas: OverseasBrokerConfig = field(default_factory=OverseasBrokerConfig)
    tr_id: TrIdConfig = field(default_factory=TrIdConfig)


@dataclass(frozen=True)
class PathsConfig:
    state_dir: str = "state"
    reports_dir: str = "reports"


@dataclass(frozen=True)
class AppConfig:
    markets: dict[str, MarketConfig]
    data: DataConfig
    features: FeatureConfig
    model: ModelConfig
    strategy: StrategyConfig
    risk: RiskConfig
    broker: BrokerConfig
    paths: PathsConfig
    root: Path = ROOT

    def market(self, key: str) -> MarketConfig:
        if key not in self.markets:
            raise ConfigError(f"알 수 없는 시장 '{key}'. 가능한 값: {sorted(self.markets)}")
        return self.markets[key]

    def path(self, relative: str) -> Path:
        p = Path(relative)
        return p if p.is_absolute() else self.root / p


def _build_market(key: str, raw: Mapping[str, Any]) -> MarketConfig:
    raw = dict(raw)
    raw.setdefault("data_delay_minutes", 0)
    where = f"markets.{key}"
    session_raw = dict(raw.pop("session", {}) or {})
    costs = _build(CostConfig, raw.pop("costs", None), f"{where}.costs")
    symbols = {str(k): str(v) for k, v in (raw.pop("symbols", None) or {}).items()}
    if not symbols:
        raise ConfigError(f"{where}.symbols 가 비어 있습니다")
    session = SessionConfig(
        open=_parse_hhmm(session_raw.pop("open", None), f"{where}.session.open"),
        close=_parse_hhmm(session_raw.pop("close", None), f"{where}.session.close"),
    )
    if session_raw:
        raise ConfigError(f"{where}.session: 알 수 없는 키 {sorted(session_raw)}")
    return _build(MarketConfig, {**raw, "key": key, "session": session, "symbols": symbols, "costs": costs}, where)


def _validate(cfg: AppConfig) -> None:
    s, r, m = cfg.strategy, cfg.risk, cfg.model
    if not 0 < s.entry_threshold < 1 or not 0 <= s.exit_threshold < 1:
        raise ConfigError("strategy.entry_threshold / exit_threshold 는 0~1 사이여야 합니다")
    if s.exit_threshold > s.entry_threshold:
        raise ConfigError("strategy.exit_threshold 는 entry_threshold 보다 클 수 없습니다")
    if not 0 < r.stop_loss_pct < 100:
        raise ConfigError("risk.stop_loss_pct 는 0~100 사이여야 합니다")
    if not 0 < r.position_size_pct <= 100:
        raise ConfigError("risk.position_size_pct 는 0~100 사이여야 합니다")
    if not 0 < r.max_drawdown_pct < 100:
        raise ConfigError("risk.max_drawdown_pct 는 0~100 사이여야 합니다")
    if r.max_trades_per_day < 1:
        raise ConfigError("risk.max_trades_per_day 는 1 이상이어야 합니다")
    if m.type not in {"hgb", "logistic", "random_forest"}:
        raise ConfigError(f"model.type '{m.type}' 는 지원하지 않습니다 (hgb | logistic | random_forest)")
    if m.min_train_days < 20 or m.retrain_every_days < 1 or m.train_window_days < 0:
        raise ConfigError("model.min_train_days >= 20, retrain_every_days >= 1, train_window_days >= 0 이어야 합니다")
    for mk in cfg.markets.values():
        c = mk.costs
        if min(c.commission_pct, c.sell_tax_pct, c.slippage_pct) < 0:
            raise ConfigError(f"markets.{mk.key}.costs 에 음수가 있습니다")
        if mk.initial_capital <= 0:
            raise ConfigError(f"markets.{mk.key}.initial_capital 은 0보다 커야 합니다")
        if mk.session.open >= mk.session.close:
            raise ConfigError(f"markets.{mk.key}.session 시작 시각이 종료 시각보다 늦습니다")
    if cfg.broker.domestic.order_type not in {"market", "limit"}:
        raise ConfigError("broker.domestic.order_type 은 market | limit")
    if cfg.broker.overseas.order_type != "limit":
        raise ConfigError("broker.overseas.order_type 은 limit 만 지원합니다 (모의투자 미국주식은 지정가만 가능)")


def load_config(path: str | Path | None = None) -> AppConfig:
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise ConfigError(f"설정 파일이 없습니다: {path}")
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    markets_raw = raw.get("markets") or {}
    if not markets_raw:
        raise ConfigError("markets 설정이 비어 있습니다")
    markets = {key: _build_market(key, val) for key, val in markets_raw.items()}

    features_raw = dict(raw.get("features") or {})
    for k in ("return_windows", "ma_windows", "volatility_windows"):
        if k in features_raw:
            features_raw[k] = tuple(int(x) for x in features_raw[k])

    broker_raw = dict(raw.get("broker") or {})
    broker = _build(
        BrokerConfig,
        {
            **broker_raw,
            "domestic": _build(DomesticBrokerConfig, broker_raw.get("domestic"), "broker.domestic"),
            "overseas": _build(OverseasBrokerConfig, broker_raw.get("overseas"), "broker.overseas"),
            "tr_id": _build(TrIdConfig, broker_raw.get("tr_id"), "broker.tr_id"),
        },
        "broker",
    )

    cfg = AppConfig(
        markets=markets,
        data=_build(DataConfig, raw.get("data"), "data"),
        features=_build(FeatureConfig, features_raw, "features"),
        model=_build(ModelConfig, raw.get("model"), "model"),
        strategy=_build(StrategyConfig, raw.get("strategy"), "strategy"),
        risk=_build(RiskConfig, raw.get("risk"), "risk"),
        broker=broker,
        paths=_build(PathsConfig, raw.get("paths"), "paths"),
        root=path.resolve().parent,
    )
    _validate(cfg)
    return cfg


# --------------------------------------------------------------------------
# .env (비밀값 / DRY_RUN)
# --------------------------------------------------------------------------
def load_env(root: Path = ROOT) -> None:
    """.env 를 환경변수로 읽는다. 이미 설정된 환경변수는 덮어쓰지 않는다."""
    load_dotenv(root / ".env", override=False)


def is_dry_run(env: Mapping[str, str] | None = None) -> bool:
    """DRY_RUN 이 명시적으로 false/0/no/off 일 때만 False. 그 외(미설정·오타 포함)는 전부 True."""
    env = os.environ if env is None else env
    value = env.get("DRY_RUN")
    if value is None:
        return True
    return value.strip().lower() not in _DRY_RUN_OFF_VALUES


@dataclass(frozen=True)
class KisCredentials:
    app_key: str = field(repr=False)
    app_secret: str = field(repr=False)
    account_no: str = field(repr=False)  # 계좌번호 앞 8자리
    product_code: str = "01"  # 계좌번호 뒤 2자리

    @property
    def masked_account(self) -> str:
        return f"{self.account_no[:2]}******-{self.product_code}"

    def __repr__(self) -> str:  # 키·시크릿·계좌번호가 로그에 찍히지 않게
        return f"KisCredentials(account={self.masked_account})"


def load_credentials(env: Mapping[str, str] | None = None) -> KisCredentials | None:
    """KIS 모의투자 자격정보. 하나라도 비어 있으면 None."""
    env = os.environ if env is None else env
    app_key = env.get("KIS_APP_KEY", "").strip()
    app_secret = env.get("KIS_APP_SECRET", "").strip()
    account = env.get("KIS_ACCOUNT_NO", "").strip().replace("-", "")
    product = env.get("KIS_ACCOUNT_PRODUCT_CODE", "").strip()
    if not (app_key and app_secret and account):
        return None
    if len(account) == 10 and not product:  # 12345678-01 형태로 한 번에 넣은 경우
        account, product = account[:8], account[8:]
    product = product or "01"
    if not (account.isdigit() and len(account) == 8):
        raise ConfigError("KIS_ACCOUNT_NO 는 계좌번호 앞 8자리 숫자여야 합니다")
    if not (product.isdigit() and len(product) == 2):
        raise ConfigError("KIS_ACCOUNT_PRODUCT_CODE 는 2자리 숫자여야 합니다 (종합계좌 01)")
    return KisCredentials(app_key=app_key, app_secret=app_secret, account_no=account, product_code=product)
