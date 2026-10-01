"""주문 모듈 안전장치 테스트. 네트워크는 전혀 쓰지 않는다 (가짜 세션)."""
import os
import stat
from dataclasses import replace

import pytest
import requests

from trader.broker import VTS_BASE_URL, KisApiError, KisClient, KisNetworkError, OrderBlockedError, make_broker
from trader.broker.domestic import DomesticBroker, krx_tick_size, round_to_tick, to_krx_code
from trader.broker.overseas import OverseasBroker, us_limit_price
from trader.config import ConfigError, KisCredentials, is_dry_run, load_credentials

CREDS = KisCredentials(app_key="APPKEY-XYZ-123", app_secret="APPSECRET-XYZ-456", account_no="12345678", product_code="01")


class NoNetwork:
    def request(self, *a, **k):
        raise AssertionError("네트워크 호출이 일어나면 안 됩니다")

    post = get = request


class FakeResp:
    def __init__(self, status, data):
        self.status_code, self._data = status, data

    def json(self):
        return self._data


class Recorder:
    """응답(또는 던질 예외)을 순서대로 돌려주는 가짜 세션."""

    def __init__(self, responses):
        self.calls, self.responses = [], list(responses)

    def request(self, method, url, **kw):
        self.calls.append((method, url, kw))
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


# --- DRY_RUN / 자격정보 ------------------------------------------------------
@pytest.mark.parametrize(
    "env, expected",
    [
        ({}, True),  # 미설정 → 켜짐
        ({"DRY_RUN": "true"}, True),
        ({"DRY_RUN": ""}, True),
        ({"DRY_RUN": "flase"}, True),  # 오타 → 안전하게 켜짐
        ({"DRY_RUN": "false"}, False),
        ({"DRY_RUN": " FALSE "}, False),
        ({"DRY_RUN": "0"}, False),
    ],
)
def test_dry_run_is_on_unless_explicitly_false(env, expected):
    assert is_dry_run(env) is expected


def test_env_example_ships_with_dry_run_true():
    with open(os.path.join(os.path.dirname(__file__), "..", ".env.example"), encoding="utf-8") as f:
        text = f.read()
    assert "DRY_RUN=true" in text
    for key in ("KIS_APP_KEY=", "KIS_APP_SECRET=", "KIS_ACCOUNT_NO="):
        line = next(l for l in text.splitlines() if l.startswith(key))
        assert line == key, "예시 파일에 실제 값이 들어 있으면 안 됩니다"


def test_load_credentials():
    assert load_credentials({}) is None
    c = load_credentials({"KIS_APP_KEY": "k", "KIS_APP_SECRET": "s", "KIS_ACCOUNT_NO": "12345678-01"})
    assert (c.account_no, c.product_code) == ("12345678", "01")
    with pytest.raises(ConfigError):
        load_credentials({"KIS_APP_KEY": "k", "KIS_APP_SECRET": "s", "KIS_ACCOUNT_NO": "1234"})
    assert "APPKEY" not in repr(CREDS) and "APPSECRET" not in repr(CREDS) and "12345678" not in repr(CREDS)


# --- 실전 차단 ---------------------------------------------------------------
def test_real_server_url_is_rejected():
    with pytest.raises(OrderBlockedError):
        KisClient(CREDS, dry_run=False, base_url="https://openapi.koreainvestment.com:9443")
    assert "openapivts" in VTS_BASE_URL


def test_non_paper_tr_id_is_blocked_even_when_dry_run_off(cfg):
    bad = replace(cfg.broker, tr_id=replace(cfg.broker.tr_id, domestic_buy="TTTC0012U"))
    client = KisClient(CREDS, dry_run=False, session=NoNetwork(), min_interval_sec=0)
    with pytest.raises(OrderBlockedError):
        DomesticBroker(client, bad, ["005930.KS"]).buy("005930.KS", 1)


# --- DRY_RUN 이면 네트워크 없음 ------------------------------------------------
def test_dry_run_orders_never_touch_network(cfg):
    client = KisClient(None, dry_run=True, session=NoNetwork())  # 자격정보 없어도 동작
    r1 = DomesticBroker(client, cfg.broker, ["005930.KS"]).buy("005930.KS", 3)
    r2 = OverseasBroker(client, cfg.broker, ["AAPL"]).sell("AAPL", 2, ref_price=180.0)
    for r in (r1, r2):
        assert r.dry_run and not r.sent and r.ok


def test_make_broker_defaults_to_dry_run(cfg):
    b = make_broker(cfg, "kr", None, dry_run=is_dry_run({}))
    assert b.client.dry_run is True


# --- 요청 형식 (공식 예제와 같은 필드) ----------------------------------------------
def test_domestic_order_request_matches_official_spec(cfg):
    client = KisClient(CREDS, dry_run=True, session=NoNetwork())
    b = DomesticBroker(client, cfg.broker, ["005930.KS"])
    buy = b.build_order("buy", "005930.KS", 3)
    sell = b.build_order("sell", "005930.KS", 3)
    assert buy.path == "/uapi/domestic-stock/v1/trading/order-cash"
    assert (buy.tr_id, sell.tr_id) == ("VTTC0012U", "VTTC0011U")
    assert set(buy.body) == {"CANO", "ACNT_PRDT_CD", "PDNO", "ORD_DVSN", "ORD_QTY", "ORD_UNPR", "EXCG_ID_DVSN_CD", "SLL_TYPE", "CNDT_PRIC"}
    assert buy.body["PDNO"] == "005930" and buy.body["ORD_DVSN"] == "01" and buy.body["ORD_UNPR"] == "0"
    assert buy.body["ORD_QTY"] == "3" and buy.body["EXCG_ID_DVSN_CD"] == "KRX"
    assert buy.body["SLL_TYPE"] == "" and sell.body["SLL_TYPE"] == "01"
    assert "12345678" not in buy.describe()  # 로그에 계좌번호 마스킹

    limit_cfg = replace(cfg.broker, domestic=replace(cfg.broker.domestic, order_type="limit"))
    lb = DomesticBroker(client, limit_cfg, ["005930.KS"]).build_order("buy", "005930.KS", 1, ref_price=70_050)
    assert lb.body["ORD_DVSN"] == "00" and int(lb.body["ORD_UNPR"]) % krx_tick_size(70_050) == 0


def test_overseas_order_request_matches_official_spec(cfg):
    client = KisClient(CREDS, dry_run=True, session=NoNetwork())
    b = OverseasBroker(client, cfg.broker, ["AAPL", "NVDA", "MSFT"])
    buy = b.build_order("buy", "AAPL", 2, ref_price=180.0)
    sell = b.build_order("sell", "AAPL", 2, ref_price=180.0)
    assert buy.path == "/uapi/overseas-stock/v1/trading/order"
    assert (buy.tr_id, sell.tr_id) == ("VTTT1002U", "VTTT1001U")
    assert set(buy.body) == {
        "CANO", "ACNT_PRDT_CD", "OVRS_EXCG_CD", "PDNO", "ORD_QTY", "OVRS_ORD_UNPR",
        "CTAC_TLNO", "MGCO_APTM_ODNO", "SLL_TYPE", "ORD_SVR_DVSN_CD", "ORD_DVSN",
    }
    assert buy.body["OVRS_EXCG_CD"] == "NASD" and buy.body["ORD_DVSN"] == "00"  # 모의투자는 지정가만
    assert buy.body["OVRS_ORD_UNPR"] == "180.54" and sell.body["OVRS_ORD_UNPR"] == "179.46"
    assert buy.body["SLL_TYPE"] == "" and sell.body["SLL_TYPE"] == "00"
    with pytest.raises(ValueError):
        b.build_order("buy", "AAPL", 1)  # 기준가 없으면 지정가를 만들 수 없음


def test_price_helpers():
    assert [krx_tick_size(p) for p in (1_999, 2_000, 4_999, 19_999, 49_999, 50_000, 199_999, 200_000, 500_000)] == [
        1, 5, 5, 10, 50, 100, 100, 500, 1_000,
    ]
    assert round_to_tick(70_123, "buy") == 70_200 and round_to_tick(70_123, "sell") == 70_100
    assert to_krx_code("000660.KS") == "000660"
    assert us_limit_price(100.0, "buy", 0.3) == 100.3 and us_limit_price(100.0, "sell", 0.3) == 99.7


# --- DRY_RUN=false 경로: 가짜 세션으로 요청 모양만 검증 (실제 전송 없음) -------------------
def test_send_path_targets_paper_server_with_paper_tr_id(cfg, tmp_path):
    session = Recorder(
        [
            FakeResp(200, {"access_token": "TOKEN-1", "access_token_token_expired": "2099-01-01 00:00:00", "expires_in": 86400}),
            FakeResp(200, {"rt_cd": "0", "msg1": "주문 전송 완료", "output": {"ODNO": "0000123"}}),
            FakeResp(200, {"rt_cd": "1", "msg_cd": "APBK0013", "msg1": "주문가능금액 부족"}),
        ]
    )
    token_file = tmp_path / "kis_token.json"
    client = KisClient(CREDS, dry_run=False, session=session, token_cache=token_file, min_interval_sec=0)
    broker = DomesticBroker(client, cfg.broker, ["005930.KS"])
    ok = broker.buy("005930.KS", 1)
    assert ok.sent and ok.ok and ok.order_no == "0000123"
    (m1, url1, kw1), (m2, url2, kw2) = session.calls
    assert url1 == VTS_BASE_URL + "/oauth2/tokenP"
    assert url2 == VTS_BASE_URL + "/uapi/domestic-stock/v1/trading/order-cash"
    assert kw2["headers"]["tr_id"] == "VTTC0012U" and kw2["headers"]["authorization"] == "Bearer TOKEN-1"
    assert stat.S_IMODE(os.stat(token_file).st_mode) == 0o600

    rejected = broker.sell("005930.KS", 1)  # 토큰은 캐시 재사용
    assert rejected.sent and not rejected.ok and "주문가능금액 부족" in rejected.message
    assert len(session.calls) == 3

    assert all(kw["allow_redirects"] is False for _, _, kw in session.calls)  # 리다이렉트는 따라가지 않음


TOKEN_OK = FakeResp(200, {"access_token": "T", "access_token_token_expired": "2099-01-01 00:00:00"})


def test_network_error_on_order_is_uncertain_and_hides_account(cfg):
    leak = requests.ReadTimeout("Read timed out: /uapi/...?CANO=12345678&ACNT_PRDT_CD=01")
    session = Recorder([TOKEN_OK, leak, leak])
    client = KisClient(CREDS, dry_run=False, session=session, min_interval_sec=0)
    broker = DomesticBroker(client, cfg.broker, ["005930.KS"])
    res = broker.buy("005930.KS", 1)
    assert res.sent and not res.ok and res.uncertain  # 접수 여부 불명 → 다음 사이클에 잔고로 확인
    assert "12345678" not in res.message
    with pytest.raises(KisNetworkError) as ei:
        broker.balance()
    assert "12345678" not in str(ei.value) and ei.value.__cause__ is None and ei.value.__suppress_context__


def test_redirect_is_rejected_not_followed(cfg):
    session = Recorder([TOKEN_OK, FakeResp(307, {}), FakeResp(302, {})])
    client = KisClient(CREDS, dry_run=False, session=session, min_interval_sec=0)
    broker = DomesticBroker(client, cfg.broker, ["005930.KS"])
    res = broker.buy("005930.KS", 1)
    assert not res.ok and not res.uncertain and "리다이렉트" in res.message
    with pytest.raises(KisApiError, match="리다이렉트"):
        broker.balance()
    assert len(session.calls) == 3  # 다른 호스트로 재전송 없음
