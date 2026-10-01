"""국내주식 주문/조회 (한국투자증권 모의투자).

공식 예제 기준 (koreainvestment/open-trading-api, examples_llm/domestic_stock):
  주문  POST /uapi/domestic-stock/v1/trading/order-cash      모의 매수 VTTC0012U / 매도 VTTC0011U
  잔고  GET  /uapi/domestic-stock/v1/trading/inquire-balance  모의 VTTC8434R
  시세  GET  /uapi/domestic-stock/v1/quotations/inquire-price 공통 FHKST01010100
"""
from __future__ import annotations

import math

from ..config import BrokerConfig
from .kis_client import Balance, KisClient, OrderRequest, OrderResult, Position, to_float

ORDER_PATH = "/uapi/domestic-stock/v1/trading/order-cash"
BALANCE_PATH = "/uapi/domestic-stock/v1/trading/inquire-balance"
PRICE_PATH = "/uapi/domestic-stock/v1/quotations/inquire-price"

# KRX 호가가격단위 (2023-01-25 개편 이후 유가·코스닥 공통) — 지정가 주문에만 사용
_KRX_TICKS = [(2_000, 1), (5_000, 5), (20_000, 10), (50_000, 50), (200_000, 100), (500_000, 500)]


def krx_tick_size(price: float) -> int:
    for upper, tick in _KRX_TICKS:
        if price < upper:
            return tick
    return 1_000


def round_to_tick(price: float, side: str) -> int:
    """매수는 위로, 매도는 아래로 호가단위에 맞춘다 (체결 쪽으로 보수적)."""
    tick = krx_tick_size(price)
    steps = math.ceil(price / tick) if side == "buy" else math.floor(price / tick)
    return int(steps * tick)


def to_krx_code(ticker: str) -> str:
    code = ticker.split(".")[0]
    if len(code) != 6 or not code.isalnum():
        raise ValueError(f"국내 종목코드 형식이 아닙니다: {ticker} (예: 005930.KS)")
    return code


class DomesticBroker:
    market = "domestic"

    def __init__(self, client: KisClient, cfg: BrokerConfig, tickers: list[str]):
        self.client = client
        self.cfg = cfg
        self._code_to_ticker = {to_krx_code(t): t for t in tickers}

    def build_order(self, side: str, ticker: str, qty: int, ref_price: float | None = None) -> OrderRequest:
        if side not in ("buy", "sell"):
            raise ValueError(side)
        if qty <= 0:
            raise ValueError("주문 수량은 1 이상이어야 합니다")
        cano, prdt = self.client.account if self.client.credentials else ("00000000", "01")
        limit = self.cfg.domestic.order_type == "limit"
        price = None
        if limit:
            if ref_price is None:
                raise ValueError("지정가 주문에는 기준가가 필요합니다")
            buf = self.cfg.domestic.limit_buffer_pct / 100
            price = round_to_tick(ref_price * (1 + buf if side == "buy" else 1 - buf), side)
        body = {
            "CANO": cano,
            "ACNT_PRDT_CD": prdt,
            "PDNO": to_krx_code(ticker),
            "ORD_DVSN": "00" if limit else "01",  # 00 지정가 / 01 시장가
            "ORD_QTY": str(int(qty)),
            "ORD_UNPR": str(price) if limit else "0",
            "EXCG_ID_DVSN_CD": self.cfg.domestic.exchange_id,
            "SLL_TYPE": "01" if side == "sell" else "",  # 01 일반매도
            "CNDT_PRIC": "",
        }
        tr_id = self.cfg.tr_id.domestic_buy if side == "buy" else self.cfg.tr_id.domestic_sell
        return OrderRequest("domestic", side, ticker, int(qty), price, ORDER_PATH, tr_id, body)

    def buy(self, ticker: str, qty: int, ref_price: float | None = None) -> OrderResult:
        return self.client.submit_order(self.build_order("buy", ticker, qty, ref_price))

    def sell(self, ticker: str, qty: int, ref_price: float | None = None) -> OrderResult:
        return self.client.submit_order(self.build_order("sell", ticker, qty, ref_price))

    def price(self, ticker: str) -> float:
        data = self.client.get(
            PRICE_PATH,
            self.cfg.tr_id.domestic_price,
            {"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": to_krx_code(ticker)},
        )
        return to_float((data.get("output") or {}).get("stck_prpr"))

    def balance(self) -> Balance:
        cano, prdt = self.client.account
        data = self.client.get(
            BALANCE_PATH,
            self.cfg.tr_id.domestic_balance,
            {
                "CANO": cano,
                "ACNT_PRDT_CD": prdt,
                "AFHR_FLPR_YN": "N",
                "OFL_YN": "",
                "INQR_DVSN": "02",
                "UNPR_DVSN": "01",
                "FUND_STTL_ICLD_YN": "N",
                "FNCG_AMT_AUTO_RDPT_YN": "N",
                "PRCS_DVSN": "00",
                "CTX_AREA_FK100": "",
                "CTX_AREA_NK100": "",
            },
        )
        positions = {}
        for row in data.get("output1") or []:
            qty = int(to_float(row.get("hldg_qty")))
            if qty <= 0:
                continue
            code = row.get("pdno", "")
            symbol = self._code_to_ticker.get(code, code)
            positions[symbol] = Position(symbol, qty, to_float(row.get("pchs_avg_pric")), to_float(row.get("prpr")))
        out2 = data.get("output2")
        summary = out2[0] if isinstance(out2, list) and out2 else (out2 if isinstance(out2, dict) else {})
        # 가수도정산금액(D+2 예수금)을 주문 가능 현금의 근사치로 사용, 없으면 예수금총금액
        cash = to_float(summary.get("prvs_rcdl_excc_amt"), to_float(summary.get("dnca_tot_amt")))
        return Balance("KRW", cash, to_float(summary.get("tot_evlu_amt")), positions)
