"""미국주식 주문/조회 (한국투자증권 모의투자).

공식 예제 기준 (koreainvestment/open-trading-api, examples_llm/overseas_stock):
  주문      POST /uapi/overseas-stock/v1/trading/order           모의 매수 VTTT1002U / 매도 VTTT1001U
  잔고      GET  /uapi/overseas-stock/v1/trading/inquire-balance  모의 VTTS3012R
  매수가능  GET  /uapi/overseas-stock/v1/trading/inquire-psamount  모의 VTTS3007R
  현재가    GET  /uapi/overseas-price/v1/quotations/price          공통 HHDFS00000300
  * 모의투자 미국 주문은 ORD_DVSN 00(지정가)만 가능 → 현재가 ± 버퍼의 지정가로 낸다.
  * 미국 매도 모의 tr_id: API 설명·Postman 모의계좌 샘플은 VTTT1001U, 예제 코드의 자동변환(T→V)은 VTTT1006U 를 만든다.
    문서 값(VTTT1001U)을 기본으로 쓰고, 거부되면 config.yaml broker.tr_id.overseas_sell 만 바꾸면 된다.
"""
from __future__ import annotations

import math

from ..config import BrokerConfig
from .kis_client import Balance, KisClient, OrderRequest, OrderResult, Position, to_float

ORDER_PATH = "/uapi/overseas-stock/v1/trading/order"
BALANCE_PATH = "/uapi/overseas-stock/v1/trading/inquire-balance"
BUYING_POWER_PATH = "/uapi/overseas-stock/v1/trading/inquire-psamount"
PRICE_PATH = "/uapi/overseas-price/v1/quotations/price"

US_ORDER_EXCHANGES = ("NASD", "NYSE", "AMEX")
QUOTE_EXCD = {"NASD": "NAS", "NYSE": "NYS", "AMEX": "AMS"}  # 시세 조회용 거래소코드는 다르다


def us_limit_price(ref_price: float, side: str, buffer_pct: float) -> float:
    """매수는 현재가보다 조금 높게(올림), 매도는 조금 낮게(내림). $1 이상은 센트 단위."""
    raw = ref_price * (1 + buffer_pct / 100 if side == "buy" else 1 - buffer_pct / 100)
    scale = 100 if raw >= 1 else 10_000
    return (math.ceil(raw * scale) if side == "buy" else math.floor(raw * scale)) / scale


class OverseasBroker:
    market = "overseas"

    def __init__(self, client: KisClient, cfg: BrokerConfig, tickers: list[str]):
        self.client = client
        self.cfg = cfg
        missing = [t for t in tickers if t not in cfg.overseas.exchanges]
        if missing:
            raise ValueError(f"config.yaml broker.overseas.exchanges 에 거래소코드가 없습니다: {missing}")
        bad = {t: x for t, x in cfg.overseas.exchanges.items() if x not in US_ORDER_EXCHANGES}
        if bad:
            raise ValueError(f"미국 거래소코드는 {US_ORDER_EXCHANGES} 중 하나여야 합니다: {bad}")
        self.tickers = list(tickers)

    def exchange(self, ticker: str) -> str:
        return self.cfg.overseas.exchanges[ticker]

    def build_order(self, side: str, ticker: str, qty: int, ref_price: float | None = None) -> OrderRequest:
        if side not in ("buy", "sell"):
            raise ValueError(side)
        if qty <= 0:
            raise ValueError("주문 수량은 1 이상이어야 합니다")
        if ref_price is None or ref_price <= 0:
            raise ValueError("미국주식 모의투자는 지정가만 가능하므로 기준가가 필요합니다")
        cano, prdt = self.client.account if self.client.credentials else ("00000000", "01")
        price = us_limit_price(ref_price, side, self.cfg.overseas.limit_buffer_pct)
        body = {
            "CANO": cano,
            "ACNT_PRDT_CD": prdt,
            "OVRS_EXCG_CD": self.exchange(ticker),
            "PDNO": ticker,
            "ORD_QTY": str(int(qty)),
            "OVRS_ORD_UNPR": f"{price:.2f}" if price >= 1 else f"{price:.4f}",
            "CTAC_TLNO": "",
            "MGCO_APTM_ODNO": "",
            "SLL_TYPE": "00" if side == "sell" else "",
            "ORD_SVR_DVSN_CD": "0",
            "ORD_DVSN": "00",  # 지정가 (모의투자는 지정가만 가능)
        }
        tr_id = self.cfg.tr_id.overseas_buy if side == "buy" else self.cfg.tr_id.overseas_sell
        return OrderRequest("overseas", side, ticker, int(qty), price, ORDER_PATH, tr_id, body)

    def buy(self, ticker: str, qty: int, ref_price: float | None = None) -> OrderResult:
        return self.client.submit_order(self.build_order("buy", ticker, qty, ref_price))

    def sell(self, ticker: str, qty: int, ref_price: float | None = None) -> OrderResult:
        return self.client.submit_order(self.build_order("sell", ticker, qty, ref_price))

    def price(self, ticker: str) -> float:
        data = self.client.get(
            PRICE_PATH,
            self.cfg.tr_id.overseas_price,
            {"AUTH": "", "EXCD": QUOTE_EXCD[self.exchange(ticker)], "SYMB": ticker},
        )
        return to_float((data.get("output") or {}).get("last"))

    def buying_power(self, ticker: str, price: float) -> float:
        cano, prdt = self.client.account
        data = self.client.get(
            BUYING_POWER_PATH,
            self.cfg.tr_id.overseas_buying_power,
            {
                "CANO": cano,
                "ACNT_PRDT_CD": prdt,
                "OVRS_EXCG_CD": self.exchange(ticker),
                "OVRS_ORD_UNPR": f"{price:.2f}",
                "ITEM_CD": ticker,
            },
        )
        out = data.get("output") or {}
        return to_float(out.get("ord_psbl_frcr_amt"), to_float(out.get("ovrs_ord_psbl_amt")))

    def balance(self) -> Balance:
        cano, prdt = self.client.account
        positions: dict[str, Position] = {}
        for exch in sorted({self.exchange(t) for t in self.tickers}):
            data = self.client.get(
                BALANCE_PATH,
                self.cfg.tr_id.overseas_balance,
                {
                    "CANO": cano,
                    "ACNT_PRDT_CD": prdt,
                    "OVRS_EXCG_CD": exch,
                    "TR_CRCY_CD": "USD",
                    "CTX_AREA_FK200": "",
                    "CTX_AREA_NK200": "",
                },
            )
            for row in data.get("output1") or []:
                qty = int(to_float(row.get("ovrs_cblc_qty")))
                if qty <= 0:
                    continue
                sym = row.get("ovrs_pdno", "")
                positions[sym] = Position(sym, qty, to_float(row.get("pchs_avg_pric")), to_float(row.get("now_pric2")))
        # 주문가능 외화금액을 현금으로 본다 (첫 종목·현재가 기준 조회)
        first = self.tickers[0]
        cash = self.buying_power(first, max(self.price(first), 0.01))
        equity = cash + sum(p.qty * p.last_price for p in positions.values())
        return Balance("USD", cash, equity, positions)
