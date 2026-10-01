"""국내주식 주문/조회 (한국투자증권 모의투자).

공식 예제 기준 (koreainvestment/open-trading-api, examples_llm/domestic_stock):
  주문  POST /uapi/domestic-stock/v1/trading/order-cash         모의 매수 VTTC0012U / 매도 VTTC0011U
  취소  POST /uapi/domestic-stock/v1/trading/order-rvsecncl     모의 VTTC0013U (정정취소가능주문조회는 모의 미지원 → 주문번호로 직접 취소)
  체결  GET  /uapi/domestic-stock/v1/trading/inquire-daily-ccld 모의 VTTC0081R (한 번에 15건, 연속조회)
  잔고  GET  /uapi/domestic-stock/v1/trading/inquire-balance    모의 VTTC8434R
  시세  GET  /uapi/domestic-stock/v1/quotations/inquire-price    공통 FHKST01010100
"""
from __future__ import annotations

import math

from ..config import BrokerConfig
from .kis_client import Balance, KisClient, OrderRequest, OrderResult, Position, to_float

ORDER_PATH = "/uapi/domestic-stock/v1/trading/order-cash"
CANCEL_PATH = "/uapi/domestic-stock/v1/trading/order-rvsecncl"
DAILY_CCLD_PATH = "/uapi/domestic-stock/v1/trading/inquire-daily-ccld"
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

    def build_order_at(self, side: str, ticker: str, qty: int, limit_price: float | None) -> OrderRequest:
        """가격을 정해서 내는 주문: limit_price 가 있으면 그 가격 지정가(00), None 이면 시장가(01). 일봉 시가 주문용."""
        if side not in ("buy", "sell"):
            raise ValueError(side)
        if qty <= 0:
            raise ValueError("주문 수량은 1 이상이어야 합니다")
        price = None
        if limit_price is not None:
            price = int(limit_price)
            if price <= 0 or price != limit_price or price % krx_tick_size(price):
                raise ValueError(f"호가단위에 맞지 않는 지정가입니다: {limit_price}")
        cano, prdt = self.client.account if self.client.credentials else ("00000000", "01")
        body = {
            "CANO": cano,
            "ACNT_PRDT_CD": prdt,
            "PDNO": to_krx_code(ticker),
            "ORD_DVSN": "01" if price is None else "00",
            "ORD_QTY": str(int(qty)),
            "ORD_UNPR": "0" if price is None else str(price),
            "EXCG_ID_DVSN_CD": self.cfg.domestic.exchange_id,
            "SLL_TYPE": "01" if side == "sell" else "",
            "CNDT_PRIC": "",
        }
        tr_id = self.cfg.tr_id.domestic_buy if side == "buy" else self.cfg.tr_id.domestic_sell
        return OrderRequest("domestic", side, ticker, int(qty), price, ORDER_PATH, tr_id, body)

    def place(self, side: str, ticker: str, qty: int, limit_price: float | None = None) -> OrderResult:
        return self.client.submit_order(self.build_order_at(side, ticker, qty, limit_price))

    def build_cancel(self, ticker: str, order_no: str, org_no: str, qty: int) -> OrderRequest:
        """미체결 잔량 전부 취소 (지정가 주문 기준)."""
        cano, prdt = self.client.account if self.client.credentials else ("00000000", "01")
        body = {
            "CANO": cano,
            "ACNT_PRDT_CD": prdt,
            "KRX_FWDG_ORD_ORGNO": org_no or "",
            "ORGN_ODNO": order_no,
            "ORD_DVSN": "00",
            "RVSE_CNCL_DVSN_CD": "02",  # 02 취소
            "ORD_QTY": str(int(qty)),
            "ORD_UNPR": "0",
            "QTY_ALL_ORD_YN": "Y",  # 잔량 전부
            "EXCG_ID_DVSN_CD": self.cfg.domestic.exchange_id,
        }
        return OrderRequest("domestic", "cancel", ticker, int(qty), None, CANCEL_PATH, self.cfg.tr_id.domestic_cancel, body)

    def cancel(self, ticker: str, order_no: str, org_no: str, qty: int) -> OrderResult:
        return self.client.submit_order(self.build_cancel(ticker, order_no, org_no, qty))

    def daily_orders(self, day: str) -> list[dict]:
        """그날(YYYYMMDD) 주문·체결 내역. 행: odno, pdno, sll_buy_dvsn_cd(01 매도/02 매수), ord_qty, tot_ccld_qty,
        avg_prvs(체결 평균가), rmn_qty(잔여), cncl_yn, ord_orgno 등 (공식 예제 컬럼명)."""
        cano, prdt = self.client.account
        params = {
            "CANO": cano,
            "ACNT_PRDT_CD": prdt,
            "INQR_STRT_DT": day,
            "INQR_END_DT": day,
            "SLL_BUY_DVSN_CD": "00",
            "PDNO": "",
            "CCLD_DVSN": "00",
            "INQR_DVSN": "01",
            "INQR_DVSN_3": "00",
            "ORD_GNO_BRNO": "",
            "ODNO": "",
            "INQR_DVSN_1": "",
            "CTX_AREA_FK100": "",
            "CTX_AREA_NK100": "",
            "EXCG_ID_DVSN_CD": self.cfg.domestic.exchange_id,
        }
        return self.client.get_all(DAILY_CCLD_PATH, self.cfg.tr_id.domestic_daily_ccld, params)

    def quote(self, ticker: str) -> dict[str, float]:
        """현재가·오늘 시가·기준가(보통 전일 종가). 시가가 아직 없으면 open=0."""
        data = self.client.get(
            PRICE_PATH,
            self.cfg.tr_id.domestic_price,
            {"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": to_krx_code(ticker)},
        )
        out = data.get("output") or {}
        return {"price": to_float(out.get("stck_prpr")), "open": to_float(out.get("stck_oprc")), "base": to_float(out.get("stck_sdpr"))}

    def buy(self, ticker: str, qty: int, ref_price: float | None = None) -> OrderResult:
        return self.client.submit_order(self.build_order("buy", ticker, qty, ref_price))

    def sell(self, ticker: str, qty: int, ref_price: float | None = None) -> OrderResult:
        return self.client.submit_order(self.build_order("sell", ticker, qty, ref_price))

    def price(self, ticker: str) -> float:
        return self.quote(ticker)["price"]

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
            sellable = int(to_float(row.get("ord_psbl_qty"), qty))
            positions[symbol] = Position(
                symbol, qty, to_float(row.get("pchs_avg_pric")), to_float(row.get("prpr")), sellable_qty=sellable
            )
        out2 = data.get("output2")
        summary = out2[0] if isinstance(out2, list) and out2 else (out2 if isinstance(out2, dict) else {})
        # 가수도정산금액(D+2 예수금)을 주문 가능 현금의 근사치로 사용, 없으면 예수금총금액
        cash = to_float(summary.get("prvs_rcdl_excc_amt"), to_float(summary.get("dnca_tot_amt")))
        return Balance("KRW", cash, to_float(summary.get("tot_evlu_amt")), positions)
