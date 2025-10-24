# server/routers/stock.py
import asyncio
import logging
from typing import Optional, List, Literal
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

# core.kiwoom_api 에서 키움 API 호출 함수 가져오기
from core.kiwoom_api import get_top_volume_stocks, get_sorted_market_cap_codes
# server.deps 에서 로깅 헬퍼 가져오기
from server.deps import get_trace_logger, new_trace_id

router = APIRouter()
log = logging.getLogger("stock_api")

# ============================================================
# 🧩 응답 모델 정의 (기존과 동일)
# ============================================================

class StockInfo(BaseModel):
    rank: int = Field(..., description="시가총액 순위")
    code: str = Field(..., description="종목 코드")
    name: str = Field(..., description="종목명")
    market: str = Field(..., description="시장명 (코스피/코스닥/ETF 등)")
    market_cap: str = Field(..., description="시가총액")
    price: str = Field(..., description="현재가")
    change_price: str = Field(..., description="전일 대비 가격")
    change_rate: str = Field(..., description="등락률 (%)")
    volume: str = Field(..., description="거래량")

# ============================================================
# 📈 시가총액 상위 종목 조회 API (수정됨)
# ============================================================

@router.get(
    "/stock/market-cap-top-codes",
    summary="시가총액 상위 종목 리스트 조회 (키움 API)",
    response_model=List[StockInfo]  # ✅ 응답 구조를 StockInfo 리스트로 지정
)
async def get_market_cap_top_codes(
    top_n: int = Query(50, ge=10, le=100, description="조회할 상위 종목 개수 (10~100)"),
    env: Literal["real", "mock"] = Query("real", description="API 환경 선택 ('real' 또는 'mock')")
):
    """
    키움증권 API(ka10099)를 호출하여 코스피/코스닥 전체 종목을
    시가총액 기준으로 정렬한 후, 상위 N개 종목의 상세 정보를 반환합니다.
    """
    trace_id = new_trace_id()
    tlog = get_trace_logger(trace_id)
    tlog.info(f"시가총액 상위 {top_n}개 종목 요청 (Env: {env})")

    try:
        # ✅ kiwoom_api의 동기(sync) 함수를 asyncio.to_thread로 감싸 비동기 호출
        top_stocks, error_message = await asyncio.to_thread(
            get_sorted_market_cap_codes, top_n=top_n, env=env
        )

        if error_message:
            tlog.error(f"시가총액 상위 종목 조회 실패: {error_message}")
            raise HTTPException(status_code=502, detail=f"종목 조회 실패: {error_message}")
        
        return top_stocks  # ✅ StockInfo 구조에 맞게 반환됨

    except HTTPException:
        raise
    except Exception as e:
        tlog.exception(f"시가총액 상위 종목 처리 중 예외 발생: {e}")
        raise HTTPException(status_code=500, detail=f"서버 내부 오류: {str(e)}")

# ============================================================
# 📊 거래량 상위 종목 조회 API (기존과 동일)
# ============================================================

@router.post("/stock/top-volume", summary="당일 거래량 상위 종목 조회 (키움 API)")
async def get_kiwoom_top_volume(
    env: Literal["real", "mock"] = Query("real", description="API 환경 선택 ('real' 또는 'mock')"),
    market_type: str = Query("000", description="시장구분 000:전체, 001:코스피, 101:코스닥"),
    sort_type: str = Query("1", description="정렬구분 1:거래량, 2:거래회전율, 3:거래대금")
):
    """
    키움증권 API(ka10030)를 직접 호출하여 당일 거래량 상위 종목 데이터를 반환합니다.
    """
    trace_id = new_trace_id()
    tlog = get_trace_logger(trace_id)
    tlog.info(f"키움 거래량 상위 요청 (Env: {env}, Market: {market_type}, Sort: {sort_type})")

    params = {
        'mrkt_tp': market_type,
        'sort_tp': sort_type,
        'mang_stk_incls': '0',
        'crd_tp': '0',
        'trde_qty_tp': '0',
        'pric_tp': '0',
        'trde_prica_tp': '0',
        'mrkt_open_tp': '0',
        'stex_tp': '1' if env == 'mock' else '3',
    }

    try:
        # ✅ 동기(sync) 함수이므로 asyncio.to_thread 사용 (기존과 동일)
        kiwoom_result = await asyncio.to_thread(get_top_volume_stocks, data=params, env=env)

        if not kiwoom_result:
            raise HTTPException(status_code=503, detail="증권사 API 호출 실패 (응답 없음)")

        status_code = kiwoom_result.get("status_code", 500)
        body = kiwoom_result.get("body")
        api_error = kiwoom_result.get("error")

        is_success = body and (body.get("rt_cd") == "0" or body.get("return_code") == 0)

        if status_code == 200 and is_success:
            return body
        elif body and body.get("rt_cd") != "0":
            msg = body.get("msg1", f"키움 API 오류 (rt_cd: {body.get('rt_cd')})")
            raise HTTPException(status_code=400, detail=f"증권사 오류: {msg}")
        elif api_error:
            raise HTTPException(status_code=502, detail=f"증권사 API 호출 실패: {api_error}")
        else:
            raise HTTPException(status_code=502, detail=f"API 응답 비정상 (Status: {status_code})")

    except HTTPException:
        raise
    except Exception as e:
        tlog.exception(f"키움 API 조회 중 예외 발생: {e}")
        raise HTTPException(status_code=500, detail=f"서버 내부 오류: {str(e)}")