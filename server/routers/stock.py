# server/routers/stock.py
import asyncio
import logging
from typing import Optional, Dict, Literal # Literal 추가
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

# core.kiwoom_api 에서 키움 API 호출 함수 가져오기
from core.kiwoom_api import get_top_volume_stocks
# server.deps 에서 로깅 헬퍼 가져오기
from server.deps import get_trace_logger, new_trace_id

router = APIRouter()
log = logging.getLogger("stock_api")

# (선택 사항) 요청 파라미터를 위한 모델 정의 (현재 사용 안 함)
# class TopVolumeParams(BaseModel):
#     market_type: str = Field("000", description="시장구분 000:전체, 001:코스피, 101:코스닥")
#     sort_type: str = Field("1", description="정렬구분 1:거래량, 2:거래회전율, 3:거래대금")

# 새로운 API 엔드포인트 정의
@router.post("/stock/top-volume", summary="당일 거래량 상위 종목 조회 (키움 API)")
async def get_kiwoom_top_volume(
    # --- Query 파라미터로 env 추가 ---
    env: Literal["real", "mock"] = Query("real", description="API 환경 선택 ('real' 또는 'mock')"), # 기본값 'real'
    # --- 기존 Query 파라미터 ---
    market_type: str = Query("000", description="시장구분 000:전체, 001:코스피, 101:코스닥"),
    sort_type: str = Query("1", description="정렬구분 1:거래량, 2:거래회전율, 3:거래대금")
    # 필요 시 다른 ka10030 파라미터도 Query로 추가 가능
):
    """
    키움증권 API(ka10030)를 직접 호출하여 당일 거래량 상위 종목 데이터를 반환합니다.
    AI 해석을 거치지 않은 원본 데이터를 제공하며, 'env' 파라미터로 실전/모의 환경 선택이 가능합니다.
    """
    trace_id = new_trace_id()
    tlog = get_trace_logger(trace_id)
    # 로그에 env 값 포함
    tlog.info(f"키움 거래량 상위 직접 조회 요청 (Env: {env}, Market: {market_type}, Sort: {sort_type})")

    # 키움 API에 전달할 파라미터 구성
    params = {
        'mrkt_tp': market_type,
        'sort_tp': sort_type,
        'mang_stk_incls': '0', # 기본값
        'crd_tp': '0',
        'trde_qty_tp': '0',
        'pric_tp': '0',
        'trde_prica_tp': '0',
        'mrkt_open_tp': '0',
        'stex_tp': '1' if env == 'mock' else '3', # 모의투자는 KRX(1)만 지원할 수 있음
    }

    try:
        # --- 여기가 수정된 부분: get_top_volume_stocks 호출 시 env=env 전달 ---
        kiwoom_result = await asyncio.to_thread(get_top_volume_stocks, data=params, env=env)
        # --- 수정 끝 ---

        # API 호출 결과 확인
        if not kiwoom_result:
            tlog.error(f"키움 API 호출 실패 (Env: {env}): 응답 없음")
            raise HTTPException(status_code=503, detail=f"증권사 API({env}) 호출에 실패했습니다 (응답 없음).")

        status_code = kiwoom_result.get("status_code", 500)
        body = kiwoom_result.get("body")
        api_error = kiwoom_result.get("error") # kiwoom_api 에서 반환하는 오류 메시지

        # 성공적인 응답 (HTTP 200 이고 키움 rt_cd 가 "0")
        is_kiwoom_success = (body and (body.get("rt_cd") == "0" or body.get("return_code") == 0))

        if status_code == 200 and is_kiwoom_success:
            tlog.info(f"키움 API 호출 성공 (Env: {env}, Status: {status_code})")
            return body # 키움 API 응답의 body 부분을 그대로 반환
        # 키움 API 자체 에러 처리 (rt_cd != "0")
        elif body and body.get("rt_cd") != "0":
             error_msg = body.get("msg1", f"키움 API 처리 오류 (rt_cd: {body.get('rt_cd')})")
             tlog.error(f"키움 API 처리 오류 (Env: {env}, Status: {status_code}): {error_msg} | 응답: {body}")
             # 클라이언트에게 키움 오류 메시지 전달
             raise HTTPException(status_code=400, detail=f"증권사 API 오류: {error_msg}") # 400 Bad Request 또는 502 Bad Gateway
        # 토큰 실패 등 API 호출 자체 실패
        elif api_error:
             tlog.error(f"키움 API 호출 실패 (Env: {env}, Status: {status_code}): {api_error} | 응답: {body}")
             # HTTP 상태코드에 따라 적절한 클라이언트 오류 반환
             client_status = 401 if status_code == 401 else 502 # 토큰 오류는 401, 나머지는 502
             raise HTTPException(status_code=client_status, detail=f"증권사 API 호출 실패: {api_error}")
        # 기타 예상치 못한 경우
        else:
             error_detail = f"증권사 API({env}) 호출 중 알 수 없는 오류 (Status: {status_code})"
             tlog.error(f"{error_detail} | 응답: {body}")
             raise HTTPException(status_code=502, detail=error_detail)

    except HTTPException as http_exc:
        # 이미 처리된 HTTP 예외는 그대로 전달
        raise http_exc
    except Exception as e:
        # 기타 서버 내부 예외 처리
        tlog.exception(f"키움 API 직접 조회 처리 중 예외 발생: {e}")
        raise HTTPException(status_code=500, detail=f"서버 내부 오류 발생: {str(e)}")