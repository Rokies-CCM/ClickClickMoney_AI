# core/kiwoom_api.py
import requests
import json
import time
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict, Literal, List
from core.config import get_settings

# 로거 설정
log = logging.getLogger("kiwoom_api")

# --- 설정 로드 (MockSettings 임시 클래스 제거) ---
try:
    cfg = get_settings()
except ImportError:
    log.warning("core.config 모듈을 찾을 수 없어 .env 파일에서 직접 설정을 로드합니다.")
    import os
    from dotenv import load_dotenv
    load_dotenv()
    
    # .env에서 직접 로드하는 간단한 객체
    class Cfg:
        KIWOOM_APP_KEY = os.getenv("KIWOOM_APP_KEY")
        KIWOOM_APP_SECRET = os.getenv("KIWOOM_APP_SECRET")
    cfg = Cfg()


# --- [수정] 실전투자 전용 설정 (상수) ---
KIWOOM_HOST = "https://api.kiwoom.com"
KIWOOM_APP_KEY = cfg.KIWOOM_APP_KEY
KIWOOM_APP_SECRET = cfg.KIWOOM_APP_SECRET

# --- [수정] 단순화된 토큰 캐시 ---
_access_token: Optional[str] = None
_token_expires_at: Optional[datetime] = None

# --- [수정] 접근 토큰 발급 함수 (env 파라미터 제거) ---
def _issue_access_token() -> Optional[Tuple[str, datetime]]:
    """키움증권 실전투자 API 접근 토큰(au10001)을 발급받습니다."""

    if not KIWOOM_APP_KEY or not KIWOOM_APP_SECRET:
        log.error("실전투자 환경의 AppKey 또는 AppSecret이 .env 파일 또는 설정에 없습니다.")
        return None

    endpoint = '/oauth2/token'
    url = KIWOOM_HOST + endpoint
    headers = { 'Content-Type': 'application/json;charset=UTF-8' }
    data = {
        'grant_type': 'client_credentials',
        'appkey': KIWOOM_APP_KEY,
        'secretkey': KIWOOM_APP_SECRET,
    }

    log.info(f"키움증권 접근 토큰 발급 요청 시작 (URL: {url})")
    response = None # 초기화
    try:
        response = requests.post(url, headers=headers, json=data, timeout=10)
        response.raise_for_status()
        result = response.json()

        token = result.get("token")
        expires_dt_str = result.get("expires_dt") # "YYYYMMDDHHMMSS" 형식

        if not token or not expires_dt_str:
             if result.get("return_code") != 0:
                 error_msg = result.get("return_msg", "알 수 없는 토큰 발급 오류")
                 log.error(f"키움 토큰 발급 실패: {error_msg} (Code: {result.get('return_code')})")
                 return None
             else:
                 log.error(f"토큰 발급 응답에서 'token' 또는 'expires_dt' 필드 누락. 응답: {result}")
                 return None

        try:
            expires_at = datetime.strptime(expires_dt_str, "%Y%m%d%H%M%S") - timedelta(minutes=1)
        except ValueError:
             log.error(f"만료 시간 형식({expires_dt_str})을 파싱할 수 없습니다.")
             return None

        log.info(f"키움증권 접근 토큰 발급 성공 (만료 예정: {expires_at})")
        return token, expires_at

    except requests.exceptions.Timeout:
        log.error(f"키움증권 토큰 발급 API 호출 시간 초과 (URL: {url})")
        return None
    except requests.exceptions.RequestException as e:
        log.error(f"키움증권 토큰 발급 API 호출 오류: {e}")
        try:
            status_code = response.status_code if response else "N/A"
            error_details = response.text if response else "No response body"
            log.error(f"HTTP 상태 코드: {status_code}, 오류 응답 내용: {error_details}")
        except: pass
        return None
    except Exception as e:
        log.exception(f"토큰 발급 처리 중 예상치 못한 오류 발생: {e}")
        return None

# --- [수정] 유효한 접근 토큰 가져오는 함수 (env 파라미터 제거) ---
def _get_valid_access_token() -> Optional[str]:
    """유효한 실전투자 접근 토큰을 반환 (캐싱 및 자동 재발급)."""
    global _access_token, _token_expires_at

    now = datetime.now()
    
    # 캐시된 토큰이 유효하면 반환
    if _access_token and _token_expires_at and now < _token_expires_at:
        log.debug("캐시된 접근 토큰 사용")
        return _access_token

    # 토큰 새로 발급
    log.info("키움증권 접근 토큰을 새로 발급합니다...")
    new_token_info = _issue_access_token()
    if new_token_info:
        _access_token, _token_expires_at = new_token_info
        return _access_token
    else:
        log.error("새 접근 토큰 발급에 실패했습니다.")
        _access_token = None
        _token_expires_at = None
        return None

# --- [수정] API 호출 공통 헬퍼 함수 (env 파라미터 제거) ---
def _call_kiwoom_api(
    endpoint: str,
    api_id: str,
    data: dict,
    cont_yn: str = 'N',
    next_key: str = ''
) -> Optional[dict]:
    """지정된 엔드포인트로 키움 API를 호출하고 결과를 반환합니다."""
    
    token = _get_valid_access_token()
    if not token:
        log.error(f"{api_id} API 호출 실패: 유효한 접근 토큰을 얻을 수 없습니다.")
        return {"status_code": 401, "error": "Failed to get valid access token", "body": None}

    url = KIWOOM_HOST + endpoint

    headers = {
        'Content-Type': 'application/json;charset=UTF-8',
        'authorization': f'Bearer {token}',
        'cont-yn': cont_yn,
        'next-key': next_key,
        'api-id': api_id,
    }

    log.info(f"키움 API 호출 시작 (ID: {api_id}, URL: {url})")
    response = None
    try:
        response = requests.post(url, headers=headers, json=data, timeout=15)
        response.raise_for_status()

        response_body = response.json()
        result = {
            "status_code": response.status_code,
            "headers": {k: response.headers.get(k) for k in ['next-key', 'cont-yn', 'api-id'] if response.headers.get(k)},
            "body": response_body
        }
        
        # [수정] rt_cd 또는 return_code 확인 로직
        rt_cd = response_body.get("rt_cd")
        return_code = response_body.get("return_code")
        is_success = (rt_cd == "0") or (str(return_code) == "0")

        if not is_success:
             code = rt_cd if rt_cd is not None else return_code
             error_msg = response_body.get("msg1", f"키움 API 처리 오류 (rt_cd/return_code: {code})")
             log.error(f"키움 API 처리 오류 (ID: {api_id}, Status: {response.status_code}): {error_msg}")
        else:
             log.info(f"키움 API 호출 성공 (ID: {api_id}, Status: {result['status_code']})")

        return result

    except requests.exceptions.Timeout:
        log.error(f"키움 API 호출 시간 초과 (ID: {api_id}, URL: {url})")
        return {"status_code": 408, "error": "Request Timeout", "body": None}
    except requests.exceptions.RequestException as e:
        status_code = response.status_code if response else 500
        error_body = None
        error_text = str(e)
        try:
            error_body = response.json() if response else None
            log.error(f"키움 API 호출 오류 (ID: {api_id}, Status: {status_code}): {e} | 응답 본문: {error_body}")
            if isinstance(error_body, dict):
                 error_text = error_body.get("error_description", error_body.get("message", str(e)))
        except:
            error_text = response.text if response else str(e)
            log.error(f"키움 API 호출 오류 (ID: {api_id}, Status: {status_code}): {e} | 응답 텍스트: {error_text}")

        if status_code == 401:
             log.warning("접근 토큰 만료 또는 무효 감지. 토큰 캐시를 초기화합니다.")
             _access_token = None
             _token_expires_at = None

        return {"status_code": status_code, "error": error_text, "body": error_body}
    except Exception as e:
        log.exception(f"API({api_id}) 처리 중 예상치 못한 오류 발생: {e}")
        return {"status_code": 500, "error": f"Unexpected server error: {str(e)}", "body": None}

# --- [수정] 종목 정보 리스트 요청 (ka10099) (env 제거) ---
def get_stock_list(
    market_type: Literal["0", "10"],
    cont_yn: str = 'N',
    next_key: str = ''
) -> Optional[dict]:
    """종목 정보 리스트(ka10099)를 조회합니다. (코스피 또는 코스닥)"""
    data = { 'mrkt_tp': market_type }
    
    # [수정] 실전투자용 엔드포인트로 고정
    endpoint = '/api/dostk/stkinfo'
    api_id = 'ka10099'

    return _call_kiwoom_api(
        endpoint=endpoint,
        api_id=api_id,
        data=data,
        cont_yn=cont_yn,
        next_key=next_key
    )

# --- [수정] 단일 종목 상세 조회 (ka10001) (env 제거) ---
def get_stock_detail(
    code: str,
    cont_yn: str = 'N',
    next_key: str = ''
) -> Optional[dict]:
    """단일 종목 상세 조회(ka10001)를 호출합니다."""
    data = {'stk_cd': code} # ka10001 파라미터
    return _call_kiwoom_api(
        endpoint='/api/dostk/stkinfo', # API 문서 기준
        api_id='ka10001',
        data=data,
        cont_yn=cont_yn,
        next_key=next_key
    )

# --- [수정] 시가총액 상위 종목 조회 (env 및 mock 로직 제거) ---
def get_sorted_market_cap_codes(
    top_n: int = 50,
) -> Tuple[List[Dict], Optional[str]]:
    """
    (리팩토링) 시가총액 상위 종목 리스트와 상세 정보를 조회합니다. (실전투자 전용)
    """
    log.info(f"시가총액 상위 {top_n}개 종목 조회 시작")

    # 1️⃣ 시총 리스트 조회 (ka10099) - 코스피('0') 고정
    list_response = get_stock_list(market_type="0") 

    if not list_response or list_response.get("status_code") != 200:
        err = list_response.get('error', 'API Error')
        log.error(f"[real] 종목 목록 조회 실패 (Status: {list_response.get('status_code')}): {err}")
        return [], f"종목 목록 조회 실패: {err}"
    
    body = list_response.get("body")
    
    rt_cd_list = body.get("rt_cd")
    return_code_list = body.get("return_code")
    is_list_success = (rt_cd_list == "0") or (str(return_code_list) == "0")

    if not body or not is_list_success:
        code = rt_cd_list if rt_cd_list is not None else return_code_list
        msg = body.get('msg1', f'API Error (Code: {code})')
        log.error(f"[real] 종목 목록 API 오류 (rt_cd/return_code: {code}): {msg}")
        return [], f"종목 목록 API 오류: {msg}"

    base_list = body.get("list", [])
    if not base_list:
        return [], "종목 데이터 없음"
    
    def get_calculated_market_cap(item):
        """
        [신규] ka10099 응답에서 시가총액을 계산합니다.
        시가총액 = 전일종가(lastPrice) * 상장주식수(listCount)
        """
        # "00006360" -> "6360"
        price_str = item.get("lastPrice", "0").lstrip('0') 
        count_str = item.get("listCount", "0").lstrip('0')
        
        try:
            # 빈 문자열일 경우 0으로 처리
            price = int(price_str) if price_str else 0 
            count = int(count_str) if count_str else 0
            return price * count
        except ValueError:
            return 0
        except Exception as e:
            log.warning(f"시가총액 계산 오류 (Code: {item.get('code')}): {e}")
            return 0

    # --- 2. 리스트 정렬 ---
    # [수정] mock 환경 ETF 필터링 로직 제거
    
    def get_market_cap(item):
        mac_str = item.get("mac", "0").replace(",", "")
        try:
            return int(mac_str)
        except ValueError:
            return 0

    for item in base_list:
        item["_calc_market_cap"] = get_calculated_market_cap(item) # <- 정의한 함수를 바로 사용

    # [신규] 리스트의 모든 항목에 대해 계산된 시가총액을 새 키에 저장
    for item in base_list:
        item["_calc_market_cap"] = get_calculated_market_cap(item)

    # [수정] 'mac' 대신 계산된 '_calc_market_cap'으로 정렬
    sorted_list = sorted(base_list, key=lambda x: x["_calc_market_cap"], reverse=True)
    top_stocks_list = sorted_list[:top_n]

    # --- 3. 상세정보 병렬 조회 (ThreadPoolExecutor 사용) ---
    def fetch_detail_sync(stock_item: Dict) -> Optional[Dict]:
        """(Sync) 단일 종목 상세 정보를 가져오는 헬퍼 함수"""
        code = stock_item.get("code")
        if not code:
            return None
        
        try:
            detail_res = get_stock_detail(code=code)
            
            if not detail_res or detail_res.get("status_code") != 200:
                log.warning(f"[real] 상세 조회 실패 ({code}): {detail_res.get('error')}")
                return None

            detail_body = detail_res.get("body")

            rt_cd_detail = detail_body.get("rt_cd")
            return_code_detail = detail_body.get("return_code")
            is_detail_success = (rt_cd_detail == "0") or (str(return_code_detail) == "0")

            if not detail_body or not is_detail_success:
                code_val = rt_cd_detail if rt_cd_detail is not None else return_code_detail
                log.warning(f"[real] 상세 조회 API 오류 ({code}): {detail_body.get('msg1')} (Code: {code_val})")
                return None
            
            output = detail_body.get("output", {})
            
            # --- [핵심 수정] ---
            # 1. ka10001 (상세)의 'mrkt_val' (실시간 시총)을 우선 사용
            market_cap_real = output.get("mrkt_val", "0").lstrip('0')
            
            if not market_cap_real or market_cap_real == "0":
                # 2. 1번이 없으면, ka10099 (목록)에서 계산한 '_calc_market_cap'을 사용
                market_cap = str(stock_item.get("_calc_market_cap", "0"))
            else:
                market_cap = market_cap_real # 1번 값 사용

            # 3. 다른 필드들도 상세 API(output) 값이 0이면 목록(stock_item) 값으로 대체
            price = output.get("stck_prpr", "0").lstrip('0')
            if not price or price == "0":
                price = stock_item.get("lastPrice", "0") # 전일종가로 대체

            change_price = output.get("prdy_vrss", "0")
            if change_price == "0":
                change_price = stock_item.get("diff", "0") 

            change_rate = output.get("prdy_ctrt", "0")
            if change_rate == "0":
                change_rate = stock_item.get("rate", "0") 

            volume = output.get("acml_vol", "0")
            if volume == "0":
                volume = stock_item.get("trd_qty", "0") 
            # --- [여기까지 핵심 수정] ---
            
            return {
                "code": code,
                "name": output.get("stk_nm", stock_item.get("name", "")),
                "market": output.get("mrkt_nm", stock_item.get("marketName", "")),
                "market_cap": market_cap,
                "price": price,
                "change_price": change_price,
                "change_rate": change_rate,
                "volume": volume,
            }
        except Exception as e:
            log.error(f"[real] 상세조회 태스크 실패 ({code}): {e}", exc_info=True)
            return None

    results = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(fetch_detail_sync, top_stocks_list))

    # --- 4. 최종 결과 정제 ---
    valid_items = [r for r in results if r]
    
    def get_final_market_cap(item):
        try: return int(item["market_cap"].replace(",", "") or 0)
        except ValueError: return 0

    valid_items.sort(key=get_final_market_cap, reverse=True)

    top_items = [
        dict(item, rank=i + 1)
        for i, item in enumerate(valid_items)
    ]

    log.info(f"[REAL] 시가총액 상위 {len(top_items)}개 종목 완료")
    return top_items, None


# --- [수정] 당일 거래량 상위 종목 조회 함수 (ka10030) (env 제거) ---
def get_top_volume_stocks(
    data: dict,
    cont_yn: str = 'N',
    next_key: str = ''
) -> Optional[dict]:
    """당일 거래량 상위 종목(ka10030)을 조회합니다."""
    return _call_kiwoom_api(
        endpoint='/api/dostk/rkinfo',
        api_id='ka10030',
        data=data,
        cont_yn=cont_yn,
        next_key=next_key
    )

# --- [삭제] get_investor_trading_details (모의투자 전용이었으므로 제거) ---
# (필요시 살릴 수 있으나, 모의투자 전용이라 주석 처리)
# def get_investor_trading_details( ... )


# --- [수정] 모듈 단독 실행 시 테스트 코드 (mock 테스트 제거) ---
if __name__ == '__main__':
    try:
        from dotenv import load_dotenv
        load_dotenv()
        logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
        import os
        # .env 값으로 상수 업데이트 (테스트 시)
        KIWOOM_APP_KEY = os.getenv("KIWOOM_APP_KEY")
        KIWOOM_APP_SECRET = os.getenv("KIWOOM_APP_SECRET")
        log.info(".env 파일 로드 및 설정 업데이트 완료.")
    except Exception as e:
        log.error(f"테스트 환경 설정 로드 중 오류: {e}", exc_info=True)

    log.info("--- 키움 API (실전 전용) 테스트 시작 ---")

    # --- 실전투자 API 테스트 ---
    print("\n--- 실전투자: 거래량 상위 조회 (ka10030) ---")
    # [수정] stex_tp: '3' (실전) 고정
    real_params_ka10030 = { 'mrkt_tp': '001', 'sort_tp': '1', 'mang_stk_incls': '0', 'crd_tp': '0', 'trde_qty_tp': '0', 'pric_tp': '0', 'trde_prica_tp': '0', 'mrkt_open_tp': '0', 'stex_tp': '3' }
    real_result_ka10030 = get_top_volume_stocks(data=real_params_ka10030)
    if real_result_ka10030:
        print(json.dumps(real_result_ka10030, indent=4, ensure_ascii=False))
    else:
        print("실전투자 ka10030 조회 실패 (API 호출 실패 또는 토큰 발급 실패)")

    # --- 시가총액 상위 테스트 ---
    print("\n--- 실전투자: 시가총액 상위 5개 조회 ---")
    top_5_real, err_real = get_sorted_market_cap_codes(top_n=5)
    if err_real:
        print(f"실전투자 시총 조회 실패: {err_real}")
    else:
        print(json.dumps(top_5_real, indent=4, ensure_ascii=False))

    log.info("--- 키움 API 테스트 종료 ---")