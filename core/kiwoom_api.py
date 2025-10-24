# core/kiwoom_api.py
import requests
import json
import time
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor # ← ThreadPoolExecutor만 사용
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict, Literal, List
from core.config import get_settings

# --- (설정 로드 및 ENV_CONFIG, 토큰 캐시 부분은 기존과 동일) ---
# 로거 설정
log = logging.getLogger("kiwoom_api")

# 설정 객체 로드 및 예외 처리
try:
    cfg = get_settings()
except ImportError:
    log.warning("core.config 모듈을 찾을 수 없어 .env 파일에서 직접 설정을 로드합니다.")
    import os
    from dotenv import load_dotenv
    load_dotenv() 

    class MockSettings: # 임시 설정 클래스
        KIWOOM_APP_KEY: Optional[str] = os.getenv("KIWOOM_APP_KEY")
        KIWOOM_APP_SECRET: Optional[str] = os.getenv("KIWOOM_APP_SECRET")
        KIWOOM_MOCK_APP_KEY: Optional[str] = os.getenv("KIWOOM_MOCK_APP_KEY")
        KIWOOM_MOCK_APP_SECRET: Optional[str] = os.getenv("KIWOOM_MOCK_APP_SECRET")
    cfg = MockSettings()


# --- 환경별 설정 ---
ENV_CONFIG = {
    "real": {
        "host": "https://api.kiwoom.com",
        "appkey": cfg.KIWOOM_APP_KEY,
        "secretkey": cfg.KIWOOM_APP_SECRET
    },
    "mock": {
        "host": "https://mockapi.kiwoom.com",
        "appkey": cfg.KIWOOM_MOCK_APP_KEY,
        "secretkey": cfg.KIWOOM_MOCK_APP_SECRET
    }
}

# --- 환경별 토큰 캐시 ---
_access_tokens: Dict[Literal["real", "mock"], Optional[str]] = {"real": None, "mock": None}
_token_expires_at: Dict[Literal["real", "mock"], Optional[datetime]] = {"real": None, "mock": None}

# --- 접근 토큰 발급 함수 (기존과 동일) ---
def _issue_access_token(env: Literal["real", "mock"] = "real") -> Optional[Tuple[str, datetime]]:
    """선택된 환경(실전/모의)에 대한 키움증권 API 접근 토큰(au10001)을 발급받습니다."""
    config = ENV_CONFIG.get(env)
    if not config:
        log.error(f"잘못된 환경 지정: {env}")
        return None

    appkey = config.get("appkey")
    secretkey = config.get("secretkey")
    host = config.get("host")

    if not appkey or not secretkey:
        log.error(f"{env} 환경의 AppKey 또는 AppSecret이 .env 파일 또는 설정에 없습니다.")
        return None
    if not host:
         log.error(f"{env} 환경의 API 호스트 주소가 설정되지 않았습니다.")
         return None

    endpoint = '/oauth2/token'
    url = host + endpoint
    headers = { 'Content-Type': 'application/json;charset=UTF-8' }
    data = {
        'grant_type': 'client_credentials',
        'appkey': appkey,
        'secretkey': secretkey,
    }

    log.info(f"키움증권 접근 토큰 발급 요청 시작 (환경: {env}, URL: {url})")
    response = None # 초기화
    try:
        response = requests.post(url, headers=headers, json=data, timeout=10) # 10초 타임아웃
        response.raise_for_status() # 4xx, 5xx 에러 시 예외 발생
        result = response.json()

        token = result.get("token")
        expires_dt_str = result.get("expires_dt") # "YYYYMMDDHHMMSS" 형식

        if not token or not expires_dt_str:
             # 키움 자체 에러 (return_code != 0) 처리
             if result.get("return_code") != 0:
                 error_msg = result.get("return_msg", "알 수 없는 토큰 발급 오류")
                 log.error(f"키움 토큰 발급 실패 (환경: {env}): {error_msg} (Code: {result.get('return_code')})")
                 return None
             else:
                 log.error(f"토큰 발급 응답에서 'token' 또는 'expires_dt' 필드 누락 (환경: {env}). 응답: {result}")
                 return None

        # 만료 시간 계산 (API 응답 시간 기준 + 여유 시간 1분 빼기)
        try:
            expires_at = datetime.strptime(expires_dt_str, "%Y%m%d%H%M%S") - timedelta(minutes=1)
        except ValueError:
             log.error(f"만료 시간 형식({expires_dt_str})을 파싱할 수 없습니다 (환경: {env}).")
             return None

        log.info(f"키움증권 접근 토큰 발급 성공 (환경: {env}, 만료 예정: {expires_at})")
        return token, expires_at

    except requests.exceptions.Timeout:
        log.error(f"키움증권 토큰 발급 API 호출 시간 초과 (환경: {env}, URL: {url})")
        return None
    except requests.exceptions.RequestException as e:
        log.error(f"키움증권 토큰 발급 API 호출 오류 (환경: {env}): {e}")
        try:
            # 오류 응답 내용 로깅 시도
            status_code = response.status_code if response else "N/A"
            error_details = response.text if response else "No response body"
            log.error(f"HTTP 상태 코드: {status_code}, 오류 응답 내용: {error_details}")
        except: pass
        return None
    except Exception as e:
        log.exception(f"토큰 발급 처리 중 예상치 못한 오류 발생 (환경: {env}): {e}")
        return None

# --- 유효한 접근 토큰 가져오는 함수 (기존과 동일) ---
def _get_valid_access_token(env: Literal["real", "mock"] = "real") -> Optional[str]:
    """선택된 환경에 대한 유효한 접근 토큰을 반환 (캐싱 및 자동 재발급)."""
    global _access_tokens, _token_expires_at

    now = datetime.now()
    current_token = _access_tokens.get(env)
    current_expiry = _token_expires_at.get(env)

    # 캐시된 토큰이 유효하면 반환
    if current_token and current_expiry and now < current_expiry:
        log.debug(f"캐시된 접근 토큰 사용 (환경: {env})")
        return current_token

    # 토큰 새로 발급
    log.info(f"키움증권 접근 토큰 ({env})을 새로 발급합니다...")
    new_token_info = _issue_access_token(env)
    if new_token_info:
        _access_tokens[env], _token_expires_at[env] = new_token_info
        return _access_tokens[env]
    else:
        # 발급 실패 시 캐시 초기화
        log.error(f"새 접근 토큰 ({env}) 발급에 실패했습니다.")
        _access_tokens[env] = None
        _token_expires_at[env] = None
        return None

# --- API 호출 공통 헬퍼 함수 (기존과 동일) ---
def _call_kiwoom_api(
    endpoint: str,
    api_id: str,
    data: dict,
    env: Literal["real", "mock"] = "real",
    cont_yn: str = 'N',
    next_key: str = ''
) -> Optional[dict]:
    """지정된 환경과 엔드포인트로 키움 API를 호출하고 결과를 반환합니다."""
    # 1. 유효 토큰 가져오기 (실패 시 None 반환)
    token = _get_valid_access_token(env)
    if not token:
        log.error(f"{api_id} API 호출 실패: 유효한 접근 토큰({env})을 얻을 수 없습니다.")
        # 실패 시 명확한 오류 구조 반환
        return {"status_code": 401, "error": f"Failed to get valid access token for env '{env}'", "body": None}

    # 2. 환경 설정 확인
    config = ENV_CONFIG.get(env)
    if not config or not config.get("host"):
        log.error(f"잘못된 환경 지정 또는 호스트 누락: {env}")
        return {"status_code": 500, "error": f"Invalid environment configuration for '{env}'", "body": None}
    host = config["host"]
    url = host + endpoint

    # 3. 헤더 구성
    headers = {
        'Content-Type': 'application/json;charset=UTF-8',
        'authorization': f'Bearer {token}',
        'cont-yn': cont_yn,
        'next-key': next_key,
        'api-id': api_id,
    }

    log.info(f"키움 API 호출 시작 (ID: {api_id}, 환경: {env}, URL: {url})")
    response = None # 초기화
    try:
        # 4. API 호출 (POST)
        response = requests.post(url, headers=headers, json=data, timeout=15) # 15초 타임아웃
        response.raise_for_status() # HTTP 4xx, 5xx 에러 시 예외 발생

        # 5. 성공 응답 처리
        response_body = response.json()
        result = {
            "status_code": response.status_code,
            "headers": {k: response.headers.get(k) for k in ['next-key', 'cont-yn', 'api-id'] if response.headers.get(k)},
            "body": response_body
        }
        
        # 키움 자체 에러 코드 확인 (rt_cd != "0")
        # rt_cd, return_code 둘 다 확인
        rt_cd = response_body.get("rt_cd")
        return_code = response_body.get("return_code") # (string or int, e.g., "0" or 0)

        # 성공 판정: rt_cd가 "0"이거나, return_code가 "0" 또는 0인 경우
        # str()로 감싸서 "0" == "0" 또는 0 == "0" (str(0)) 모두 처리
        is_success = (rt_cd == "0") or (str(return_code) == "0")

        if not is_success:
             # 에러 메시지 추출
             code = rt_cd if rt_cd is not None else return_code
             error_msg = response_body.get("msg1", f"키움 API 처리 오류 (rt_cd/return_code: {code})")
             log.error(f"키움 API 처리 오류 (ID: {api_id}, 환경: {env}, Status: {response.status_code}): {error_msg}")
        else:
             log.info(f"키움 API 호출 성공 (ID: {api_id}, 환경: {env}, Status: {result['status_code']})")

        return result

    except requests.exceptions.Timeout:
        log.error(f"키움 API 호출 시간 초과 (ID: {api_id}, 환경: {env}, URL: {url})")
        return {"status_code": 408, "error": "Request Timeout", "body": None}
    except requests.exceptions.RequestException as e:
        status_code = response.status_code if response else 500
        error_body = None
        error_text = str(e)
        try: # JSON 오류 응답 파싱 시도
            error_body = response.json() if response else None
            log.error(f"키움 API 호출 오류 (ID: {api_id}, 환경: {env}, Status: {status_code}): {e} | 응답 본문: {error_body}")
            if isinstance(error_body, dict): # 오류 메시지 추출 시도
                 error_text = error_body.get("error_description", error_body.get("message", str(e)))
        except: # JSON 파싱 실패 시 텍스트 응답 로깅
            error_text = response.text if response else str(e)
            log.error(f"키움 API 호출 오류 (ID: {api_id}, 환경: {env}, Status: {status_code}): {e} | 응답 텍스트: {error_text}")

        # 토큰 만료(401) 시 캐시 초기화
        if status_code == 401:
             log.warning(f"접근 토큰({env}) 만료 또는 무효 감지. 토큰 캐시를 초기화합니다.")
             _access_tokens[env] = None
             _token_expires_at[env] = None

        return {"status_code": status_code, "error": error_text, "body": error_body}
    except Exception as e:
        log.exception(f"API({api_id}, {env}) 처리 중 예상치 못한 오류 발생: {e}")
        return {"status_code": 500, "error": f"Unexpected server error: {str(e)}", "body": None}

# --- 종목 정보 리스트 요청 (ka10099) (기존과 동일) ---
def get_stock_list(
    market_type: Literal["0", "10"],
    env: Literal["real", "mock"] = "real",
    cont_yn: str = 'N',
    next_key: str = ''
) -> Optional[dict]:
    """종목 정보 리스트(ka10099)를 조회합니다. (코스피 또는 코스닥)"""
    data = { 'mrkt_tp': market_type }

    # ✅ 환경별 엔드포인트 차이 반영
    endpoint = '/api/dostk/stkinfo'
    api_id = 'ka10099'

    return _call_kiwoom_api(
        endpoint=endpoint,
        api_id=api_id,
        data=data,
        env=env,
        cont_yn=cont_yn,
        next_key=next_key
    )

# --- (신규) 단일 종목 상세 조회 (ka10001) ---
def get_stock_detail(
    code: str,
    env: Literal["real", "mock"] = "real",
    cont_yn: str = 'N',
    next_key: str = ''
) -> Optional[dict]:
    """
    (신규) 단일 종목 상세 조회(ka10001)를 _call_kiwoom_api 헬퍼를 통해 호출합니다.
    """
    data = {'stk_cd': code} # ka10001 파라미터
    return _call_kiwoom_api(
        endpoint='/api/dostk/stkinfo', # API 문서 기준
        api_id='ka10001',
        data=data,
        env=env,
        cont_yn=cont_yn,
        next_key=next_key
    )

# --- (리팩토링) 코스피/코스닥 종목 리스트를 합치고 시가총액으로 정렬하는 헬퍼 함수 ---
def get_sorted_market_cap_codes(
    top_n: int = 50,
    env: Literal["real", "mock"] = "real"
) -> Tuple[List[Dict], Optional[str]]: # ✅ async 제거, 반환 타입 수정
    """
    (리팩토링) 시가총액 상위 종목 리스트와 상세 정보를 조회합니다.
    이 함수는 동기적으로 동작하며, 내부적으로 ThreadPoolExecutor를 사용해 상세 조회를 병렬화합니다.
    """
    log.info(f"시가총액 상위 {top_n}개 종목 조회 시작 (환경: {env})")

    # 1️⃣ 시총 리스트 조회 (ka10099)
    list_response = get_stock_list(market_type="0", env=env) # ✅ await 제거

    if not list_response or list_response.get("status_code") != 200:
        err = list_response.get('error', 'API Error')
        log.error(f"[{env}] 종목 목록 조회 실패 (Status: {list_response.get('status_code')}): {err}")
        return [], f"종목 목록 조회 실패: {err}"
    
    body = list_response.get("body")
    rt_cd_list = body.get("rt_cd")
    return_code_list = body.get("return_code")
    is_list_success = (rt_cd_list == "0") or (str(return_code_list) == "0")

    if not body or not is_list_success:
        code = rt_cd_list if rt_cd_list is not None else return_code_list
        msg = body.get('msg1', f'API Error (Code: {code})')
        log.error(f"[{env}] 종목 목록 API 오류 (rt_cd/return_code: {code}): {msg}")
        return [], f"종목 목록 API 오류: {msg}"

    base_list = body.get("list", [])
    if not base_list:
        return [], "종목 데이터 없음"

    # --- 2. 리스트 필터링 및 정렬 ---
    # (모의투자 시 ETF/ETN 제외 로직 유지)
    if env == "mock":
        base_list = [
            item for item in base_list
            if not any(keyword in item.get("name", "") for keyword in ["ETF", "ETN", "TIGER", "KODEX"])
        ]

    # 시총 기준 정렬
    def get_market_cap(item):
        mac_str = item.get("mac", "0").replace(",", "")
        try:
            return int(mac_str)
        except ValueError:
            return 0

    sorted_list = sorted(base_list, key=get_market_cap, reverse=True)
    top_stocks_list = sorted_list[:top_n]

    # --- 3. 상세정보 병렬 조회 (ThreadPoolExecutor 사용) ---
    def fetch_detail_sync(stock_item: Dict) -> Optional[Dict]:
        """(Sync) 단일 종목 상세 정보를 가져오는 헬퍼 함수"""
        code = stock_item.get("code")
        if not code:
            return None
        
        try:
            # 동기 get_stock_detail 함수 호출
            detail_res = get_stock_detail(code=code, env=env)
            
            if not detail_res or detail_res.get("status_code") != 200:
                log.warning(f"[{env}] 상세 조회 실패 ({code}): {detail_res.get('error')}")
                return None

            detail_body = detail_res.get("body")

            # ✅ [수정] rt_cd 또는 return_code 확인
            rt_cd_detail = detail_body.get("rt_cd")
            return_code_detail = detail_body.get("return_code")
            is_detail_success = (rt_cd_detail == "0") or (str(return_code_detail) == "0")

            if not detail_body or not is_detail_success:
                code_val = rt_cd_detail if rt_cd_detail is not None else return_code_detail
                log.warning(f"[{env}] 상세 조회 API 오류 ({code}): {detail_body.get('msg1')} (Code: {code_val})")
                return None
            
            output = detail_body.get("output", {})
            
            # 리스트 정보와 상세 정보를 조합
            return {
                "code": code,
                "name": output.get("stk_nm", stock_item.get("name", "")),
                "market": output.get("mrkt_nm", stock_item.get("marketName", "")),
                "market_cap": output.get("mrkt_val", stock_item.get("mac", "0")),
                "price": output.get("stck_prpr", stock_item.get("lastPrice", "0")),
                "change_price": output.get("prdy_vrss", stock_item.get("diff", "0")),
                "change_rate": output.get("prdy_ctrt", stock_item.get("rate", "0")),
                "volume": output.get("acml_vol", stock_item.get("trd_qty", "0")),
            }
        except Exception as e:
            log.error(f"[{env}] 상세조회 태스크 실패 ({code}): {e}", exc_info=True)
            return None

    # ✅ ThreadPoolExecutor로 병렬 실행
    results = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        # map을 사용하여 순서를 유지하면서 병렬 실행
        results = list(executor.map(fetch_detail_sync, top_stocks_list))

    # --- 4. 최종 결과 정제 ---
    valid_items = [r for r in results if r]
    
    # 상세 조회된 정보(market_cap) 기준으로 최종 정렬
    def get_final_market_cap(item):
        try: return int(item["market_cap"].replace(",", "") or 0)
        except ValueError: return 0

    valid_items.sort(key=get_final_market_cap, reverse=True)

    # 순위(rank) 매기기
    top_items = [
        dict(item, rank=i + 1)
        for i, item in enumerate(valid_items)
    ]

    log.info(f"[{env.upper()}] 시가총액 상위 {len(top_items)}개 종목 완료")
    return top_items, None


# --- 당일 거래량 상위 종목 조회 함수 (ka10030) (기존과 동일) ---
def get_top_volume_stocks(
    data: dict,
    env: Literal["real", "mock"] = "real",
    cont_yn: str = 'N',
    next_key: str = ''
) -> Optional[dict]:
    """당일 거래량 상위 종목(ka10030)을 조회합니다."""
    return _call_kiwoom_api(
        endpoint='/api/dostk/rkinfo',
        api_id='ka10030',
        data=data,
        env=env,
        cont_yn=cont_yn,
        next_key=next_key
    )

# --- 국내주식 시간별 투자자매매 현황 상세 요청 (ka00198) (기존과 동일) ---
def get_investor_trading_details(
    data: dict,
    env: Literal["real", "mock"] = "mock",
    cont_yn: str = 'N',
    next_key: str = ''
) -> Optional[dict]:
    """국내주식 시간별 투자자매매 현황 상세(ka00198)를 조회합니다."""
    if env == "real":
        log.warning("ka00198 API는 현재 모의투자(mock) 환경에서만 테스트되었습니다. 실전(real) 호출 시 동작을 보장할 수 없습니다.")

    return _call_kiwoom_api(
        endpoint='/api/dostk/stkinfo', # API 문서 기준 엔드포인트
        api_id='ka00198',
        data=data,
        env=env, # 전달받은 env 사용
        cont_yn=cont_yn,
        next_key=next_key
    )

# --- (삭제) get_stock_list_v2, get_stock_detail_v2, get_top_marketcap_realtime ---
# 위 함수들은 get_stock_list, get_stock_detail, get_sorted_market_cap_codes로
# 기능이 통합되거나 대체되었으므로 삭제되었습니다.


# --- 모듈 단독 실행 시 테스트 코드 (기존과 동일) ---
if __name__ == '__main__':
    # .env 파일 로드 및 로깅 설정
    try:
        from dotenv import load_dotenv
        load_dotenv()
        logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
        # 설정 변수 로드 (위의 try-except 블록에서 이미 로드됨)
        # ENV_CONFIG 업데이트 (dotenv 로드 후 값으로)
        import os
        ENV_CONFIG["real"]["appkey"] = os.getenv("KIWOOM_APP_KEY")
        ENV_CONFIG["real"]["secretkey"] = os.getenv("KIWOOM_APP_SECRET")
        ENV_CONFIG["mock"]["appkey"] = os.getenv("KIWOOM_MOCK_APP_KEY")
        ENV_CONFIG["mock"]["secretkey"] = os.getenv("KIWOOM_MOCK_APP_SECRET")
        log.info(".env 파일 로드 및 설정 업데이트 완료.")
    except Exception as e:
        log.error(f"테스트 환경 설정 로드 중 오류: {e}", exc_info=True)

    log.info("--- 키움 API 테스트 시작 ---")

    # --- 실전투자 API 테스트 ---
    print("\n--- 실전투자: 거래량 상위 조회 (ka10030) ---")
    real_params_ka10030 = { 'mrkt_tp': '001', 'sort_tp': '1', 'mang_stk_incls': '0', 'crd_tp': '0', 'trde_qty_tp': '0', 'pric_tp': '0', 'trde_prica_tp': '0', 'mrkt_open_tp': '0', 'stex_tp': '3' }
    real_result_ka10030 = get_top_volume_stocks(data=real_params_ka10030, env="real")
    if real_result_ka10030:
        print(json.dumps(real_result_ka10030, indent=4, ensure_ascii=False))
    else:
        print("실전투자 ka10030 조회 실패 (API 호출 실패 또는 토큰 발급 실패)")

    # --- 모의투자 API 테스트 ---
    print("\n--- 모의투자: 거래량 상위 조회 (ka10030) ---")
    mock_params_ka10030 = { 'mrkt_tp': '001', 'sort_tp': '1', 'mang_stk_incls': '0', 'crd_tp': '0', 'trde_qty_tp': '0', 'pric_tp': '0', 'trde_prica_tp': '0', 'mrkt_open_tp': '0', 'stex_tp': '1' } # 모의투자는 KRX(1)만 지원 가능성 있음
    mock_result_ka10030 = get_top_volume_stocks(data=mock_params_ka10030, env="mock")
    if mock_result_ka10030:
        print(json.dumps(mock_result_ka10030, indent=4, ensure_ascii=False))
    else:
        print("모의투자 ka10030 조회 실패 (API 호출 실패 또는 토큰 발급 실패)")

    print("\n--- 모의투자: 투자자매매 현황 상세 (ka00198) ---")
    mock_params_ka00198 = { 'qry_tp': '1' } # 1분 단위
    mock_result_ka00198 = get_investor_trading_details(data=mock_params_ka00198, env="mock")
    if mock_result_ka00198:
        print(json.dumps(mock_result_ka00198, indent=4, ensure_ascii=False))
    else:
        print("모의투자 ka00198 조회 실패 (API 호출 실패 또는 토큰 발급 실패)")
        
    # --- (추가) 리팩토링된 시가총액 상위 테스트 ---
    print("\n--- 모의투자: 시가총액 상위 5개 조회 (리팩토링) ---")
    top_5_mock, err_mock = get_sorted_market_cap_codes(top_n=5, env="mock") 
    if err_mock:
        print(f"모의투자 시총 조회 실패: {err_mock}")
    else:
        print(json.dumps(top_5_mock, indent=4, ensure_ascii=False))

    log.info("--- 키움 API 테스트 종료 ---")