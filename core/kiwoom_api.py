# core/kiwoom_api.py
import requests
import json
import time
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict, Literal, List
from core.config import get_settings

# 로거 설정
log = logging.getLogger("kiwoom_api")

# 설정 객체 로드 및 예외 처리
try:
    cfg = get_settings()
except ImportError:
    log.warning("core.config 모듈을 찾을 수 없어 .env 파일에서 직접 설정을 로드합니다.")
    # 실제 환경에서는 core.config 사용 필수
    import os
    from dotenv import load_dotenv
    load_dotenv() # .env 파일 로드 시도

    class MockSettings: # 임시 설정 클래스
        KIWOOM_APP_KEY: Optional[str] = os.getenv("KIWOOM_APP_KEY")
        KIWOOM_APP_SECRET: Optional[str] = os.getenv("KIWOOM_APP_SECRET")
        KIWOOM_MOCK_APP_KEY: Optional[str] = os.getenv("KIWOOM_MOCK_APP_KEY")
        KIWOOM_MOCK_APP_SECRET: Optional[str] = os.getenv("KIWOOM_MOCK_APP_SECRET")
    cfg = MockSettings()


# --- 환경별 설정 ---
# 설정 파일에서 키 값들을 가져와 ENV_CONFIG 구성
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

# --- 접근 토큰 발급 함수 ---
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

# --- 유효한 접근 토큰 가져오는 함수 ---
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

# --- API 호출 공통 헬퍼 함수 ---
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
        if response_body.get("rt_cd") != "0":
             error_msg = response_body.get("msg1", f"키움 API 처리 오류 (rt_cd: {response_body.get('rt_cd')})")
             log.error(f"키움 API 처리 오류 (ID: {api_id}, 환경: {env}, Status: {response.status_code}): {error_msg}")
             # body에 오류 내용이 있으므로 그대로 반환하되, 로그는 에러로 남김
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

# --- (신규) 종목 정보 리스트 요청 (ka10099) ---
def get_stock_list(
    market_type: Literal["0", "10"], # 0: 코스피, 10: 코스닥
    env: Literal["real", "mock"] = "real",
    cont_yn: str = 'N',
    next_key: str = ''
) -> Optional[dict]:
    """종목 정보 리스트(ka10099)를 조회합니다. (코스피 또는 코스닥)"""
    # ka10099 API에 필요한 data 파라미터 구성
    data = {
        'mrkt_tp': market_type
        # 필요 시 다른 ka10099 파라미터 추가 가능 (예: 관리종목 제외 등)
    }
    return _call_kiwoom_api(
        endpoint='/api/dostk/stkinfo', # ka10099 엔드포인트
        api_id='ka10099',
        data=data,
        env=env,
        cont_yn=cont_yn,
        next_key=next_key
    )

# --- (신규) 코스피/코스닥 종목 리스트를 합치고 시가총액으로 정렬하는 헬퍼 함수 ---
async def get_sorted_market_cap_codes(
    top_n: int = 50,
    env: Literal["real", "mock"] = "real"
) -> Tuple[List[Dict[str, str]], Optional[str]]:
    """
    코스피/코스닥 종목 중 시가총액 상위 N개를 조회하여
    순위, 종목명, 시가총액, 현재가, 전일대비, 등락률, 거래량 정보를 반환합니다.
    """
    log.info(f"시가총액 상위 {top_n}개 종목 정보 조회 시작 (환경: {env})")

    try:
        # 코스피/코스닥 동시 요청 (스레드로 병렬 처리)
        kospi_task = asyncio.to_thread(get_stock_list, market_type="0", env=env)
        kosdaq_task = asyncio.to_thread(get_stock_list, market_type="10", env=env)
        kospi_result, kosdaq_result = await asyncio.gather(kospi_task, kosdaq_task, return_exceptions=True)

        combined_list = []
        error_messages = []

        # 안전하게 리스트 추출
        def extract_list(result):
            if not result or not isinstance(result, dict):
                return []
            body = result.get("body", {})
            items = body.get("list", [])
            return items if isinstance(items, list) else []

        combined_list.extend(extract_list(kospi_result))
        combined_list.extend(extract_list(kosdaq_result))

        if not combined_list:
            return [], "코스피/코스닥 종목 데이터가 없습니다."

        # 시가총액 기준 정렬
        def get_market_cap(item):
            mac_str = item.get("mac", "0").replace(",", "")
            try:
                return int(mac_str)
            except ValueError:
                return 0

        sorted_list = sorted(combined_list, key=get_market_cap, reverse=True)

        # 상위 N개만 정제해서 반환
        top_items = []
        for rank, item in enumerate(sorted_list[:top_n], start=1):
            top_items.append({
                "rank": rank,
                "code": item.get("code", ""),
                "name": item.get("name", ""),
                "market": item.get("marketName", ""),
                "market_cap": item.get("mac", "0"),
                "price": item.get("lastPrice", "0"),
                "change_price": item.get("prdy_vrss", item.get("diff", "0")),  # 전일대비 가격
                "change_rate": item.get("prdy_ctrt", item.get("rate", "0")),   # 등락률
                "volume": item.get("trd_qty", item.get("listCount", "0"))      # 거래량(필드명 환경마다 다름)
            })

        log.info(f"시가총액 상위 {len(top_items)}개 종목 정보 정제 완료 (환경: {env})")
        return top_items, None

    except Exception as e:
        log.exception(f"시가총액 정렬 중 오류 발생: {e}")
        return [], f"서버 오류 발생: {str(e)}"

# --- 당일 거래량 상위 종목 조회 함수 (ka10030) ---
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

# --- (신규) 국내주식 시간별 투자자매매 현황 상세 요청 (ka00198) ---
def get_investor_trading_details(
    data: dict,
    env: Literal["real", "mock"] = "mock",
    cont_yn: str = 'N',
    next_key: str = ''
) -> Optional[dict]:
    """국내주식 시간별 투자자매매 현황 상세(ka00198)를 조회합니다."""
    if env == "real":
        log.warning("ka00198 API는 현재 모의투자(mock) 환경에서만 테스트되었습니다. 실전(real) 호출 시 동작을 보장할 수 없습니다.")
        # 필요 시 실전 호출 차단:
        # return {"status_code": 400, "error": "ka00198 is only supported in mock environment", "body": None}

    return _call_kiwoom_api(
        endpoint='/api/dostk/stkinfo', # API 문서 기준 엔드포인트
        api_id='ka00198',
        data=data,
        env=env, # 전달받은 env 사용
        cont_yn=cont_yn,
        next_key=next_key
    )

# --- 모듈 단독 실행 시 테스트 코드 ---
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

    log.info("--- 키움 API 테스트 종료 ---")