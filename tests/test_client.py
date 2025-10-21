# 
import requests
import json

BASE_URL = "http://localhost:8000"
ENDPOINT = "/llm-request"

def print_response(title, response):
    """응답 결과를 예쁘게 출력하는 함수"""
    print(f"🚀 {title}")
    print(f"Status Code: {response.status_code}")
    try:
        print("Response JSON:")
        print(json.dumps(response.json(), indent=2, ensure_ascii=False))
    except json.JSONDecodeError:
        print("Response Text:")
        print(response.text)
    print("-" * 50)

# --- 테스트에 사용할 샘플 거래 데이터 ---
sample_transactions = [
    {"category": "쇼핑", "amount": 125000, "date": "2025-09-05"},
    {"category": "배달", "amount": 28000, "date": "2025-09-08"},
    {"category": "교통", "amount": 15000, "date": "2025-09-10"},
    {"category": "카페/간식", "amount": 7500, "date": "2025-09-12"},
    {"category": "쇼핑", "amount": 49000, "date": "2025-09-18"},
    {"category": "문화/여가", "amount": 22000, "date": "2025-09-21"},
    {"category": "배달", "amount": 19000, "date": "2025-09-25"},
    # 정기 결제 테스트를 위한 데이터
    {"category": "OTT/음악", "amount": 17000, "date": "2025-09-05", "merchant_name": "넷플릭스"},
    {"category": "OTT/음악", "amount": 10900, "date": "2025-09-10", "merchant_name": "유튜브 프리미엄"},
    {"category": "OTT/음악", "amount": 11500, "date": "2025-09-15", "merchant_name": "멜론"},
]

def run_tests():
    try:
        # --- [Test 1] 지출 분석 기능 테스트 ---
        analysis_payload = { "question": "9월 소비 내역을 분석하고 요약해줘.", "items": sample_transactions }
        response1 = requests.post(f"{BASE_URL}{ENDPOINT}", json=analysis_payload)
        print_response("[Test 1] 지출 분석 요청", response1)

        # --- [Test 2] 챗봇 기능 테스트 ---
        chat_payload = { "question": "배달 음식을 너무 많이 시켜 먹는 것 같은데, 줄일 수 있는 좋은 방법 없을까?", "items": sample_transactions }
        response2 = requests.post(f"{BASE_URL}{ENDPOINT}", json=chat_payload)
        print_response("[Test 2] 챗봇 질문 요청", response2)

        # --- [Test 3] 퀴즈 기능 테스트 ---
        quiz_payload = { "question": "절약 퀴즈 3개만 내줘!", "items": [] }
        response3 = requests.post(f"{BASE_URL}{ENDPOINT}", json=quiz_payload)
        print_response("[Test 3] 절약 퀴즈 요청", response3)
        
        # --- [Test 4] 정기 결제 분석 기능 테스트 ---
        #subscription_payload = { "question": "내 정기결제 내역 좀 분석해줘", "items": sample_transactions }
        #response4 = requests.post(f"{BASE_URL}{ENDPOINT}", json=subscription_payload)
        #print_response("[Test 4] 정기 결제 분석 요청", response4)

    except requests.exceptions.ConnectionError:
        print(f"🚨 에러: AI 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인하세요.")
        print(f"   (uvicorn ai_service:app --reload)")
    except Exception as e:
        print(f"🚨 예외 발생: {e}")

if __name__ == "__main__":
    run_tests()