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

# --- 기획서에 맞춘 풍부한 샘플 거래 데이터 ---
# category_name, category_type, merchant 필드 포함
sample_transactions = [
    {"category_name": "쇼핑", "category_type": "변동비", "amount": 125000, "date": "2025-09-05", "merchant": "쿠팡"},
    {"category_name": "배달", "category_type": "변동비", "amount": 28000, "date": "2025-09-08", "merchant": "배달의민족"},
    {"category_name": "교통", "category_type": "고정비", "amount": 15000, "date": "2025-09-10", "merchant": "코레일"},
    {"category_name": "카페/간식", "category_type": "변동비", "amount": 7500, "date": "2025-09-12", "merchant": "스타벅스"},
    {"category_name": "쇼핑", "category_type": "변동비", "amount": 49000, "date": "2025-09-18", "merchant": "무신사"},
    {"category_name": "문화/여가", "category_type": "변동비", "amount": 22000, "date": "2025-09-21", "merchant": "CGV"},
    # 정기 결제 테스트를 위한 데이터
    {"category_name": "OTT/음악", "category_type": "고정비", "amount": 17000, "date": "2025-09-05", "merchant": "넷플릭스"},
    {"category_name": "OTT/음악", "category_type": "고정비", "amount": 11500, "date": "2025-09-15", "merchant": "멜론"},
]

def run_all_tests():
    """모든 AI 기능에 대한 테스트를 순차적으로 실행합니다."""
    try:
        # --- [Test 1] 지출 분석 기능 테스트 ---
        payload1 = { "question": "9월 소비 내역을 상세히 분석하고 요약해줘.", "items": sample_transactions }
        response1 = requests.post(f"{BASE_URL}{ENDPOINT}", json=payload1, timeout=60)
        print_response("[Test 1] 지출 분석 요청", response1)

        # --- [Test 2] 챗봇 기능 테스트 (구조화된 답변) ---
        payload2 = { "question": "배달 음식을 너무 많이 시켜 먹는 것 같은데, 줄일 수 있는 좋은 방법 없을까?", "items": sample_transactions }
        response2 = requests.post(f"{BASE_URL}{ENDPOINT}", json=payload2, timeout=60)
        print_response("[Test 2] 챗봇 질문 요청", response2)

        # --- [Test 3] 퀴즈 기능 테스트 ---
        payload3 = { "question": "절약 퀴즈 3개만 내줘!", "items": [] }
        response3 = requests.post(f"{BASE_URL}{ENDPOINT}", json=payload3, timeout=60)
        print_response("[Test 3] 절약 퀴즈 요청", response3)
        
        # --- [Test 4] 정기 결제 분석 기능 테스트 ---
        payload4 = { "question": "내 정기결제 내역에서 중복된 거 없는지 분석해줘", "items": sample_transactions }
        response4 = requests.post(f"{BASE_URL}{ENDPOINT}", json=payload4, timeout=60)
        print_response("[Test 4] 정기 결제 분석 요청", response4)

        # --- [Test 5] 대시보드 브리핑 기능 테스트 ---
        payload5 = { "question": "오늘의 대시보드 브리핑을 만들어줘", "items": sample_transactions }
        response5 = requests.post(f"{BASE_URL}{ENDPOINT}", json=payload5, timeout=60)
        print_response("[Test 5] 대시보드 브리핑 요청", response5)

    except requests.exceptions.ConnectionError:
        print(f"🚨 에러: AI 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인하세요.")
        print(f"   (uvicorn ai_service:app --reload)")
    except requests.exceptions.Timeout:
        print(f"🚨 에러: AI 서버로부터 응답 시간 초과 (Timeout). 서버가 정상 작동 중인지 확인하세요.")
    except Exception as e:
        print(f"🚨 예외 발생: {e}")

if __name__ == "__main__":
    run_all_tests()