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
    # 정기 결제 테스트를 위한 데이터
    {"category": "OTT/음악", "amount": 17000, "date": "2025-09-05"},
    {"category": "OTT/음악", "amount": 11500, "date": "2025-09-15"},
]

def run_agent_tests():
    """Agent가 각 Tool을 올바르게 호출하는지 테스트합니다."""
    try:
        # --- [Test 1] analysis_tool 호출 테스트 ---
        # '분석', '요약' 키워드를 포함하여 Agent가 analysis_tool을 선택하도록 유도
        payload1 = { "question": "9월 소비 내역을 분석하고 요약해줘.", "items": sample_transactions }
        response1 = requests.post(f"{BASE_URL}{ENDPOINT}", json=payload1, timeout=60)
        print_response("[Test 1] '분석' Tool 호출 테스트", response1)

        # --- [Test 2] general_chat_tool (RAG) 호출 테스트 ---
        # 다른 Tool의 키워드를 포함하지 않는 일반적인 질문
        payload2 = { "question": "배달 음식을 너무 많이 시켜 먹는 것 같은데, 줄일 수 있는 좋은 방법 없을까?", "items": sample_transactions }
        response2 = requests.post(f"{BASE_URL}{ENDPOINT}", json=payload2, timeout=60)
        print_response("[Test 2] '일반 챗봇(RAG)' Tool 호출 테스트", response2)

        # --- [Test 3] quiz_tool 호출 테스트 ---
        # '퀴즈' 키워드를 포함하여 Agent가 quiz_tool을 선택하도록 유도
        payload3 = { "question": "절약 관련 퀴즈 하나만 내줘!", "items": [] }
        response3 = requests.post(f"{BASE_URL}{ENDPOINT}", json=payload3, timeout=60)
        print_response("[Test 3] '퀴즈' Tool 호출 테스트", response3)
        
        # --- [Test 4] subscription_analysis_tool 호출 테스트 ---
        # '정기결제' 키워드를 포함하여 Agent가 subscription_analysis_tool을 선택하도록 유도
        payload4 = { "question": "내 정기결제 내역 좀 분석해줘", "items": sample_transactions }
        response4 = requests.post(f"{BASE_URL}{ENDPOINT}", json=payload4, timeout=60)
        print_response("[Test 4] '정기 결제 분석' Tool 호출 테스트", response4)

    except requests.exceptions.ConnectionError:
        print(f"🚨 에러: AI 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인하세요.")
        print(f"   (uvicorn ai_service:app --reload)")
    except requests.exceptions.Timeout:
        print(f"🚨 에러: AI 서버로부터 응답 시간 초과 (Timeout). 서버가 정상 작동 중인지, LLM API 키가 유효한지 확인하세요.")
    except Exception as e:
        print(f"🚨 예외 발생: {e}")

if __name__ == "__main__":
    run_agent_tests()