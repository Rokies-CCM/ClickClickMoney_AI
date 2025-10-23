# server/routers/quiz.py
from __future__ import annotations
import time
import random
import uuid
import json # JSON 파싱을 위해 추가
import logging # 로깅 추가
import asyncio # 병렬 처리를 위해 추가
import re # JSON 추출을 위해 추가
from typing import Dict, List, Optional

from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel, Field

from core.models import build_openai_client
from core.prompts import QUIZ_INSTRUCTIONS
from server.deps import get_trace_logger, new_trace_id

router = APIRouter()
log = logging.getLogger("quiz")

# --- AI 클라이언트 초기화 ---
# OpenAI 클라이언트를 빌드합니다. 실패 시 None으로 설정하고 에러 로깅.
try:
    ai_client = build_openai_client() #
    if not ai_client:
        log.error("AI Client 초기화 실패: build_openai_client()가 None을 반환했습니다.")
except Exception as e:
    log.error(f"AI Client 초기화 중 예외 발생: {e}")
    ai_client = None

# ---------------- Models ----------------
class QuizNextPayload(BaseModel):
    difficulty: Optional[str] = Field(default="easy", description="easy|medium|hard")
    tag: Optional[str] = Field(default=None, description="ex) savings|points|fuel|ledger|subscription|energy|transport|meal")

class QuizNextResponse(BaseModel):
    question_id: str
    difficulty: str
    tag: str
    question: str
    options: Dict[str, str]
    correct_key: str
    explanation: str
    points_base: int

# Batch API에서 반환될 개별 퀴즈 항목 모델
class QuizItem(BaseModel):
    question_id: str
    difficulty: str
    tag: str
    question: str
    options: Dict[str, str]
    correct_key: str
    explanation: str
    points_base: int

# Batch API의 최종 응답 모델 (퀴즈 목록 포함)
class QuizBatchResponse(BaseModel):
    quizzes: List[QuizItem]

class QuizAnswerPayload(BaseModel):
    question_id: str
    user_answer: str  # A|B|C|D

class QuizAnswerResponse(BaseModel):
    is_correct: bool
    correct_key: str
    explanation: str
    points_awarded: int
    streak: int

# ---------------- In-memory state ----------------
POINTS_BY_DIFF = {"easy": 10, "medium": 15, "hard": 20}
PENDING: Dict[str, Dict] = {}     # question_id -> quiz item
STREAKS: Dict[str, int] = {}      # client_ip -> streak

def _client_key(req: Request) -> str:
    ip = (req.client.host if req and req.client else "anonymous") or "anonymous"
    return ip

# ---------------- Question bank ----------------
# 도메인: 절약/포인트/연료/가계부/구독/전기/교통/식비
BANK: List[Dict] = [
    {
        "tag":"savings","difficulty":"easy",
        "question":"한 달 고정비를 줄이는 데 가장 먼저 할 일은?",
        "options":{"A":"정기결제/구독 전체 목록 수집","B":"무지출 챌린지부터 시작","C":"가계부 테마 색상 변경","D":"커피 하루 두 잔으로 줄이기"},
        "correct_key":"A",
        "explanation":"구독·정기결제 전체 목록을 모아 금액/해지여부를 먼저 점검하면 고정비 절감 효과가 가장 큽니다."
    },
    {
        "tag":"points","difficulty":"easy",
        "question":"적립률 1.0% 카드와 1.5% 카드 중, 월 50만원 사용 시 추가 적립액은?",
        "options":{"A":"1,000원","B":"2,500원","C":"5,000원","D":"7,500원"},
        "correct_key":"B",
        "explanation":"차이 0.5% × 500,000원 = 2,500원입니다."
    },
    {
        "tag":"fuel","difficulty":"easy",
        "question":"주유비 절약에 가장 효과적인 방법은?",
        "options":{"A":"집에서 먼 주유소 이용","B":"가격 비교앱으로 근처 최저가 주유소 이용","C":"항상 고급유 주유","D":"연료등 켜질 때만 주유"},
        "correct_key":"B",
        "explanation":"동일 지역 내 가격 편차가 커서 비교 후 최저가 주유소 이용이 가장 효과적입니다."
    },
    {
        "tag":"ledger","difficulty":"medium",
        "question":"비상금 권장 수준으로 가장 적절한 것은?",
        "options":{"A":"한 달 생활비의 1/2","B":"한 달 생활비의 1배","C":"3~6개월치 생활비","D":"1년치 생활비"},
        "correct_key":"C",
        "explanation":"일반적으로 3~6개월치 생활비를 권장합니다."
    },
    {
        "tag":"subscription","difficulty":"easy",
        "question":"구독 서비스 관리에서 가장 먼저 해야 할 일은?",
        "options":{"A":"쿠폰 모으기","B":"구독별 결제일/금액 파악","C":"앱 삭제","D":"비밀번호 변경"},
        "correct_key":"B",
        "explanation":"결제일·금액을 파악하고 중복/미사용을 선별해 해지 우선순위를 정합니다."
    },
    {
        "tag":"energy","difficulty":"medium",
        "question":"여름 냉방 전기요금 절감을 위해 가장 영향력이 큰 행동은?",
        "options":{"A":"냉장고 문 자주 열기","B":"에어컨 26~28℃ 설정 + 선풍기 병행","C":"형광등으로 교체","D":"에어컨 ON/OFF 반복"},
        "correct_key":"B",
        "explanation":"적정온도 유지와 송풍 병행이 전력 사용을 안정화합니다."
    },
    {
        "tag":"transport","difficulty":"medium",
        "question":"교통비 절감을 위해 가장 합리적인 선택은?",
        "options":{"A":"항상 택시 이용","B":"정기권/정액권 사용","C":"최고급 좌석 고집","D":"이동마다 앱 설치"},
        "correct_key":"B",
        "explanation":"자주 이용 노선은 정기권/정액권이 단가를 낮춥니다."
    },
    {
        "tag":"meal","difficulty":"easy",
        "question":"식비 절감에 가장 효과적인 방법은?",
        "options":{"A":"즉흥적으로 장보기","B":"식단 계획 + 장보기 리스트 작성","C":"외식만 하기","D":"간식 늘리기"},
        "correct_key":"B",
        "explanation":"계획형 장보기가 충동구매를 줄이고 재고·낭비를 감소시킵니다."
    },
]

def _pick_question(difficulty: str, tag: Optional[str]) -> Dict:
    cands = [q for q in BANK if (q["difficulty"] == difficulty)]
    if tag:
        tagged = [q for q in cands if q["tag"] == tag]
        if tagged:
            cands = tagged
    if not cands:
        cands = BANK
    return random.choice(cands)

# --- AI를 사용하여 단일 퀴즈 생성하는 비동기 함수 (수정됨) ---
async def _generate_quiz_with_ai(difficulty: str, tag: Optional[str]) -> Optional[Dict]:
    """주어진 난이도와 태그로 AI를 호출하여 퀴즈 하나를 생성합니다."""
    # AI 클라이언트가 없으면 에러 로깅 후 None 반환
    if not ai_client:
        log.error("AI 클라이언트가 초기화되지 않아 퀴즈를 생성할 수 없습니다.")
        return None

    # 로깅을 위한 trace ID 생성
    trace_id = new_trace_id()
    tlog = get_trace_logger(trace_id)

    # AI에게 전달할 사용자 프롬프트 구성
    prompt = f"난이도 '{difficulty}'"
    if tag:
        prompt += f", 주제 '{tag}'"
    prompt += "에 관한 금융 퀴즈 1개를 생성해주세요. 반드시 지정된 JSON 형식으로만 답변해주세요."

    # 시스템 프롬프트는 퀴즈 지침 사용
    system_prompt = QUIZ_INSTRUCTIONS #
    # 개발자 프롬프트로 JSON 출력 형식 명시
    developer_prompt = "출력 형식: {question: str, options: {A:str, B:str, C:str, D:str}, correct_key: str('A'|'B'|'C'|'D'), explanation: str}"

    try:
        t0 = time.monotonic()
        # AI 클라이언트의 generate 메소드 호출하여 퀴즈 생성
        ai_response_text = await ai_client.generate(
            system=system_prompt,
            developer=developer_prompt,
            user=prompt,
            context=None # 퀴즈 생성에는 별도 컨텍스트 불필요
        )
        t_gen = (time.monotonic() - t0) * 1000
        tlog.info(f"AI 퀴즈 생성 완료 (난이도: {difficulty}, 태그: {tag}, 시간: {t_gen:.1f}ms)")

        # AI 응답 텍스트에서 JSON 부분만 추출 (```json ... ``` 또는 {...} 형태 처리)
        json_match = re.search(r'```json\s*(\{.*?\})\s*```|\{.*\}', ai_response_text, re.DOTALL)
        if not json_match:
            tlog.error(f"AI 응답에서 유효한 JSON 형식을 찾을 수 없습니다 (난이도: {difficulty}): {ai_response_text}")
            return None

        # 추출된 JSON 문자열 파싱
        quiz_data_str = json_match.group(1) if json_match.group(1) else json_match.group(0)
        quiz_data = json.loads(quiz_data_str)

        # --- 생성된 퀴즈 데이터 유효성 검사 ---
        required_keys = {"question", "options", "correct_key", "explanation"}
        if not all(key in quiz_data and quiz_data[key] for key in required_keys):
            tlog.error(f"AI 생성 퀴즈에 필수 키 또는 값이 누락되었습니다 (난이도: {difficulty}): {quiz_data}")
            return None
        if not isinstance(quiz_data.get("options"), dict) or len(quiz_data["options"]) != 4 or not all(k in ["A", "B", "C", "D"] for k in quiz_data["options"]):
            tlog.error(f"AI 생성 퀴즈의 보기(options) 형식이 잘못되었습니다 (난이도: {difficulty}): {quiz_data.get('options')}")
            return None
        if quiz_data.get("correct_key") not in ["A", "B", "C", "D"]:
            tlog.error(f"AI 생성 퀴즈의 정답 키(correct_key)가 잘못되었습니다 (난이도: {difficulty}): {quiz_data.get('correct_key')}")
            return None

        # 요청받은 tag와 difficulty를 데이터에 추가
        quiz_data["tag"] = tag if tag else "general" # 태그 없으면 "general" 사용
        quiz_data["difficulty"] = difficulty

        return quiz_data

    except json.JSONDecodeError as e:
        tlog.error(f"AI 응답 JSON 파싱 실패 (난이도: {difficulty}): {e}\n응답 내용: {ai_response_text}")
        return None
    except Exception as e:
        tlog.exception(f"AI 퀴즈 생성 중 예상치 못한 오류 발생 (난이도: {difficulty}): {e}")
        return None

# ---------------- Routes ----------------
@router.post("/quiz/next", response_model=QuizNextResponse)
async def quiz_next(payload: QuizNextPayload, req: Request):
    diff = (payload.difficulty or "easy").lower()
    if diff not in POINTS_BY_DIFF:
        diff = "easy"
    q = _pick_question(diff, payload.tag)
    qid = f"{q['tag']}-{diff}-{uuid.uuid4().hex[:8]}-{int(time.time())}"
    item = {
        "question_id": qid,
        "difficulty": diff,
        "tag": q["tag"],
        "question": q["question"],
        "options": q["options"],
        "correct_key": q["correct_key"],
        "explanation": q["explanation"],
        "points_base": POINTS_BY_DIFF[diff],
    }
    PENDING[qid] = item
    return item

@router.post("/quiz/batch", response_model=QuizBatchResponse)
async def quiz_batch(payload: QuizNextPayload, req: Request):
    """지정된 태그에 대해 쉬움, 보통, 어려움 퀴즈 3개를 동시에 생성하여 반환합니다."""
    tag = payload.tag # 요청 본문에서 태그 가져오기
    difficulties = ["easy", "medium", "hard"] # 생성할 난이도 목록

    trace_id = new_trace_id()
    tlog = get_trace_logger(trace_id)
    tlog.info(f"퀴즈 Batch 생성 요청 시작 (tag: {tag})")

    # --- AI 호출 작업을 리스트로 준비 ---
    # 각 난이도별로 _generate_quiz_with_ai 함수를 호출하는 비동기 작업 생성
    tasks = [_generate_quiz_with_ai(diff, tag) for diff in difficulties]

    t0 = time.monotonic()
    # --- asyncio.gather 를 사용하여 3개의 작업을 병렬로 실행 ---
    # return_exceptions=True 로 설정하여 일부 작업 실패 시에도 전체가 중단되지 않도록 함
    results = await asyncio.gather(*tasks, return_exceptions=True)
    t_batch = (time.monotonic() - t0) * 1000
    tlog.info(f"퀴즈 Batch 생성 AI 호출 완료 ({len(results)}개 결과, 시간: {t_batch:.1f}ms)")

    # --- 결과 처리 및 응답 구성 ---
    generated_quizzes: List[QuizItem] = []

    # 병렬 실행 결과 순회
    for i, result in enumerate(results):
        diff = difficulties[i] # 현재 결과에 해당하는 난이도

        # 결과가 Exception 객체이거나 None이면 (생성 실패) 에러 로깅 후 건너뜀
        if isinstance(result, Exception) or not result:
            error_msg = str(result) if isinstance(result, Exception) else "None 반환됨"
            tlog.error(f"'{diff}' 난이도 퀴즈 생성 실패: {error_msg}")
            continue # 다음 결과로 넘어감

        q_data = result # 성공적으로 생성된 퀴즈 데이터 (딕셔너리)

        # 퀴즈 고유 ID 생성 (태그-난이도-UUID-타임스탬프 형식)
        qid = f"{q_data.get('tag', 'general')}-{diff}-{uuid.uuid4().hex[:8]}-{int(time.time())}"

        # 응답 모델(QuizItem) 형식에 맞게 데이터 구성
        item_data = {
            "question_id": qid,
            "difficulty": diff,
            "tag": q_data.get("tag", "general"),
            "question": q_data.get("question", "질문 생성 오류"),
            "options": q_data.get("options", {"A": "-", "B": "-", "C": "-", "D": "-"}),
            "correct_key": q_data.get("correct_key", "A"),
            "explanation": q_data.get("explanation", "설명 생성 오류"),
            "points_base": POINTS_BY_DIFF[diff], # 난이도별 기본 점수
        }

        # 생성된 퀴즈를 PENDING에 저장 (나중에 정답 확인용)
        PENDING[qid] = item_data

        # 최종 응답 목록에 추가 (QuizItem 모델로 변환)
        try:
            generated_quizzes.append(QuizItem(**item_data))
        except Exception as pydantic_error:
             tlog.error(f"QuizItem 모델 변환 실패 (난이도: {diff}): {pydantic_error} | 데이터: {item_data}")
             continue # 모델 변환 실패 시 해당 퀴즈는 제외

    # 성공적으로 생성된 퀴즈가 하나도 없으면 500 에러 반환
    if not generated_quizzes:
         tlog.error(f"모든 난이도의 퀴즈 생성에 실패했습니다 (tag: {tag}).")
         raise HTTPException(status_code=500, detail="퀴즈를 생성하는 중 오류가 발생했습니다.")

    tlog.info(f"퀴즈 Batch 생성 성공 ({len(generated_quizzes)}개 반환, tag: {tag})")
    # 최종적으로 생성된 퀴즈 목록을 포함하는 응답 반환
    return QuizBatchResponse(quizzes=generated_quizzes)

@router.post("/quiz/answer", response_model=QuizAnswerResponse)
async def quiz_answer(payload: QuizAnswerPayload, req: Request):
    qid = payload.question_id
    ua  = (payload.user_answer or "").strip().upper()
    if qid not in PENDING:
        raise HTTPException(status_code=404, detail="question_id not found or expired")
    item = PENDING.pop(qid)

    correct = (ua == item["correct_key"])
    key = _client_key(req)
    streak = STREAKS.get(key, 0)
    if correct:
        streak += 1
    else:
        streak = 0
    STREAKS[key] = streak

    # 간단 포인트: 기본점수 + 연속정답 보너스(최대 +5)
    points = item["points_base"] + min(streak, 5) if correct else 0

    return QuizAnswerResponse(
        is_correct=correct,
        correct_key=item["correct_key"],
        explanation=item["explanation"],
        points_awarded=points,
        streak=streak,
    )

@router.get("/quiz/health")
async def quiz_health():
    return {"ok": True}
