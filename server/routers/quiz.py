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
    # /quiz/batch 에서는 이 페이로드의 값을 사용하지 않습니다.
    difficulty: Optional[str] = Field(default="easy", description="easy|medium|hard")
    tag: Optional[str] = Field(default=None, description="ex) savings|points|fuel|ledger|subscription|energy|transport|meal")

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
PENDING: Dict[str, Dict] = {}      # question_id -> quiz item
STREAKS: Dict[str, int] = {}       # client_ip -> streak

def _client_key(req: Request) -> str:
    ip = (req.client.host if req and req.client else "anonymous") or "anonymous"
    return ip

# --- [추가] 사용 가능한 전체 퀴즈 태그 목록 ---
ALL_QUIZ_TAGS = [
    "savings", "points", "fuel", "ledger", 
    "subscription", "energy", "transport", "meal"
]
# ---------------------------------------------


# --- AI를 사용하여 단일 퀴즈 생성하는 비동기 함수 (수정됨) ---
async def _generate_quiz_with_ai(difficulty: str, tag: Optional[str]) -> Optional[Dict]:
    """주어진 난이도와 태그로 AI를 호출하여 퀴즈 하나를 생성합니다."""
    if not ai_client:
        log.error("AI 클라이언트가 초기화되지 않아 퀴즈를 생성할 수 없습니다.")
        return None

    trace_id = new_trace_id()
    tlog = get_trace_logger(trace_id)

    # AI에게 전달할 사용자 프롬프트 구성
    prompt = f"난이도 '{difficulty}'"
    if tag:
        prompt += f", 주제 '{tag}'"
    
    # --- [수정됨] 창의성/다양성 규칙을 마지막 프롬프트에 다시 한번 강조 ---
    prompt += "에 관한 **창의적이고 독창적인** 금융 퀴즈 1개를 생성해주세요."
    prompt += " **'다음 중 ...이 아닌 것은?' 같은 단순한 패턴은 절대 사용하지 마세요.**"
    prompt += " 반드시 지정된 JSON 형식으로만 답변해주세요."
    # --- [수정 끝] ---

    system_prompt = QUIZ_INSTRUCTIONS #
    
    developer_prompt = "출력 형식: {\"question\": \"str\", \"options\": {\"A\":\"str\", \"B\":\"str\", \"C\":\"str\", \"D\":\"str\"}, \"correct_key\": \"A\", \"explanation\": \"str\"}. 반드시 이 JSON 형식을 따르세요."

    try:
        t0 = time.monotonic()
        ai_response_text = await ai_client.generate(
            system=system_prompt,
            developer=developer_prompt,
            user=prompt,
            context=None, # 퀴즈 생성에는 별도 컨텍스트 불필요
            
        )
        t_gen = (time.monotonic() - t0) * 1000
        tlog.info(f"AI 퀴즈 생성 완료 (난이도: {difficulty}, 태그: {tag}, 시간: {t_gen:.1f}ms)")

        json_match = re.search(r'```json\s*(\{.*?\})\s*```|\{.*\}', ai_response_text, re.DOTALL)
        if not json_match:
            tlog.error(f"AI 응답에서 유효한 JSON 형식을 찾을 수 없습니다 (난이도: {difficulty}): {ai_response_text}")
            return None

        quiz_data_str = json_match.group(1) if json_match.group(1) else json_match.group(0)

        try:
            quiz_data_str_fixed = re.sub(r'([{,]\s*)(\w+)(\s*:)', r'\1"\2"\3', quiz_data_str)
            quiz_data = json.loads(quiz_data_str_fixed)
        except json.JSONDecodeError as e:
            tlog.error(f"AI 응답 JSON 파싱 실패 (수정 시도 후) (난이도: {difficulty}): {e}\n수정 시도한 내용: {quiz_data_str_fixed}\n원본 응답 내용: {quiz_data_str}")
            return None

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

        # AI가 tag를 생성하지 않으므로, 요청한 tag를 명시적으로 추가
        quiz_data["tag"] = tag if tag else "general" 
        quiz_data["difficulty"] = difficulty

        return quiz_data

    except json.JSONDecodeError as e:
        tlog.error(f"AI 응답 JSON 파싱 실패 (난이도: {difficulty}): {e}\n응답 내용: {ai_response_text}")
        return None
    except Exception as e:
        tlog.exception(f"AI 퀴즈 생성 중 예상치 못한 오류 발생 (난이도: {difficulty}): {e}")
        return None

# ---------------- Routes ----------------

# --- [ /quiz/batch 라우트 전체 수정 ] ---
@router.post("/quiz/batch", response_model=QuizBatchResponse)
async def quiz_batch(payload: QuizNextPayload, req: Request):
    """[수정] easy, medium, hard 난이도별로 각각 '다른 분야(태그)' 퀴즈 3개를 생성합니다."""
    
    # 1. 난이도 결정 (고정)
    difficulties = ["easy", "medium", "hard"]
    
    # 2. 태그 결정 (전체 목록에서 3개 랜덤 샘플링)
    # 페이로드의 tag, difficulty 값은 이 엔드포인트에서 무시됩니다.
    if len(ALL_QUIZ_TAGS) < 3:
        raise HTTPException(status_code=500, detail="Not enough tags to sample from.")
    
    try:
        # random.sample을 사용하여 중복되지 않는 3개의 태그 선택
        selected_tags = random.sample(ALL_QUIZ_TAGS, 3)
    except ValueError:
        selected_tags = ALL_QUIZ_TAGS[:3] # 3개 미만이면 그냥 앞에서 3개

    trace_id = new_trace_id()
    tlog = get_trace_logger(trace_id)
    # [수정] 로그 메시지에 조합 정보 표시
    request_pairs = list(zip(difficulties, selected_tags))
    tlog.info(f"퀴즈 Batch 생성 요청 시작 (조합: {request_pairs})")

    # 3. AI 호출 작업을 리스트로 준비 (난이도 3개, 각기 다른 태그 3개)
    tasks = [_generate_quiz_with_ai(diff, tag) for diff, tag in request_pairs]

    t0 = time.monotonic()
    # --- asyncio.gather 를 사용하여 3개의 작업을 병렬로 실행 ---
    results = await asyncio.gather(*tasks, return_exceptions=True)
    t_batch = (time.monotonic() - t0) * 1000
    tlog.info(f"퀴즈 Batch 생성 AI 호출 완료 ({len(results)}개 결과, 시간: {t_batch:.1f}ms)")

    # 4. 결과 처리 및 응답 구성
    generated_quizzes: List[QuizItem] = []

    # 병렬 실행 결과 순회
    for i, result in enumerate(results):
        diff = difficulties[i] # 현재 결과에 해당하는 '난이도'
        tag = selected_tags[i] # 현재 결과에 해당하는 '태그'

        # 결과가 Exception 객체이거나 None이면 (생성 실패) 에러 로깅 후 건너뜀
        if isinstance(result, Exception) or not result:
            error_msg = str(result) if isinstance(result, Exception) else "None 반환됨"
            tlog.error(f"'{diff}' 난이도, '{tag}' 태그 퀴즈 생성 실패: {error_msg}")
            continue # 다음 결과로 넘어감

        q_data = result # 성공적으로 생성된 퀴즈 데이터 (딕셔너리)

        # 퀴즈 고유 ID 생성 (태그-난이도-UUID-타임스탬프 형식)
        qid = f"{q_data.get('tag', tag)}-{diff}-{uuid.uuid4().hex[:8]}-{int(time.time())}"

        # 응답 모델(QuizItem) 형식에 맞게 데이터 구성
        item_data = {
            "question_id": qid,
            "difficulty": diff, # [easy, medium, hard] 중 하나
            "tag": q_data.get("tag", tag), # 랜덤 선택된 태그
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
            tlog.error(f"QuizItem 모델 변환 실패 (태그: {tag}, 난이도: {diff}): {pydantic_error} | 데이터: {item_data}")
            continue # 모델 변환 실패 시 해당 퀴즈는 제외

    # 성공적으로 생성된 퀴즈가 하나도 없으면 500 에러 반환
    if not generated_quizzes:
        tlog.error(f"모든 퀴즈 생성에 실패했습니다 (요청 조합: {request_pairs}).")
        raise HTTPException(status_code=500, detail="퀴즈를 생성하는 중 오류가 발생했습니다.")

    tlog.info(f"퀴즈 Batch 생성 성공 ({len(generated_quizzes)}개 반환, 조합: {request_pairs})")
    # 최종적으로 생성된 퀴즈 목록을 포함하는 응답 반환
    return QuizBatchResponse(quizzes=generated_quizzes)

# --- [ /quiz/batch 수정 끝 ] ---


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