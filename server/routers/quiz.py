# server/routers/quiz.py
from __future__ import annotations
import time
import random
import uuid
from typing import Dict, List, Optional

from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel, Field

router = APIRouter()

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
