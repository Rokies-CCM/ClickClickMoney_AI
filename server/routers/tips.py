# server/routers/tips.py
from __future__ import annotations

import os
import re
import json
import logging
from typing import List, Optional

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from server.deps import get_http_client  # httpx.AsyncClient 재사용

logger = logging.getLogger("server")
router = APIRouter()

# ---------- Request/Response 모델 ----------
class CategoryItem(BaseModel):
  name: Optional[str] = Field(default=None, description="카테고리명 (또는 category)")
  category: Optional[str] = Field(default=None, description="카테고리명 (백워드 호환)")
  amount: float = 0

class TipsRequest(BaseModel):
  yearMonth: str
  totalAmount: float = 0
  budget: Optional[float] = None
  byCategory: List[CategoryItem] = []

# ---------- 유틸 ----------
def _normalize_name(it: CategoryItem) -> str:
  return (it.name or it.category or "").strip() or "기타"

_bullet_re = re.compile(r"^\s*(?:[-•*]|\d+\.)\s*(.+)$", re.MULTILINE)

def _extract_bullets(text: str) -> List[str]:
  items = [m.group(1).strip() for m in _bullet_re.finditer(text)]
  seen = set()
  out: List[str] = []
  for t in items:
    t = t[:120]
    if t and t not in seen:
      out.append(t)
      seen.add(t)
    if len(out) >= 8:
      break
  return out

def _round_int(v: float) -> int:
  try:
    return int(round(float(v)))
  except Exception:
    return int(v or 0)

# ---------- 휴리스틱: 절약 팁 ----------
def _heuristic_tips(req: TipsRequest) -> List[str]:
  spent = float(req.totalAmount or 0)
  budget = float(req.budget or 0)
  over = budget > 0 and spent > budget

  cat = [{"name": _normalize_name(it), "amount": float(it.amount or 0)} for it in (req.byCategory or [])]
  cat.sort(key=lambda x: x["amount"], reverse=True)
  top3 = cat[:3]
  total = spent if spent > 0 else (sum(x["amount"] for x in cat) or 1)

  tips: List[str] = []
  if over:
    pct = ((spent - budget) / max(budget, 1)) * 100
    tips.append(f"이번 달 지출이 예산을 {pct:.0f}% 초과했어요. 남은 기간 일일 한도를 정해보세요.")
  if top3:
    t0 = top3[0]
    tips.append(
      f"가장 큰 지출은 '{t0['name']}'로 비중 {round((t0['amount']/total)*100)}%. "
      f"1주일 15%만 줄이면 약 {_round_int(t0['amount']*0.15)}원 절약!"
    )
  if len(top3) > 1:
    t1 = top3[1]
    tips.append(f"두 번째 '{t1['name']}'는 구독/빈도 점검이 효과적입니다. 반복 결제 확인!")
  if len(top3) > 2:
    t2 = top3[2]
    tips.append(f"세 번째 '{t2['name']}'는 브랜드 변경/다른 상점으로 5~10% 절감해보세요.")
  if not over and budget > 0:
    # 예산 대비 진행률 / 남은 기간 일 평균 한도
    try:
      import calendar
      from datetime import datetime
      y, m = map(int, req.yearMonth.split("-"))
      last_day = calendar.monthrange(y, m)[1]
      today = datetime.today()
      days_left = max(last_day - min(today.day, last_day) + 1, 1)
    except Exception:
      days_left = 15
    remain = max(budget - spent, 0)
    tips.append(f"예산 대비 {round((spent/max(budget,1))*100)}% 사용. 남은 기간 일 평균 한도 ≈ {_round_int(remain/days_left)}원.")
  if len(tips) < 5:
    tips.append("큰 지출엔 메모를 남겨 일회성/반복성 구분을 해두면 다음 달 계획에 도움이 됩니다.")
  return tips[:8]

# ---------- 휴리스틱: 주의(경고) ----------
def _heuristic_cautions(req: TipsRequest) -> List[str]:
  spent = float(req.totalAmount or 0)
  budget = float(req.budget or 0)
  cat = [{"name": _normalize_name(it), "amount": float(it.amount or 0)} for it in (req.byCategory or [])]
  total = spent if spent > 0 else (sum(x["amount"] for x in cat) or 1)
  cat.sort(key=lambda x: x["amount"], reverse=True)

  out: List[str] = []
  if budget > 0 and spent > budget * 1.1:
    out.append(f"지출이 예산 대비 {round((spent/budget)*100)}% 입니다. 초과분을 줄일 계획이 필요해요.")
  if cat:
    share = (cat[0]["amount"] / total) * 100
    if share >= 45:
      out.append(f"'{cat[0]['name']}' 카테고리 비중이 {round(share)}%로 과도합니다. 한도 설정/대체 전략을 검토하세요.")
  for c in cat[:3]:
    n = c["name"]
    if n in ("쇼핑", "카페/간식"):
      out.append(f"'{n}' 지출이 높습니다. 충동구매/빈도 점검과 주간 한도 설정을 권장해요.")
    if n in ("식비", "외식"):
      out.append("식비가 큰 비중을 차지합니다. 주간 식단 계획/대량 장보기로 빈도/단가를 낮춰보세요.")
    if n in ("교통",):
      out.append("교통비가 부담됩니다. 대중교통/정기권/이동 병합으로 횟수를 줄여보세요.")
  if not out:
    out.append("이번 기간엔 특별한 이상 징후가 크지 않지만, 반복 결제(구독)와 일회성 지출을 구분해 관리하세요.")
  return out[:8]

# ---------- 공통 LLM 호출 ----------
async def _llm_chat(messages: List[dict], model_default: str = "gpt-4o-mini") -> Optional[str]:
  api_key = os.getenv("OPENAI_API_KEY", "").strip()
  if not api_key:
    return None
  base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
  model = os.getenv("OPENAI_MODEL", model_default)

  body = {"model": model, "temperature": 0.2, "messages": messages}
  try:
    client = get_http_client()
    res = await client.post(
      f"{base_url}/chat/completions",
      headers={"Authorization": f"Bearer {api_key}"},
      json=body,
      timeout=30.0,
    )
    if res.status_code != 200:
      logger.warning("LLM HTTP %s: %s", res.status_code, res.text[:300])
      return None
    data = res.json()
    return (data.get("choices", [{}])[0].get("message", {}).get("content", "")) or ""
  except Exception as e:
    logger.exception("LLM call failed: %s", e)
    return None

async def _llm_tips(req: TipsRequest) -> Optional[List[str]]:
  cat = [{"name": _normalize_name(x), "amount": float(x.amount or 0)} for x in (req.byCategory or [])]
  cat_sorted = sorted(cat, key=lambda x: x["amount"], reverse=True)
  payload = {
    "yearMonth": req.yearMonth,
    "budget": req.budget,
    "totalAmount": req.totalAmount,
    "topCategories": cat_sorted[:5],
  }
  system = (
    "너는 개인 소비 데이터로 절약 팁을 만들어 주는 재무 코치야. "
    "한국어로 간결하고 실천 가능한 문장을 3~5개 bullet로 만들어. "
    "각 bullet은 70자 이내. 과도한 추측 금지."
  )
  user = "아래 JSON을 참고해 이번 기간 절약 팁을 만들어 줘.\n" + json.dumps(payload, ensure_ascii=False) + "\n\n" \
         "출력 형식: '- '로 시작하는 bullet들만."
  text = await _llm_chat([
    {"role": "system", "content": system},
    {"role": "user", "content": user},
  ])
  return _extract_bullets(text or "") if text is not None else None

async def _llm_cautions(req: TipsRequest) -> Optional[List[str]]:
  cat = [{"name": _normalize_name(x), "amount": float(x.amount or 0)} for x in (req.byCategory or [])]
  cat_sorted = sorted(cat, key=lambda x: x["amount"], reverse=True)
  payload = {
    "period": req.yearMonth,
    "totalAmount": req.totalAmount,
    "budget": req.budget,
    "topCategories": cat_sorted[:5],
  }
  system = (
    "너는 개인 소비 데이터에서 리스크를 찾아주는 재무 감사관이야. "
    "과도한 비중, 예산 초과, 불필요 반복비용 등을 짚어 '주의/경고'를 3~5개 bullet로 제시해. "
    "각 bullet은 70자 이내, 모호한 표현은 피하고 근거(비중, 추정)를 간단히 포함."
  )
  user = "다음 JSON을 검토해 위험 신호(주의)를 한국어 bullet로 출력해줘.\n" + json.dumps(payload, ensure_ascii=False) + "\n\n" \
         "출력 형식: '- '로 시작하는 bullet들만."
  text = await _llm_chat([
    {"role": "system", "content": system},
    {"role": "user", "content": user},
  ])
  return _extract_bullets(text or "") if text is not None else None

# ---------- 엔드포인트 ----------
@router.post("/tips")
async def generate_tips(
  req: TipsRequest,
  llm: bool = Query(default=os.getenv("ENABLE_TIPS_LLM", "false").lower() == "true",
                    description="모델 호출 사용 여부(기본: ENV ENABLE_TIPS_LLM)"),
  mode: str = Query(default="tips", description="생성 모드: tips | caution"),
):
  """
  - mode=tips    : 절약 팁 생성
  - mode=caution : 경고/주의 생성
  - llm=true     : LLM 호출 시도 → 실패 시 휴리스틱 폴백
  """
  mode = (mode or "tips").lower()
  if mode not in ("tips", "caution"):
    mode = "tips"

  if mode == "tips":
    heuristic = _heuristic_tips(req)
    if not llm:
      return {"tips": heuristic, "yearMonth": req.yearMonth, "source": "heuristic"}
    llm_items = await _llm_tips(req)
    if llm_items:
      return {"tips": llm_items, "yearMonth": req.yearMonth, "source": "llm", "fallback": heuristic}
    return {"tips": heuristic, "yearMonth": req.yearMonth, "source": "heuristic"}

  # mode == "caution"
  heuristic = _heuristic_cautions(req)
  if not llm:
    return {"tips": heuristic, "yearMonth": req.yearMonth, "source": "heuristic"}
  llm_items = await _llm_cautions(req)
  if llm_items:
    return {"tips": llm_items, "yearMonth": req.yearMonth, "source": "llm", "fallback": heuristic}
  return {"tips": heuristic, "yearMonth": req.yearMonth, "source": "heuristic"}
