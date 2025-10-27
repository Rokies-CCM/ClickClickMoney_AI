# server/routers/chat.py
import re
import json
import time
import asyncio
from typing import Optional, Dict, List, Set, Tuple
from collections import deque
from urllib.parse import urlparse
from collections import defaultdict

from fastapi import APIRouter, Request, Body
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

from core.config import get_settings
from core.metrics import append_bench_cache
from core.utils import normalize_question
from evals.bench_offline import rouge_l_approx, compute_bleu, compute_evidence_score
from server.deps import get_trace_logger, new_trace_id
from core.prompts import SYSTEM_PROMPT, DEVELOPER_PROMPT, get_domain_prompt
from core.router import choose_model
from core.cache import response_cache, semantic_cache
from core.tavily import web_search_snippets, is_tavily_enabled
from core.rag import build_context
from core.quant_guard import extract_numbers, sanitize_numbers
from core.evidence import analyze_question, normalize_citations
# 소비 분석 기능
from core.backend_client import get_consumption_data, format_consumption_for_llm
from datetime import datetime, timedelta
# 근거 꼬리 렌더러는 더 이상 사용하지 않으므로 import 제거
# from core.answer import render_answer, render_tail

# 최대 12개의 대화를 기록(이전 대화 내역은 10개) 늘리기를 원할 시 최대 10개 권장
MAX_HISTORY_TURNS = 5
MAX_HISTORY_MESSAGES = MAX_HISTORY_TURNS * 2

try:
    from core.utils import count_tokens  # tiktoken 기반
except Exception:
    count_tokens = lambda text, *_: max(1, len(text) // 4)

cfg = get_settings()

SMALLTALK_RE = re.compile(
    r"^\s*(안녕|안녕하세요|반가워|ㅎ+ㅇ+|하이|hello|hi|hey|"
    r"누구(야|세요)|자기소개|넌\s*뭘\s*도와줄|무엇을\s*도와줄|what can you do)\b",
    re.IGNORECASE,
)
TAIL_SUPPRESS_RE = re.compile(
    r"(출처|근거\s*요약)\s*(?:을?\s*)?(?:붙이|넣|달|쓰|표시|포함)\s*지\s*말(?:고|아)?|"
    r"(출처|근거\s*요약)\s*없[이이]|"
    r"(출처|근거\s*요약)\s*빼고|"
    r"(출처|근거\s*요약)\s*제외|"
    r"\b(no\s+sources?|without\s+(sources?|citations?|evidence))\b",
    re.IGNORECASE
)

router = APIRouter()

# ------------------------ Metrics ------------------------
_METRICS = {
    "total_requests": 0,
    "simcache_hits": 0,
    "provider_counts": {"groq": 0, "openai": 0},
    "total_prompt_tokens": 0,
    "total_completion_tokens": 0,
}
def _update_metrics(provider: str, prompt_tks: int, compl_tks: int, sim_hit: bool = False):
    _METRICS["total_requests"] += 1
    _METRICS["provider_counts"][provider] = _METRICS["provider_counts"].get(provider, 0) + 1
    _METRICS["total_prompt_tokens"] += max(0, int(prompt_tks))
    _METRICS["total_completion_tokens"] += max(0, int(compl_tks))
    if sim_hit:
        _METRICS["simcache_hits"] += 1

_MAX_RECENTS = 200
_RECENTS: deque = deque(maxlen=_MAX_RECENTS)
def _push_recent(entry: dict):
    _RECENTS.append(entry)

# ------------------------ Schema ------------------------
class ChatPayload(BaseModel):
    question: str
    context: Optional[str] = ""
    domain: Optional[str] = Field(default=None, description="apptech|ledger|invest|points|consumption")
    top_k: Optional[int] = Field(default=None, ge=1, le=10)
    max_context_chars: Optional[int] = Field(default=None, ge=200, le=8000)
    stream: Optional[bool] = Field(default=True)
    facts: Optional[Dict[str, str]] = Field(default=None, description="사용자 제공 수치")
    chat_history: List[Dict[str, str]] = Field(default=[], description="이전 대화 기록")

# ------------------------ URL / 수치 가드 ------------------------
_URL_RE = re.compile(r"https?://[^\s\])}>,]+", re.IGNORECASE)
def _normalize_host(url: str) -> Optional[str]:
    try:
        u = urlparse(url)
        if u.scheme in ("http", "https") and u.netloc:
            host = u.netloc.lower()
            if host.startswith("www."):
                host = host[4:]
            return host
    except Exception:
        return None
    return None

def _collect_allowed_hosts(passages: List[Dict], *extra_texts: str) -> Set[str]:
    hosts: Set[str] = set()
    for p in passages or []:
        for key in ("source", "url"):
            v = p.get(key)
            if isinstance(v, str):
                for u in _URL_RE.findall(v):
                    h = _normalize_host(u)
                    if h:
                        hosts.add(h)
    for t in extra_texts or []:
        if not t:
            continue
        for u in _URL_RE.findall(t):
            h = _normalize_host(u)
            if h:
                hosts.add(h)
    return hosts

def _sanitize_urls(text: str, allowed_hosts: Optional[Set[str]]) -> str:
    if not text:
        return text
    if allowed_hosts is None:
        return text
    return _URL_RE.sub(lambda m: m.group(0) if (_normalize_host(m.group(0)) in allowed_hosts) else "[링크 생략]", text)

def _sanitize_text_with_numbers(text: str, allowed_hosts: Optional[Set[str]], allowed_numbers: set[str]) -> str:
    try:
        return sanitize_numbers(_sanitize_urls(text, allowed_hosts), allowed_numbers)
    except Exception:
        return text

# -------- 모델이 써버린 증거/근거부족 라인 & inline 토큰 제거 --------
_BEGIN_EVD_LINE = re.compile(
    r"^\s*(?:\[?\s*근거\s*요약\s*\]?\s*:?|"
    r"\[?\s*출처\s*\]?\s*:?|"
    r"근거\s*:|"
    r"\[?\s*근거\s*부족\s*\]?)",
    re.IGNORECASE
)
_INADEQUATE_LINE = re.compile(r"^\s*\[\s*근거\s*부족\s*\]\s*$", re.IGNORECASE)
_EVD_INLINE = re.compile(r"\[?\s*(근거\s*요약|출처|근거\s*부족)\s*\]?\s*:?", re.IGNORECASE)

def cut_inline_evidence_tokens(text: str) -> str:
    if not text:
        return text
    m = _EVD_INLINE.search(text)
    if not m:
        return text.strip()
    return text[:m.start()].rstrip()

def strip_evidence_lines(text: str) -> str:
    if not text:
        return text
    lines = text.splitlines()
    out, skipping = [], False
    for ln in lines:
        if _BEGIN_EVD_LINE.match(ln):
            skipping = True
            continue
        if not skipping:
            out.append(ln)
    return "\n".join(out).rstrip()

def strip_inadequate_line(text: str) -> str:
    if not text:
        return text
    lines = [ln for ln in text.splitlines() if not _INADEQUATE_LINE.match(ln)]
    return "\n".join(lines).strip()

def sanitize_body_text(text: str) -> str:
    s = strip_inadequate_line(
        strip_evidence_lines(
            cut_inline_evidence_tokens(text)
        )
    )
    # 통화/퍼센트 앞 불필요 공백 제거: "103,500 원"→"103,500원", "58.8 %"→"58.8%"
    s = re.sub(r"\s+원\b", "원", s)
    s = re.sub(r"\s+%", "%", s)

    # 라인 끝 공백 제거 + 중복 공백 최소화
    s = "\n".join(line.rstrip() for line in s.splitlines())
    s = re.sub(r"[ \t]{2,}", " ", s)

    return s.strip()

# -------- 질의에서 '출처 N개'·'공공기관만' 요구 파싱 --------
_SRC_LIMIT_RE = re.compile(r"(?:출처|source)\s*([0-9]{1,2})\s*개", re.IGNORECASE)
def _desired_source_limit(q: str) -> Optional[int]:
    m = _SRC_LIMIT_RE.search(q or "")
    if not m:
        return None
    try:
        n = int(m.group(1))
        return min(10, max(1, n))
    except Exception:
        return None

def _require_public_sources(q: str) -> bool:
    return bool(re.search(r"(공공기관|정부|관계부처|공공|gov)", q or "", re.IGNORECASE))

def _is_public_host(host: str) -> bool:
    if not host:
        return False
    if host.endswith(".go.kr"):
        return True
    wl = set((cfg.TAVILY_DOMAINS or [])) | {"opinet.co.kr","bok.or.kr","kostat.go.kr","kosis.kr","msit.go.kr","kcc.go.kr","kca.go.kr"}
    return (host in wl) or host.endswith(".or.kr")

def _filter_sources(sources: List[Dict], limit: Optional[int], public_only: bool) -> List[Dict]:
    seen_hosts: Set[str] = set()
    picked: List[Dict] = []
    for s in sources:
        host = (s.get("site") or "").lower()
        if public_only and not _is_public_host(host):
            continue
        if host in seen_hosts:
            continue
        picked.append(s)
        seen_hosts.add(host)
        if limit and len(picked) >= limit:
            break
    if limit:
        return picked[:limit]
    return picked

# -------------- 출처 버튼 생성 유틸 --------------
def _norm_button_title(t: str, max_len: int = 80) -> str:
    if not t:
        return ""
    t = re.sub(r"\s+", " ", t).strip()
    if len(t) > max_len:
        t = t[: max_len - 1].rstrip() + "…"
    return t

def _ensure_http(url: str) -> str:
    if not url:
        return url
    if re.match(r"^https?://", url, re.IGNORECASE):
        return url
    return "https://" + url

def _build_source_buttons(passages_used: List[Dict], limit: Optional[int], public_only: bool) -> List[Dict]:
    """
    passages_used를 기반으로 dedup + host 필터 반영한 버튼 목록 생성.
    반환: [{title, url, host}]
    """
    try:
        _, uniq_sources = normalize_citations(passages_used or [], max_sources=(limit or cfg.MAX_SOURCES))
    except Exception:
        uniq_sources = []

    uniq_sources = _filter_sources(uniq_sources, limit=limit, public_only=public_only)

    buttons: List[Dict] = []
    seen_urls: Set[str] = set()

    for s in uniq_sources:
        url = _ensure_http((s.get("url") or "").strip())
        if not url:
            continue
        if url in seen_urls:
            continue
        seen_urls.add(url)
        host = _normalize_host(url) or (s.get("site") or "").lower()
        title = _norm_button_title(s.get("title") or s.get("name") or s.get("url") or host or "출처")
        buttons.append({"title": title, "url": url, "host": host})

        if limit and len(buttons) >= limit:
            break

    return buttons

# ledger 파생 수치
import re as _re
WON_NUM = _re.compile(r"(?:[₩￦]?\s?\d{1,3}(?:,\d{3})+|\d+)\s?원")
def _parse_won(s: str) -> int | None:
    if not isinstance(s, str):
        return None
    m = WON_NUM.search(s)
    if not m:
        return None
    try:
        return int(_re.sub(r"[^\d]", "", m.group(0)))
    except Exception:
        return None

def _fmt_won(n: int) -> str:
    return f"{n:,}원"

def _derive_ledger_numbers(facts: Dict[str, str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for key in ["교통비", "식비"]:
        base = _parse_won(facts.get(key, "")) if facts else None
        if base is None:
            continue
        for delta in (50_000, 100_000):
            out[f"{key} 한도 제안(-{_fmt_won(delta)})"] = _fmt_won(max(0, base - delta))
    return out

def _compose_prompts(domain: Optional[str]) -> tuple[str, str]:
    dom = get_domain_prompt(domain)
    brief_rule = "\n[지시] 답변은 2~4문장 이내 핵심 위주. **본문만 작성**하고 근거 섹션은 쓰지 마세요."
    facts_rule = (
        "\n[사실우선] 아래 [사용자 제공 수치]/[파생 수치]만 근거로 사용하세요. "
        "여기에 없는 카테고리·비율·금액을 추정하거나 만들어내지 마세요."
    )
    number_rule = (
        "\n[수치지침] 제공된 문맥/사실(facts)에 없는 임의의 비율(%)·금액을 만들지 마세요. "
        "비율 표현이 필요하면 '비중이 높음/낮음'처럼 서술하세요. 제공된 수치는 그대로 사용하세요."
    )
    return SYSTEM_PROMPT + brief_rule + facts_rule + number_rule + ("\n" + dom if dom else ""), DEVELOPER_PROMPT

# ------------------------ 안전 웹 검색 래퍼(재시도 포함) ------------------------
def _safe_web_search(q: str, k: int, log) -> List[Dict]:
    """Tavily 장애(HTTP 5xx/네트워크) 시에도 빈 리스트로 폴백. 간단 재시도 포함."""
    if not is_tavily_enabled():
        return []
    delay = cfg.WEB_BACKOFF_SEC
    for attempt in range(1, cfg.WEB_RETRIES + 1):
        try:
            return web_search_snippets(q, max_results=k)
        except Exception as e:
            try:
                log.warning(f"[web] tavily search failed({attempt}/{cfg.WEB_RETRIES}): {type(e).__name__}: {e}")
            except Exception:
                pass
            if attempt < cfg.WEB_RETRIES:
                try:
                    time.sleep(delay)
                except Exception:
                    pass
                delay *= 1.5
    return []

# ------------------------ Delta 계산 헬퍼 (중복 송출 방지) ------------------------
_WS = re.compile(r"\s+")
def _collapse_ws(s: str) -> str:
    return _WS.sub("", s or "")

def _cut_idx_ws_prefix(early: str, final: str) -> int:
    """
    공백/개행/중복 스페이스를 무시하며 early가 final의 prefix인지 검사하고,
    원본(final)에서 잘라낼 인덱스를 반환. 실패 시 -1.
    """
    if not early:
        return 0
    i = j = 0
    le, lf = len(early), len(final)
    while i < le and j < lf:
        ce, cf = early[i], final[j]
        if ce.isspace():
            i += 1
            continue
        if cf.isspace():
            j += 1
            continue
        if ce != cf:
            return -1
        i += 1
        j += 1
    # early의 non-space 문자를 모두 소비했는가?
    while i < le and early[i].isspace():
        i += 1
    if i == le:
        # j는 final에서 early가 끝난 직후의 위치(원본 인덱스)
        return j
    return -1

def _map_norm_to_original_index(s: str, norm_end: int) -> int:
    """
    collapse_ws(s)에서 norm_end 위치(비공백 문자 개수 기준)에 해당하는
    원문 인덱스를 반환. (문자 바로 뒤 index)
    """
    if norm_end <= 0:
        return 0
    count = 0
    for idx, ch in enumerate(s):
        if not ch.isspace():
            count += 1
            if count == norm_end:
                return idx + 1
    return len(s)

def _compute_delta(early_sent_parts: List[str], final_text: str, need_web: bool) -> str:
    """
    - need_web=False: 파이널 본문 재송출 금지(빈 문자열 반환)
    - 그 외:
        1) exact startswith
        2) 공백 무시 prefix 비교
        3) early 마지막 120자 suffix로 final 내 위치 검색(원문/공백무시 모두)
    - 모두 실패 시 안전하게 재송출 생략("")하여 중복을 100% 차단
    """
    if not need_web:
        return ""  # 로컬/스몰톡은 파이널 본문 재송출 금지

    early_text = "".join(early_sent_parts).strip()
    if not early_text:
        return final_text  # early 스트림 없음 → 파이널 전체 송출

    # 1) exact startswith
    if final_text.startswith(early_text):
        return final_text[len(early_text):].lstrip("\n")

    # 2) 공백 무시 prefix 비교
    cut_idx = _cut_idx_ws_prefix(early_text, final_text)
    if cut_idx >= 0:
        return final_text[cut_idx:].lstrip("\n")

    # 3) suffix fallback (마지막 120자)
    suf = early_text[-120:].strip()
    if suf:
        pos = final_text.rfind(suf)
        if pos != -1:
            return final_text[pos + len(suf):].lstrip("\n")
        # 공백무시 suffix
        f_norm = _collapse_ws(final_text)
        s_norm = _collapse_ws(suf)
        pos_norm = f_norm.rfind(s_norm)
        if pos_norm != -1:
            cut = _map_norm_to_original_index(final_text, pos_norm + len(s_norm))
            return final_text[cut:].lstrip("\n")

    # 실패 시 보수적으로 아무 것도 재송출하지 않음(중복 100% 차단)
    return ""

# ------------------------ Route ------------------------
@router.post("/chat")
async def chat(req: Request):
    body = await req.json()
    payload = ChatPayload(**body)
    if isinstance(payload.stream, str):
        payload.stream = payload.stream.lower() == "true"

    question_raw = (payload.question or "").strip()
    question = normalize_question(question_raw)
    domain = (payload.domain or "").lower()
    client_ctx = payload.context or ""
    facts = payload.facts or {}
    chat_history = payload.chat_history

    if len(chat_history) > MAX_HISTORY_MESSAGES:
        chat_history = chat_history[-MAX_HISTORY_MESSAGES:]

    trace_id = new_trace_id()
    log = get_trace_logger(trace_id)

    if not question:
        return JSONResponse(status_code=400, content={
            "error": {"code": "BAD_REQUEST", "message": "question is required", "trace_id": trace_id}
        })

    # ========== 소비 분석 기능 (전달 비교 포함) ==========
    consumption_keywords = ["소비", "지출", "돈", "얼마", "쓴", "spending", "expense"]
    comparison_keywords = ["비교", "대비", "전달", "지난달", "변화", "차이"]
    
    if any(kw in question for kw in consumption_keywords):
        log.info(f"[Consumption] Detected consumption question")

        # JWT 토큰 추출
        auth_header = req.headers.get("authorization", "")
        token = auth_header.replace("Bearer ", "").strip() if auth_header.startswith("Bearer ") else None

        # 날짜 추출 (두 달 비교 지원)
        def extract_dates_from_question(q: str) -> tuple:
            today = datetime.now()
            
            # "X월과 Y월 비교" 또는 "X월 Y월 비교" 패턴 찾기
            two_months = re.findall(r'(\d{1,2})월', q)
            
            if len(two_months) >= 2:
                # 두 달이 명시된 경우
                month1 = int(two_months[0])
                month2 = int(two_months[1])
                
                # 더 최근 달이 current, 이전 달이 previous
                if month1 > month2:
                    current_month, prev_month = month1, month2
                else:
                    current_month, prev_month = month2, month1
                
                # 연도 계산 (올해 기준)
                current_year = today.year if current_month <= today.month else today.year - 1
                prev_year = today.year if prev_month <= today.month else today.year - 1
                
                # 날짜 범위 계산
                current_start = datetime(current_year, current_month, 1).strftime("%Y-%m-%d")
                if current_month == 12:
                    current_end = datetime(current_year, 12, 31).strftime("%Y-%m-%d")
                else:
                    current_end = (datetime(current_year, current_month + 1, 1) - timedelta(days=1)).strftime("%Y-%m-%d")
                
                prev_start = datetime(prev_year, prev_month, 1).strftime("%Y-%m-%d")
                if prev_month == 12:
                    prev_end = datetime(prev_year, 12, 31).strftime("%Y-%m-%d")
                else:
                    prev_end = (datetime(prev_year, prev_month + 1, 1) - timedelta(days=1)).strftime("%Y-%m-%d")
                
                return current_start, current_end, prev_start, prev_end
            
            elif len(two_months) == 1:
                # 한 달만 명시된 경우
                month = int(two_months[0])
                year = today.year if month <= today.month else today.year - 1
                start = datetime(year, month, 1)
                end = (datetime(year, month + 1, 1) - timedelta(days=1)) if month < 12 else datetime(year, 12, 31)
                return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), None, None
            
            # "이번" 키워드
            if "이번" in q:
                start = datetime(today.year, today.month, 1)
                return start.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d"), None, None
            
            # 기본값: 최근 30일
            start = today - timedelta(days=30)
            return start.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d"), None, None

        date_result = extract_dates_from_question(question)
        
        # 비교 요청 여부 확인
        is_comparison = any(kw in question for kw in comparison_keywords)
        model_name = "gpt-4o-mini"
        
        try:
            if len(date_result) == 4 and date_result[2] is not None:
                # 두 달이 명시된 경우 (자동 비교)
                current_start, current_end, prev_start, prev_end = date_result
                is_comparison = True
                log.info(f"[Consumption] Manual comparison: {current_start}~{current_end} vs {prev_start}~{prev_end}")
                
                # 두 달 데이터 직접 가져오기
                
                
                current_data = await get_consumption_data(
                    start_date=current_start,
                    end_date=current_end,
                    token=token
                )
                
                prev_data = await get_consumption_data(
                    start_date=prev_start,
                    end_date=prev_end,
                    token=token
                )
                
                # 데이터 파싱
                def parse_data(data):
                    if not data or "data" not in data:
                        return {}, 0, 0
                    content = data["data"].get("content", [])
                    category_totals = defaultdict(int)
                    total = 0
                    for item in content:
                        cat = item.get("categoryName") or "기타"
                        amt = int(item.get("amount", 0))
                        category_totals[cat] += amt
                        total += amt
                    return category_totals, total, len(content)
                
                current_cats, current_total, current_count = parse_data(current_data)
                prev_cats, prev_total, prev_count = parse_data(prev_data)
                
                # 비교 텍스트 생성
                lines = [
                    f"[{current_start[:7]} vs {prev_start[:7]} 비교]",
                    "",
                    f"**{current_start[:7]}**",
                    f"- 총 지출: {current_total:,}원",
                    f"- 거래 건수: {current_count}건",
                    "",
                    f"**{prev_start[:7]}**",
                    f"- 총 지출: {prev_total:,}원",
                    f"- 거래 건수: {prev_count}건",
                    "",
                ]
                
                # 증감 분석
                if prev_total > 0:
                    diff = current_total - prev_total
                    diff_pct = (diff / prev_total) * 100
                    
                    if diff > 0:
                        lines.append(f"💡 총 지출이 {diff:,}원 증가했어요 (+{diff_pct:.1f}%)")
                    elif diff < 0:
                        lines.append(f"💡 총 지출이 {abs(diff):,}원 감소했어요 ({diff_pct:.1f}%)")
                    else:
                        lines.append("💡 총 지출이 동일해요")
                
                lines.append("")
                lines.append("[카테고리별 변화]")
                
                # 카테고리별 증감
                all_cats = set(current_cats.keys()) | set(prev_cats.keys())
                changes = []
                
                for cat in all_cats:
                    curr = current_cats.get(cat, 0)
                    prev = prev_cats.get(cat, 0)
                    diff = curr - prev
                    
                    if prev > 0:
                        diff_pct = (diff / prev) * 100
                        changes.append((cat, diff, diff_pct, curr, prev))
                    elif curr > 0:
                        changes.append((cat, diff, 0, curr, prev))
                
                changes.sort(key=lambda x: abs(x[1]), reverse=True)
                
                for cat, diff, diff_pct, curr, prev in changes[:5]:
                    if diff > 0:
                        lines.append(f"- {cat}: {curr:,}원 (이전 {prev:,}원, +{diff:,}원 ↑{diff_pct:.1f}%)")
                    elif diff < 0:
                        lines.append(f"- {cat}: {curr:,}원 (이전 {prev:,}원, {diff:,}원 ↓{abs(diff_pct):.1f}%)")
                    else:
                        lines.append(f"- {cat}: {curr:,}원 (동일)")
                
                formatted_data = "\n".join(lines)
                
            else:
                # 한 달만 명시된 경우
                start_date, end_date = date_result[0], date_result[1]
                log.info(f"[Consumption] Date range: {start_date} ~ {end_date}")
                
                if is_comparison:
                    # 전달 비교 분석
                    from core.backend_client import get_consumption_comparison
                    formatted_data = await get_consumption_comparison(
                        current_start=start_date,
                        current_end=end_date,
                        token=token
                    )
                else:
                    # 일반 소비 분석
                    data = await get_consumption_data(
                        start_date=start_date,
                        end_date=end_date,
                        token=token
                    )
                    
                    if not data or "data" not in data:
                        answer = "소비 데이터를 불러올 수 없습니다. 로그인 또는 서버 연결을 확인해주세요."
                        formatted_data = ""
                    else:
                        formatted_data = format_consumption_for_llm(data)

            # AI 분석 (비교 데이터 또는 일반 데이터)
            if formatted_data:
                analysis_prompt = f"""
당신은 개인 재정 분석가이자 절약 코치입니다.
다음은 사용자의 실제 소비 데이터입니다.

<소비 데이터>
{formatted_data}
</소비 데이터>

질문: "{question}"

위 데이터를 기반으로 다음을 분석해 주세요.

**중요**: 
- "총 지출"과 "카테고리별 지출" 섹션의 금액을 정확히 사용하세요
- "최근 거래" 섹션은 개별 거래 내역일 뿐, 카테고리 총액이 아닙니다
- 예: 식비 총액은 108,300원이지, 최근 거래의 16,800원이 아닙니다
- 비교 데이터가 있다면 증감 패턴을 정확히 언급하세요

분석 내용:
1. 사용자가 가장 많이 쓴 지출 카테고리와 금액 (정확한 숫자 사용)
2. {"비교 분석 및 변화 이유" if is_comparison else "해당 항목이 높은 이유 추정"}
3. 절약을 위한 구체적인 제안 2~3가지

출력은 간결하고 자연스러운 한국어 문장으로 해주세요.
"""
                
                choice = choose_model(question, formatted_data)
                provider_name, model_name, model_client = choice.name, choice.model, choice.client
                result = await model_client.generate(
                    analysis_prompt, "", "", "", chat_history=[]
                )
                
                answer = result.strip()
            else:
                answer = "소비 데이터를 분석할 수 없습니다."

            # 스트리밍 응답 (SSE)
            if payload.stream:
                async def consumption_stream(model_name_local: str):
                    for ch in answer:
                        yield f"data: {ch}\n\n"

                    meta = {
                        "provider": "consumption_analysis",
                        "model": model_name,
                        "usage": {
                            "prompt_tokens": 0,
                            "completion_tokens": len(answer),
                            "total_tokens": len(answer)
                        },
                        "source_buttons": [],
                        "chat_history": []
                    }
                    yield f"event: done\ndata: {json.dumps(meta, ensure_ascii=False)}\n\n"
                    yield "event: end\ndata: [DONE]\n\n"

                return StreamingResponse(
                    consumption_stream(model_name),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "X-Accel-Buffering": "no"
                    }
                )
            else:
                return {"answer": answer, "chat_history": chat_history}

        except Exception as e:
            log.error(f"[Consumption] Failed: {e}")
            err = f"소비 데이터를 분석하는 중 오류가 발생했습니다: {e}"
            if payload.stream:
                async def err_stream():
                    for ch in err:
                        yield f"data: {ch}\n\n"
                    yield f"event: done\ndata: {{}}\n\n"
                    yield "event: end\ndata: [DONE]\n\n"
                return StreamingResponse(err_stream(), media_type="text/event-stream")
            return {"answer": err}
    # ========== 소비 분석 끝 ==========


    # facts → context
    if facts:
        client_ctx += "\n\n[사용자 제공 수치]\n" + "\n".join([f"- {k}: {v}" for k, v in facts.items()])
    if domain == "ledger" and facts:
        d = _derive_ledger_numbers(facts)
        if d:
            client_ctx += "\n\n[파생 수치]\n" + "\n".join([f"- {k}: {v}" for k, v in d.items()])
    if domain == "consumption" and facts:
        client_ctx += f"\n\n[소비 요약]\n이번 달 소비 요약: " + ", ".join([f"{k}: {v}" for k, v in facts.items()])

    # ---------------- Grounding policy ----------------
    need_web, news_like, explicit_src = analyze_question(question_raw)
    suppress_tail = True  # 본문 꼬리(근거/출처) 완전 억제
    is_smalltalk = bool(SMALLTALK_RE.match(question_raw))
    if domain in {"ledger", "consumption", "invest", "points", "apptech"} and not (news_like or explicit_src):
        need_web = False

    # (evd_min은 현재 내부 사용 X, 호환 유지)
    evd_min = cfg.EVIDENCE_MIN
    if news_like and getattr(cfg, "EVIDENCE_MIN_NEWS", None) is not None:
        evd_min = max(evd_min, cfg.EVIDENCE_MIN_NEWS)
    elif news_like:
        evd_min = max(evd_min, 0.50)
    if explicit_src:
        evd_min = max(evd_min, 0.45)

    desired_limit = _desired_source_limit(question_raw)
    public_only = _require_public_sources(question_raw)

    # ---------------- Semantic cache fast-path ----------------
    try:
        cached = await semantic_cache.get(question, facts)
    except Exception:
        cached = None

    if cached:
        used_passages_cached = cached.get("sources", [])
        ctx_cached = cached.get("context_used", "") or ""
        allowed_hosts_cached = _collect_allowed_hosts(used_passages_cached, ctx_cached, client_ctx)
        allowed_numbers_cached = extract_numbers(ctx_cached + " " + " ".join([s.get("text", "") for s in used_passages_cached]) + " " + client_ctx)
        answer_cached = _sanitize_text_with_numbers(cached.get("answer", ""), allowed_hosts_cached, allowed_numbers_cached)
        answer_cached = sanitize_body_text(answer_cached)
        model_cached = (cached.get("model") or "-")
        provider_cached = (cached.get("provider") or "openai")
        usage_cached = cached.get("usage") or {
            "prompt_tokens": count_tokens((SYSTEM_PROMPT or "") + (DEVELOPER_PROMPT or "") + question + ctx_cached, model_cached),
            "completion_tokens": count_tokens(answer_cached, model_cached),
            "total_tokens": count_tokens((SYSTEM_PROMPT or "") + (DEVELOPER_PROMPT or "") + question + ctx_cached, model_cached) + count_tokens(answer_cached, model_cached),
        }
        updated_history_cached = chat_history + [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer_cached}
        ]

        # 버튼 생성 (캐시 결과 기반)
        try:
            source_buttons_cached = _build_source_buttons(used_passages_cached, desired_limit, public_only)
        except Exception:
            source_buttons_cached = []

        if payload.stream:
            async def event_stream_cache():
                yield f"event: start\ndata: {trace_id}\n\n"
                for line in (answer_cached or "").splitlines():
                    if line.strip():
                        yield f"data: {line}\n\n"
                meta = {
                    "provider": provider_cached, "model": model_cached,
                    "usage": usage_cached, "context_used": ctx_cached,
                    "sources": used_passages_cached, "source_buttons": source_buttons_cached,
                    "cost_usd_estimate": None,
                    "chat_history": updated_history_cached,
                    "from_cache": True,
                }
                yield f"event: done\ndata: {json.dumps(meta, ensure_ascii=False)}\n\n"
                yield "event: end\ndata: [DONE]\n\n"
            return StreamingResponse(event_stream_cache(), media_type="text/event-stream")

        return JSONResponse(content={
            "trace_id": trace_id,
            "answer": answer_cached,
            "context_used": ctx_cached,
            "sources": used_passages_cached,
            "source_buttons": source_buttons_cached,
            "model": model_cached,
            "provider": provider_cached,
            "usage": usage_cached,
            "chat_history": updated_history_cached,
            "from_cache": True,
        })

    # ---------------- Retrieval ----------------
    t_retr_s = time.monotonic()
    brief_mode = (len(question) < 40) and (not news_like) and (not explicit_src)

    initial = [{"text": client_ctx, "source": "client", "title": "client"}] if client_ctx else []

    if (not need_web and brief_mode) or is_smalltalk:
        ctx, used_passages = (client_ctx, initial)
    else:
        initial_k = cfg.NEWS_MIN_K if (news_like or explicit_src) else 2
        web_snips = _safe_web_search(question_raw, initial_k, log) if need_web else []
        ctx, used_passages = build_context(
            question_raw,
            web_snippets=(initial + web_snips),
            top_k=payload.top_k,
            max_context_chars=payload.max_context_chars,
            min_k=(initial_k if (news_like or explicit_src) else None),
            prefer_web_first=(news_like or explicit_src)
        )
    retrieval_ms = round((time.monotonic() - t_retr_s) * 1000, 1)

    # Guards allow-list
    allowed_hosts = _collect_allowed_hosts(used_passages, ctx or "", client_ctx)
    allowed_numbers = extract_numbers((ctx or "") + " " + " ".join([s.get("text", "") for s in used_passages]) + " " + client_ctx)

    # Routing (GPT only)
    choice = choose_model(question, ctx)
    provider_name, model_name, model_client = choice.name, choice.model, choice.client
    sys_p, dev_p = _compose_prompts(domain)

    # ------------------------ Streaming ------------------------
    if payload.stream:
        async def event_stream():
            t_gen_s = time.monotonic()
            full_chunks: List[str] = []
            did_finalize = False

            # need_web=True(뉴스/근거요청)면 초안은 스트림하지 않고 파이널만 송출
            stream_early = not (need_web and not is_smalltalk)

            # 초반에 보낸 본문 누적 → 파이널 중복 방지
            sent_parts: List[str] = []

            def _usage_obj(full_text: str) -> Dict:
                p = count_tokens(sys_p + dev_p + question + (ctx or ""), model_name)
                c = max(1, len(full_text) // 4)
                return {"prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c}

            async def _finalize_and_get_meta(full_text: str) -> Tuple[str, Dict, str]:
                nonlocal allowed_hosts, allowed_numbers

                final_text = sanitize_body_text(full_text)
                ctx_used = ctx
                passages_used = used_passages

                # 품질 보강 재검색(본문만): tail 부착과 무관하게 수행
                if need_web and not is_smalltalk:
                    try:
                        evd0 = compute_evidence_score(ctx_used, final_text)
                    except Exception:
                        evd0 = 0.0
                    try:
                        web3 = _safe_web_search(question_raw, cfg.WEB_MAX_RESULTS, log)
                        if web3:
                            initial3 = [{"text": client_ctx, "source": "client", "title": "client"}] if client_ctx else []
                            ctx3, used3 = build_context(
                                question_raw,
                                web_snippets=(initial3 + web3),
                                top_k=cfg.TOP_K,
                                max_context_chars=cfg.MAX_CONTEXT_CHARS,
                                min_k=cfg.NEWS_MIN_K,
                                prefer_web_first=True
                            )
                            safer = await model_client.generate(sys_p, dev_p, question, ctx3, chat_history=chat_history)
                            safer = sanitize_body_text(_sanitize_text_with_numbers(safer, _collect_allowed_hosts(used3, ctx3), extract_numbers(ctx3)))
                            _, s3 = normalize_citations(used3, max_sources=cfg.MAX_SOURCES)
                            if (compute_evidence_score(ctx3, safer) >= evd0) or (len(s3) > 0):
                                final_text = safer
                                allowed_hosts = _collect_allowed_hosts(used3, ctx3)
                                allowed_numbers = extract_numbers(ctx3)
                                ctx_used = ctx3
                                passages_used = used3
                    except Exception:
                        pass

                # ✅ 본문 꼬리([출처]/[근거 요약])는 더 이상 붙이지 않음

                # 이미 보낸 본문(early)을 제외한 차이만 송출
                resend = _compute_delta(sent_parts if stream_early else [], final_text, need_web)

                generation_ms = round((time.monotonic() - t_gen_s) * 1000, 1)
                usage = _usage_obj(final_text)

                # 버튼 생성
                try:
                    source_buttons = _build_source_buttons(passages_used, desired_limit, public_only)
                except Exception:
                    source_buttons = []

                # 메트릭/벤치 기록은 실제 사용된 컨텍스트 기준
                try:
                    _update_metrics(provider_name, usage["prompt_tokens"], usage["completion_tokens"], sim_hit=False)
                    _push_recent({
                        "provider": provider_name, "model": model_name,
                        "total_ms": retrieval_ms + generation_ms,
                        "retrieval_ms": retrieval_ms, "generation_ms": generation_ms,
                        "tokens_in": usage["prompt_tokens"], "tokens_out": usage["completion_tokens"],
                        "q": question, "trace_id": trace_id,
                    })
                except Exception:
                    pass

                try:
                    append_bench_cache({
                        "q": question, "answer": final_text,
                        "rougeL": round(rouge_l_approx("", final_text), 3),
                        "bleu": round(compute_bleu("", final_text), 3),
                        "evidence_score": round(compute_evidence_score(ctx_used, final_text), 3),
                        "provider": provider_name, "model": model_name,
                        "total_ms": retrieval_ms + generation_ms,
                        "retrieval_ms": retrieval_ms, "generation_ms": generation_ms,
                        "tokens_in": usage["prompt_tokens"], "tokens_out": usage["completion_tokens"],
                    })
                except Exception:
                    pass

                try:
                    cost = (usage["total_tokens"] / 1000) * (0.003)
                except Exception:
                    cost = None
                updated_history = chat_history + [
                    {"role": "user", "content": question_raw},
                    {"role": "assistant", "content": final_text}
                ]
                meta = {
                    "provider": provider_name, "model": model_name,
                    "usage": usage, "context_used": ctx_used, "sources": passages_used,
                    "source_buttons": source_buttons,
                    "cost_usd_estimate": round(cost, 6) if cost is not None else None,
                    "chat_history": updated_history,
                    "from_cache": False,
                }

                try:
                    await semantic_cache.set(question, {
                        "answer": final_text, "provider": provider_name, "model": model_name,
                        "sources": passages_used, "context_used": ctx_used, "usage": usage,
                    }, facts=facts)
                except Exception:
                    pass

                return resend, meta, final_text

            try:
                yield f"event: start\ndata: {trace_id}\n\n"
                if stream_early:
                    buf = ""
                    async for chunk in model_client.stream(sys_p, dev_p, question, ctx, chat_history=chat_history):
                        if not chunk:
                            continue
                        full_chunks.append(chunk)
                        buf += chunk
                        while "\n" in buf:
                            line, buf = buf.split("\n", 1)
                            line = line.strip()
                            if not line:
                                continue
                            line = cut_inline_evidence_tokens(line)
                            if not line:
                                continue
                            try:
                                safe = _sanitize_text_with_numbers(line, allowed_hosts, allowed_numbers)
                            except Exception:
                                safe = line
                            yield f"data: {safe}\n\n"
                            sent_parts.append(safe + "\n")
                    if buf.strip():
                        tail_line = cut_inline_evidence_tokens(buf.strip())
                        if tail_line:
                            try:
                                safe_tail = _sanitize_text_with_numbers(tail_line, allowed_hosts, allowed_numbers)
                            except Exception:
                                safe_tail = tail_line
                            yield f"data: {safe_tail}\n\n"
                            sent_parts.append(safe_tail + "\n")
                else:
                    async for chunk in model_client.stream(sys_p, dev_p, question, ctx, chat_history=chat_history):
                        if not chunk:
                            continue
                        full_chunks.append(chunk)

                did_finalize = True
                resend, meta, final_text_all = await _finalize_and_get_meta("".join(full_chunks))
                if resend:
                    for ln in resend.splitlines():
                        if ln.strip():
                            yield f"data: {ln}\n\n"
                yield f"event: done\ndata: {json.dumps(meta, ensure_ascii=False)}\n\n"
                yield "event: end\ndata: [DONE]\n\n"

            except (asyncio.CancelledError, GeneratorExit):
                try:
                    if not did_finalize:
                        await _finalize_and_get_meta("".join(full_chunks))
                except Exception:
                    pass
                raise
            except Exception as e:
                try:
                    yield f"event: error\ndata: {str(e)}\n\n"
                except Exception:
                    pass

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    # ------------------------ JSON ------------------------
    t_gen_s = time.monotonic()
    answer = await model_client.generate(sys_p, dev_p, question, ctx, chat_history=chat_history)
    answer = sanitize_body_text(answer)

    try:
        # 품질 보강 재검색(본문만)
        if need_web and not is_smalltalk:
            evd0 = compute_evidence_score(ctx, answer)

            webx = _safe_web_search(question_raw, cfg.WEB_MAX_RESULTS, log)
            if webx:
                initialx = [{"text": client_ctx, "source": "client", "title": "client"}] if client_ctx else []
                ctxx, usedx = build_context(
                    question_raw,
                    web_snippets=(initialx + webx),
                    top_k=payload.top_k,
                    max_context_chars=payload.max_context_chars,
                    min_k=cfg.NEWS_MIN_K,
                    prefer_web_first=True
                )
                ans2 = await model_client.generate(sys_p, dev_p, question, ctxx, chat_history=chat_history)
                ans2 = sanitize_body_text(ans2)
                if compute_evidence_score(ctxx, ans2) >= evd0:
                    answer, used_passages, ctx = ans2, usedx, ctxx

            # ✅ 본문 꼬리([출처]/[근거 요약])는 붙이지 않음
    except Exception:
        pass

    generation_ms = round((time.monotonic() - t_gen_s) * 1000, 1)
    allowed_hosts = _collect_allowed_hosts(used_passages, ctx or "", client_ctx)
    allowed_numbers = extract_numbers((ctx or "") + " " + " ".join([s.get("text", "") for s in used_passages]) + " " + client_ctx)
    answer = _sanitize_text_with_numbers(answer, allowed_hosts, allowed_numbers)

    # 버튼 생성
    try:
        source_buttons_final = _build_source_buttons(used_passages, desired_limit, public_only)
    except Exception:
        source_buttons_final = []

    usage = {
        "prompt_tokens": count_tokens(sys_p + dev_p + question + (ctx or ""), model_name),
        "completion_tokens": count_tokens(answer, model_name),
    }
    usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]

    try:
        _update_metrics(provider_name, usage["prompt_tokens"], usage["completion_tokens"], sim_hit=False)
        _push_recent({
            "provider": provider_name, "model": model_name,
            "total_ms": retrieval_ms + generation_ms,
            "retrieval_ms": retrieval_ms, "generation_ms": generation_ms,
            "tokens_in": usage["prompt_tokens"], "tokens_out": usage["completion_tokens"],
            "q": question, "trace_id": trace_id,
        })
    except Exception:
        pass

    try:
        append_bench_cache({
            "q": question, "answer": answer,
            "rougeL": round(rouge_l_approx("", answer), 3),
            "bleu": round(compute_bleu("", answer), 3),
            "evidence_score": round(compute_evidence_score(ctx, answer), 3),
            "provider": provider_name, "model": model_name,
            "total_ms": retrieval_ms + generation_ms,
            "retrieval_ms": retrieval_ms, "generation_ms": generation_ms,
            "tokens_in": usage["prompt_tokens"], "tokens_out": usage["completion_tokens"],
        })
    except Exception:
        pass

    try:
        await semantic_cache.set(question, {
            "answer": answer, "provider": provider_name, "model": model_name,
            "sources": used_passages, "context_used": ctx, "usage": usage,
        }, facts=facts)
    except Exception:
        pass

    updated_history = chat_history + [
        {"role": "user", "content": question_raw},
        {"role": "assistant", "content": answer}
    ]

    return JSONResponse(content={
        "trace_id": trace_id,
        "answer": answer,
        "context_used": ctx,
        "sources": used_passages,
        "source_buttons": source_buttons_final,
        "model": model_name,
        "provider": provider_name,
        "usage": usage,
        "chat_history": updated_history
    })

# ------------------------ Utils ------------------------
@router.get("/chat/health")
async def health():
    return {"ok": True}

@router.get("/metrics")
async def metrics():
    total = max(1, _METRICS["total_requests"])
    return {
        "total_requests": _METRICS["total_requests"],
        "simcache_hits": _METRICS["simcache_hits"],
        "hit_rate": round(_METRICS["simcache_hits"] / total, 4),
        "provider_counts": _METRICS["provider_counts"],
        "avg_prompt_tokens": round(_METRICS["total_prompt_tokens"] / total, 2),
        "avg_completion_tokens": round(_METRICS["total_completion_tokens"] / total, 2),
    }