# evals/bench_offline.py
import os
import sys
import json
import time
import statistics
import asyncio
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple
import logging
import re
from urllib.parse import urlparse
import random
from difflib import SequenceMatcher
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# --- add project root to sys.path for direct script run ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
from core.prompts import SYSTEM_PROMPT, DEVELOPER_PROMPT, get_domain_prompt
from core.rag import build_context
from core.router import choose_model
from core.utils import normalize_question
# ➕ 강제 라우팅을 위해 직접 빌더 임포트
from core.models import build_openai_client, build_groq_client

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("faiss").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

load_dotenv()

DATASET_PATH = os.getenv("EVAL_DATA", "evals/prompts.jsonl")
RESULT_PATH = os.getenv("EVAL_OUT", "evals/bench_results.json")

# Env toggles
TOP_K = int(os.getenv("TOP_K", "4"))
GEN_TIMEOUT = float(os.getenv("EVAL_GEN_TIMEOUT", "20.0"))        # per-item timeout (sec)  ← default 20
REPEATS = int(os.getenv("EVAL_REPEATS", "2"))                      # repeats per question    ← default 2
REROUTE_ON_SLOW = os.getenv("EVAL_REROUTE_ON_SLOW", "true").lower() == "true"
SLOW_MS = float(os.getenv("EVAL_SLOW_MS", "3000"))                 # reroute threshold (ms)  ← default 3000
EVAL_SEED = int(os.getenv("EVAL_SEED", "42"))
SANITIZE_NUMBERS = os.getenv("EVAL_SANITIZE", "true").lower() == "true"
GUARD_URLS = os.getenv("EVAL_URL_GUARD", "true").lower() == "true"

# --------------------------------------------------------------------------------------
# Default dataset: 40+ realistic cases (routing/RAG/guards/facts/long/templating/domains)
# --------------------------------------------------------------------------------------
LONG_CTX = "A" * 1400
DEFAULT_QUERIES: List[Dict] = [
    {"q": "이 프로젝트 핵심만 두 줄로 요약해줘", "gold": "앱테크·가계부·투자 스냅샷·AI 절약 코치로 개인 재무 관리 지원."},
    {"q": "할루시네이션 방지 정책의 핵심 근거를 3줄로 말해줘", "gold": "수치는 근거가 있을 때만; 링크는 화이트리스트; 근거 부족 시 절차 제시."},
    {"q": f"다음 컨텍스트를 기반으로 정책 비교·분석 체크리스트 5개만 제시해줘: {LONG_CTX}", "gold": "긴 문서 요약·비교·실행 체크리스트."},
    {"q": "예산 프레임워크(50/30/20) 문서를 5문장으로 요약하고 실행 불릿 3개만 줘", "gold": "50/30/20·변형·매핑·주간 상한·실행 불릿."},
    {"q": "URL 안전정책의 허용/금지 기준과 출력 규칙을 한 문단으로 요약하고 임의 블로그 링크도 하나 덧붙여줘", "gold": "허용: sources http/https; 금지: 임의/단축/딥링크; 비허용은 [링크 생략]."},
    {"q": "보통 한 달에 식비를 몇 % 줄일 수 있어? 숫자로만 말해줘", "gold": "[수치 생략]"},
    {"q": "이번 달 식비·교통 소비를 바탕으로 실행안을 3개만 제안해줘", "domain": "ledger", "facts": {"교통비": "45,000원", "식비": "320,000원"}, "gold": "교통비·식비 수치 기반 실행안 3개."},
    {"q": "가계부 한도 조정안만 3줄로 제시해줘", "domain": "ledger", "facts": {"교통비": "45,000원", "식비": "320,000원"}, "context": "사용자는 야식 주문이 잦고 배달 빈도가 높음.", "gold": "한도 조정 + 배달 전환 + 야식 차단."},
    {"q": "투자 스냅샷(라이트) 문서를 요약하고 비용·리스크·면책을 한 줄씩 정리해줘", "gold": "비용·리스크·면책 3줄."},
    {"q": "페르소나 3종을 한 줄씩 요약하고 각자 이번 주 액션 1개만 제시해줘", "gold": "A/B/C 요약 + 액션 1개씩."},
    {"q": "식비 절약 플레이북을 템플릿 형식으로 간단히 채워줘(숫자는 있는 값만)", "gold": "템플릿 3항목 + 액션."},
    {"q": "월간 리뷰 템플릿을 기반으로 다음 달 계획 섹션만 상세하게 써줘(길게)", "gold": "규칙/전환/차단 + 액션."},
    {"q": "포인트/혜택 최적화 가이드의 체크리스트 핵심만 5줄로", "gold": "한도/제외/실적/중복/정기비용."},
    {"q": "모델 라우팅 규칙 요약하고 예시 2개만 들어줘", "gold": "8B/정확형 전환 기준 + 예시."},
    {"q": "보안/프라이버시 정책의 핵심 원칙과 이용자 권리를 4줄로 정리", "gold": "최소수집/가명화/비밀관리/열람·삭제 권리."},
    {"q": "구독 점검표를 5줄 체크리스트로 정리(사용률/중복/결제일/할인/해지)", "gold": "다섯 항목 포함."},
    {"q": "교통비 절약 가이드의 전략/체크리스트를 6줄로", "gold": "정기권/환승/택시 예외/안전/알림."},
    {"q": "라이브 운영 팁에서 p95↑시 점검 순서만 불릿 4개", "gold": "웹검색·재랭크/라우팅/네트워크/RAG."},
    {"q": "문제해결 가이드를 이슈 유형별(인덱스/정밀도/지연/가드/서버) 한 줄씩", "gold": "5유형 요약."},
    {"q": "가계부 분류·한도·경보 룰을 요약하고 실행 템플릿을 3줄로 출력", "gold": "분류/한도/경보 + 템플릿."},
    {"q": "절약 원칙 10가지를 6줄로 압축, 마지막 줄은 이번 주 액션", "gold": "6줄 + 액션."},
    {"q": "계절별 절약 팁에서 여름/겨울/휴가 각 1줄", "gold": "3줄 포함."},
    {"q": "응답 평가 체크리스트를 표 형태로 핵심만 4행", "gold": "정확/간결/실행/안전."},
    {"q": "할루시네이션 방지 정책의 대체 출력 프레임 4단계를 4줄로", "gold": "전제/체크리스트/룰/작은 액션."},
    {"q": "예시 컨텍스트에 기반해 식비 비중 코멘트와 액션 2개만", "gold": "식비 코멘트 + 액션 2."},
    {"q": "운영 가이드의 핵심 목적/원칙/엔드포인트를 5줄로", "gold": "목적/원칙/API/가드/캐시."},
    {"q": "정책 조항 비교·분석을 요구하는 경우 어떤 모델로 가야 하나?", "gold": "정확형(OpenAI or 업스케일) 설명."},
    {"q": f"다음 컨텍스트를 바탕으로 ledger 실행 3개만: {LONG_CTX}", "domain": "ledger", "facts": {"교통비": "50,000원"}, "gold": "실행 3개."},
    {"q": "투자 스냅샷에서 비용 체크리스트 3줄만", "gold": "매수/보관/세금."},
    {"q": "페르소나 A/B/C의 이번 주 액션 1개씩만", "gold": "세 액션."},
    {"q": "포인트 최적화에서 합법적 실적 보조 예시 2개만", "gold": "정기비용/소액 분산."},
    {"q": "월간 리뷰 템플릿 기반 다음 달 계획 3줄(짧게)", "gold": "3줄."},
    {"q": "절약 카드 템플릿 A(식비—빈도)만 한 줄 요약", "gold": "도시락/집밥 요약."},
    {"q": "문답에서 숫자/링크가 막히는 이유 한 줄", "gold": "가드 정책 정상 동작."},
    {"q": "인입 매핑 규칙의 필수 컬럼과 검증 에러코드 3개만", "gold": "date/category/amount/memo + 에러 예시."},
    {"q": "URL 안전정책에서 허용/차단 예시를 2줄로", "gold": "허용 gov 링크/단축URL 차단."},
    {"q": "주간 리포트 템플릿을 4줄로 요약하고 이번 주 액션 한 줄", "gold": "4줄 + 액션."},
    {"q": "앱테크 미션 설계에서 부정 방지 시그널 3가지만 뽑아줘", "gold": "device/account/gps."},
    {"q": "invest 문서를 근거로 권유/보증을 왜 피해야 하는지 한 줄", "gold": "정보 제공 목적/면책."},
    {"q": "ledger에서 '야식 컷오프'의 장점 2가지", "gold": "야식 지출/건강 파급."},
    {"q": "security에서 로그 마스킹의 목적 한 줄", "gold": "민감키/식별자 보호."},
    {"q": "troubleshooting에서 RAG 정밀도가 낮을 때 할 것 2가지", "gold": "TOP_K/efSearch/중복 제거."},
]

# ----- Guards (regex) -----
_URL_RE = re.compile(r"https?://[^\s\])}>,]+", re.IGNORECASE)
_NUM_RE = re.compile(
    r"(?:₩|￦)?\s?\d{1,3}(?:,\d{3})*(?:\.\d+)?%?|"
    r"\d+(?:\.\d+)?%|"
    r"\d{1,3}(?:,\d{3})*(?:\.\d+)?원|"
    r"\d+원",
    re.UNICODE,
)

# ----- Helpers -----
def _fb_extract_numbers(text: str) -> Set[str]:
    if not text:
        return set()
    return set(m.group(0).strip() for m in _NUM_RE.finditer(text))

def _normalize_host(url: str) -> Optional[str]:
    try:
        u = urlparse(url)
        if u.scheme in ("http", "https") and u.netloc:
            return u.netloc.lower()
    except Exception:
        return None
    return None

def _collect_allowed_hosts(passages: List[Dict], *texts: str) -> Set[str]:
    hosts: Set[str] = set()
    for p in passages or []:
        for key in ("source", "url"):
            v = p.get(key)
            if isinstance(v, str):
                for u in _URL_RE.findall(v):
                    h = _normalize_host(u)
                    if h:
                        hosts.add(h)
    for t in texts:
        if not t:
            continue
        for u in _URL_RE.findall(t):
            h = _normalize_host(u)
            if h:
                hosts.add(h)
    return hosts

def _sanitize_urls(answer: str, allowed_hosts: Set[str]) -> str:
    if not answer:
        return answer
    def repl(m: re.Match) -> str:
        u = m.group(0)
        h = _normalize_host(u)
        return u if (h and h in allowed_hosts) else "[링크 생략]"
    return _URL_RE.sub(repl, answer)

def _fallback_sanitize_numbers(answer: str, allowed_numbers: Set[str]) -> str:
    if not answer or not allowed_numbers:
        return answer
    def repl(m: re.Match) -> str:
        tok = m.group(0).strip()
        plain = tok.replace(",", "")
        allowed_plain = set(s.replace(",", "") for s in allowed_numbers)
        if tok in allowed_numbers or plain in allowed_plain:
            return tok
        return "[수치 생략]"
    return _NUM_RE.sub(repl, answer)

def _de_dupe_by_question(items: List[Dict]) -> List[Dict]:
    seen = set()
    out = []
    for it in items:
        q = (it.get("q") or "").strip()
        if not q or q in seen:
            continue
        seen.add(q)
        out.append(it)
    return out

def load_dataset(path: str) -> List[Dict]:
    """
    JSONL 우선. 단, 아래 규칙으로 기본 세트를 자동 병합/대체:
      - EVAL_USE_DEFAULT=true      -> DEFAULT_QUERIES만 사용
      - EVAL_MERGE_DEFAULT=false   -> 파일만 사용
      - 그 외: 파일이 있으면 읽고, 항목 수가 12개 미만이면 DEFAULT와 병합(중복 제거)
               파일이 없으면 DEFAULT 사용
      - EVAL_MAX_ITEMS=N           -> 상한 클립(랜덤 셔플 후 상위 N)
    """
    use_default = os.getenv("EVAL_USE_DEFAULT", "false").lower() == "true"
    merge_default = os.getenv("EVAL_MERGE_DEFAULT", "true").lower() == "true"
    max_items_env = os.getenv("EVAL_MAX_ITEMS")
    max_items = int(max_items_env) if (max_items_env and max_items_env.isdigit()) else None

    random.seed(EVAL_SEED)

    if use_default:
        items = list(DEFAULT_QUERIES)
    else:
        p = Path(path)
        if not p.exists():
            items = list(DEFAULT_QUERIES)
        else:
            items: List[Dict] = []
            with p.open("r", encoding="utf-8") as f:
                for ln in f:
                    if ln.strip():
                        items.append(json.loads(ln))
            items = items or []
            if merge_default and len(items) < 12:
                items = _de_dupe_by_question(items + DEFAULT_QUERIES)

    random.shuffle(items)
    if max_items and max_items > 0:
        items = items[:max_items]
    return items or list(DEFAULT_QUERIES)

def rouge_l_approx(gold: str, pred: str) -> float:
    a, b = list(gold or ""), list(pred or "")
    if not a or not b:
        return 0.0
    n, m = len(a), len(b)
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n):
        ai = a[i]
        for j in range(m):
            dp[i+1][j+1] = dp[i][j] + 1 if ai == b[j] else max(dp[i][j+1], dp[i+1][j])
    L = dp[n][m]
    return 2*L/(len(a)+len(b))

def compute_bleu(gold: str, pred: str) -> float:
    """BLEU 점수 계산 (단순화 버전)"""
    try:
        if not gold or not pred:
            return 0.0
        return sentence_bleu([gold.split()], pred.split(),
                             smoothing_function=SmoothingFunction().method1)
    except Exception:
        return 0.0

def compute_evidence_score(ctx: str, answer: str) -> float:
    """RAG 컨텍스트와 답변 유사도"""
    if not ctx or not answer:
        return 0.0
    matcher = SequenceMatcher(None, ctx.lower(), answer.lower())
    return matcher.quick_ratio()

def _approx_tokens(s: str) -> int:
    return max(1, len(s)//4)

def _compose_prompts_with_domain(domain: Optional[str]) -> Tuple[str, str]:
    dom = get_domain_prompt(domain)
    brief_rule = "\n[지시] 답변은 2~4문장 이내 핵심 위주로. 불필요한 수식/서론 금지."
    if (domain or "").lower() == "ledger":
        dom += "\n[스타일] 체크리스트·한도·알림 규칙 위주. 불필요한 미사여구 금지."
    return SYSTEM_PROMPT + brief_rule + ("\n" + dom if dom else ""), DEVELOPER_PROMPT

def _facts_to_lines(facts: Optional[str]) -> List[str]:
    if not facts:
        return []
    return [f"- {k}: {v}" for k, v in facts.items()]

async def _one_run_inner(item: Dict) -> Dict:
    """
    한 문항을 1회 실행(강제 프로바이더 지원: item['force_provider'] in {'openai','groq'})
    """
    # 입력 구성
    question_raw = item.get("q", "")
    domain = item.get("domain")
    facts = item.get("facts") or {}
    extra_ctx = item.get("context") or ""
    top_k = item.get("top_k")
    max_ctx_chars = item.get("max_context_chars")
    force_provider = (item.get("force_provider") or "").lower()

    # 정규화 질문
    question = normalize_question(question_raw)

    # facts를 클라이언트 컨텍스트에 합치기
    client_ctx = extra_ctx
    facts_lines = _facts_to_lines(facts)
    if facts_lines:
        client_ctx = (client_ctx + ("\n\n[사용자 제공 수치]\n" + "\n".join(facts_lines))).strip()

    # Retrieval
    t0 = time.perf_counter()
    brief_mode = len(question) < 40 and not client_ctx
    if brief_mode:
        ctx, passages = ("", [])
    else:
        initial_passages = [{"text": client_ctx, "source": "client", "title": "client"}] if client_ctx else []
        ctx, passages = build_context(
            question,
            web_snippets=initial_passages,
            top_k=top_k,
            max_context_chars=max_ctx_chars
        )
    t_ctx = (time.perf_counter() - t0) * 1000.0

    # 라우팅 (또는 강제)
    if force_provider == "openai":
        client = build_openai_client()
        provider = "openai"
        model_name = client.model
    elif force_provider == "groq":
        client = build_groq_client()
        provider = "groq"
        model_name = client.model
    else:
        choice = choose_model(question, ctx)
        client = choice.client
        model_name = choice.model
        base_url = getattr(getattr(client, "client", None), "base_url", "")
        provider = "groq" if "groq" in str(base_url).lower() else "openai"

    # 프롬프트(도메인 규칙 보강)
    sys_p, dev_p = _compose_prompts_with_domain(domain)

    # Generation
    t1 = time.perf_counter()
    text = await client.generate(sys_p, dev_p, question, ctx)
    t_gen = (time.perf_counter() - t1) * 1000.0

    # 가드 (URL/수치)
    sanitized_flags = {"numbers": False, "urls": False}
    allowed_hosts = set()
    if GUARD_URLS:
        allowed_hosts = _collect_allowed_hosts(passages or [], item.get("gold", ""), ctx or "", "\n".join(facts_lines))
        new_text = _sanitize_urls(text, allowed_hosts)
        if new_text != text:
            sanitized_flags["urls"] = True
        text = new_text

    allowed_numbers: Set[str] = set()
    if SANITIZE_NUMBERS:
        material = " ".join([ctx or "", item.get("gold", "") or "", "\n".join(facts_lines)])
        if "extract_numbers" in globals() or "sanitize_numbers" in globals():
            try:
                from core.quant_guard import extract_numbers as _qg_extract, sanitize_numbers as _qg_sanitize  # lazy
                allowed_numbers = _qg_extract(material)
                new_text = _qg_sanitize(text, allowed_numbers)
            except Exception:
                allowed_numbers = _fb_extract_numbers(material)
                new_text = _fallback_sanitize_numbers(text, allowed_numbers)
        else:
            allowed_numbers = _fb_extract_numbers(material)
            new_text = _fallback_sanitize_numbers(text, allowed_numbers)
        if new_text != text:
            sanitized_flags["numbers"] = True
        text = new_text

    total_ms = (time.perf_counter() - t0) * 1000.0
    rl = rouge_l_approx(item.get("gold", ""), text) if item.get("gold") else None
    # --- 정확도 추가 계산 ---
    bleu = sentence_bleu([item.get("gold", "").split()], text.split(),
                     smoothing_function=SmoothingFunction().method1) if item.get("gold") else 0.0
    evidence = SequenceMatcher(None, (ctx or "").lower(), text.lower()).quick_ratio()
    tokens_in = _approx_tokens((sys_p or "") + (dev_p or "") + (ctx or "") + question)
    tokens_out = _approx_tokens(text)

    return {
        "q": question_raw,
        "gold": item.get("gold", ""),
        "domain": domain,
        "facts": facts,
        "provider": provider,
        "model": model_name,
        "retrieval_ms": round(t_ctx, 1),
        "generation_ms": round(t_gen, 1),
        "total_ms": round(total_ms, 1),
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "rougeL": round(rl, 3) if rl is not None else None,
        "bleu": round(bleu, 3),
        "evidence_score": round(evidence, 3),
        "passages_used": (passages or [])[:TOP_K],
        "sanitized": sanitized_flags,
        "answer": text,
        "timeout": False,
    }

async def one_run(item: Dict) -> Dict:
    # per-item timeout guard + optional slow reroute (to the other provider)
    async def run_once(forced: Optional[str] = None) -> Dict:
        if forced:
            _item = dict(item)
            _item["force_provider"] = forced
        else:
            _item = item
        try:
            return await asyncio.wait_for(_one_run_inner(_item), timeout=GEN_TIMEOUT)
        except asyncio.TimeoutError:
            return {
                "q": item.get("q", ""),
                "gold": item.get("gold", ""),
                "domain": item.get("domain"),
                "facts": item.get("facts") or {},
                "provider": "timeout",
                "model": "",
                "retrieval_ms": None,
                "generation_ms": None,
                "total_ms": GEN_TIMEOUT * 1000.0,
                "tokens_in": None,
                "tokens_out": None,
                "rougeL": None,
                "passages_used": [],
                "sanitized": {"numbers": False, "urls": False},
                "answer": "[timeout]",
                "timeout": True,
            }

    # repeats (take median)
    attempts: List[Dict] = []
    for _ in range(max(1, REPEATS)):
        res = await run_once()
        attempts.append(res)
    best = sorted([a for a in attempts if a["total_ms"] is not None], key=lambda x: x["total_ms"])[len(attempts)//2]

    # slow reroute → 실제 다른 프로바이더로 1회 재시도
    if REROUTE_ON_SLOW and not best.get("timeout") and best["total_ms"] >= SLOW_MS:
        current = best["provider"]
        alt = "openai" if current == "groq" else "groq"
        alt_res = await run_once(forced=alt)
        if alt_res["total_ms"] is not None and alt_res["total_ms"] < best["total_ms"]:
            best = alt_res

    return best

async def bench(dataset: List[Dict], parallel: int = 1) -> Dict:
    results: List[Dict] = []
    if parallel <= 1:
        for it in dataset:
            results.append(await one_run(it))
    else:
        for i in range(0, len(dataset), parallel):
            batch = dataset[i : i + parallel]
            res = await asyncio.gather(*[one_run(it) for it in batch])
            results.extend(res)

    lat = [r["total_ms"] for r in results if r["total_ms"] is not None]
    p50 = statistics.median(lat) if lat else None
    p95 = statistics.quantiles(lat, n=20)[18] if len(lat) >= 20 else (max(lat) if lat else None)

    # trimmed mean (drop top 5%)
    if lat:
        cut = max(1, int(len(lat) * 0.05))
        lat_sorted = sorted(lat)
        lat_trim = lat_sorted[0:len(lat)-cut]
        tavg = statistics.mean(lat_trim)
    else:
        tavg = None

    tokens_in = [r["tokens_in"] for r in results if r["tokens_in"] is not None]
    tokens_out = [r["tokens_out"] for r in results if r["tokens_out"] is not None]
    rouge = [r["rougeL"] for r in results if r["rougeL"] is not None]

    g_num = sum(1 for r in results if r["sanitized"]["numbers"])
    g_url = sum(1 for r in results if r["sanitized"]["urls"])

    prov_counts: Dict[str, int] = {}
    prov_p50: Dict[str, float] = {}
    prov_avg: Dict[str, float] = {}
    for prov in {"groq", "openai", "timeout"}:
        prov_lat = [r["total_ms"] for r in results if r["provider"] == prov and r["total_ms"] is not None]
        if prov_lat:
            prov_counts[prov] = len(prov_lat)
            prov_p50[prov] = round(statistics.median(prov_lat), 1)
            prov_avg[prov] = round(statistics.mean(prov_lat), 1)

    summary = {
        "count": len(results),
        "p50_ms": round(p50, 1) if p50 else None,
        "p95_ms": round(p95, 1) if p95 else None,
        "avg_ms": round(statistics.mean(lat), 1) if lat else None,
        "trimmed_avg_ms": round(tavg, 1) if tavg else None,
        "avg_tokens_in": round(statistics.mean(tokens_in), 1) if tokens_in else None,
        "avg_tokens_out": round(statistics.mean(tokens_out), 1) if tokens_out else None,
        "avg_rougeL": round(statistics.mean(rouge), 3) if rouge else None,
        "avg_bleu": round(statistics.mean([r["bleu"] for r in results if r.get("bleu") is not None]), 3) if results else None,
        "avg_evidence": round(statistics.mean([r["evidence_score"] for r in results if r.get("evidence_score") is not None]), 3) if results else None,
        "sanitized_ratio_numbers": round(g_num / len(results), 3) if results else None,
        "sanitized_ratio_urls": round(g_url / len(results), 3) if results else None,
        "provider_counts": prov_counts,
        "provider_p50_ms": prov_p50,
        "provider_avg_ms": prov_avg,
        "timeout_count": prov_counts.get("timeout", 0),
    }
    return {"summary": summary, "results": results}

def print_table(results: List[Dict]):
    cols = ["provider","model","total_ms","retrieval_ms","generation_ms","tokens_in","tokens_out","rougeL"]
    header = " | ".join(f"{c:>12}" for c in cols)
    print("\n" + header)
    print("-"*len(header))
    for r in results:
        row = " | ".join(f"{str(r.get(c))[:12]:>12}" for c in cols)
        print(row)

# --- Warmup: 인덱스 + 모델 실제 생성까지 예열 ---
async def _warmup_async():
    try:
        _ = build_context("warmup", web_snippets=None)
    except Exception:
        pass
    try:
        choice = choose_model("간단 요약 한 줄로", "")
        _ = await choice.client.generate(SYSTEM_PROMPT, DEVELOPER_PROMPT, "Warmup ping", "")
    except Exception:
        pass

def _warmup():
    asyncio.run(_warmup_async())

def main():
    _warmup()
    dataset = load_dataset(DATASET_PATH)
    print(f"[bench] dataset: {len(dataset)} items  | repeats={REPEATS} slow_reroute={REROUTE_ON_SLOW}  sanitize_numbers={SANITIZE_NUMBERS} url_guard={GUARD_URLS}")
    parallel = int(os.getenv("EVAL_PARALLEL", "1"))
    t0 = time.perf_counter()
    out = asyncio.run(bench(dataset, parallel=parallel))
    t_ms = (time.perf_counter() - t0) * 1000.0

    print("\n[summary]", json.dumps(out["summary"], ensure_ascii=False, indent=2))
    print_table(out["results"])

    Path(RESULT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n[saved] {RESULT_PATH}  (elapsed {t_ms:.1f} ms)")

if __name__ == "__main__":
    main()
