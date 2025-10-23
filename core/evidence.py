# core/evidence.py
from __future__ import annotations
import re
from typing import List, Tuple, Dict, Iterable
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode
from datetime import datetime

from core.config import get_settings

cfg = get_settings()

# ---- 질의 특성 판별 ----
NEWS_HINTS = re.compile(r"(최근|최신|오늘|어제|발표|속보|인상|인하|지표|통계|지수|요금|가격)", re.IGNORECASE)
ASK_SRC_RE = re.compile(r"(근거|출처|링크|evidence|source|references?)", re.IGNORECASE)
FACTY_RE   = re.compile(r"(기준금리|금리|물가|cpi|실업률|gdp|환율|주가|날씨|뉴스|속보)", re.IGNORECASE)
DATEY_RE = re.compile(r"(언제|시점|날짜|연월일)", re.IGNORECASE)

def requires_web(question: str) -> tuple[bool, bool]:
    q = (question or "").strip()
    news_like = bool(NEWS_HINTS.search(q))
    explicit  = bool(ASK_SRC_RE.search(q))
    facty     = bool(FACTY_RE.search(q))
    need = news_like or explicit or facty
    return need, news_like

def analyze_question(question: str) -> tuple[bool, bool, bool]:
    q = (question or "").strip()
    news_like = bool(NEWS_HINTS.search(q))
    explicit  = bool(ASK_SRC_RE.search(q))
    facty     = bool(FACTY_RE.search(q))
    datey = bool(DATEY_RE.search(q))
    need = news_like or explicit or facty or datey
    return need, news_like, explicit

# ---- URL/호스트 유틸 ----
_URL_RE = re.compile(r"https?://[^\s\])}>,]+", re.IGNORECASE)

def _host(u: str) -> str | None:
    try:
        p = urlparse(u)
        if p.scheme in ("http", "https") and p.netloc:
            return p.netloc.lower()
    except Exception:
        return None
    return None

def _is_public_host(host: str) -> bool:
    """공공/준공공 도메인 판별: .go.kr, .or.kr, 화이트리스트 포함"""
    if not host:
        return False
    if host.endswith(".go.kr") or host.endswith(".or.kr"):
        return True
    prefers = set(cfg.TAVILY_DOMAINS or [])
    return host in prefers

# ---- (하위호환) 단순 (title, url) 목록 ----
def normalize_citations_from_passages(passages: List[Dict], max_n: int | None = None) -> List[Tuple[str, str]]:
    prefers = set(cfg.TAVILY_DOMAINS or [])
    items: List[Tuple[str, str]] = []
    for p in passages or []:
        cand = (p.get("source") or "").strip()
        if not cand.startswith("http"):
            u2 = (p.get("url") or "").strip()
            cand = u2 if u2.startswith("http") else ""
        if cand:
            title = (p.get("title") or "").strip() or cand
            items.append((title, cand))
    seen = set()
    uniq: List[Tuple[str, str]] = []
    for t, u in items:
        if u in seen:
            continue
        uniq.append((t, u))
        seen.add(u)
    def _score(u: str) -> tuple[int, int]:
        return (0 if any(d in u for d in prefers) else 1, len(u))
    uniq.sort(key=lambda tu: _score(tu[1]))
    limit = max_n if max_n is not None else cfg.MAX_SOURCES
    return uniq[:limit]

# ---- 근거 요약(요약문 잡음 제거 + 보정) ----

# 포털/공지/뉴스 페이지 잡음 + 구조 텍스트 + 잔여 토큰
_NOISE_PATTS = [
    r"메뉴\s*바로가기", r"본문\s*바로가기", r"주요\s*메뉴", r"상단\s*메뉴", r"하단\s*메뉴",
    r"검색\s*메뉴", r"전체보기", r"바로가기", r"인쇄", r"스크랩", r"공유", r"구독",
    r"최신\s*뉴스", r"속보", r"관련\s*기사", r"기사\s*원문", r"기사\s*입력",
    r"보도자료\s*목록", r"저작권", r"무단전재", r"재배포", r"Copyright", r"All rights reserved",
    r"Read\s*more", r"Click\s*here", r"로그인", r"회원가입", r"앵커멘트", r"앵커",
    r"송고\s*\d{4}", r"승인\s*\d{4}", r"사진\s*=", r"영상\s*=", r"핫뉴스",
    r"본문\s*글자\s*크기", r"글자\s*크기", r"제보", r"원문보기", r"광고문의",
    # spaced English letters (e.g., L P G)
    r"(?:\b[A-Za-z]\b\s*){3,}",
    # spaced Korean letters (e.g., 전 철 료)
    r"(?:[가-힣]\s){2,}[가-힣]",
    # hash banners like ### ###
    r"#{3,}",
    # residual numeric tokens like '7 6 )' or '1 2 ]'
    r"\b\d+(?:\s+\d+){1,}\s*[\)\]]",
]
_NOISE_RE = re.compile("|".join(_NOISE_PATTS), re.IGNORECASE)

def _is_noise_sentence(s: str) -> bool:
    if _NOISE_RE.search(s):
        return True
    # 기호/숫자 비율이 과도하면 제거(문자 적고 비문자/숫자 비중 높을 때)
    letters = len(re.findall(r"[가-힣A-Za-z]", s))
    non_alnum = len(re.findall(r"[^가-힣A-Za-z0-9\s]", s))
    digits = len(re.findall(r"\d", s))
    total = max(1, len(s))
    if (non_alnum + digits) / total > 0.35 and letters < 10:
        return True
    return False

def _fix_sentence_punct_and_paren(s: str) -> str:
    """마침표/괄호 보정: 불필요 기호 제거, 문장부호 보강, 열림 괄호 단절 제거."""
    s = re.sub(r"\s+", " ", (s or "")).strip()
    # 앞쪽의 단독 닫힘 괄호 제거
    s = re.sub(r"^[\)\]\}]+", "", s)
    # 끝부분 열림 괄호로 끝나는 경우 제거
    s = re.sub(r"[\(\[\{][^)\]\}]*$", "", s)
    # 문장부호 없으면 마침표 보강
    if s and not s.endswith((".", "!", "?", "…")):
        s += "."
    return s

def summarize_passages(passages: List[Dict], max_chars: int = 220) -> str:
    """상위 몇 개 passage에서 핵심 2~3문장을 추출하되 포털 잡음 문구 제거 및 보정."""
    if not passages:
        return ""
    text_pool: List[str] = []
    for p in passages[:6]:
        t = (p.get("text") or "").strip()
        if t:
            text_pool.append(t)
    if not text_pool:
        return ""
    joined = " ".join(text_pool)[:2000]

    # 문장 단위 분할(국/영 혼합)
    sents = re.split(r"(?<=[\.\?\!])\s+|(?<=다\.)\s+|(?<=요\.)\s+", joined)

    # 우선 키워드가 포함된 문장 위주로 선별하되 잡음 제거
    KEY = re.compile(r"(\d|년|월|일|인상|인하|증가|감소|발표|동결|인상폭|하향|상향|가격|요금)")
    candidates: List[str] = []
    for s in sents:
        ss = s.strip()
        if len(ss) < 20:
            continue
        if _is_noise_sentence(ss):
            continue
        if KEY.search(ss):
            candidates.append(ss)
        if len(candidates) >= 3:
            break

    # 백업: 일반 문장 중 비잡음 2~3개
    if not candidates:
        candidates = [s.strip() for s in sents if len(s.strip()) > 20 and not _is_noise_sentence(s)]

    # 문장 보정 및 길이 제한
    cleaned = [_fix_sentence_punct_and_paren(c) for c in candidates[:3]]
    out = " ".join(cleaned)[:max_chars].strip()
    return out

# ---- 출처 정규화/중복제거/정렬 ----
DROP_PARAMS = {"utm_source","utm_medium","utm_campaign","utm_term","utm_content","gclid","fbclid","igshid","si"}

def _canonical_url(u: str) -> str:
    try:
        p = urlparse(u.strip())
        if not p.scheme or not p.netloc:
            return u.strip()
        scheme = "https"
        netloc = p.netloc.lower()
        path = re.sub(r"/+$", "", p.path or "")
        q = [(k, v) for k, v in parse_qsl(p.query, keep_blank_values=False) if k not in DROP_PARAMS]
        return urlunparse((scheme, netloc, path, "", urlencode(q), ""))
    except Exception:
        return u.strip()

def _norm_title(t: str) -> str:
    t = re.sub(r"\s+", " ", (t or "")).strip()
    return t.casefold()

def _parse_published_ts(s: str) -> int:
    """다양한 날짜 표기 파싱 → epoch(초). 실패 시 0."""
    if not s:
        return 0
    s = s.strip()
    fmts = [
        "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M", "%Y.%m.%d %H:%M", "%Y/%m/%d %H:%M",
        "%Y-%m-%d", "%Y.%m.%d", "%Y/%m/%d",
    ]
    for f in fmts:
        try:
            dt = datetime.strptime(s[:len(s)], f)
            return int(dt.timestamp())
        except Exception:
            pass
    # 한국어 표기: 2025년 10월 21일
    m = re.search(r"(\d{4})\s*년\s*(\d{1,2})\s*월\s*(\d{1,2})\s*일", s)
    if m:
        try:
            y, mth, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
            return int(datetime(y, mth, d).timestamp())
        except Exception:
            return 0
    # yyyy.mm.dd 또는 yyyy-mm-dd 유사 패턴 느슨 파싱
    m2 = re.search(r"(\d{4})[.\-\/](\d{1,2})[.\-\/](\d{1,2})", s)
    if m2:
        try:
            y, mth, d = int(m2.group(1)), int(m2.group(2)), int(m2.group(3))
            return int(datetime(y, mth, d).timestamp())
        except Exception:
            return 0
    return 0

def _extract_sources(passages: List[Dict]) -> List[Dict]:
    out: List[Dict] = []
    for p in passages or []:
        url = (p.get("source") or "").strip()
        if not url.startswith("http"):
            u2 = (p.get("url") or "").strip()
            url = u2 if u2.startswith("http") else ""
        if not url:
            continue
        site = _host(url) or ""
        out.append({
            "title": (p.get("title") or "").strip() or url,
            "url": url,
            "published_time": (p.get("published_time") or p.get("date") or "").strip(),
            "site": site,
            "score": p.get("score"),
        })
    return out

def _uniq_sources(items: Iterable[Dict], max_k: int, prefers: Iterable[str]) -> List[Dict]:
    prefer_set = set(prefers or [])
    seen = set()
    uniq: List[Dict] = []
    for it in items:
        url = _canonical_url(it.get("url", ""))
        title = it.get("title") or url
        key = (_norm_title(title), url)
        if key in seen:
            continue
        it = dict(it)
        it["url"] = url
        it["title"] = title
        uniq.append(it)
        seen.add(key)

    def _rank(s: Dict) -> tuple:
        site = (s.get("site") or "").lower()
        url = s.get("url", "")
        ts  = _parse_published_ts(s.get("published_time", ""))
        public_rank = 0 if _is_public_host(site) else 1
        prefer_rank = 0 if any(dom in url for dom in prefer_set) else 1
        # 정렬: 최신 우선(-ts) → 공공 도메인 우선 → 화이트리스트 선호 → URL 길이
        return (-ts, public_rank, prefer_rank, len(url))

    uniq.sort(key=_rank)
    return uniq[:max_k]

def _build_bullets(passages: List[Dict], max_chars: int = 220, max_bullets: int = 3) -> List[str]:
    raw = summarize_passages(passages, max_chars=max_chars)
    if not raw:
        return []
    # 말줄임/괄호 등 보정
    raw = re.sub(r"[\[\(]$", "", raw.strip())
    sents = re.split(r"(?<=[\.\!\?])\s+|(?<=다\.)\s+|(?<=요\.)\s+", raw)

    bullets: List[str] = []
    for s in sents:
        # 불릿 기호/잡음 제거
        s = re.sub(r"^\s*[\-\u2022·•\*]+\s*", "", s.strip())
        if not s or _is_noise_sentence(s):
            continue
        s = _fix_sentence_punct_and_paren(s)
        bullets.append(s)
        if len(bullets) >= max_bullets:
            break
    return bullets

def normalize_citations(passages: List[Dict], max_sources: int = 5) -> Tuple[List[str], List[Dict]]:
    """
    반환:
      - bullets: 근거 요약 문장 리스트(잡음 제거/보정)
      - clean_sources: 중복 제거된 출처(최신/공공/선호 도메인 우선 정렬)
    """
    sources = _extract_sources(passages)
    clean_sources = _uniq_sources(sources, max_k=max_sources, prefers=cfg.TAVILY_DOMAINS or [])
    bullets = _build_bullets(passages, max_chars=220, max_bullets=3)
    return bullets, clean_sources
