# core/answer.py
from __future__ import annotations
from typing import List, Dict, Optional
from urllib.parse import urlparse
from datetime import datetime
import re

# ---------- helpers ----------
def _host(u: str) -> str:
    try:
        netloc = urlparse(u).netloc.lower()
        if not netloc:
            return ""
        # strip port
        if ":" in netloc:
            netloc = netloc.split(":", 1)[0]
        # strip common prefix
        if netloc.startswith("www."):
            netloc = netloc[4:]
        return netloc
    except Exception:
        return ""

def _norm_title(t: str, max_len: int = 120) -> str:
    t = re.sub(r"\s+", " ", (t or "")).strip()
    if not t:
        return ""
    if len(t) > max_len:
        t = t[: max_len - 1].rstrip() + "…"
    return t

_DATE_FMTS = [
    "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M", "%Y.%m.%d %H:%M", "%Y/%m/%d %H:%M",
    "%Y-%m-%d", "%Y.%m.%d", "%Y/%m/%d",
]

def _norm_date_display(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    # try common formats
    for f in _DATE_FMTS:
        try:
            dt = datetime.strptime(s, f)
            return dt.strftime("%Y-%m-%d")
        except Exception:
            pass
    # 한국어 표기: 2025년 10월 21일
    m = re.search(r"(\d{4})\s*년\s*(\d{1,2})\s*월\s*(\d{1,2})\s*일", s)
    if m:
        try:
            y, mth, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
            return f"{y:04d}-{mth:02d}-{d:02d}"
        except Exception:
            return s
    # 느슨한 yyyy.mm.dd or yyyy-mm-dd
    m2 = re.search(r"(\d{4})[.\-\/](\d{1,2})[.\-\/](\d{1,2})", s)
    if m2:
        try:
            y, mth, d = int(m2.group(1)), int(m2.group(2)), int(m2.group(3))
            return f"{y:04d}-{mth:02d}-{d:02d}"
        except Exception:
            return s
    return s

def _format_source(idx: int, s: Dict) -> List[str]:
    title = _norm_title(s.get("title") or s.get("name") or s.get("url") or "")
    url = (s.get("url") or "").strip()
    host = _host(url)
    date = _norm_date_display(s.get("published_time") or s.get("date") or "")
    meta = ""
    if host:
        meta = f" — {host}"
    if date:
        meta += f" ({date})"
    line1 = f"{idx}. {title}{meta}".rstrip()
    if url:
        line2 = f"   {url}"
        return [line1, line2]
    return [line1]

def _dedup_bullets(bullets: Optional[List[str]]) -> List[str]:
    out: List[str] = []
    seen = set()
    for b in bullets or []:
        bb = re.sub(r"\s+", " ", (b or "").strip())
        if not bb:
            continue
        if bb in seen:
            continue
        seen.add(bb)
        # 보수적 마침표 보강
        if not bb.endswith((".", "!", "?", "…")):
            bb += "."
        out.append(bb)
    return out

# ---------- public renderers ----------
def render_answer(core_text: str, bullets: Optional[List[str]] = None, sources: Optional[List[Dict]] = None) -> str:
    """
    최종 응답 본문에 '근거 요약'은 붙이지 않고, 필요 시 '[출처]'만 붙인다.
    """
    core_text = (core_text or "").strip()
    parts: List[str] = []
    if core_text:
        parts.append(core_text)

    # 근거 요약(불릿) 섹션은 더 이상 출력하지 않음

    if sources:
        parts.append("\n[출처]")
        idx = 1
        for s in sources:
            lines = _format_source(idx, s or {})
            for ln in lines:
                parts.append(ln)
            idx += 1

    return "\n".join(parts).strip()

def render_tail(bullets: Optional[List[str]] = None, sources: Optional[List[Dict]] = None) -> str:
    """
    스트리밍/파이널 뒤에 붙는 꼬리 섹션에서 '근거 요약'은 제외하고,
    '[출처]'만 두 줄 개행 후에 붙인다.
    """
    parts: List[str] = []

    # 근거 요약(불릿) 섹션은 더 이상 출력하지 않음

    if sources:
        parts.append("\n\n[출처]")
        idx = 1
        for s in sources:
            lines = _format_source(idx, s or {})
            for ln in lines:
                parts.append(ln)
            idx += 1

    return "\n".join(parts)
