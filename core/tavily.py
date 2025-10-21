# core/tavily.py
from __future__ import annotations
from typing import List, Dict
from core.config import get_settings

try:
    from tavily import TavilyClient
except Exception:
    TavilyClient = None

def is_tavily_enabled() -> bool:
    cfg = get_settings()
    return bool(cfg.TAVILY_API_KEY) and TavilyClient is not None and cfg.USE_TAVILY

def _env_list_to_list(xs) -> List[str]:
    if not xs:
        return []
    return [t.strip() for t in xs if str(t).strip()]

def web_search_snippets(query: str, max_results: int = 2) -> List[Dict]:
    """
    Tavily 검색 결과를 컨텍스트용 스니펫으로 변환.
    - 스니펫 길이: cfg.TAVILY_SNIPPET_CHARS
    - 도메인 화이트리스트: cfg.TAVILY_DOMAINS
    - search_depth: cfg.TAVILY_SEARCH_DEPTH ("basic" | "advanced")
    - (가능 시) API의 answer 필드도 스니펫으로 선반영
    """
    if not is_tavily_enabled():
        return []

    cfg = get_settings()
    client = TavilyClient(api_key=cfg.TAVILY_API_KEY)
    snippet_chars = int(cfg.TAVILY_SNIPPET_CHARS or 900)
    include_domains = _env_list_to_list(cfg.TAVILY_DOMAINS)
    search_depth = (cfg.TAVILY_SEARCH_DEPTH or "basic").lower()

    # 클라이언트 버전 편차 대비
    res = None
    try:
        res = client.search(
            query=query,
            max_results=max_results,
            include_answer=True,  # ✅ 가능하면 answer 요청
            search_depth=search_depth,
            include_domains=include_domains or None,
            include_images=False,
        )
    except TypeError:
        try:
            res = client.search(query=query, max_results=max_results, include_answer=True)
        except TypeError:
            res = client.search(query=query, max_results=max_results, include_answer=False)
    except Exception:
        return []

    out: List[Dict] = []

    # ✅ answer가 있으면 최상단 스니펫으로 추가(링크는 없음)
    ans = (res or {}).get("answer")
    if isinstance(ans, str) and ans.strip():
        out.append({"text": ans[:snippet_chars], "source": "", "title": "tavily:answer"})

    for item in (res or {}).get("results", [])[:max_results]:
        snippet = (item.get("content") or "")[:snippet_chars]
        out.append({
            "text": snippet,
            "source": item.get("url", "") or "",
            "title": item.get("title", "") or ""
        })
    return out
