# core/config.py
from __future__ import annotations
import os
from functools import lru_cache
from typing import List, Optional

def _env_bool(key: str, default: bool = False) -> bool:
    val = os.getenv(key)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}

def _env_int(key: str, default: int) -> int:
    val = os.getenv(key)
    if val is None:
        return default
    try:
        return int(val)
    except Exception:
        return default

def _env_float(key: str, default: float) -> float:
    val = os.getenv(key)
    if val is None:
        return default
    try:
        return float(val)
    except Exception:
        return default

def _env_str(key: str, default: Optional[str] = None) -> Optional[str]:
    val = os.getenv(key)
    if val is None or val.strip() == "":
        return default
    return val

def _env_list(key: str, default: List[str] | None = None) -> List[str]:
    raw = os.getenv(key)
    if raw is None or raw.strip() == "":
        return list(default or [])
    return [tok.strip() for tok in raw.split(",") if tok.strip()]

class Settings:
    # ------------------------------------------------------------------
    # Provider Keys
    # ------------------------------------------------------------------
    GROQ_API_KEY: Optional[str] = _env_str("GROQ_API_KEY")
    OPENAI_API_KEY: Optional[str] = _env_str("OPENAI_API_KEY")
    UPSTAGE_API_KEY: Optional[str] = _env_str("UPSTAGE_API_KEY") or _env_str("Upstage_API_KEY")
    GOOGLE_API_KEY: Optional[str] = _env_str("GOOGLE_API_KEY")
    TAVILY_API_KEY: Optional[str] = _env_str("TAVILY_API_KEY")

    # ------------------------------------------------------------------
    # Global/Runtime
    # ------------------------------------------------------------------
    DEBUG: bool = _env_bool("DEBUG", False)
    CORS_ORIGINS: List[str] = _env_list("CORS_ORIGINS", default=["*"])

    # ✅ 근거 강제 여부(기본: 끔). false면 어떤 경우에도 "[근거 부족]" 문구를 출력하지 않음.
    EVIDENCE_ENFORCE: bool = _env_bool("EVIDENCE_ENFORCE", False)

    # ------------------------------------------------------------------
    # Vector / RAG
    # ------------------------------------------------------------------
    VECTOR_BACKEND: str = (_env_str("VECTOR_BACKEND", "faiss") or "faiss").lower()
    VECTOR_DB: str = _env_str("VECTOR_DB", "data/chroma") or "data/chroma"
    DOCS_DIR: str = _env_str("DOCS_DIR", "./data/docs") or "./data/docs"
    EMBEDDING_MODEL: str = _env_str("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2") or \
                           "sentence-transformers/all-MiniLM-L6-v2"
    CHUNK_SIZE: int = _env_int("CHUNK_SIZE", 1000)
    CHUNK_OVERLAP: int = _env_int("CHUNK_OVERLAP", 200)
    TOP_K: int = _env_int("TOP_K", 4)
    MAX_CONTEXT_CHARS: int = _env_int("MAX_CONTEXT_CHARS", 1200)

    # 검색 정밀도 (FAISS HNSW 등)
    FAISS_EF_SEARCH: int = _env_int("FAISS_EF_SEARCH", 160)

    # ------------------------------------------------------------------
    # Behavior / Output control
    # ------------------------------------------------------------------
    REPLY_MAX_TOKENS: int = _env_int("REPLY_MAX_TOKENS", 360)
    STOP_TOKENS: List[str] = _env_list("STOP_TOKENS", default=["출처:", "참고:", "References:", "Sources:"])

    # ------------------------------------------------------------------
    # Evidence / Web search thresholds
    # ------------------------------------------------------------------
    EVIDENCE_MIN: float = _env_float("EVIDENCE_MIN", 0.35)
    # 뉴스성일 때 상향 임계치(미설정 시 0.50 이상으로 자동 상향)
    EVIDENCE_MIN_NEWS: Optional[float] = _env_float("EVIDENCE_MIN_NEWS", None)  # type: ignore
    MAX_SOURCES: int = _env_int("MAX_SOURCES", 5)
    NEWS_MIN_K: int = _env_int("NEWS_MIN_K", 6)
    WEB_MAX_RESULTS: int = _env_int("WEB_MAX_RESULTS", 8)
    WEB_RETRIES: int = _env_int("WEB_RETRIES", 3)
    WEB_BACKOFF_SEC: float = _env_float("WEB_BACKOFF_SEC", 0.6)

    # ------------------------------------------------------------------
    # Feature switches
    # ------------------------------------------------------------------
    USE_TAVILY: bool = _env_bool("USE_TAVILY", False)
    USE_RERANK: bool = _env_bool("USE_RERANK", False)

    # ------------------------------------------------------------------
    # Tavily preferences
    # ------------------------------------------------------------------
    TAVILY_DOMAINS: List[str] = _env_list("TAVILY_DOMAINS", default=[])
    TAVILY_SEARCH_DEPTH: str = _env_str("TAVILY_SEARCH_DEPTH", "advanced") or "advanced"
    TAVILY_SNIPPET_CHARS: int = _env_int("TAVILY_SNIPPET_CHARS", 1500)

    # ------------------------------------------------------------------
    # Tracing / Observability
    # ------------------------------------------------------------------
    LANGSMITH_TRACING: bool = _env_bool("LANGSMITH_TRACING", False)
    LANGSMITH_ENDPOINT: Optional[str] = _env_str("LANGSMITH_ENDPOINT")
    LANGSMITH_API_KEY: Optional[str] = _env_str("LANGSMITH_API_KEY")
    LANGSMITH_PROJECT: Optional[str] = _env_str("LANGSMITH_PROJECT", "ai-chatbot") or "ai-chatbot"

    # ------------------------------------------------------------------
    # Routing candidates (optional)
    # ------------------------------------------------------------------
    GROQ_MODEL_UPSCALE: Optional[str] = _env_str("GROQ_MODEL_UPSCALE", "llama-3.1-8b-instant")
    GROQ_FALLBACK_MODEL: Optional[str] = _env_str("GROQ_FALLBACK_MODEL", "llama-3.1-8b-instant")

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """프로세스 생애주기 동안 단일 인스턴스 유지."""
    return Settings()
