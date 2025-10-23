# server/deps.py
import os
import logging
import uuid
from typing import Optional

from dotenv import load_dotenv
import httpx

# ---------------------------------------------------------------------
# Load .env & LangSmith flags early
# ---------------------------------------------------------------------
load_dotenv()
os.environ.setdefault("LANGCHAIN_PROJECT", os.getenv("LANGSMITH_PROJECT", "highperf-chatbot"))
os.environ.setdefault("LANGCHAIN_TRACING_V2", os.getenv("LANGSMITH_TRACING", "false"))

# ---------------------------------------------------------------------
# Centralized settings from core.config
# ---------------------------------------------------------------------
try:
    from core.config import get_settings  # single source of truth
except Exception as _e:  # pragma: no cover
    get_settings = None

# Optional Redis import (no hard dependency)
try:
    import redis.asyncio as aioredis  # type: ignore
except Exception:
    aioredis = None  # gracefully handle absence

# ---------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------
logger = logging.getLogger("ai-chatbot")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

# ---------------------------------------------------------------------
# Per-process settings snapshot
#   - Provider/API keys 등은 core.config에서 가져옴
#   - Base URL/timeout/redis_url 등 런타임 튜닝 파라미터는 ENV 직독
# ---------------------------------------------------------------------
_core = get_settings() if get_settings else None

OPENAI_API_KEY = getattr(_core, "OPENAI_API_KEY", None) or os.getenv("OPENAI_API_KEY", "")
GROQ_API_KEY = getattr(_core, "GROQ_API_KEY", None) or os.getenv("GROQ_API_KEY", "")
TAVILY_API_KEY = getattr(_core, "TAVILY_API_KEY", None) or os.getenv("TAVILY_API_KEY", "")
UPSTAGE_API_KEY = getattr(_core, "UPSTAGE_API_KEY", None) or os.getenv("UPSTAGE_API_KEY", "")

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

REQUEST_TIMEOUT_SEC = float(os.getenv("HTTPX_REQUEST_TIMEOUT_SEC", "60.0"))
CONNECT_TIMEOUT_SEC = float(os.getenv("HTTPX_CONNECT_TIMEOUT_SEC", "5.0"))

# ---------------------------------------------------------------------
# Shared HTTP client (HTTP/2 + keep-alive)
# ---------------------------------------------------------------------
_client: Optional[httpx.AsyncClient] = None

def get_http_client() -> httpx.AsyncClient:
    """
    재사용 가능한 AsyncClient.
    HTTP/2 활성화, keep-alive, 여유있는 타임아웃으로 스트리밍 지연 최소화.
    """
    global _client
    if _client is None or _client.is_closed:
        limits = httpx.Limits(max_keepalive_connections=100, max_connections=200)
        timeout = httpx.Timeout(
            connect=CONNECT_TIMEOUT_SEC,
            read=REQUEST_TIMEOUT_SEC,
            write=REQUEST_TIMEOUT_SEC,
            pool=5.0,
        )
        _client = httpx.AsyncClient(
            timeout=timeout,
            limits=limits,
            follow_redirects=True,
            http2=True,
        )
    return _client

# ---------------------------------------------------------------------
# Optional Redis getter (raise only when caller truly needs Redis)
# ---------------------------------------------------------------------
_redis = None

def get_redis():
    """
    Returns an aioredis.Redis if redis package is installed.
    If not installed, raise RuntimeError. Callers are expected to handle this.
    (core/cache.py should fall back to in-memory when this raises.)
    """
    global _redis
    if aioredis is None:
        raise RuntimeError("Redis client not available (redis package not installed).")
    if _redis is None:
        _redis = aioredis.from_url(REDIS_URL, encoding="utf-8", decode_responses=True)
    return _redis

# ---------------------------------------------------------------------
# Trace logger helpers
# ---------------------------------------------------------------------
class TraceLoggerAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        trace_id = self.extra.get("trace_id", "-")
        return f"[{trace_id}] {msg}", kwargs

def get_trace_logger(trace_id: str) -> TraceLoggerAdapter:
    return TraceLoggerAdapter(logger, {"trace_id": trace_id})

def new_trace_id() -> str:
    return uuid.uuid4().hex[:12]
