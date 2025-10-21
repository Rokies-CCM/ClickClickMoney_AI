# server/main.py
import os
import time
import uuid
import logging
from typing import List
from collections import defaultdict
from fastapi import FastAPI, Request
from contextlib import asynccontextmanager
from server.deps import get_http_client
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import ORJSONResponse, JSONResponse
from core.metrics import get_bench_results
from server.routers import chat
from server.routers import ingest
from server.routers.quiz import router as quiz_router

# Optional: core.config 사용 (없어도 동작하도록 ENV 우선)
try:
    from core.config import get_settings  # type: ignore
except Exception:  # pragma: no cover
    get_settings = None  # fallback


# ------------------------------------------------------------------------------
# App init
# ------------------------------------------------------------------------------
APP_TITLE = "High-Performance AI Chatbot"
APP_VERSION = "1.3.0"

@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    _ = get_http_client()
    try:
        yield
    finally:
    # shutdown
        try:
            client = get_http_client()
            await client.aclose()
        except Exception:
            pass

app = FastAPI(
    title=APP_TITLE,
    version=APP_VERSION,
    default_response_class=ORJSONResponse,  # 빠른 직렬화
    lifespan=lifespan,
)

# ------------------------------------------------------------------------------
# ENV / Settings
# ------------------------------------------------------------------------------
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

# CORS_ORIGINS="https://your.app,https://admin.your.app"
def _load_cors_origins() -> List[str]:
    env_val = os.getenv("CORS_ORIGINS", "").strip()
    if env_val:
        return [o.strip() for o in env_val.split(",") if o.strip()]
    # settings에 정의되어 있으면 사용
    if get_settings is not None:
        try:
            s = get_settings()
            cors = getattr(s, "CORS_ORIGINS", None)
            if isinstance(cors, (list, tuple)) and cors:
                return list(cors)
            if isinstance(cors, str) and cors.strip():
                return [x.strip() for x in cors.split(",")]
        except Exception:
            pass
    # 기본값: 개발 편의상 와일드카드, 운영에선 ENV로 제한 권장
    return ["*"]

CORS_ORIGINS = _load_cors_origins()

# ------------------------------------------------------------------------------
# Middleware
# ------------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1024)

logger = logging.getLogger("server")
logger.setLevel(logging.INFO)


@app.middleware("http")
async def add_request_context(request: Request, call_next):
    """
    - 모든 요청에 trace_id/X-Request-Id를 부여
    - 처리 시간 측정 → 응답 헤더 X-Process-Time(ms)
    """
    trace_id = request.headers.get("X-Request-Id") or str(uuid.uuid4())
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception as exc:
        # 예외는 전역 핸들러에서 처리되지만, 여기서도 최소한의 로그를 남김
        logger.exception(f"[{trace_id}] Unhandled exception during request: {request.url}")
        raise exc
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        try:
            response.headers["X-Request-Id"] = trace_id
            response.headers["X-Process-Time"] = f"{elapsed_ms:.1f}ms"
        except Exception:
            pass
    return response


# ------------------------------------------------------------------------------
# Exception Handler (보안 강화)
# ------------------------------------------------------------------------------
@app.exception_handler(Exception)
async def all_exception_handler(request: Request, exc: Exception):
    """
    운영 모드: 내부 메시지/스택 미노출, trace_id만 반환
    DEBUG=true: 간단 메시지 노출(개발 편의)
    """
    trace_id = request.headers.get("X-Request-Id") or str(uuid.uuid4())
    logger.exception(f"[{trace_id}] {request.method} {request.url} -> 500 INTERNAL ERROR")

    content = {
        "error": {
            "code": "INTERNAL",
            "message": str(exc) if DEBUG else "Internal server error",
            "trace_id": trace_id,
        }
    }
    return ORJSONResponse(status_code=500, content=content)


# ------------------------------------------------------------------------------
# Health Endpoint
# ------------------------------------------------------------------------------
@app.get("/v1/health")
async def health():
    """
    간단한 헬스체크: 상태/버전/서버시간(ms)
    """
    return ORJSONResponse(
        {
            "status": "ok",
            "version": APP_VERSION,
            "server_time_ms": int(time.time() * 1000),
        }
    )

# ----------------------------------------------------------------------
# Rate limiting / Body size limit (✅ 중복 제거)
# ----------------------------------------------------------------------
_RATE_BUCKET = defaultdict(lambda: {"tokens": 20, "ts": time.time()})
RATE_CAP = int(os.getenv("RATE_CAP", "20"))      # 분당 요청 수
RATE_REFILL = int(os.getenv("RATE_REFILL", "20"))
MAX_BODY_BYTES = int(os.getenv("MAX_BODY_BYTES", str(256 * 1024)))

@app.middleware("http")
async def rate_limit(request: Request, call_next):
    # ✅ 특정 경로는 rate limit 제외 (대시보드/헬스/메트릭)
    if request.method == "OPTIONS":
        return await call_next(request)
    if request.url.path.startswith("/v1/health"):
        return await call_next(request)
    if request.url.path.startswith("/v1/metrics"):
        return await call_next(request)
    if request.url.path.startswith("/v1/bench/recent"):
        return await call_next(request)

    ip = request.client.host if request.client else "unknown"
    slot = _RATE_BUCKET[ip]
    now = time.time()
    if now - slot["ts"] >= 60:
        slot["tokens"] = RATE_REFILL
        slot["ts"] = now
    if slot["tokens"] <= 0:
        return ORJSONResponse(status_code=429, content={
            "error": {"code": "RATE_LIMIT", "message": "Too many requests",
                      "trace_id": request.headers.get("X-Request-Id", "-")}
        })
    slot["tokens"] -= 1
    return await call_next(request)

@app.middleware("http")
async def limit_body_size(request: Request, call_next):
    try:
        length = int(request.headers.get("content-length", "0"))
        if length > MAX_BODY_BYTES:
            return ORJSONResponse(status_code=413, content={
                "error": {"code": "PAYLOAD_TOO_LARGE", "message": "Request body too large", "trace_id": request.headers.get("X-Request-Id","-")}
            })
    except Exception:
        pass
    return await call_next(request)

# ------------------------------------------------------------------------------
# Routers
# ------------------------------------------------------------------------------
app.include_router(chat.router, prefix="/v1")
app.include_router(ingest.router, prefix="/v1")
app.include_router(quiz_router, prefix="/v1")

# ----------------------------------------------------------------------
# 실시간 벤치 결과 (Dashboard용)
# ----------------------------------------------------------------------
@app.get("/v1/bench/recent")
async def bench_recent():
    """
    Dashboard가 실시간으로 가져갈 벤치마크 요약 + 결과
    """
    data = get_bench_results()
    return JSONResponse(content=data)
