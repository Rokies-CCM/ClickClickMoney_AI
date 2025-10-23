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
from server.routers import chat, ingest
from server.routers.quiz import router as quiz_router
from server.routers.tips import router as tips_router  # ← 추가

try:
    from core.config import get_settings  # type: ignore
except Exception:  # pragma: no cover
    get_settings = None  # fallback

APP_TITLE = "High-Performance AI Chatbot"
APP_VERSION = "1.3.1"

@asynccontextmanager
async def lifespan(app: FastAPI):
    _ = get_http_client()
    try:
        yield
    finally:
        try:
            client = get_http_client()
            await client.aclose()
        except Exception:
            pass

app = FastAPI(
    title=APP_TITLE,
    version=APP_VERSION,
    default_response_class=ORJSONResponse,
    lifespan=lifespan,
)

# ----------------------------------------------------------------------
# ENV / CORS
# ----------------------------------------------------------------------
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

def _load_cors_origins() -> List[str]:
    env_val = os.getenv("CORS_ORIGINS", "").strip()
    if env_val:
        return [o.strip() for o in env_val.split(",") if o.strip()]
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
    return [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8080",
    ]

CORS_ORIGINS = _load_cors_origins()

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

# ----------------------------------------------------------------------
# Middlewares
# ----------------------------------------------------------------------
@app.middleware("http")
async def add_request_context(request: Request, call_next):
    trace_id = request.headers.get("X-Request-Id") or str(uuid.uuid4())
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception as exc:
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

@app.exception_handler(Exception)
async def all_exception_handler(request: Request, exc: Exception):
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

# ----------------------------------------------------------------------
# Health
# ----------------------------------------------------------------------
@app.get("/health")
async def health_root():
    return ORJSONResponse({"status": "ok", "version": APP_VERSION, "server_time_ms": int(time.time() * 1000)})

@app.get("/v1/health")
async def health_v1():
    return ORJSONResponse({"status": "ok", "version": APP_VERSION, "server_time_ms": int(time.time() * 1000)})

# ----------------------------------------------------------------------
# Rate limiting / Body size limit
# ----------------------------------------------------------------------
_RATE_BUCKET = defaultdict(lambda: {"tokens": 20, "ts": time.time()})
RATE_CAP = int(os.getenv("RATE_CAP", "20"))
RATE_REFILL = int(os.getenv("RATE_REFILL", "20"))
MAX_BODY_BYTES = int(os.getenv("MAX_BODY_BYTES", str(256 * 1024)))

def _is_exempt(path: str) -> bool:
    return (
        path.startswith("/health")
        or path.startswith("/v1/health")
        or path.startswith("/v1/metrics")
        or path.startswith("/v1/bench/recent")
        or path.startswith("/docs")
        or path.startswith("/openapi.json")
    )

@app.middleware("http")
async def rate_limit(request: Request, call_next):
    if request.method == "OPTIONS" or _is_exempt(request.url.path):
        return await call_next(request)
    ip = request.client.host if request.client else "unknown"
    slot = _RATE_BUCKET[ip]
    now = time.time()
    if now - slot["ts"] >= 60:
        slot["tokens"] = RATE_REFILL
        slot["ts"] = now
    if slot["tokens"] <= 0:
        return ORJSONResponse(
            status_code=429,
            content={"error": {"code": "RATE_LIMIT", "message": "Too many requests", "trace_id": request.headers.get("X-Request-Id", "-")}},
        )
    slot["tokens"] -= 1
    return await call_next(request)

@app.middleware("http")
async def limit_body_size(request: Request, call_next):
    try:
        length = int(request.headers.get("content-length", "0"))
        if length > MAX_BODY_BYTES:
            return ORJSONResponse(
                status_code=413,
                content={"error": {"code": "PAYLOAD_TOO_LARGE", "message": "Request body too large", "trace_id": request.headers.get("X-Request-Id", "-")}},
            )
    except Exception:
        pass
    return await call_next(request)

# ----------------------------------------------------------------------
# Routers
# ----------------------------------------------------------------------
app.include_router(chat.router,   prefix="/v1")
app.include_router(ingest.router, prefix="/v1")
app.include_router(quiz_router,   prefix="/v1")
app.include_router(tips_router,   prefix="/v1")  # /v1/tips 로 제공

# ----------------------------------------------------------------------
# Bench results
# ----------------------------------------------------------------------
@app.get("/v1/bench/recent")
async def bench_recent():
    data = get_bench_results()
    return JSONResponse(content=data)
