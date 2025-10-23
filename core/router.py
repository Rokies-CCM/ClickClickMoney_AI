# core/router.py
from __future__ import annotations
from dataclasses import dataclass
from core.models import build_openai_client

@dataclass
class ModelChoice:
    name: str
    model: str
    client: any

def choose_model(question: str, ctx: str | None = None) -> ModelChoice:
    """
    GPT 단일 경로 라우팅:
    - 항상 OpenAI 클라이언트를 사용한다.
    - Groq 비활성화 정책(USE_GROQ=false)에 맞춰 고정.
    """
    oai = build_openai_client()
    return ModelChoice("openai", oai.model, oai)
